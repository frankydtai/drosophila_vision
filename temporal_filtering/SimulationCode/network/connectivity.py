# -*- coding: utf-8 -*-
"""Connectivity backends for the medulla simulation.

Interface:

    conn.exc_inh_drive(x, syn_strength) -> (g_exc, g_inh)  # network / ScatterConn
    conn.signed_drive(x)  -> g_signed
    conn.n_units
    conn.node_type

``x`` is the presynaptic output already scaled by the per-source out_gain, i.e.
``rectsyn(Vm, trld) * out_gain`` for the conductance model or
``relu(activity) * out_gain`` for the adaptive model. The post-synaptic input
gain (``in_gain``) is applied by the caller AFTER these calls.

Conductance type→type scaling ``syn_strength`` (length ``n_pairs``) multiplies
each edge as \(\alpha_{t_{\mathrm{src}},t_{\mathrm{tar}}}\), shared across columns.
Network path only (:class:`ScatterConn`).

Both backends operate on the LAST axis (the units), so a plain 1-D ``(N,)`` state
and a batched ``(B, N)`` state work without change in the caller.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from training_config import SIM_DTYPE_DEFAULT


def _as_long(t, device) -> torch.Tensor:
    return torch.as_tensor(t, dtype=torch.long, device=device)


def build_type_pair_index(src_type, tar_type, n_types: int):
    """Unique directed ``(source_type, target_type)`` codes → per-edge pair index.

    Returns
    -------
    pair_idx : (E,) int64
    n_pairs : int
    pair_keys : list[(src_type, tar_type)] in index order
    """
    src_type = np.asarray(src_type, dtype=np.int64)
    tar_type = np.asarray(tar_type, dtype=np.int64)
    codes = src_type * int(n_types) + tar_type
    uniq, inv = np.unique(codes, return_inverse=True)
    pair_keys = [(int(c // n_types), int(c % n_types)) for c in uniq]
    return inv.astype(np.int64), int(len(uniq)), pair_keys


class DenseConn:
    """Dense connectivity backend (historical 5-column ``multi_colM`` path).

    Conductance ``exc_inh_drive`` with type→type ``syn_strength`` is network-only;
    this class does not implement it.
    """

    def __init__(
        self,
        M_exc: torch.Tensor,
        M_inh: torch.Tensor,
        M_signed: torch.Tensor,
        node_type: torch.Tensor,
    ) -> None:
        self.M_exc = M_exc
        self.M_inh = M_inh
        self.M_signed = M_signed
        self.node_type = node_type.to(M_exc.device)
        self.n_units = M_exc.shape[0]

    @staticmethod
    def _mv(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, M.transpose(-1, -2))

    def exc_inh_drive(self, x: torch.Tensor, syn_strength: torch.Tensor):
        raise NotImplementedError(
            "conductance type-pair syn_strength requires network ScatterConn"
        )

    def signed_drive(self, x: torch.Tensor) -> torch.Tensor:
        return self._mv(self.M_signed, x)


class ScatterConn:
    """Edge-list connectivity backend (connectome sub-graph or full graph).

    Built from parallel arrays describing directed synaptic edges ``source ->
    target`` with a signed weight ``base_w = sign * n_syn``. Excitatory and
    inhibitory drives are accumulated with ``scatter_add`` over the target index.
    ``syn_strength[pair_idx[e]]`` scales edge ``e`` by its
    ``(source_type, target_type)`` group.
    """

    def __init__(
        self,
        src_idx,
        tar_idx,
        base_w,
        n_units: int,
        node_type,
        exc_scale: float = 1.0,
        inh_scale: float = 1.0,
        device: Optional[str] = None,
        dtype: torch.dtype = SIM_DTYPE_DEFAULT,
    ) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.n_units = int(n_units)
        self.src_idx = _as_long(src_idx, device)
        self.tar_idx = _as_long(tar_idx, device)
        self.node_type = _as_long(node_type, device)

        base_w = torch.as_tensor(base_w, dtype=dtype, device=device)
        pos = base_w > 0
        neg = base_w < 0
        self.w_exc = torch.where(pos, base_w, torch.zeros_like(base_w)) * exc_scale
        self.w_inh = torch.where(neg, -base_w, torch.zeros_like(base_w)) * inh_scale
        self.w_signed = base_w * exc_scale

        n_types = int(self.node_type.max().item()) + 1 if self.n_units else 0
        src_t = self.node_type[self.src_idx].detach().cpu().numpy()
        tar_t = self.node_type[self.tar_idx].detach().cpu().numpy()
        pair_idx_np, n_pairs, pair_keys = build_type_pair_index(src_t, tar_t, n_types)
        self.pair_idx = torch.as_tensor(pair_idx_np, dtype=torch.long, device=device)
        self.n_pairs = int(n_pairs)
        self.pair_keys = pair_keys

    def _scatter(self, vals: torch.Tensor) -> torch.Tensor:
        out_shape = vals.shape[:-1] + (self.n_units,)
        out = torch.zeros(out_shape, dtype=vals.dtype, device=vals.device)
        idx = self.tar_idx.expand(vals.shape)
        out.scatter_add_(-1, idx, vals)
        return out

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self.src_idx)

    def _edge_alpha(self, syn_strength: torch.Tensor) -> torch.Tensor:
        if syn_strength.shape[-1] != self.n_pairs:
            raise ValueError(
                f"syn_strength length {syn_strength.shape[-1]} != n_pairs {self.n_pairs}"
            )
        return syn_strength.index_select(-1, self.pair_idx)

    def exc_inh_drive(
        self, x: torch.Tensor, syn_strength: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self._gather(x)
        a = self._edge_alpha(syn_strength)
        return self._scatter(xs * self.w_exc * a), self._scatter(xs * self.w_inh * a)

    def signed_drive(self, x: torch.Tensor) -> torch.Tensor:
        return self._scatter(self._gather(x) * self.w_signed)
