# -*- coding: utf-8 -*-
"""Connectivity backends for the medulla simulation.

Interface:

    conn.exc_inh_drive(x, syn_strength) -> (g_exc, g_inh)  # network / ScatterConn
    conn.signed_drive(x, syn_strength)  -> g_signed
    conn.n_units
    conn.node_type

``x`` is the presynaptic output already scaled by the per-source out_gain, i.e.
``rectsyn(v, v_th) * out_gain`` for the borst model or
``relu(activity) * out_gain`` for hp_lp. The post-synaptic input
gain (``in_gain``) is applied by the caller AFTER these calls.

Synaptic scaling multiplies each edge: length ``n_pairs`` type→type α
(``syn_strength``, ``--syn-mode type_pair``) or length ``n_edges`` per-edge
magnitude (``edge_weight``, ``--syn-mode per_edge``). Network path only
(:class:`ScatterConn`).

Both backends operate on the LAST axis (the units), so a plain 1-D ``(N,)`` state
and a batched ``(B, N)`` state work without change in the caller.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


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


class ScatterConn:
    """Edge-list connectivity backend (connectome sub-graph or full graph).

    Built from parallel arrays describing directed synaptic edges ``source ->
    target`` with a signed weight ``base_w`` (``sign * n_syn`` for type_pair,
    ``sign`` for per_edge). Excitatory and inhibitory drives are accumulated with
    ``scatter_add`` over the target index. Scaling is either type-pair
    ``syn_strength[pair_idx[e]]`` or per-edge ``edge_weight[e]``.
    """

    def __init__(
        self,
        src_idx,
        tar_idx,
        base_w,
        n_units: int,
        node_type,
        *,
        dtype: torch.dtype,
        exc_scale: float = 1.0,
        inh_scale: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.n_units = int(n_units)
        self.src_idx = _as_long(src_idx, device)
        self.tar_idx = _as_long(tar_idx, device)
        self.node_type = _as_long(node_type, device)
        self.n_edges = int(self.src_idx.numel())

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

    def _edge_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        n = int(alpha.shape[-1])
        if n == self.n_edges:
            return alpha
        if n == self.n_pairs:
            return alpha.index_select(-1, self.pair_idx)
        raise ValueError(
            f"synaptic scale length {n} != n_edges {self.n_edges} "
            f"or n_pairs {self.n_pairs}"
        )

    def exc_inh_drive(
        self, x: torch.Tensor, alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self._gather(x)
        a = self._edge_alpha(alpha)
        return self._scatter(xs * self.w_exc * a), self._scatter(xs * self.w_inh * a)

    def signed_drive(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        a = self._edge_alpha(alpha)
        return self._scatter(self._gather(x) * self.w_signed * a)
