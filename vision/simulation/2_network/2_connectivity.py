# -*- coding: utf-8 -*-
"""Connectivity backend for the medulla simulation.

Interface (:class:`ScatterConn`):

    conn.exc_inh_drive(v_out, syn_strength) -> (g_exc, g_inh)
    conn.signed_drive(v_out, syn_strength)  -> g_signed
    conn.n_nodes
    conn.node_cell

``v_out`` is the presynaptic output already scaled by the per-source a_out, i.e.
``relu(v − v_th)·a_out`` (borst / hp_lp). The post-synaptic input
gain (``a_in``) is applied by the caller AFTER these calls.

Synaptic scaling multiplies each edge: length ``n_pairs`` type→type
(``syn_strength_cell``, ``--syn-mode per_cell``) or length ``n_edges`` per-edge
(``syn_strength_edge``, ``--syn-mode per_edge``).

Operates on the LAST axis (the nodes), so a plain 1-D ``(N,)`` state
and a batched ``(B, N)`` state work without change in the caller.
"""
from __future__ import annotations

import numpy as np
import torch


def _as_long(t, device) -> torch.Tensor:
    return torch.as_tensor(t, dtype=torch.long, device=device)


def build_cell_pair_index(src_cell, tar_cell, n_cells: int):
    """Unique directed ``(source_cell, target_cell)`` codes → per-edge pair index.

    Returns
    -------
    pair_index : (E,) int64
    n_pairs : int
    pair_keys : list[(src_cell, tar_cell)] in index order
    """
    src_cell = np.asarray(src_cell, dtype=np.int64)
    tar_cell = np.asarray(tar_cell, dtype=np.int64)
    codes = src_cell * int(n_cells) + tar_cell
    uniq, inv = np.unique(codes, return_inverse=True)
    pair_keys = [(int(c // n_cells), int(c % n_cells)) for c in uniq]
    return inv.astype(np.int64), int(len(uniq)), pair_keys


class ScatterConn:
    """Edge-list connectivity backend (connectome sub-graph or full graph).

    Built from parallel arrays describing directed synaptic edges ``source ->
    target`` with a signed weight ``edge_weight`` (``syn_sign * n_syn`` for per_cell,
    ``syn_sign`` for per_edge). Excitatory and inhibitory drives are accumulated with
    ``scatter_add`` over the target index. Scaling is either type-pair
    ``syn_strength_cell[pair_index[e]]`` or per-edge ``syn_strength_edge[e]``.
    """

    def __init__(
        self,
        source_index,
        target_index,
        edge_weight,
        n_nodes: int,
        node_cell,
        *,
        dtype: torch.dtype,
        syn_scale_exc: float = 1.0,
        syn_scale_inh: float = 1.0,
        device: str | None = None,
    ) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.n_nodes = int(n_nodes)
        self.source_index = _as_long(source_index, device)
        self.target_index = _as_long(target_index, device)
        self.node_cell = _as_long(node_cell, device)
        self.n_edges = int(self.source_index.numel())

        edge_weight = torch.as_tensor(edge_weight, dtype=dtype, device=device)
        self.w_exc = edge_weight.clamp(min=0) * syn_scale_exc
        self.w_inh = (-edge_weight).clamp(min=0) * syn_scale_inh
        self.w_signed = self.w_exc - self.w_inh

        n_cells = int(self.node_cell.max().item()) + 1 if self.n_nodes else 0
        src_t = self.node_cell[self.source_index].detach().cpu().numpy()
        tar_t = self.node_cell[self.target_index].detach().cpu().numpy()
        pair_index_np, n_pairs, pair_keys = build_cell_pair_index(src_t, tar_t, n_cells)
        self.pair_index = torch.as_tensor(pair_index_np, dtype=torch.long, device=device)
        self.n_pairs = int(n_pairs)
        self.pair_keys = pair_keys

    def _scatter(self, vals: torch.Tensor) -> torch.Tensor:
        out_shape = vals.shape[:-1] + (self.n_nodes,)
        out = torch.zeros(out_shape, dtype=vals.dtype, device=vals.device)
        target_index_expanded = self.target_index.expand(vals.shape)
        out.scatter_add_(-1, target_index_expanded, vals)
        return out

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self.source_index)

    def _edge_syn_strength(self, syn_strength: torch.Tensor) -> torch.Tensor:
        n = int(syn_strength.shape[-1])
        if n == self.n_edges:
            return syn_strength
        if n == self.n_pairs:
            return syn_strength.index_select(-1, self.pair_index)
        raise ValueError(
            f"synaptic scale length {n} != n_edges {self.n_edges} "
            f"or n_pairs {self.n_pairs}"
        )

    def _pre_edges(self, v_out: torch.Tensor, syn_strength: torch.Tensor):
        return self._gather(v_out), self._edge_syn_strength(syn_strength)

    def exc_inh_drive(
        self, v_out: torch.Tensor, syn_strength: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, syn_strength = self._pre_edges(v_out, syn_strength)
        return (
            self._scatter(xs * self.w_exc * syn_strength),
            self._scatter(xs * self.w_inh * syn_strength),
        )

    def signed_drive(self, v_out: torch.Tensor, syn_strength: torch.Tensor) -> torch.Tensor:
        xs, syn_strength = self._pre_edges(v_out, syn_strength)
        return self._scatter(xs * self.w_signed * syn_strength)
