# -*- coding: utf-8 -*-
"""Connectivity backend for the medulla simulation.

Interface (:class:`ScatterConn`):

    conn.exc_inh_g(v_out, syn_strength) -> (g_exc, g_inh)
    conn.signed_g(v_out, syn_strength)  -> g_signed
    conn.n_nodes
    conn.node_cells

``v_out`` is the presynaptic output already multiplied by the per-source a_out, i.e.
``relu(v − v_th)·a_out`` (borst / hp_lp). The post-synaptic input
``a_in`` is applied by the caller AFTER these calls.

``syn_strength`` multiplies each edge: length ``n_pairs`` type→type
(``syn_strength_cell``, ``--syn-mode per_cell``) or length ``n_edges`` per-edge
(``syn_strength_edge``, ``--syn-mode per_edge``).

Operates on the LAST axis (the nodes), so a plain 1-D ``(N,)`` state
and a ``(n_b, N)`` state work without change in the caller.
"""
from __future__ import annotations

import numpy as np
import torch


def _as_long(t, device) -> torch.Tensor:
    return torch.as_tensor(t, dtype=torch.long, device=device)


def build_cell_pair_indices(src_cell, tar_cell, n_cells: int):
    """Unique directed ``(source_cell, target_cell)`` codes → per-edge pair index.

    Returns
    -------
    pair_indices : (E,) int64
    n_pairs : int
    pairs : list[(src_cell, tar_cell)] in index order
    """
    src_cell = np.asarray(src_cell, dtype=np.int64)
    tar_cell = np.asarray(tar_cell, dtype=np.int64)
    codes = src_cell * int(n_cells) + tar_cell
    uniq, inv = np.unique(codes, return_inverse=True)
    pairs = [(int(code // n_cells), int(code % n_cells)) for code in uniq]
    return inv.astype(np.int64), int(len(uniq)), pairs


class ScatterConn:
    """Edge-list connectivity backend (connectome sub-graph or full graph).

    Built from parallel arrays describing directed synaptic edges ``source ->
    target`` with a signed weight ``edge_weights`` (``syn_sign * n_syn`` for per_cell,
    ``syn_sign`` for per_edge). Excitatory and inhibitory ``g`` are accumulated with
    ``scatter_add_`` onto target nodes. ``syn_strength`` is either type-pair
    ``syn_strength_cell[pair_indices[e]]`` or per-edge ``syn_strength_edge[e]``.
    """

    def __init__(
        self,
        source_indices,
        target_indices,
        edge_weights,
        n_nodes: int,
        node_cells,
        *,
        dtype: torch.dtype,
        a_syn_exc: float = 1.0,
        a_syn_inh: float = 1.0,
        device: str | None = None,
    ) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.n_nodes = int(n_nodes)
        self.source_indices = _as_long(source_indices, device)
        self.target_indices = _as_long(target_indices, device)
        self.node_cells = _as_long(node_cells, device)
        self.n_edges = int(self.source_indices.numel())

        edge_weights = torch.as_tensor(edge_weights, dtype=dtype, device=device)
        self.w_exc = edge_weights.clamp(min=0) * a_syn_exc
        self.w_inh = (-edge_weights).clamp(min=0) * a_syn_inh
        self.w_signed = self.w_exc - self.w_inh

        n_cells = int(self.node_cells.max().item()) + 1 if self.n_nodes else 0
        src_t = self.node_cells[self.source_indices].detach().cpu().numpy()
        tar_t = self.node_cells[self.target_indices].detach().cpu().numpy()
        pair_indices_np, n_pairs, pairs = build_cell_pair_indices(src_t, tar_t, n_cells)
        self.pair_indices = torch.as_tensor(pair_indices_np, dtype=torch.long, device=device)
        self.n_pairs = int(n_pairs)
        self.pairs = pairs

    def _accumulate_on_target(self, vals: torch.Tensor) -> torch.Tensor:
        out_shape = vals.shape[:-1] + (self.n_nodes,)
        out = torch.zeros(out_shape, dtype=vals.dtype, device=vals.device)
        target_indices_expanded = self.target_indices.expand(vals.shape)
        out.scatter_add_(-1, target_indices_expanded, vals)
        return out

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self.source_indices)

    def _edge_syn_strength(self, syn_strength: torch.Tensor) -> torch.Tensor:
        n = int(syn_strength.shape[-1])
        if n == self.n_edges:
            return syn_strength
        if n == self.n_pairs:
            return syn_strength.index_select(-1, self.pair_indices)
        raise ValueError(
            f"syn_strength length {n} != n_edges {self.n_edges} "
            f"or n_pairs {self.n_pairs}"
        )

    def _pre_edges(self, v_out: torch.Tensor, syn_strength: torch.Tensor):
        return self._gather(v_out), self._edge_syn_strength(syn_strength)

    def exc_inh_g(
        self, v_out: torch.Tensor, syn_strength: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, syn_strength = self._pre_edges(v_out, syn_strength)
        return (
            self._accumulate_on_target(xs * self.w_exc * syn_strength),
            self._accumulate_on_target(xs * self.w_inh * syn_strength),
        )

    def signed_g(self, v_out: torch.Tensor, syn_strength: torch.Tensor) -> torch.Tensor:
        xs, syn_strength = self._pre_edges(v_out, syn_strength)
        return self._accumulate_on_target(xs * self.w_signed * syn_strength)
