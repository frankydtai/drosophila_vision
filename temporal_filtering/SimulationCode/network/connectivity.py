# -*- coding: utf-8 -*-
"""Connectivity backends for the medulla simulation.

Two interchangeable backends expose the SAME interface so the simulator core can
stay agnostic about whether it runs the historical 5-column dense matrix or an
arbitrary connectome read from an edge list:

    conn.exc_inh_drive(x) -> (g_exc, g_inh)   # both >= 0, post-synaptic drive
    conn.signed_drive(x)  -> g_signed         # signed current-based drive
    conn.n_units                              # number of cells (state width)
    conn.node_type                            # (n_units,) long: cell-type index

``x`` is the presynaptic output already scaled by the per-source out_gain, i.e.
``rectsyn(Vm, trld) * out_gain`` for the conductance model or
``relu(activity) * out_gain`` for the adaptive model. The post-synaptic input
gain (``inp_gain``) is applied by the caller AFTER these calls, exactly as in the
original ``torch.mv(M_exc, ...)`` code.

Both backends operate on the LAST axis (the units), so a plain 1-D ``(N,)`` state
and a batched ``(B, N)`` state (multi-column / multi-shift training) work without
any change in the caller.

  - :class:`DenseConn` wraps the existing ``multi_colM``-derived matrices and is
    *bit-identical* to the original ``torch.mv`` path (``M @ x``).
  - :class:`ScatterConn` stores the connectome as an edge list and uses
    ``index_select`` + ``scatter_add`` (O(E) memory) so a 668-node sub-graph or a
    full 721-column graph trains without materialising an N x N matrix.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def _as_long(t, device) -> torch.Tensor:
    return torch.as_tensor(t, dtype=torch.long, device=device)


class DenseConn:
    """Dense connectivity backend (the historical 5-column ``multi_colM`` path).

    Stores the same three matrices the original core built and reproduces
    ``torch.mv(M, x)`` exactly. ``M`` is indexed ``[target, source]`` so the drive
    on unit ``i`` is ``sum_j M[i, j] * x[j]``; ``x @ M.T`` evaluates this for both
    a 1-D ``(N,)`` and a batched ``(B, N)`` ``x``.
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
        # x @ M.T == torch.mv(M, x) for 1-D x, and the batched generalisation for
        # (B, N) x. Kept as matmul so a single line covers both ranks.
        return torch.matmul(x, M.transpose(-1, -2))

    def exc_inh_drive(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._mv(self.M_exc, x), self._mv(self.M_inh, x)

    def signed_drive(self, x: torch.Tensor) -> torch.Tensor:
        return self._mv(self.M_signed, x)


class ScatterConn:
    """Edge-list connectivity backend (connectome sub-graph or full graph).

    Built from parallel arrays describing directed synaptic edges ``source ->
    target`` with a signed weight ``base_w = sign * n_syn``. Excitatory and
    inhibitory drives are accumulated with ``scatter_add`` over the target index,
    mirroring ``DenseConn``'s split (``M_exc`` uses ``exc_scale``, ``M_inh`` uses
    ``inh_scale``, both producing non-negative conductances; ``signed`` uses
    ``exc_scale`` like the original ``M_signed``).
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
        dtype: torch.dtype = torch.float64,
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
        # exc / inh drives are BOTH >= 0 (conductances); signed keeps its sign.
        self.w_exc = torch.where(pos, base_w, torch.zeros_like(base_w)) * exc_scale
        self.w_inh = torch.where(neg, -base_w, torch.zeros_like(base_w)) * inh_scale
        self.w_signed = base_w * exc_scale

    def _scatter(self, vals: torch.Tensor) -> torch.Tensor:
        # vals: (..., E) per-edge contributions -> (..., n_units) summed by target.
        out_shape = vals.shape[:-1] + (self.n_units,)
        out = torch.zeros(out_shape, dtype=vals.dtype, device=vals.device)
        idx = self.tar_idx.expand(vals.shape)
        out.scatter_add_(-1, idx, vals)
        return out

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_units) -> (..., E) presynaptic value on each edge's source.
        return x.index_select(-1, self.src_idx)

    def exc_inh_drive(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self._gather(x)
        return self._scatter(xs * self.w_exc), self._scatter(xs * self.w_inh)

    def signed_drive(self, x: torch.Tensor) -> torch.Tensor:
        return self._scatter(self._gather(x) * self.w_signed)
