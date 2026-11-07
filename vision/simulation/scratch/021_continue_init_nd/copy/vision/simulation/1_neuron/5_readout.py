# -*- coding: utf-8 -*-
"""Pack cost-trace selection: which nodes and which time samples from absolute ``v``.

Owns the time-axis gather shared by the continuous moving-bar cost_window
(``cost_t0s``) and the plain post-onset spot pack gather. Takes duck-typed
``pack`` objects and primitive tensors only -- it never imports
``Pack`` or the session/train layer, so ``neuron`` stays below
``train`` in the import graph.

Sparse time-point subsampling (``cost_ts``) is applied at cost time in
``train.cost`` on the post-onset ``ms_response`` window returned here
(``gts.shape[1]`` samples from ``pack_t_onset``; excludes spot ``ms_post``).
"""
from __future__ import annotations

import torch

from neuron.forward import pack_t_onset


def window_time_traces(trace, entry_bs, entry_nodes, cost_t0s, n_t, *, t_onset=0):
    """Extract per-entry time slices from ``trace`` ``(B, n_t, N)``.

    ``cost_t0s`` is each entry's window-start ``t``; sample ``k`` uses
    ``cost_t0s + k`` for ``k`` in ``0 .. n_t - 1``. Samples before ``t_onset``
    are zeroed (cost alignment).
    """
    device = trace.device
    ts = (
        cost_t0s[:, None].to(device=device, dtype=torch.long)
        + torch.arange(int(n_t), dtype=torch.long, device=device)[None, :]
    )
    v_readout = trace[
        entry_bs[:, None],
        ts.clamp(0, trace.shape[1] - 1),
        entry_nodes[:, None],
    ]
    return torch.where(ts < int(t_onset), torch.zeros_like(v_readout), v_readout)


def pack_cost_window_t_min(pack) -> int:
    """Lowest absolute sample retained by a pack's aligned cost windows.

    Most packs use onset-relative traces and therefore suppress samples before
    ``pack_t_onset``.  Moving-bar windows contain a real pre-stimulus baseline
    for every local hex onset, so those samples must be retained.
    """
    return (
        pack_t_onset(pack)
        if getattr(pack, "cost_zero_before_t_onset", True)
        else 0
    )


def pack_traces(trace, pack, *, b_offset=0):
    """Select MSE traces for cost nodes; uses ``cost_t0s`` when set."""
    entry_bs = pack.entry_bs if b_offset == 0 else pack.entry_bs + b_offset
    t_onset = pack_t_onset(pack)
    n_t = int(pack.gts.shape[1])
    if getattr(pack, "cost_t0s", None) is None:
        return trace[entry_bs, t_onset:t_onset + n_t, pack.entry_nodes]
    return window_time_traces(
        trace, entry_bs, pack.entry_nodes, pack.cost_t0s,
        n_t, t_onset=pack_cost_window_t_min(pack),
    )
