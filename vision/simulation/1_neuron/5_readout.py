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


def pack_needs_waveform_mse(pack) -> bool:
    """Whether ``pack`` needs waveform MSE (``pack.waveform_mse``)."""
    return bool(pack.waveform_mse)


def window_time_traces(trace, bs, nodes, t0, n_t, *, t_onset=0):
    """Extract per-entry time slices from ``trace`` ``(B, n_t, N)``.

    ``t0`` is the absolute simulation time of slice start (``t`` uses
    ``t0 + t``). ``n_t`` is how many time samples to gather. Samples with
    ``t0 + t < t_onset`` are zeroed (cost alignment).
    """
    n_t = int(n_t)
    device = trace.device
    t = torch.arange(n_t, dtype=torch.long, device=device)
    t_abs = t0[:, None].to(device=device, dtype=torch.long) + t[None, :]
    t_safe = t_abs.clamp(0, trace.shape[1] - 1)
    v_readout = trace[bs[:, None], t_safe, nodes[:, None]]
    return torch.where(
        t_abs < int(t_onset), torch.zeros_like(v_readout), v_readout,
    )


def pack_traces(trace, pack):
    """Select MSE traces for cost nodes; uses ``cost_t0s`` when set."""
    t0 = pack_t_onset(pack)
    n_t = int(pack.gts.shape[1])
    if pack.cost_t0s is None:
        return trace[pack.entry_bs, t0:t0 + n_t, pack.entry_nodes]
    return window_time_traces(
        trace, pack.entry_bs, pack.entry_nodes, pack.cost_t0s,
        n_t, t_onset=t0,
    )
