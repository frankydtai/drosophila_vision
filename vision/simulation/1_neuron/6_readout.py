# -*- coding: utf-8 -*-
"""Pack cost-trace selection: which nodes and which time samples from absolute ``v``.

Owns the time-axis gather shared by the continuous moving-bar cost_window
(``cost_t0s``) and the plain post-onset spot pack gather. Takes duck-typed
``pack`` objects and primitive tensors only -- it never imports
``Pack`` or the session/train layer, so ``neuron`` stays below
``train`` in the import graph.

Sparse time-point subsampling (``cost_time_indices``) is applied at cost time in
``train.cost`` on the post-onset ``ms_response`` window returned here
(``gts.shape[1]`` samples from ``pack_t_onset``; excludes spot ``ms_post``).
"""
from __future__ import annotations

import torch

from neuron.forward import forward_full, pack_t_onset


def pack_needs_waveform_mse(pack) -> bool:
    """Whether ``pack`` needs waveform MSE (``pack.waveform_mse``)."""
    return bool(pack.waveform_mse)


def window_time_traces(trace_full, b_idx, u_idx, t0, n_t, *, t_onset=0):
    """Extract per-entry time slices from ``trace_full`` ``(B, n_t, N)``.

    ``t0`` is the absolute simulation time of slice start (``t`` uses
    ``t0 + t``). ``n_t`` is how many time samples to gather. Samples with
    ``t0 + t < t_onset`` are zeroed (cost alignment).
    """
    n_t = int(n_t)
    dev = trace_full.device
    t = torch.arange(n_t, dtype=torch.long, device=dev)
    t_abs = t0[:, None].to(device=dev, dtype=torch.long) + t[None, :]
    t_safe = t_abs.clamp(0, trace_full.shape[1] - 1)
    v_readout = trace_full[b_idx[:, None], t_safe, u_idx[:, None]]
    return torch.where(
        t_abs < int(t_onset), torch.zeros_like(v_readout), v_readout,
    )


def pack_traces(trace_full, pack):
    """Select MSE traces for cost nodes; uses ``cost_t0s`` when set."""
    t0 = pack_t_onset(pack)
    n_t = int(pack.gts.shape[1])
    if pack.cost_t0s is None:
        return trace_full[pack.entry_bs, t0:t0 + n_t, pack.entry_nodes]
    return window_time_traces(
        trace_full, pack.entry_bs, pack.entry_nodes, pack.cost_t0s,
        n_t, t_onset=t0,
    )


def pack_cost_traces(params, pack, session, b=None):
    """Shared pack cost traces via ``forward_full`` (waveform MSE only when needed)."""
    i_sti = pack.i_sti if b is None else pack.i_sti[b:b + 1]
    trace_full = forward_full(session, params, i_sti, pack=pack)
    t0 = pack_t_onset(pack)
    n_t = int(pack.gts.shape[1])
    need_mse = pack_needs_waveform_mse(pack)
    if b is None:
        v_dsi = trace_full[pack.entry_bs, t0:t0 + n_t, pack.entry_nodes]
        if not need_mse:
            return None, v_dsi
        if pack.cost_t0s is None:
            return v_dsi, v_dsi
        return pack_traces(trace_full, pack), v_dsi
    mask = pack.entry_bs == int(b)
    u_m = pack.entry_nodes[mask]
    v_dsi = trace_full[0, t0:t0 + n_t, u_m].transpose(0, 1)
    if not need_mse:
        return None, v_dsi
    if pack.cost_t0s is None:
        return v_dsi, v_dsi
    return window_time_traces(
        trace_full, torch.zeros_like(u_m), u_m, pack.cost_t0s[mask],
        n_t, t_onset=t0,
    ), v_dsi


# Model → pack cost traces (both share ``pack_cost_traces``; b_axis in train.cost).
CA_PACK_COST_TRACES = {"borst": pack_cost_traces, "hp_lp": pack_cost_traces}
