# -*- coding: utf-8 -*-
"""Pack readout selection: which nodes and which time samples from absolute ``v``.

Owns the time-axis gather shared by the continuous moving-bar window
(``cost_t0``) and the plain post-onset spot readout. Takes duck-typed
``pack`` objects and primitive tensors only -- it never imports
``ReadoutPack`` or the session/training layer, so ``neuron`` stays below
``training`` in the import graph.

Sparse time-point subsampling (``cost_time_ix``) is applied at cost time in
``training.cost`` on the post-onset response window returned here
(``gt.shape[1]`` samples from ``pack_t_onset``; excludes spot ``ms_post``).
"""
from __future__ import annotations

import torch

from neuron.forward import forward_full, pack_t_onset


def pack_needs_waveform_mse(pack) -> bool:
    """Whether ``pack`` needs a waveform MSE readout (``pack.waveform_mse``)."""
    return bool(pack.waveform_mse)


def window_time_traces(trace_full, b_idx, u_idx, t0, win, *, t_onset=0):
    """Extract per-readout windows from ``trace_full`` ``(B, n_t, N)``.

    ``t0`` is the absolute simulation time of window start (slot ``k`` uses
    ``t0 + k``). Slots with ``t0 + k < t_onset`` are zeroed (cost alignment).
    """
    win = int(win)
    dev = trace_full.device
    k = torch.arange(win, dtype=torch.long, device=dev)
    t_idx = t0[:, None].to(device=dev, dtype=torch.long) + k[None, :]
    t_safe = t_idx.clamp(0, trace_full.shape[1] - 1)
    v_readout = trace_full[b_idx[:, None], t_safe, u_idx[:, None]]
    return torch.where(
        t_idx < int(t_onset), torch.zeros_like(v_readout), v_readout,
    )


def readout_pack_traces(trace_full, pack):
    """Select MSE traces for cost nodes; windowed when ``pack.cost_t0`` is set."""
    t0 = pack_t_onset(pack)
    win = int(pack.gt.shape[1])
    if pack.cost_t0 is None:
        return trace_full[pack.readout_batch, t0:t0 + win, pack.readout_node]
    return window_time_traces(
        trace_full, pack.readout_batch, pack.readout_node, pack.cost_t0,
        win, t_onset=t0,
    )


def pack_readout(p, pack, session, batch_idx=None):
    """Shared pack readout via ``forward_full`` (waveform MSE only when needed)."""
    i_sti = pack.i_sti if batch_idx is None else pack.i_sti[batch_idx:batch_idx + 1]
    trace_full = forward_full(session, p, i_sti, pack=pack)
    t0 = pack_t_onset(pack)
    win = int(pack.gt.shape[1])
    need_mse = pack_needs_waveform_mse(pack)
    if batch_idx is None:
        v_dsi = trace_full[pack.readout_batch, t0:t0 + win, pack.readout_node]
        if not need_mse:
            return None, v_dsi
        if pack.cost_t0 is None:
            return v_dsi, v_dsi
        return readout_pack_traces(trace_full, pack), v_dsi
    mask = pack.readout_batch == int(batch_idx)
    u_m = pack.readout_node[mask]
    v_dsi = trace_full[0, t0:t0 + win, u_m].transpose(0, 1)
    if not need_mse:
        return None, v_dsi
    if pack.cost_t0 is None:
        return v_dsi, v_dsi
    return window_time_traces(
        trace_full, torch.zeros_like(u_m), u_m, pack.cost_t0[mask],
        win, t_onset=t0,
    ), v_dsi


# Model → pack readout (both share ``pack_readout``; batching in training.cost).
CA_PACK_READOUTS = {"borst": pack_readout, "hp_lp": pack_readout}
