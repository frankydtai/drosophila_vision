# -*- coding: utf-8 -*-
"""Pack readout selection: which model trace, which units, which time samples.

Owns the time-axis gather shared by the continuous moving-bar window
(``cost_t0``) and the plain post-onset spot readout, plus the ca/v dispatch
(``readout_kind``). Takes duck-typed ``pack`` objects and primitive tensors only
-- it never imports ``TargetPack`` or the session/training layer, so
``neuron_model`` stays below ``training`` in the import graph.

Sparse time-point subsampling (``cost_time_ix``) is applied at cost time in
``training.cost`` on the full post-onset trace returned here, so the
``t_on = signal.shape[1] - data.shape[1]`` convention stays intact.
"""
from __future__ import annotations

import torch

from neuron_model.forward import run_full


def readout_kind(pack) -> str:
    """'ca' (default) or 'v' delta-Vm readout for this pack."""
    return getattr(pack, "readout_kind", None) or "ca"


def pack_trace_full(session, p, sig, pack):
    """Full ``(B, T, N)`` model trace: v_delta for ``readout_kind='v'`` else Ca."""
    if readout_kind(pack) == "v":
        return run_full(session, p, sig, pack=pack, return_v_delta=True)
    return run_full(session, p, sig, pack=pack)


def pack_needs_waveform_mse(pack) -> bool:
    """Spot always; moving-bar only when cost-window targets were built.

    Encoded on the pack (``always_waveform_mse``) so ``neuron_model`` needs no
    knowledge of which paradigm names are moving bars.
    """
    if getattr(pack, "always_waveform_mse", True):
        return True
    return pack.cost_t0 is not None


def window_time_traces(trace_full, b_idx, u_idx, t0, win=None, *, t_on=0):
    """Extract per-readout windows from ``trace_full`` ``(B, maxtime, N)``.

    ``t0`` is the absolute simulation step of window start (slot ``k`` uses
    ``t0 + k``). Slots with ``t0 + k < t_on`` are zeroed (cost alignment).
    """
    if win is None:
        raise ValueError("window length win required")
    win = int(win)
    dev = trace_full.device
    k = torch.arange(win, dtype=torch.long, device=dev)
    t_idx = t0[:, None].to(device=dev, dtype=torch.long) + k[None, :]
    t_max = trace_full.shape[1] - 1
    t_safe = t_idx.clamp(0, t_max)
    sel = trace_full[b_idx[:, None], t_safe, u_idx[:, None]]
    pre = t_idx < int(t_on)
    return torch.where(pre, torch.zeros_like(sel), sel)


def readout_pack_traces(trace_full, pack):
    """Select MSE traces for cost cells; windowed when ``pack.cost_t0`` is set."""
    pack_t_on = int(pack.signal.shape[1] - pack.data.shape[1])
    if pack.cost_t0 is None:
        return trace_full[pack.readout_batch, pack_t_on:, pack.readout_unit]
    return window_time_traces(
        trace_full, pack.readout_batch, pack.readout_unit, pack.cost_t0,
        win=pack.data.shape[1], t_on=pack_t_on,
    )


def pack_readout(p, pack, session, batch_idx=None):
    """Shared pack readout via ``run_full`` (waveform MSE only when needed)."""
    sig = pack.signal if batch_idx is None else pack.signal[batch_idx:batch_idx + 1]
    trace_full = pack_trace_full(session, p, sig, pack)
    need_mse = pack_needs_waveform_mse(pack)
    t_on = int(pack.signal.shape[1] - pack.data.shape[1])
    if batch_idx is None:
        dsi_sel = trace_full[pack.readout_batch, t_on:, pack.readout_unit]
        if not need_mse:
            return None, dsi_sel
        return readout_pack_traces(trace_full, pack), dsi_sel
    mask = pack.readout_batch == int(batch_idx)
    u_m = pack.readout_unit[mask]
    dsi_sel = trace_full[0, t_on:, u_m].transpose(0, 1)
    if not need_mse:
        return None, dsi_sel
    if pack.cost_t0 is None:
        return dsi_sel, dsi_sel
    b_zero = torch.zeros_like(u_m)
    mse_sel = window_time_traces(
        trace_full, b_zero, u_m, pack.cost_t0[mask],
        win=pack.data.shape[1], t_on=t_on,
    )
    return mse_sel, dsi_sel


# Register pack readouts here -- batching stays in training cost.
CA_PACK_READOUTS = {
    "conductance": pack_readout,
    "hp_lp": pack_readout,
}
