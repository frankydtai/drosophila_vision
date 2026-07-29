# -*- coding: utf-8 -*-
"""Shared full-T Ca forward for all neuron models.

Per-model modules supply only ``prepare_signal`` / ``init_state`` / ``step``.
This module owns the time loop, ``t_on`` reference, and Ca readout contract.
"""
from __future__ import annotations

import torch

from neuron_model.constants import Ca_tau, deltat
from neuron_model import conductance as _conductance
from neuron_model import hp_lp as _hp_lp

# Per-model dynamics for ``run_full`` (prepare_signal / init_state / step only).
MODEL_DRIVERS = {
    "conductance": _conductance,
    "hp_lp": _hp_lp,
}


def ca_readout_step(ca, v, v_ref):
    """One Ca low-pass step on ``v - v_ref`` (shared by all models)."""
    return deltat / Ca_tau * (v - v_ref - ca) + ca


def run_full(session, p, sig, *, return_ref=False, return_v_delta=False, pack=None):
    """Shared full-T forward for every ``session.model``.

    Time index ``t`` is post-update at step ``t``. Membrane drive comes from
    ``MODEL_DRIVERS[model].prepare_signal`` / ``init_state`` / ``step``. Ca
    runs for all ``t``; the integrator resets at ``t_on`` (same contract for
    every model).

    Returns
    -------
    ca_full ``(B, T, N)``, or with ``return_v_delta``: ``v_delta`` / ``(v_delta,
    v_ref, v_full)``.
    """
    import FiveCol_MedSim_Pytorch as fc

    try:
        drv = MODEL_DRIVERS[session.model]
    except KeyError as exc:
        raise ValueError(
            f"no MODEL_DRIVERS entry for model={session.model!r}; "
            f"expected one of {tuple(MODEL_DRIVERS)}"
        ) from exc

    pack = pack or session.primary_pack
    x = drv.prepare_signal(session, p, sig, pack)
    B, t_end, _n = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    t_on = int(pack.signal.shape[1] - pack.data.shape[1])
    state, v = drv.init_state(session, p, B)
    v_rows = [v]
    for t in range(1, t_end):
        state, v = drv.step(state, v, p, x[:, t - 1], session)
        v_rows.append(v)
    v_full = torch.stack(v_rows, dim=1)
    v_ref = v_full[:, t_on - 1, :].clone()
    v_delta = v_full - v_ref.unsqueeze(1)

    ca_rows = [
        torch.zeros(
            (B, session.backend.n_units),
            dtype=session.sim_dtype,
            device=v_ref.device,
        ),
    ]
    ca = 0
    for t in range(1, t_end):
        if t == t_on:
            ca = 0
        ca = ca_readout_step(ca, v_full[:, t], v_ref)
        ca_rows.append(ca)
    ca_full = torch.stack(ca_rows, dim=1)

    if return_v_delta:
        if return_ref:
            return v_delta, v_ref, v_full
        return v_delta
    if return_ref:
        return ca_full, v_ref
    return ca_full


def run_units(
    session, p, neuron_index=None, return_ref=False, sig=None, pack=None,
    *, return_v_delta=False,
):
    """``run_full`` then index units; squeeze when ``sig`` is ``(T, N)``."""
    pack = pack or session.primary_pack
    if neuron_index is None:
        neuron_index = pack.readout_unit
    if sig is None:
        sig = session.pack_signal(pack)
    squeeze = sig.dim() == 2
    sig_b = sig.unsqueeze(0) if squeeze else sig
    if return_v_delta:
        out, v_ref, _v_full = run_full(
            session, p, sig_b, return_ref=True, return_v_delta=True, pack=pack,
        )
    else:
        out, v_ref = run_full(session, p, sig_b, return_ref=True, pack=pack)
    out = out[:, :, neuron_index]
    v_ref = v_ref[:, neuron_index]
    if squeeze:
        out = out.squeeze(0)
        v_ref = v_ref.squeeze(0)
    if return_ref:
        return out, v_ref
    return out


def pack_readout(p, pack, session, batch_idx=None):
    """Shared pack readout via ``run_full`` (waveform MSE only when needed)."""
    import FiveCol_MedSim_Pytorch as fc

    sig = pack.signal if batch_idx is None else pack.signal[batch_idx:batch_idx + 1]
    ca_full = run_full(session, p, sig, pack=pack)
    need_mse = fc._pack_needs_waveform_mse(pack)
    t_on = int(pack.signal.shape[1] - pack.data.shape[1])
    if batch_idx is None:
        dsi_sel = ca_full[pack.readout_batch, t_on:, pack.readout_unit]
        if not need_mse:
            return None, dsi_sel
        return fc._readout_ca_traces_pack(ca_full, pack), dsi_sel
    mask = pack.readout_batch == int(batch_idx)
    u_m = pack.readout_unit[mask]
    dsi_sel = ca_full[0, t_on:, u_m].transpose(0, 1)
    if not need_mse:
        return None, dsi_sel
    if pack.cost_t0 is None:
        return dsi_sel, dsi_sel
    b_zero = torch.zeros_like(u_m)
    mse_sel = fc._window_time_traces(
        ca_full, b_zero, u_m, pack.cost_t0[mask],
        win=pack.data.shape[1],
    )
    return mse_sel, dsi_sel


# Register pack readouts here — batching stays in FiveCol ``_pack_cost``.
CA_PACK_READOUTS = {
    "conductance": pack_readout,
    "hp_lp": pack_readout,
}
