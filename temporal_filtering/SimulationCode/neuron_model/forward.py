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


def ca_readout_step(model, V, V_ref):
    """One Ca low-pass step on ``V - V_ref`` (shared by all models)."""
    return deltat / Ca_tau * (V - V_ref - model) + model


def run_full(session, p, sig, *, return_ref=False, return_vm=False, pack=None):
    """Shared full-T forward for every ``session.model``.

    Time index ``t`` is post-update at step ``t``. Membrane drive comes from
    ``MODEL_DRIVERS[model].prepare_signal`` / ``init_state`` / ``step``. Ca
    runs for all ``t``; the integrator resets at ``t_on`` (same contract for
    every model).

    Returns
    -------
    ca_full ``(B, T, N)``, or with ``return_vm``: ``vm_delta`` / ``(vm_delta,
    V_ref, vm_full)``.
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
    state, V = drv.init_state(session, p, B)
    vm_rows = [V]
    for t in range(1, t_end):
        state, V = drv.step(state, V, p, x[:, t - 1], session)
        vm_rows.append(V)
    vm_full = torch.stack(vm_rows, dim=1)
    V_ref = vm_full[:, t_on - 1, :].clone()
    vm_delta = vm_full - V_ref.unsqueeze(1)

    ca_rows = [
        torch.zeros(
            (B, session.backend.n_units),
            dtype=session.sim_dtype,
            device=V_ref.device,
        ),
    ]
    model = 0
    for t in range(1, t_end):
        if t == t_on:
            model = 0
        model = ca_readout_step(model, vm_full[:, t], V_ref)
        ca_rows.append(model)
    ca_full = torch.stack(ca_rows, dim=1)

    if return_vm:
        if return_ref:
            return vm_delta, V_ref, vm_full
        return vm_delta
    if return_ref:
        return ca_full, V_ref
    return ca_full


def run_units(
    session, p, neuron_index=None, return_ref=False, sig=None, pack=None,
    *, return_vm=False,
):
    """``run_full`` then index units; squeeze when ``sig`` is ``(T, N)``."""
    pack = pack or session.primary_pack
    if neuron_index is None:
        neuron_index = pack.readout_unit
    if sig is None:
        sig = session.pack_signal(pack)
    squeeze = sig.dim() == 2
    sig_b = sig.unsqueeze(0) if squeeze else sig
    if return_vm:
        out, V_ref, _vm_full = run_full(
            session, p, sig_b, return_ref=True, return_vm=True, pack=pack,
        )
    else:
        out, V_ref = run_full(session, p, sig_b, return_ref=True, pack=pack)
    out = out[:, :, neuron_index]
    V_ref = V_ref[:, neuron_index]
    if squeeze:
        out = out.squeeze(0)
        V_ref = V_ref.squeeze(0)
    if return_ref:
        return out, V_ref
    return out


def pack_readout(p, pack, session, batch_idx=None):
    """Shared pack readout via ``run_full`` (waveform MSE only when needed)."""
    import FiveCol_MedSim_Pytorch as fc

    sig = pack.signal if batch_idx is None else pack.signal[batch_idx:batch_idx + 1]
    model_full = run_full(session, p, sig, pack=pack)
    need_mse = fc._pack_needs_waveform_mse(pack)
    t_on = int(pack.signal.shape[1] - pack.data.shape[1])
    if batch_idx is None:
        dsi_sel = model_full[pack.readout_batch, t_on:, pack.readout_unit]
        if not need_mse:
            return None, dsi_sel
        return fc._readout_model_traces_pack(model_full, pack), dsi_sel
    mask = pack.readout_batch == int(batch_idx)
    u_m = pack.readout_unit[mask]
    dsi_sel = model_full[0, t_on:, u_m].transpose(0, 1)
    if not need_mse:
        return None, dsi_sel
    if pack.cost_t0 is None:
        return dsi_sel, dsi_sel
    b_zero = torch.zeros_like(u_m)
    mse_sel = fc._window_time_traces(
        model_full, b_zero, u_m, pack.cost_t0[mask],
        win=pack.data.shape[1],
    )
    return mse_sel, dsi_sel


# Register pack readouts here — batching stays in FiveCol ``_pack_cost``.
MODEL_PACK_READOUTS = {
    "conductance": pack_readout,
    "hp_lp": pack_readout,
}
