# -*- coding: utf-8 -*-
"""HP-then-membrane-LP neuron (``--model hp_lp``).

    τ_HP da/dt = X − a
    τ_m  dV/dt = −(V − bias) + G (X − a)

with X = bias + syn + x_t, syn from relu(V)·out_gain scaled by type→type syn_strength.
"""
from __future__ import annotations

import torch

from neuron_model.constants import STATE_CLAMP, Ca_tau, deltat


def update_state_hp_lp(V, a, p, x_t, backend):
    """One HP→LP membrane step; returns (V, a)."""
    bias = p["bias"]
    tau_m = torch.clamp(p["tau_m"], min=deltat)
    tau_hp = torch.clamp(p["tau_hp"], min=deltat)
    G = p["hp_gain"]

    syn = p["in_gain"] * backend.conn.signed_drive(
        torch.relu(V) * p["out_gain"], p["syn_strength"],
    )
    X = bias + syn + x_t
    a = a + deltat / tau_hp * (X - a)
    V = V + deltat / tau_m * (-(V - bias) + G * (X - a))

    a = torch.clamp(a, -STATE_CLAMP, STATE_CLAMP)
    V = torch.clamp(V, -STATE_CLAMP, STATE_CLAMP)
    return V, a


def run_hp_lp(p, session, neuron_index=None, return_ref=False, sig=None, pack=None):
    import FiveCol_MedSim_Pytorch as fc

    backend = session.backend
    mt = session.maxtime
    t_on = fc.t_on
    pack = pack or session.primary_pack
    if neuron_index is None:
        neuron_index = pack.readout_unit
    if sig is None:
        sig = session.pack_signal(pack)
    bias = p["bias"]
    x_signal = sig / fc._pack_signal_scale(pack, session)

    V = bias.clone()
    a = bias.clone()

    act_ref = None
    model = 0
    rows = []
    for t in range(1, mt):
        x_t = x_signal[t - 1]
        V, a = update_state_hp_lp(V, a, p, x_t, backend)
        if t == t_on - 1:
            act_ref = 1.0 * V[neuron_index]
        elif t >= t_on:
            model = deltat / Ca_tau * (V[neuron_index] - act_ref - model) + model
            rows.append(model)
    model = torch.stack(rows)
    if return_ref:
        return model, act_ref
    return model


def _window_activity_traces(model, t0, win):
    """Windowed readout from activity-model output ``(T', K)``."""
    from network.moving_bar_target import moving_bar_window_t_rel_torch
    import FiveCol_MedSim_Pytorch as fc

    dev = fc.active_device()
    t_rel, pre = moving_bar_window_t_rel_torch(t0, int(fc.t_on), int(win), device=dev)
    t_max = model.shape[0] - 1
    t_safe = t_rel.clamp(0, t_max)
    k_idx = torch.arange(model.shape[1], dtype=torch.long, device=dev)
    sel = model[t_safe, k_idx[:, None]]
    return torch.where(pre, torch.zeros_like(sel), sel)


def pack_readout(p, pack, session, batch_idx=None):
    """hp_lp forward; waveform MSE readout only when pack needs it."""
    import FiveCol_MedSim_Pytorch as fc

    need_mse = fc._pack_needs_waveform_mse(pack)
    if batch_idx is not None:
        sig = pack.signal[batch_idx]
        mask = pack.readout_batch == int(batch_idx)
        u = pack.readout_unit[mask]
        t0 = pack.cost_t0[mask] if pack.cost_t0 is not None else None
        model = run_hp_lp(p, session, neuron_index=u, sig=sig, pack=pack)
        dsi_sel = model.transpose(0, 1)
        if not need_mse:
            return None, dsi_sel
        if t0 is None:
            return dsi_sel, dsi_sel
        return _window_activity_traces(model, t0, win=pack.data.shape[1]), dsi_sel

    sig = session.pack_signal(pack)
    if sig.dim() == 2:
        model = run_hp_lp(p, session, neuron_index=pack.readout_unit, sig=sig, pack=pack)
        dsi_sel = model.transpose(0, 1)
        if not need_mse:
            return None, dsi_sel
        if pack.cost_t0 is None:
            return dsi_sel, dsi_sel
        return (
            _window_activity_traces(model, pack.cost_t0, win=pack.data.shape[1]),
            dsi_sel,
        )

    row_indices = []
    mse_parts = []
    dsi_parts = []
    for b in pack.readout_batch.unique(sorted=True).tolist():
        mask = pack.readout_batch == int(b)
        rows = torch.nonzero(mask, as_tuple=False).reshape(-1)
        model = run_hp_lp(
            p, session, neuron_index=pack.readout_unit[mask],
            sig=sig[int(b)], pack=pack,
        )
        row_indices.append(rows)
        dsi_parts.append(model.transpose(0, 1))
        if need_mse:
            if pack.cost_t0 is None:
                mse_parts.append(model.transpose(0, 1))
            else:
                mse_parts.append(_window_activity_traces(
                    model, pack.cost_t0[mask], win=pack.data.shape[1],
                ))
    row_order = torch.cat(row_indices).argsort()
    dsi_sel = torch.cat(dsi_parts, dim=0).index_select(0, row_order)
    if not need_mse:
        return None, dsi_sel
    mse_sel = torch.cat(mse_parts, dim=0).index_select(0, row_order)
    return mse_sel, dsi_sel
