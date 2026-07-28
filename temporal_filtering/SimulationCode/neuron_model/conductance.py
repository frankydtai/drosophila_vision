# -*- coding: utf-8 -*-
"""Conductance-based neuron + Ih (``--model conductance``)."""
from __future__ import annotations

import torch

from neuron_model.constants import (
    E_IH_OFF,
    E_Ih,
    E_exc,
    E_inh,
    IH_OFF_DEFAULT,
    Ih_gain,
    cdt,
    deltat,
    g_leak,
)
from neuron_model.schema import conductance_ih_off_kwargs


def rectsyn(x, thrld):
    result = x - thrld
    return result * (result > 0)


def _ih_gate_step(
    Vm, u_on, u_off, Ih_gmax, Ih_gmax_off,
    Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
):
    """Advance Ih gate states and conductances for active columns only."""
    slope_on = Ih_slope
    slope_off = -Ih_slope_off
    Ih_ss_on = 1.0 / (1.0 + torch.exp((Ih_midv - Vm) * slope_on))
    Ih_ss_off = 1.0 / (1.0 + torch.exp((Ih_midv_off - Vm) * slope_off))
    tau_on = (
        1.5 / (torch.exp(-0.1 * (Vm - tau_midv)) + torch.exp(+0.1 * (Vm - tau_midv))) * 1000.0
        + 100.0
    )
    tau_off = (
        1.5
        / (torch.exp(-0.1 * (Vm - tau_midv_off)) + torch.exp(+0.1 * (Vm - tau_midv_off)))
        * 1000.0
        + 100.0
    )
    u_on = deltat / tau_on * (Ih_ss_on - u_on) + u_on
    u_off = deltat / tau_off * (Ih_ss_off - u_off) + u_off
    g_Ih_on = u_on * Ih_gmax * Ih_gain
    g_Ih_off = u_off * Ih_gmax_off * Ih_gain
    return u_on, u_off, g_Ih_on, g_Ih_off


def update_Vm(
    Vm, u_on, u_off, in_gain, out_gain, syn_strength, v_th, Ih_gmax, Ih_gmax_off,
    Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
    signal, backend, *, return_budget: bool = False,
):
    """One conductance step."""
    e_leak = backend.e_leak
    conn = backend.conn
    ih_active = (Ih_gmax + Ih_gmax_off) != 0
    g_Ih_on = u_on.new_zeros(u_on.shape)
    g_Ih_off = u_off.new_zeros(u_off.shape)
    if ih_active.any():
        ih_kw = dict(
            Ih_midv=Ih_midv, Ih_slope=Ih_slope, tau_midv=tau_midv,
            Ih_midv_off=Ih_midv_off, Ih_slope_off=Ih_slope_off, tau_midv_off=tau_midv_off,
        )
        if ih_active.all():
            u_on, u_off, g_Ih_on, g_Ih_off = _ih_gate_step(
                Vm, u_on, u_off, Ih_gmax, Ih_gmax_off, **ih_kw)
        else:
            idx = ih_active
            u_on_a, u_off_a, g_on_a, g_off_a = _ih_gate_step(
                Vm[:, idx], u_on[:, idx], u_off[:, idx],
                Ih_gmax[idx], Ih_gmax_off[idx], **{k: v[idx] for k, v in ih_kw.items()},
            )
            u_on = u_on.clone()
            u_off = u_off.clone()
            u_on[:, idx] = u_on_a
            u_off[:, idx] = u_off_a
            g_Ih_on[:, idx] = g_on_a
            g_Ih_off[:, idx] = g_off_a
    g_Ih = g_Ih_on + g_Ih_off

    g_exc, g_inh = conn.exc_inh_drive(rectsyn(Vm, v_th) * out_gain, syn_strength)
    g_exc = g_exc * in_gain
    g_inh = g_inh * in_gain

    Vm = (
        g_exc * E_exc + g_inh * E_inh + g_leak * e_leak
        + E_Ih * g_Ih_on + E_IH_OFF * g_Ih_off + cdt * Vm + signal
    )
    Vm = Vm / (g_exc + g_inh + g_Ih + g_leak + cdt)

    if return_budget:
        return Vm, u_on, u_off, g_exc, g_inh, g_Ih_on, g_Ih_off
    return Vm, u_on, u_off


def vm_budget_from_g(Vm_pre, g_exc, g_inh, g_Ih_on, g_Ih_off, signal, e_leak):
    """Numerator / denom terms matching ``update_Vm`` (torch or numpy)."""
    return {
        "num_exc": g_exc * E_exc,
        "num_inh": g_inh * E_inh,
        "num_leak": g_leak * e_leak,
        "num_ihon": g_Ih_on * E_Ih,
        "num_ihoff": g_Ih_off * E_IH_OFF,
        "num_cdt": cdt * Vm_pre,
        "num_sig": signal,
        "den": g_exc + g_inh + g_Ih_on + g_Ih_off + g_leak + cdt,
    }


def _ca_readout_step(model, Vm, Vm_ref):
    from neuron_model.constants import Ca_tau

    return deltat / Ca_tau * (Vm - Vm_ref - model) + model


def run_conductance_full(session, p, sig, return_ref=False, *, return_vm=False):
    """Conductance forward; ``model_full`` time index ``t`` is post-update at step ``t``."""
    import FiveCol_MedSim_Pytorch as fc

    backend = session.backend
    ih_off = (session.train_opts or {}).get("ih_off", IH_OFF_DEFAULT)
    in_gain, out_gain = p["in_gain"], p["out_gain"]
    syn_strength = p["syn_strength"]
    v_th = p["v_th"]
    Ih_gmax = p["Ih_gmax"]
    Ih_gmax_off, Ih_midv_off, Ih_slope_off, tau_midv_off = conductance_ih_off_kwargs(p, ih_off)
    Ih_midv, Ih_slope, tau_midv = p["Ih_midv"], p["Ih_slope"], p["tau_midv"]
    B = sig.shape[0]
    t_end = sig.shape[1]
    t_on = fc.t_on
    dev = backend.conn.node_type.device
    u_on = u_off = torch.zeros((B, backend.n_units), dtype=session.sim_dtype, device=dev)
    Vm = backend.e_leak.expand(B, backend.n_units).clone()
    vm_rows = [Vm]
    for t in range(1, t_end):
        Vm, u_on, u_off = update_Vm(
            Vm, u_on, u_off, in_gain, out_gain, syn_strength, v_th, Ih_gmax, Ih_gmax_off,
            Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
            sig[:, t - 1], backend)
        vm_rows.append(Vm)
    vm_full = torch.stack(vm_rows, dim=1)
    Vm_ref = vm_full[:, t_on - 1, :].clone()
    vm_delta = vm_full - Vm_ref.unsqueeze(1)

    ca_rows = [torch.zeros((B, backend.n_units), dtype=session.sim_dtype, device=dev)]
    model = 0
    for t in range(1, t_end):
        if t == t_on:
            model = 0
        model = _ca_readout_step(model, vm_full[:, t], Vm_ref)
        ca_rows.append(model)
    ca_full = torch.stack(ca_rows, dim=1)

    if return_vm:
        if return_ref:
            return vm_delta, Vm_ref, vm_full
        return vm_delta
    if return_ref:
        return ca_full, Vm_ref
    return ca_full


def run_conductance(session, p, neuron_index=None, return_ref=False, sig=None, pack=None, *, return_vm=False):
    if neuron_index is None:
        pack = pack or session.primary_pack
        neuron_index = pack.readout_unit
    if sig is None:
        sig = session.pack_signal(pack)
    squeeze = sig.dim() == 2
    sig_b = sig.unsqueeze(0) if squeeze else sig
    if return_vm:
        out, vm_ref, _vm_full = run_conductance_full(session, p, sig_b, return_ref=True, return_vm=True)
    else:
        out, vm_ref = run_conductance_full(session, p, sig_b, return_ref=True)
    out = out[:, :, neuron_index]
    vm_ref = vm_ref[:, neuron_index]
    if squeeze:
        out = out.squeeze(0)
        vm_ref = vm_ref.squeeze(0)
    if return_ref:
        return out, vm_ref
    return out


def pack_readout(p, pack, session, batch_idx=None):
    """Conductance forward; waveform MSE readout only when pack needs it."""
    import FiveCol_MedSim_Pytorch as fc

    sig = pack.signal if batch_idx is None else pack.signal[batch_idx:batch_idx + 1]
    model_full = run_conductance_full(session, p, sig)
    need_mse = fc._pack_needs_waveform_mse(pack)
    t_on = fc.t_on
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
