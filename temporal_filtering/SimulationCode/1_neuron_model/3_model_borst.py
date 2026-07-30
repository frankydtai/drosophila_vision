# -*- coding: utf-8 -*-
"""Borst neuron + Ih (``--model borst``).

Dynamics only: ``prepare_signal`` / ``init_state`` / ``step``. Full-T Ca
forward lives in ``neuron_model.forward``.
"""
from __future__ import annotations

import torch

from neuron_model.param import (
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
from neuron_model.schema import borst_ih_off_kwargs, synaptic_scale


def rectsyn(x, thrld):
    result = x - thrld
    return result * (result > 0)


def _ih_gate_step(
    v, u_on, u_off, Ih_gmax, Ih_gmax_off,
    Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
):
    """Advance Ih gate states and channel conductances for active columns only."""
    slope_on = Ih_slope
    slope_off = -Ih_slope_off
    Ih_ss_on = 1.0 / (1.0 + torch.exp((Ih_midv - v) * slope_on))
    Ih_ss_off = 1.0 / (1.0 + torch.exp((Ih_midv_off - v) * slope_off))
    tau_on = (
        1.5 / (torch.exp(-0.1 * (v - tau_midv)) + torch.exp(+0.1 * (v - tau_midv))) * 1000.0
        + 100.0
    )
    tau_off = (
        1.5
        / (torch.exp(-0.1 * (v - tau_midv_off)) + torch.exp(+0.1 * (v - tau_midv_off)))
        * 1000.0
        + 100.0
    )
    u_on = deltat / tau_on * (Ih_ss_on - u_on) + u_on
    u_off = deltat / tau_off * (Ih_ss_off - u_off) + u_off
    g_Ih_on = u_on * Ih_gmax * Ih_gain
    g_Ih_off = u_off * Ih_gmax_off * Ih_gain
    return u_on, u_off, g_Ih_on, g_Ih_off


def update_v(
    v, u_on, u_off, in_gain, out_gain, syn_strength, v_th, Ih_gmax, Ih_gmax_off,
    Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
    signal, backend, *, return_budget: bool = False,
):
    """One borst step."""
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
                v, u_on, u_off, Ih_gmax, Ih_gmax_off, **ih_kw)
        else:
            idx = ih_active
            u_on_a, u_off_a, g_on_a, g_off_a = _ih_gate_step(
                v[:, idx], u_on[:, idx], u_off[:, idx],
                Ih_gmax[idx], Ih_gmax_off[idx], **{k: val[idx] for k, val in ih_kw.items()},
            )
            u_on = u_on.clone()
            u_off = u_off.clone()
            u_on[:, idx] = u_on_a
            u_off[:, idx] = u_off_a
            g_Ih_on[:, idx] = g_on_a
            g_Ih_off[:, idx] = g_off_a
    g_Ih = g_Ih_on + g_Ih_off

    g_exc, g_inh = conn.exc_inh_drive(rectsyn(v, v_th) * out_gain, syn_strength)
    g_exc = g_exc * in_gain
    g_inh = g_inh * in_gain

    v = (
        g_exc * E_exc + g_inh * E_inh + g_leak * e_leak
        + E_Ih * g_Ih_on + E_IH_OFF * g_Ih_off + cdt * v + signal
    )
    v = v / (g_exc + g_inh + g_Ih + g_leak + cdt)

    if return_budget:
        return v, u_on, u_off, g_exc, g_inh, g_Ih_on, g_Ih_off
    return v, u_on, u_off


def v_budget_from_g(v_pre, g_exc, g_inh, g_Ih_on, g_Ih_off, signal, e_leak):
    """Numerator / denom terms matching ``update_v`` (torch or numpy)."""
    return {
        "num_exc": g_exc * E_exc,
        "num_inh": g_inh * E_inh,
        "num_leak": g_leak * e_leak,
        "num_ihon": g_Ih_on * E_Ih,
        "num_ihoff": g_Ih_off * E_IH_OFF,
        "num_cdt": cdt * v_pre,
        "num_sig": signal,
        "den": g_exc + g_inh + g_Ih_on + g_Ih_off + g_leak + cdt,
    }


def prepare_signal(session, p, sig, pack):
    """PR current ``(B, T, N)`` as membrane drive (no rescale)."""
    del p, pack
    return sig.unsqueeze(0) if sig.dim() == 2 else sig


def init_state(session, p, B):
    """``(u_on, u_off)``, ``v0 = e_leak``."""
    del p
    backend = session.backend
    dev = backend.conn.node_type.device
    dtype = session.sim_dtype
    n = backend.n_units
    u_on = u_off = torch.zeros((B, n), dtype=dtype, device=dev)
    v = backend.e_leak.expand(B, n).clone()
    return (u_on, u_off), v


def step(state, v, p, x_t, session):
    """One borst update; returns ``((u_on, u_off), v)``."""
    u_on, u_off = state
    ih_off = (session.train_opts or {}).get("ih_off", IH_OFF_DEFAULT)
    Ih_gmax_off, Ih_midv_off, Ih_slope_off, tau_midv_off = borst_ih_off_kwargs(
        p, ih_off,
    )
    v, u_on, u_off = update_v(
        v, u_on, u_off,
        p["in_gain"], p["out_gain"], synaptic_scale(p), p["v_th"],
        p["Ih_gmax"], Ih_gmax_off,
        p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
        Ih_midv_off, Ih_slope_off, tau_midv_off,
        x_t, session.backend,
    )
    return (u_on, u_off), v
