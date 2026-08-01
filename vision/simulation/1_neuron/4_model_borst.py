# -*- coding: utf-8 -*-
"""Borst neuron + Ih (``--model borst``).

Dynamics only: ``prepare_i_sti`` / ``init_state`` / ``step``. Full-T Ca
forward lives in ``neuron.forward``. Membrane scalars are injected kwargs
(from ``session`` flat fields), never a Physics bag.
"""
from __future__ import annotations

import torch

from neuron.params import e_ih_off, membrane_cdt
from neuron.schema import borst_ih_off_kwargs, syn_strength


def rectsyn(x, thrld):
    result = x - thrld
    return result * (result > 0)


def _ih_gate_step(
    v, u_on, u_off, Ih_gmax, Ih_gmax_off,
    Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
    *,
    delta_ms: float,
    Ih_gain: float,
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
    dt = float(delta_ms)
    u_on = dt / tau_on * (Ih_ss_on - u_on) + u_on
    u_off = dt / tau_off * (Ih_ss_off - u_off) + u_off
    g_Ih_on = u_on * Ih_gmax * float(Ih_gain)
    g_Ih_off = u_off * Ih_gmax_off * float(Ih_gain)
    return u_on, u_off, g_Ih_on, g_Ih_off


def update_v(
    v, u_on, u_off, in_gain, out_gain, syn_strength, v_th, Ih_gmax, Ih_gmax_off,
    Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
    i_sti, backend, *,
    delta_ms: float,
    capac: float,
    g_leak: float,
    E_exc: float,
    E_inh: float,
    E_Ih: float,
    E_LEAK_REST: float,
    Ih_gain: float,
    return_component: bool = False,
):
    """One borst step; membrane / reversal scalars are required kwargs."""
    e_leak = backend.e_leak
    conn = backend.conn
    ih_active = (Ih_gmax + Ih_gmax_off) != 0
    g_Ih_on = u_on.new_zeros(u_on.shape)
    g_Ih_off = u_off.new_zeros(u_off.shape)
    ih_kw_common = dict(delta_ms=delta_ms, Ih_gain=Ih_gain)
    if ih_active.any():
        ih_kw = dict(
            Ih_midv=Ih_midv, Ih_slope=Ih_slope, tau_midv=tau_midv,
            Ih_midv_off=Ih_midv_off, Ih_slope_off=Ih_slope_off, tau_midv_off=tau_midv_off,
        )
        if ih_active.all():
            u_on, u_off, g_Ih_on, g_Ih_off = _ih_gate_step(
                v, u_on, u_off, Ih_gmax, Ih_gmax_off, **ih_kw_common, **ih_kw)
        else:
            idx = ih_active
            u_on_a, u_off_a, g_on_a, g_off_a = _ih_gate_step(
                v[:, idx], u_on[:, idx], u_off[:, idx],
                Ih_gmax[idx], Ih_gmax_off[idx],
                **ih_kw_common,
                **{k: val[idx] for k, val in ih_kw.items()},
            )
            u_on = u_on.clone()
            u_off = u_off.clone()
            u_on[:, idx] = u_on_a.to(dtype=u_on.dtype)
            u_off[:, idx] = u_off_a.to(dtype=u_off.dtype)
            g_Ih_on[:, idx] = g_on_a.to(dtype=g_Ih_on.dtype)
            g_Ih_off[:, idx] = g_off_a.to(dtype=g_Ih_off.dtype)
    g_Ih = g_Ih_on + g_Ih_off

    g_exc, g_inh = conn.exc_inh_drive(rectsyn(v, v_th) * out_gain, syn_strength)
    g_exc = g_exc * in_gain
    g_inh = g_inh * in_gain

    cdt = membrane_cdt(capac, delta_ms)
    E_IH_OFF = e_ih_off(E_LEAK_REST, E_Ih)
    v = (
        g_exc * E_exc + g_inh * E_inh + g_leak * e_leak
        + E_Ih * g_Ih_on + E_IH_OFF * g_Ih_off + cdt * v + i_sti
    )
    v = v / (g_exc + g_inh + g_Ih + g_leak + cdt)

    if return_component:
        return v, u_on, u_off, g_exc, g_inh, g_Ih_on, g_Ih_off
    return v, u_on, u_off


def v_component_from_g(
    v_pre, g_exc, g_inh, g_Ih_on, g_Ih_off, i_sti, e_leak, *,
    delta_ms: float,
    capac: float,
    g_leak: float,
    E_exc: float,
    E_inh: float,
    E_Ih: float,
    E_LEAK_REST: float,
):
    """Numerator / denom terms matching ``update_v`` (torch or numpy)."""
    cdt = membrane_cdt(capac, delta_ms)
    E_IH_OFF = e_ih_off(E_LEAK_REST, E_Ih)
    return {
        "num_exc": g_exc * E_exc,
        "num_inh": g_inh * E_inh,
        "num_leak": g_leak * e_leak,
        "num_ihon": g_Ih_on * E_Ih,
        "num_ihoff": g_Ih_off * E_IH_OFF,
        "num_cdt": cdt * v_pre,
        "i_sti": i_sti,
        "den": g_exc + g_inh + g_Ih_on + g_Ih_off + g_leak + cdt,
    }


def prepare_i_sti(session, p, i_sti, pack):
    """PR current ``(B, T, N)`` as membrane drive (no rescale)."""
    del p, pack
    return i_sti.unsqueeze(0) if i_sti.dim() == 2 else i_sti


def init_state(session, p, B):
    """``(u_on, u_off)``, ``v0 = e_leak``."""
    del p
    backend = session.backend
    dev = backend.conn.node_cell.device
    dtype = session.sim_dtype
    n = backend.n_nodes
    u_on = u_off = torch.zeros((B, n), dtype=dtype, device=dev)
    v = backend.e_leak.expand(B, n).clone()
    return (u_on, u_off), v


def _membrane_kwargs(session):
    return dict(
        delta_ms=session.delta_ms,
        capac=session.capac,
        g_leak=session.g_leak,
        E_exc=session.E_exc,
        E_inh=session.E_inh,
        E_Ih=session.E_Ih,
        E_LEAK_REST=session.E_LEAK_REST,
        Ih_gain=session.Ih_gain,
    )


def step(state, v, p, i_sti, session, *, return_component: bool = False):
    """One borst update; returns ``((u_on, u_off), v)`` or + g component tuple."""
    u_on, u_off = state
    ih_off = (session.train_opts or {})["ih_off"]
    Ih_gmax_off, Ih_midv_off, Ih_slope_off, tau_midv_off = borst_ih_off_kwargs(
        p, ih_off,
    )
    out = update_v(
        v, u_on, u_off,
        p["in_gain"], p["out_gain"], syn_strength(p), p["v_th"],
        p["Ih_gmax"], Ih_gmax_off,
        p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
        Ih_midv_off, Ih_slope_off, tau_midv_off,
        i_sti, session.backend,
        **_membrane_kwargs(session),
        return_component=return_component,
    )
    if return_component:
        v, u_on, u_off, g_exc, g_inh, g_Ih_on, g_Ih_off = out
        return (u_on, u_off), v, (g_exc, g_inh, g_Ih_on, g_Ih_off)
    v, u_on, u_off = out
    return (u_on, u_off), v
