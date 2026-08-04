# -*- coding: utf-8 -*-
"""Borst neuron + Ih (``--model borst``).

Dynamics only: ``prepare_i_sti`` / ``pre_steady`` / ``step``. Full-T Ca
forward lives in ``neuron.forward``. Membrane scalars are injected kwargs
(from ``session`` flat fields), never a Physics bag.

t=0 membrane state uses ``session.pre_steady`` (``--pre-steady borst=…``);
only ``e_leak`` is defined (``v0 = e_leak``, ``u_on/u_off = 0``).

Membrane Euler (``session.euler`` = ``implicit`` | ``explicit``):

    C dv/dt = i_sti + Σ g_i (E_i − v)

    implicit:  v ← (v + α (i_sti + Σ gE)) / (1 + α Σ g),  α = Δt/C
    explicit:  v ← v + α (i_sti + Σ gE − Σ g · v)

Ih gate kinetics are always explicit Euler, independent of ``euler``.
"""
from __future__ import annotations

import torch

from neuron.params import e_ih_off, expand_euler, membrane_dt_over_c
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
    """Advance Ih gate states and channel conductances for active columns only.

    Gate ODE uses explicit Euler regardless of membrane ``euler``.
    """
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
    v, u_on, u_off, a_in, a_out, syn_strength, v_th, Ih_gmax, Ih_gmax_off,
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
    euler: str,
    return_component: bool = False,
):
    """One borst step; membrane / reversal scalars are required kwargs."""
    euler = expand_euler(euler)
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

    g_exc, g_inh = conn.exc_inh_drive(rectsyn(v, v_th) * a_out, syn_strength)
    g_exc = g_exc * a_in
    g_inh = g_inh * a_in

    dt_over_c = membrane_dt_over_c(capac, delta_ms)
    E_IH_OFF = e_ih_off(E_LEAK_REST, E_Ih)
    sum_gE = (
        g_exc * E_exc + g_inh * E_inh + g_leak * e_leak
        + E_Ih * g_Ih_on + E_IH_OFF * g_Ih_off
    )
    sum_g = g_exc + g_inh + g_Ih + g_leak
    if euler == "implicit":
        v = (v + dt_over_c * (i_sti + sum_gE)) / (1.0 + dt_over_c * sum_g)
    else:
        v = v + dt_over_c * (i_sti + sum_gE - sum_g * v)

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
    euler: str,
):
    """Numerator / denom terms matching ``update_v`` (torch or numpy)."""
    euler = expand_euler(euler)
    dt_over_c = membrane_dt_over_c(capac, delta_ms)
    E_IH_OFF = e_ih_off(E_LEAK_REST, E_Ih)
    num_exc = g_exc * E_exc
    num_inh = g_inh * E_inh
    num_leak = g_leak * e_leak
    num_ihon = g_Ih_on * E_Ih
    num_ihoff = g_Ih_off * E_IH_OFF
    sum_gE = num_exc + num_inh + num_leak + num_ihon + num_ihoff
    sum_g = g_exc + g_inh + g_Ih_on + g_Ih_off + g_leak
    if euler == "implicit":
        num = v_pre + dt_over_c * (i_sti + sum_gE)
        den = 1.0 + dt_over_c * sum_g
        num_v = v_pre
    else:
        num = v_pre + dt_over_c * (i_sti + sum_gE - sum_g * v_pre)
        den = 1.0
        num_v = v_pre * (1.0 - dt_over_c * sum_g)
    return {
        "num_exc": num_exc,
        "num_inh": num_inh,
        "num_leak": num_leak,
        "num_ihon": num_ihon,
        "num_ihoff": num_ihoff,
        "num_v": num_v,
        "i_sti": i_sti,
        "sum_gE": sum_gE,
        "sum_g": sum_g,
        "dt_over_c": dt_over_c,
        "num": num,
        "den": den,
    }


def prepare_i_sti(session, p, i_sti, pack):
    """PR current ``(B, T, N)`` as membrane drive (no rescale)."""
    del p, pack
    return i_sti.unsqueeze(0) if i_sti.dim() == 2 else i_sti


def pre_steady(session, p, B, i_sti=None):
    """``(u_on, u_off)``, ``v`` at t=0 from ``session.pre_steady``."""
    del p, i_sti
    mode = str(session.pre_steady)
    if mode != "e_leak":
        raise ValueError(
            f"borst pre_steady must be e_leak; got {mode!r}"
        )
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
        euler=session.euler,
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
        p["a_in"], p["a_out"], syn_strength(p), p["v_th"],
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
