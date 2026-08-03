# -*- coding: utf-8 -*-
"""HP-then-membrane-LP neuron (``--model hp_lp``).

    τ_HP d v_slow / dt = v_tot − v_slow
    τ_lp dv/dt = −(v − v_rest) + G (v_tot − v_slow)

with v_tot = v_rest + v_in + v_sti, v_sti = i_sti/g_in (g_in in nS converts pA → mV),
v_in from relu(v+bias)·out_gain scaled by syn_strength_cell (per_cell) or
syn_strength_edge (per_edge). v_hp = v_tot − v_slow is the HP output.

Membrane / HP Euler (``session.euler`` = ``implicit`` | ``explicit``):

    α_hp = Δt/τ_hp,  α_lp = Δt/τ_lp

    implicit HP:  v_slow ← (v_slow + α_hp v_tot) / (1 + α_hp)
    explicit HP:  v_slow ← v_slow + α_hp (v_tot − v_slow)

    implicit LP:  v ← (v + α_lp (v_rest + G v_hp)) / (1 + α_lp)
    explicit LP:  v ← v + α_lp (−(v − v_rest) + G v_hp)

Dynamics only: ``prepare_i_sti`` / ``init_state`` / ``step``. Full-T ``v``
forward lives in ``neuron.forward``. Scalars from ``session`` flat fields.
``init_state`` starts at the pre steady state: ``v0 = v_rest``,
``v_slow0 = v_tot0`` with ``v_tot0`` from ``v_rest`` and ``i_sti[t=0]``
(HP contributes 0 at DC).
"""
from __future__ import annotations

import torch

from neuron.params import expand_euler
from neuron.schema import syn_strength


def update_state_hp_lp(
    v, v_slow, p, i_sti, backend, *, delta_ms, state_clamp, g_in, euler,
    return_component: bool = False,
):
    """One HP→LP membrane step; returns ``(v, v_slow)`` or component extras."""
    euler = expand_euler(euler)
    v_rest = p["v_rest"]
    dt = float(delta_ms)
    tau_lp = torch.clamp(p["tau_lp"], min=dt)
    tau_hp = torch.clamp(p["tau_hp"], min=dt)
    G = p["hp_gain"]
    clamp = float(state_clamp)
    g_in = float(g_in)
    if g_in == 0.0:
        raise ValueError("g_in must be non-zero")

    pre = torch.relu(v + p["bias"]) * p["out_gain"]
    w = syn_strength(p)
    v_in = p["in_gain"] * backend.conn.signed_drive(pre, w)
    v_sti = i_sti / g_in
    v_tot = v_rest + v_in + v_sti
    hp_dt_over_tau = dt / tau_hp
    if euler == "implicit":
        v_slow = (v_slow + hp_dt_over_tau * v_tot) / (1.0 + hp_dt_over_tau)
    else:
        v_slow = v_slow + hp_dt_over_tau * (v_tot - v_slow)
    # LP uses post-HP ``v_slow`` (same as prior identity).
    v_hp = v_tot - v_slow
    lp_dt_over_tau = dt / tau_lp
    if euler == "implicit":
        scale = lp_dt_over_tau / (1.0 + lp_dt_over_tau)
        dv_leak = scale * (-(v - v_rest))
        dv_hp = scale * G * v_hp
        v = (v + lp_dt_over_tau * (v_rest + G * v_hp)) / (1.0 + lp_dt_over_tau)
    else:
        dv_leak = lp_dt_over_tau * (-(v - v_rest))
        dv_hp = lp_dt_over_tau * G * v_hp
        v = v + dv_leak + dv_hp

    v_slow = torch.clamp(v_slow, -clamp, clamp)
    v = torch.clamp(v, -clamp, clamp)

    if not return_component:
        return v, v_slow

    g_exc, g_inh = backend.conn.exc_inh_drive(pre, w)
    v_in_exc = p["in_gain"] * g_exc
    v_in_inh = p["in_gain"] * g_inh
    return v, v_slow, {
        "v_in": v_in,
        "v_in_exc": v_in_exc,
        "v_in_inh": v_in_inh,
        "v_slow": v_slow,
        "v_tot": v_tot,
        "v_hp": v_hp,
        "dv_leak": dv_leak,
        "dv_hp": dv_hp,
        "i_sti": i_sti,
        "v_sti": v_sti,
    }


def prepare_i_sti(session, p, i_sti, pack):
    """PR current ``(B, T, N)`` as membrane drive (no rescale)."""
    del p, pack
    return i_sti.unsqueeze(0) if i_sti.dim() == 2 else i_sti


def init_state(session, p, B, i_sti=None):
    """``(v_slow,)`` at pre steady state: ``v0 = v_rest``, ``v_slow0 = v_tot0``."""
    if i_sti is None:
        raise TypeError("hp_lp init_state requires i_sti")
    v_rest = p["v_rest"]
    backend = session.backend
    n = backend.n_nodes
    g_in = float(session.g_in)
    if g_in == 0.0:
        raise ValueError("g_in must be non-zero")
    v = v_rest.expand(B, n).clone()
    pre = torch.relu(v + p["bias"]) * p["out_gain"]
    w = syn_strength(p)
    v_in = p["in_gain"] * backend.conn.signed_drive(pre, w)
    v_sti = i_sti[:, 0, :] / g_in
    v_slow = v_rest + v_in + v_sti
    return (v_slow,), v


def step(state, v, p, i_sti, session, *, return_component: bool = False):
    """One hp_lp update; returns ``((v_slow,), v)`` or + component dict."""
    v_slow, = state
    out = update_state_hp_lp(
        v, v_slow, p, i_sti, session.backend,
        delta_ms=session.delta_ms, state_clamp=session.STATE_CLAMP,
        g_in=session.g_in, euler=session.euler,
        return_component=return_component,
    )
    if return_component:
        v, v_slow, component = out
        return (v_slow,), v, component
    v, v_slow = out
    return (v_slow,), v
