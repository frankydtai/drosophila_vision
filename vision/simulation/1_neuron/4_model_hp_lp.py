# -*- coding: utf-8 -*-
"""HP-then-membrane-LP neuron (``--model hp_lp``).

    τ_HP da/dt = X − a
    τ_lp dv/dt = −(v − v_rest) + G (X − a)

with X = v_rest + v_in + v_sti, v_sti = i_sti/g_in (g_in in nS converts pA → mV),
v_in from relu(v+bias)·out_gain scaled by syn_strength_cell (per_cell) or
syn_strength_edge (per_edge).

Dynamics only: ``prepare_i_sti`` / ``init_state`` / ``step``. Full-T ``v``
forward lives in ``neuron.forward``. Scalars from ``session`` flat fields.
``init_state`` starts at the pre steady state: ``v0 = v_rest``, ``a0 = X0``
with ``X0`` from ``v_rest`` and ``i_sti[t=0]`` (HP contributes 0 at DC).
"""
from __future__ import annotations

import torch

from neuron.schema import syn_strength


def update_state_hp_lp(
    v, a, p, i_sti, backend, *, delta_ms, state_clamp, g_in, return_component: bool = False,
):
    """One HP→LP membrane step; returns ``(v, a)`` or component extras."""
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
    X = v_rest + v_in + v_sti
    a = a + dt / tau_hp * (X - a)
    # LP uses post-HP ``a`` (same as prior identity).
    X_minus_a = X - a
    dv_leak = dt / tau_lp * (-(v - v_rest))
    dv_hp = dt / tau_lp * G * X_minus_a
    v = v + dv_leak + dv_hp

    a = torch.clamp(a, -clamp, clamp)
    v = torch.clamp(v, -clamp, clamp)

    if not return_component:
        return v, a

    g_exc, g_inh = backend.conn.exc_inh_drive(pre, w)
    v_in_exc = p["in_gain"] * g_exc
    v_in_inh = p["in_gain"] * g_inh
    return v, a, {
        "v_in": v_in,
        "v_in_exc": v_in_exc,
        "v_in_inh": v_in_inh,
        "a": a,
        "X": X,
        "X_minus_a": X_minus_a,
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
    """``(a,)`` at pre steady state: ``v0 = v_rest``, ``a0 = X0``."""
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
    a = v_rest + v_in + v_sti
    return (a,), v


def step(state, v, p, i_sti, session, *, return_component: bool = False):
    """One hp_lp update; returns ``((a,), v)`` or + component dict."""
    a, = state
    out = update_state_hp_lp(
        v, a, p, i_sti, session.backend,
        delta_ms=session.delta_ms, state_clamp=session.STATE_CLAMP,
        g_in=session.g_in,
        return_component=return_component,
    )
    if return_component:
        v, a, component = out
        return (a,), v, component
    v, a = out
    return (a,), v
