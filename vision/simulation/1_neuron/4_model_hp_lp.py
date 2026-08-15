# -*- coding: utf-8 -*-
"""HP-then-membrane-LP neuron (``--model hp_lp``).

    τ_HP(e_HP) d v_slow / dt = v_in − v_slow
    e_HP = v_in − v_slow
    τ_HP(e_HP) = τ_HP,rise if e_HP ≥ 0 else τ_HP,fall
    v_hp = v_in − a_h v_slow
    τ_lp dv/dt = −(v − e_leak) + v_hp

with v_in = v_syn + v_sti (no e_leak; leak alone sets rest), v_sti = i_sti/g_leak
(g_leak in nS converts pA → mV; same scalar as borst),
v_out = relu(v − v_th)·a_out, v_syn = a_in · signed_drive(v_out, syn_strength)
(syn_strength_cell per_cell or syn_strength_edge per_edge). a_h = 1
recovers classical HP (v_hp = v_in − v_slow); DC then has v_hp → 0 when
v_slow → v_in, so v → e_leak.

Membrane / HP Euler (``session.euler`` = ``implicit`` | ``explicit``):

    implicit HP:  v_slow ← (v_slow + (Δt/τ_HP) v_in) / (1 + Δt/τ_HP)
    explicit HP:  v_slow ← v_slow + (Δt/τ_HP) (v_in − v_slow)

    implicit LP:  v ← (v + (Δt/τ_lp) (e_leak + v_hp)) / (1 + Δt/τ_lp)
    explicit LP:  v ← v + (Δt/τ_lp) (−(v − e_leak) + v_hp)

Dynamics only: ``standardize_i_sti`` / ``pre_steady`` / ``step``. Full-T ``v``
forward lives in ``neuron.forward``. Scalars from ``session`` flat fields.

t=0 membrane state uses ``session.pre_steady`` (``--pre-steady …``):

* ``solve`` (default): fixed-iter DC map with ``session.pre_steady_iters`` /
  ``session.pre_steady_damp`` (under-relaxation; not part of dynamics)
* ``probe``: one ``v_syn`` from ``e_leak`` (legacy)
"""
from __future__ import annotations

import torch

from neuron.param import expand_euler
from neuron.schema import syn_strength


def _syn_drive(v, params, backend):
    """``v_out``, ``w``, ``v_syn = a_in · signed_drive(v_out, w)``."""
    v_out = torch.relu(v - params["v_th"]) * params["a_out"]
    w = syn_strength(params)
    return v_out, w, params["a_in"] * backend.conn.signed_drive(v_out, w)


def update_state_hp_lp(
    v, v_slow, params, i_sti, backend, *, delta_ms, state_clamp, g_leak, euler,
    return_component: bool = False,
):
    """One HP→LP membrane step; returns ``(v, v_slow)`` or component extras."""
    euler = expand_euler(euler)
    e_leak = params["e_leak"]
    dt = float(delta_ms)
    tau_lp = torch.clamp(params["tau_lp"], min=dt)
    tau_hp_rise = torch.clamp(params["tau_hp_rise"], min=dt)
    tau_hp_fall = torch.clamp(params["tau_hp_fall"], min=dt)
    a_h = params["a_h"]
    clamp = float(state_clamp)
    g_leak = float(g_leak)
    if g_leak == 0.0:
        raise ValueError("g_leak must be non-zero")

    v_out, w, v_syn = _syn_drive(v, params, backend)
    v_sti = i_sti / g_leak
    v_in = v_syn + v_sti
    tau_hp = torch.where(v_in >= v_slow, tau_hp_rise, tau_hp_fall)
    if euler == "implicit":
        v_slow = (v_slow + (dt / tau_hp) * v_in) / (1.0 + dt / tau_hp)
    else:
        v_slow = v_slow + (dt / tau_hp) * (v_in - v_slow)
    # LP uses post-HP ``v_slow`` (same as prior identity).
    v_hp = v_in - a_h * v_slow
    dt_over_tau_lp = dt / tau_lp
    if return_component:
        if euler == "implicit":
            dv_leak = (dt_over_tau_lp / (1.0 + dt_over_tau_lp)) * (-(v - e_leak))
            dv_hp = (dt_over_tau_lp / (1.0 + dt_over_tau_lp)) * v_hp
        else:
            dv_leak = dt_over_tau_lp * (-(v - e_leak))
            dv_hp = dt_over_tau_lp * v_hp
        v = v + dv_leak + dv_hp
    elif euler == "implicit":
        v = (v + dt_over_tau_lp * (e_leak + v_hp)) / (1.0 + dt_over_tau_lp)
    else:
        v = v + dt_over_tau_lp * (-(v - e_leak) + v_hp)

    v_slow = torch.clamp(v_slow, -clamp, clamp)
    v = torch.clamp(v, -clamp, clamp)

    if not return_component:
        return v, v_slow

    g_exc, g_inh = backend.conn.exc_inh_drive(v_out, w)
    return v, v_slow, {
        "v_syn": v_syn,
        "v_syn_exc": params["a_in"] * g_exc,
        "v_syn_inh": params["a_in"] * g_inh,
        "v_slow": v_slow,
        "v_in": v_in,
        "v_hp": v_hp,
        "dv_leak": dv_leak,
        "dv_hp": dv_hp,
        "i_sti": i_sti,
        "v_sti": v_sti,
    }


def standardize_i_sti(session, params, i_sti, pack):
    """Sti current ``(B, T, N)`` as membrane drive (no rescale)."""
    del params, pack
    return i_sti.unsqueeze(0) if i_sti.dim() == 2 else i_sti


def _dc_v_star(v, params, v_sti, backend):
    """Algebraic DC target: ``v★ = e_leak + (1−a_h)·v_in(v)``."""
    _, _, v_syn = _syn_drive(v, params, backend)
    v_in = v_syn + v_sti
    return params["e_leak"] + (1.0 - params["a_h"]) * v_in, v_in


def pre_steady(session, params, n_b, i_sti=None):
    """``(v_slow,)``, ``v`` at t=0 from ``session.pre_steady``."""
    if i_sti is None:
        raise TypeError("hp_lp pre_steady requires i_sti")
    pre_steady = str(session.pre_steady)
    if pre_steady not in ("probe", "solve"):
        raise ValueError(f"hp_lp pre_steady must be probe|solve; got {pre_steady!r}")
    backend = session.backend
    g_leak = float(session.g_leak)
    if g_leak == 0.0:
        raise ValueError("g_leak must be non-zero")
    v_sti = i_sti[:, 0, :] / g_leak
    e_leak = params["e_leak"]
    v = e_leak.expand(n_b, backend.n_nodes).clone()
    if pre_steady == "probe":
        v_star, v_in = _dc_v_star(v, params, v_sti, backend)
        return (v_in,), v_star
    damp = float(session.pre_steady_damp)
    for _ in range(int(session.pre_steady_iters)):
        v_star, _ = _dc_v_star(v, params, v_sti, backend)
        v = v + damp * (v_star - v)
    _, v_in = _dc_v_star(v, params, v_sti, backend)
    return (v_in,), v


def step(state, v, params, i_sti, session, *, delta_ms: float, return_component: bool = False):
    """One hp_lp update; returns ``((v_slow,), v)`` or + component dict."""
    (v_slow,) = state
    out = update_state_hp_lp(
        v, v_slow, params, i_sti, session.backend,
        delta_ms=float(delta_ms),
        state_clamp=session.state_clamp,
        g_leak=session.g_leak,
        euler=session.euler,
        return_component=return_component,
    )
    if return_component:
        v, v_slow, component = out
        return (v_slow,), v, component
    v, v_slow = out
    return (v_slow,), v
