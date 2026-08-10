# -*- coding: utf-8 -*-
"""HP-then-membrane-LP neuron (``--model hp_lp``).

    τ_HP d v_slow / dt = v_in − v_slow
    v_hp = v_in − a_h v_slow
    τ_lp dv/dt = −(v − e_leak) + v_hp

with v_in = v_syn + v_sti (no e_leak; leak alone sets rest), v_sti = i_sti/g_leak
(g_leak in nS converts pA → mV; same scalar as borst),
v_out = max(v−v_th, 0)·a_out, v_syn = a_in · signed_drive(v_out, syn_strength)
(syn_strength_cell per_cell or syn_strength_edge per_edge). a_h = 1
recovers classical HP (v_hp = v_in − v_slow); DC then has v_hp → 0 when
v_slow → v_in, so v → e_leak.

Membrane / HP Euler (``session.euler`` = ``implicit`` | ``explicit``):

    α_hp = Δt/τ_hp,  α_lp = Δt/τ_lp

    implicit HP:  v_slow ← (v_slow + α_hp v_in) / (1 + α_hp)
    explicit HP:  v_slow ← v_slow + α_hp (v_in − v_slow)

    implicit LP:  v ← (v + α_lp (e_leak + v_hp)) / (1 + α_lp)
    explicit LP:  v ← v + α_lp (−(v − e_leak) + v_hp)

Dynamics only: ``prepare_i_sti`` / ``pre_steady`` / ``step``. Full-T ``v``
forward lives in ``neuron.forward``. Scalars from ``session`` flat fields.

t=0 membrane state uses ``session.pre_steady`` (``--pre-steady …``):

* ``solve`` (default): fixed-iter DC map with ``session.pre_steady_iters`` /
  ``session.pre_steady_damp`` (under-relaxation; not part of dynamics)
* ``probe``: one ``v_syn`` from ``e_leak`` (legacy)
"""
from __future__ import annotations

import torch

from neuron.params import expand_euler
from neuron.schema import syn_strength


def _syn_drive(v, p, backend):
    """``v_out``, ``w``, ``v_syn = a_in · signed_drive(v_out, w)``."""
    v_out = torch.relu(v - p["v_th"]) * p["a_out"]
    w = syn_strength(p)
    return v_out, w, p["a_in"] * backend.conn.signed_drive(v_out, w)


def update_state_hp_lp(
    v, v_slow, p, i_sti, backend, *, delta_ms, state_clamp, g_leak, euler,
    return_component: bool = False,
):
    """One HP→LP membrane step; returns ``(v, v_slow)`` or component extras."""
    euler = expand_euler(euler)
    e_leak = p["e_leak"]
    dt = float(delta_ms)
    tau_lp = torch.clamp(p["tau_lp"], min=dt)
    tau_hp = torch.clamp(p["tau_hp"], min=dt)
    a_h = p["a_h"]
    clamp = float(state_clamp)
    g_leak = float(g_leak)
    if g_leak == 0.0:
        raise ValueError("g_leak must be non-zero")

    v_out, w, v_syn = _syn_drive(v, p, backend)
    v_sti = i_sti / g_leak
    v_in = v_syn + v_sti
    hp_dt_over_tau = dt / tau_hp
    hp_scale = (
        hp_dt_over_tau / (1.0 + hp_dt_over_tau) if euler == "implicit" else hp_dt_over_tau
    )
    v_slow = v_slow + hp_scale * (v_in - v_slow)
    # LP uses post-HP ``v_slow`` (same as prior identity).
    v_hp = v_in - a_h * v_slow
    lp_dt_over_tau = dt / tau_lp
    lp_scale = (
        lp_dt_over_tau / (1.0 + lp_dt_over_tau) if euler == "implicit" else lp_dt_over_tau
    )
    if return_component:
        dv_leak = lp_scale * (-(v - e_leak))
        dv_hp = lp_scale * v_hp
        v = v + dv_leak + dv_hp
    else:
        v = v + lp_scale * (-(v - e_leak) + v_hp)

    v_slow = torch.clamp(v_slow, -clamp, clamp)
    v = torch.clamp(v, -clamp, clamp)

    if not return_component:
        return v, v_slow

    g_exc, g_inh = backend.conn.exc_inh_drive(v_out, w)
    return v, v_slow, {
        "v_syn": v_syn,
        "v_syn_exc": p["a_in"] * g_exc,
        "v_syn_inh": p["a_in"] * g_inh,
        "v_slow": v_slow,
        "v_in": v_in,
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


def _dc_v_star(v, p, v_sti, backend):
    """Algebraic DC target: ``v★ = e_leak + (1−a_h)·v_in(v)``."""
    _, _, v_syn = _syn_drive(v, p, backend)
    v_in = v_syn + v_sti
    return p["e_leak"] + (1.0 - p["a_h"]) * v_in, v_in


def pre_steady(session, p, B, i_sti=None):
    """``(v_slow,)``, ``v`` at t=0 from ``session.pre_steady``."""
    if i_sti is None:
        raise TypeError("hp_lp pre_steady requires i_sti")
    mode = str(session.pre_steady)
    if mode not in ("probe", "solve"):
        raise ValueError(f"hp_lp pre_steady must be probe|solve; got {mode!r}")
    backend = session.backend
    g_leak = float(session.g_leak)
    if g_leak == 0.0:
        raise ValueError("g_leak must be non-zero")
    v_sti = i_sti[:, 0, :] / g_leak
    e_leak = p["e_leak"]
    v = e_leak.expand(B, backend.n_nodes).clone()
    if mode == "probe":
        v_star, v_in = _dc_v_star(v, p, v_sti, backend)
        return (v_in,), v_star
    damp = float(session.pre_steady_damp)
    for _ in range(int(session.pre_steady_iters)):
        v_star, _ = _dc_v_star(v, p, v_sti, backend)
        v = v + damp * (v_star - v)
    _, v_in = _dc_v_star(v, p, v_sti, backend)
    return (v_in,), v


def step(state, v, p, i_sti, session, *, delta_ms: float, return_component: bool = False):
    """One hp_lp update; returns ``((v_slow,), v)`` or + component dict."""
    (v_slow,) = state
    out = update_state_hp_lp(
        v, v_slow, p, i_sti, session.backend,
        delta_ms=float(delta_ms),
        state_clamp=session.STATE_CLAMP,
        g_leak=session.g_leak,
        euler=session.euler,
        return_component=return_component,
    )
    if return_component:
        v, v_slow, component = out
        return (v_slow,), v, component
    v, v_slow = out
    return (v_slow,), v
