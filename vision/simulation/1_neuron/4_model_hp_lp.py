# -*- coding: utf-8 -*-
"""HP-then-membrane-LP neuron (``--model hp_lp``).

    τ_HP d v_slow / dt = v_tot − v_slow
    v_hp = v_tot − a_slow v_slow
    τ_lp dv/dt = −(v − v_rest) + v_hp

with v_tot = v_rest + v_in + v_sti, v_sti = i_sti/g_in (g_in in nS converts pA → mV),
v_in from relu(v+bias_out)·a_out scaled by syn_strength_cell (per_cell) or
syn_strength_edge (per_edge). a_slow = 1 recovers classical HP
(v_hp = v_tot − v_slow); DC then has v_hp → 0 when v_slow → v_tot.

Membrane / HP Euler (``session.euler`` = ``implicit`` | ``explicit``):

    α_hp = Δt/τ_hp,  α_lp = Δt/τ_lp

    implicit HP:  v_slow ← (v_slow + α_hp v_tot) / (1 + α_hp)
    explicit HP:  v_slow ← v_slow + α_hp (v_tot − v_slow)

    implicit LP:  v ← (v + α_lp (v_rest + v_hp)) / (1 + α_lp)
    explicit LP:  v ← v + α_lp (−(v − v_rest) + v_hp)

Dynamics only: ``prepare_i_sti`` / ``pre_steady`` / ``step``. Full-T ``v``
forward lives in ``neuron.forward``. Scalars from ``session`` flat fields.

t=0 membrane state uses ``session.pre_steady`` (``--pre-steady …``):

* ``solve`` (default): fixed-iter DC map with ``session.pre_steady_iters`` /
  ``session.pre_steady_damp`` (under-relaxation; not part of dynamics)
* ``probe``: one ``v_in`` from ``v_rest`` (legacy)
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
    a_slow = p["a_slow"]
    clamp = float(state_clamp)
    g_in = float(g_in)
    if g_in == 0.0:
        raise ValueError("g_in must be non-zero")

    pre = torch.relu(v + p["bias_out"]) * p["a_out"]
    w = syn_strength(p)
    v_in = p["a_in"] * backend.conn.signed_drive(pre, w)
    v_sti = i_sti / g_in
    v_tot = v_rest + v_in + v_sti
    hp_dt_over_tau = dt / tau_hp
    if euler == "implicit":
        v_slow = (v_slow + hp_dt_over_tau * v_tot) / (1.0 + hp_dt_over_tau)
    else:
        v_slow = v_slow + hp_dt_over_tau * (v_tot - v_slow)
    # LP uses post-HP ``v_slow`` (same as prior identity).
    v_hp = v_tot - a_slow * v_slow
    lp_dt_over_tau = dt / tau_lp
    if euler == "implicit":
        scale = lp_dt_over_tau / (1.0 + lp_dt_over_tau)
        dv_leak = scale * (-(v - v_rest))
        dv_hp = scale * v_hp
        v = (v + lp_dt_over_tau * (v_rest + v_hp)) / (1.0 + lp_dt_over_tau)
    else:
        dv_leak = lp_dt_over_tau * (-(v - v_rest))
        dv_hp = lp_dt_over_tau * v_hp
        v = v + dv_leak + dv_hp

    v_slow = torch.clamp(v_slow, -clamp, clamp)
    v = torch.clamp(v, -clamp, clamp)

    if not return_component:
        return v, v_slow

    g_exc, g_inh = backend.conn.exc_inh_drive(pre, w)
    v_in_exc = p["a_in"] * g_exc
    v_in_inh = p["a_in"] * g_inh
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


def _v_in_from_v(v, p, backend):
    pre = torch.relu(v + p["bias_out"]) * p["a_out"]
    w = syn_strength(p)
    return p["a_in"] * backend.conn.signed_drive(pre, w)


def _dc_v_star(v, p, v_sti, backend):
    """Algebraic DC target: ``v★ = v_rest + (1−a_slow)·v_tot(v)``."""
    v_rest = p["v_rest"]
    v_tot = v_rest + _v_in_from_v(v, p, backend) + v_sti
    return v_rest + (1.0 - p["a_slow"]) * v_tot, v_tot


def _pre_steady_probe(p, B, v_sti, backend):
    """One-shot: ``v_in`` from ``v=v_rest`` (not self-consistent)."""
    v_rest = p["v_rest"]
    n = backend.n_nodes
    v_probe = v_rest.expand(B, n).clone()
    v_tot = v_rest + _v_in_from_v(v_probe, p, backend) + v_sti
    v_slow = v_tot
    v = v_rest + v_tot - p["a_slow"] * v_slow
    return (v_slow,), v


def _pre_steady_solve(p, B, v_sti, backend, *, iters: int, damp: float):
    """Fixed-iter under-relaxed solve of the DC map (not time stepping)."""
    v_rest = p["v_rest"]
    n = backend.n_nodes
    v = v_rest.expand(B, n).clone()
    v_tot = v_rest + _v_in_from_v(v, p, backend) + v_sti
    for _ in range(int(iters)):
        v_star, v_tot = _dc_v_star(v, p, v_sti, backend)
        v = v + float(damp) * (v_star - v)
    v_slow = v_tot
    return (v_slow,), v


def pre_steady(session, p, B, i_sti=None):
    """``(v_slow,)``, ``v`` at t=0 from ``session.pre_steady``."""
    if i_sti is None:
        raise TypeError("hp_lp pre_steady requires i_sti")
    backend = session.backend
    g_in = float(session.g_in)
    if g_in == 0.0:
        raise ValueError("g_in must be non-zero")
    v_sti = i_sti[:, 0, :] / g_in
    mode = str(session.pre_steady)
    if mode == "probe":
        return _pre_steady_probe(p, B, v_sti, backend)
    if mode == "solve":
        return _pre_steady_solve(
            p, B, v_sti, backend,
            iters=int(session.pre_steady_iters),
            damp=float(session.pre_steady_damp),
        )
    raise ValueError(
        f"hp_lp pre_steady must be probe|solve; got {mode!r}"
    )


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
