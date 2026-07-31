# -*- coding: utf-8 -*-
"""HP-then-membrane-LP neuron (``--model hp_lp``).

    τ_HP da/dt = X − a
    τ_lp dv/dt = −(v − v_rest) + G (X − a)

with X = v_rest + syn + x_t, syn from relu(v)·out_gain scaled by syn_strength
(type_pair) or edge_weight (per_edge).

Dynamics only: ``prepare_signal`` / ``init_state`` / ``step``. Full-T ``v``
forward lives in ``neuron.forward``. Scalars from ``session`` flat fields.
"""
from __future__ import annotations

import torch

from neuron.schema import synaptic_scale


def update_state_hp_lp(v, a, p, x_t, backend, *, delta_ms, state_clamp):
    """One HP→LP membrane step; returns (v, a)."""
    v_rest = p["v_rest"]
    dt = float(delta_ms)
    tau_lp = torch.clamp(p["tau_lp"], min=dt)
    tau_hp = torch.clamp(p["tau_hp"], min=dt)
    G = p["hp_gain"]
    clamp = float(state_clamp)

    syn = p["in_gain"] * backend.conn.signed_drive(
        torch.relu(v) * p["out_gain"], synaptic_scale(p),
    )
    X = v_rest + syn + x_t
    a = a + dt / tau_hp * (X - a)
    v = v + dt / tau_lp * (-(v - v_rest) + G * (X - a))

    a = torch.clamp(a, -clamp, clamp)
    v = torch.clamp(v, -clamp, clamp)
    return v, a


def prepare_signal(session, p, sig, pack):
    """PR current scaled by peak ``i_*`` → activity-model drive ``(B, T, N)``.

    Scale is ``pack.signal_scale`` (stamped at session build); no training import.
    """
    scale = float(getattr(pack, "signal_scale", 1.0) or 1.0)
    if scale == 0.0:
        raise ValueError("pack.signal_scale must be non-zero")
    return sig / scale


def init_state(session, p, B):
    """``(a,)``, ``v0 = v_rest``."""
    v_rest = p["v_rest"]
    n = session.backend.n_units
    v = v_rest.expand(B, n).clone()
    a = v_rest.expand(B, n).clone()
    return (a,), v


def step(state, v, p, x_t, session):
    a, = state
    v, a = update_state_hp_lp(
        v, a, p, x_t, session.backend,
        delta_ms=session.delta_ms, state_clamp=session.STATE_CLAMP,
    )
    return (a,), v
