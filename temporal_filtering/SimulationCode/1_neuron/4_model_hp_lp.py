# -*- coding: utf-8 -*-
"""HP-then-membrane-LP neuron (``--model hp_lp``).

    τ_HP da/dt = X − a
    τ_lp dv/dt = −(v − bias) + G (X − a)

with X = bias + syn + x_t, syn from relu(v)·out_gain scaled by syn_strength
(type_pair) or edge_weight (per_edge).

Dynamics only: ``prepare_signal`` / ``init_state`` / ``step``. Full-T Ca
forward lives in ``neuron.forward``.
"""
from __future__ import annotations

import torch

from neuron.params import STATE_CLAMP, deltat
from neuron.schema import synaptic_scale


def update_state_hp_lp(v, a, p, x_t, backend):
    """One HP→LP membrane step; returns (v, a)."""
    bias = p["bias"]
    tau_lp = torch.clamp(p["tau_lp"], min=deltat)
    tau_hp = torch.clamp(p["tau_hp"], min=deltat)
    G = p["hp_gain"]

    syn = p["in_gain"] * backend.conn.signed_drive(
        torch.relu(v) * p["out_gain"], synaptic_scale(p),
    )
    X = bias + syn + x_t
    a = a + deltat / tau_hp * (X - a)
    v = v + deltat / tau_lp * (-(v - bias) + G * (X - a))

    a = torch.clamp(a, -STATE_CLAMP, STATE_CLAMP)
    v = torch.clamp(v, -STATE_CLAMP, STATE_CLAMP)
    return v, a


def prepare_signal(session, p, sig, pack):
    """PR current scaled by peak ``i_*`` → activity-model drive ``(B, T, N)``.

    Scale is ``pack.signal_scale`` (stamped at session build); no training import.
    """
    del p
    pack = pack or session.primary_pack
    x = sig.unsqueeze(0) if sig.dim() == 2 else sig
    scale = float(getattr(pack, "signal_scale", 1.0) or 1.0)
    return x / scale


def init_state(session, p, B):
    """``(a,)``, ``v0 = bias``."""
    bias = p["bias"]
    n = session.backend.n_units
    v = bias.expand(B, n).clone()
    a = bias.expand(B, n).clone()
    return (a,), v


def step(state, v, p, x_t, session):
    """One HP→LP update; returns ``((a,), v)``."""
    (a,) = state
    v, a = update_state_hp_lp(v, a, p, x_t, session.backend)
    return (a,), v
