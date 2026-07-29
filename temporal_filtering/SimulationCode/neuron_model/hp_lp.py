# -*- coding: utf-8 -*-
"""HP-then-membrane-LP neuron (``--model hp_lp``).

    τ_HP da/dt = X − a
    τ_lp dV/dt = −(V − bias) + G (X − a)

with X = bias + syn + x_t, syn from relu(V)·out_gain scaled by syn_strength
(type_pair) or edge_weight (per_edge).

Dynamics only: ``prepare_signal`` / ``init_state`` / ``step``. Full-T Ca
forward lives in ``neuron_model.forward``.
"""
from __future__ import annotations

import torch

from neuron_model.constants import STATE_CLAMP, deltat
from neuron_model.schema import synaptic_scale


def update_state_hp_lp(V, a, p, x_t, backend):
    """One HP→LP membrane step; returns (V, a)."""
    bias = p["bias"]
    tau_lp = torch.clamp(p["tau_lp"], min=deltat)
    tau_hp = torch.clamp(p["tau_hp"], min=deltat)
    G = p["hp_gain"]

    syn = p["in_gain"] * backend.conn.signed_drive(
        torch.relu(V) * p["out_gain"], synaptic_scale(p),
    )
    X = bias + syn + x_t
    a = a + deltat / tau_hp * (X - a)
    V = V + deltat / tau_lp * (-(V - bias) + G * (X - a))

    a = torch.clamp(a, -STATE_CLAMP, STATE_CLAMP)
    V = torch.clamp(V, -STATE_CLAMP, STATE_CLAMP)
    return V, a


def prepare_signal(session, p, sig, pack):
    """PR current scaled by peak ``i_*`` → activity-model drive ``(B, T, N)``."""
    del p
    import FiveCol_MedSim_Pytorch as fc

    pack = pack or session.primary_pack
    x = sig.unsqueeze(0) if sig.dim() == 2 else sig
    return x / fc._pack_signal_scale(pack, session)


def init_state(session, p, B):
    """``(a,)``, ``V0 = bias``."""
    bias = p["bias"]
    n = session.backend.n_units
    V = bias.expand(B, n).clone()
    a = bias.expand(B, n).clone()
    return (a,), V


def step(state, V, p, x_t, session):
    """One HP→LP update; returns ``((a,), V)``."""
    (a,) = state
    V, a = update_state_hp_lp(V, a, p, x_t, session.backend)
    return (a,), V
