# -*- coding: utf-8 -*-
"""Shiu et al. whole-brain leaky integrate-and-fire neuron (``model=shiu``).

The fixed biophysical parameters reproduce the model in
Shiu et al., Nature 2024:

    dv/dt = (V_REST - v + g + v_sti) / T_MEMBRANE_MS
    dg/dt = -g / T_SYNAPSE_MS

An incoming spike adds ``n_syn * W_SYNAPSE_MV`` to ``g``.  Connectivity
signs select excitation or inhibition.  A neuron crossing ``V_THRESHOLD_MV``
spikes, clears ``g``, resets to ``V_RESET_MV``, and is refractory for
``T_REFRACTORY_MS``.  The previous step's spikes drive the next step, which
approximates the paper's 1.8 ms synaptic delay at the default 2 ms step.

The paper stimulates selected neurons with Poisson spikes.  This simulation's
task interface instead supplies deterministic current ``i_sti``; it is
converted to a voltage drive with ``v_sti = i_sti / session.g_leak``.

``pre_steady`` / ``step`` intentionally match the three-state driver interface
used by ``neuron.forward``: ``(g, refractory_ms, v)``.
"""
from __future__ import annotations

import torch

from neuron.borst import syn_strength


V_REST_MV = -52.0
V_RESET_MV = -52.0
V_THRESHOLD_MV = -45.0
T_MEMBRANE_MS = 20.0
T_SYNAPSE_MS = 5.0
T_REFRACTORY_MS = 2.2
T_DELAY_MS = 1.8
W_SYNAPSE_MV = 0.275


def _synaptic_jump(spiked, params, session):
    """Signed Shiu ``g`` jump from a binary presynaptic spike tensor.

    ``ScatterConn`` has already multiplied edge counts by ``a_syn_exc`` /
    ``a_syn_inh``.  Divide those construction scales back out before applying
    the paper's fixed 0.275 mV per anatomical synapse.
    """
    a_exc = float(session.a_syn_exc)
    a_inh = float(session.a_syn_inh)
    if a_exc == 0.0 or a_inh == 0.0:
        raise ValueError("shiu requires non-zero a_syn_exc and a_syn_inh")
    g_exc, g_inh = session.connectome.conn.exc_inh_g(
        spiked, syn_strength(params),
    )
    return params["a_in"] * W_SYNAPSE_MV * (
        g_exc / a_exc - g_inh / a_inh
    )


def update_v(
    g,
    refractory_ms,
    v,
    params,
    i_sti,
    session,
    *,
    delta_ms: float,
    return_component: bool = False,
):
    """Advance one discretized Shiu LIF step.

    The suprathreshold voltage is retained for one output sample so
    ``forward_v`` exposes spike events.  It is reset at the start of the next
    step, before any further membrane integration.
    """
    dt = float(delta_ms)
    if dt <= 0.0:
        raise ValueError(f"delta_ms must be > 0; got {dt}")
    g_leak = float(session.g_leak)
    if g_leak == 0.0:
        raise ValueError("shiu requires non-zero g_leak")

    # ``refractory_ms == T_REFRACTORY_MS`` marks a spike emitted on the
    # preceding sample.  Use a dtype-scaled tolerance only for round-off.
    eps = torch.finfo(refractory_ms.dtype).eps * max(T_REFRACTORY_MS, 1.0) * 8.0
    spiked_pre = refractory_ms >= T_REFRACTORY_MS - eps
    v = torch.where(spiked_pre, v.new_tensor(V_RESET_MV), v)

    syn_jump = _synaptic_jump(spiked_pre.to(v.dtype), params, session)
    dt_over_tau_syn = dt / T_SYNAPSE_MS
    if session.euler == "implicit":
        g = g / (1.0 + dt_over_tau_syn)
    else:
        g = g + dt_over_tau_syn * (-g)
    g = g + syn_jump

    active = refractory_ms <= 0.0
    refractory_next = torch.clamp(refractory_ms - dt, min=0.0)
    v_sti = i_sti / g_leak
    dt_over_tau_m = dt / T_MEMBRANE_MS
    if session.euler == "implicit":
        v_candidate = (
            v + dt_over_tau_m * (V_REST_MV + g + v_sti)
        ) / (1.0 + dt_over_tau_m)
    else:
        v_candidate = v + dt_over_tau_m * (
            V_REST_MV - v + g + v_sti
        )
    v = torch.where(active, v_candidate, v.new_tensor(V_RESET_MV))

    spiked = active & (v > V_THRESHOLD_MV)
    g = torch.where(spiked, torch.zeros_like(g), g)
    refractory_ms = torch.where(
        spiked,
        refractory_next.new_tensor(T_REFRACTORY_MS),
        refractory_next,
    )

    if return_component:
        return g, refractory_ms, v, {
            "spiked": spiked,
            "spiked_pre": spiked_pre,
            "syn_jump": syn_jump,
            "g": g,
            "refractory_ms": refractory_ms,
            "i_sti": i_sti,
            "v_sti": v_sti,
        }
    return g, refractory_ms, v


def pre_steady(session, params, n_b, i_sti=None):
    """Initialize the paper's zero-Hz baseline at ``V_REST_MV``."""
    if i_sti is None:
        raise TypeError("shiu pre_steady requires i_sti")
    mode = str(session.pre_steady)
    if mode not in ("probe", "solve"):
        raise ValueError(f"shiu pre_steady must be probe|solve; got {mode!r}")
    shape = (int(n_b), int(session.connectome.n_node))
    v = i_sti.new_full(shape, V_REST_MV)
    g = torch.zeros_like(v)
    refractory_ms = torch.zeros_like(v)
    return g, refractory_ms, v


def step(
    g,
    refractory_ms,
    v,
    params,
    i_sti,
    session,
    *,
    delta_ms: float,
    return_component: bool = False,
):
    """One Shiu update; return ``(g, refractory_ms, v)``."""
    return update_v(
        g,
        refractory_ms,
        v,
        params,
        i_sti,
        session,
        delta_ms=delta_ms,
        return_component=return_component,
    )
