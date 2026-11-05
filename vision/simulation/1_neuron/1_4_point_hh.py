# -*- coding: utf-8 -*-
"""Differentiable single-compartment Hodgkin-Huxley dynamics.

The numerical interface uses ``ms``, ``mV``, ``pA``, ``pF`` and ``nS``.
Membrane parameters are per node and ionotropic parameters are per directed
edge.  The module contains no fitting or gene-mapping policy.

``pre_steady`` / ``step`` form the model-driver API used by
``neuron.forward``. ``initial_state`` is also public so the kernel can be
tested independently of a training session.
"""
from __future__ import annotations

from typing import Mapping

import torch


STATE_KEYS = ("v", "m", "h", "n", "s")
_MEMBRANE_PARAM_NAMES = ("cap", "gNa", "gK", "gLeak", "eNa", "eK", "eLeak")
_SYNAPSE_PARAM_NAMES = (
    "gS", "e_syn", "v_th_syn", "delta_syn", "k_minus_syn",
)
_PREPARED_KEY = "_point_hh_prepared"


def stable_vtrap(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Return ``x / (exp(x / scale) - 1)`` with its analytic zero limit.

    The regular branch receives a safe nonzero argument.  This is important in
    PyTorch because an unselected ``where`` branch containing ``0 / 0`` can
    still contaminate gradients.
    """
    scale_t = x.new_tensor(float(scale))
    z = x / scale_t
    small = torch.abs(z) < 1.0e-7
    safe_z = torch.where(small, torch.ones_like(z), z)
    regular = x / torch.expm1(safe_z)
    series = scale_t * (1.0 - z / 2.0 + z.square() / 12.0)
    return torch.where(small, series, regular)


def gate_rates(v: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Stable classic-HH ``alpha/beta`` rates in inverse milliseconds."""
    alpha_m = 0.1 * stable_vtrap(-(v + 40.0), 10.0)
    beta_m = 4.0 * torch.exp(-(v + 65.0) / 18.0)
    alpha_h = 0.07 * torch.exp(-(v + 65.0) / 20.0)
    beta_h = 1.0 / (torch.exp(-(v + 35.0) / 10.0) + 1.0)
    alpha_n = 0.01 * stable_vtrap(-(v + 55.0), 10.0)
    beta_n = 0.125 * torch.exp(-(v + 65.0) / 80.0)
    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


def steady_gates(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``m_inf, h_inf, n_inf`` at voltage ``v``."""
    am, bm, ah, bh, an, bn = gate_rates(v)
    return am / (am + bm), ah / (ah + bh), an / (an + bn)


def _rush_larsen(
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    dt_ms: float,
) -> torch.Tensor:
    rate = alpha + beta
    inf = alpha / rate
    updated = inf + (value - inf) * torch.exp(-float(dt_ms) * rate)
    return torch.clamp(updated, 0.0, 1.0)


def _as_param(
    params: Mapping[str, torch.Tensor],
    name: str,
    reference: torch.Tensor,
    length: int,
) -> torch.Tensor:
    if name not in params:
        raise KeyError(f"point_hh requires parameter {name!r}")
    if params.get(_PREPARED_KEY, False):
        return params[name]
    value = torch.as_tensor(params[name], dtype=reference.dtype, device=reference.device)
    if value.numel() == 1:
        return value.reshape(1).expand(int(length))
    value = value.reshape(-1)
    if value.numel() != int(length):
        raise ValueError(
            f"point_hh {name} length {value.numel()} != expected {int(length)}"
        )
    return value


def _edge_indices(
    session,
    device,
    params: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if params is not None and params.get(_PREPARED_KEY, False):
        return params["_source_idx"], params["_target_idx"]
    conn = session.connectome.conn
    source = torch.as_tensor(conn.source_idxs, dtype=torch.long, device=device)
    target = torch.as_tensor(conn.target_idxs, dtype=torch.long, device=device)
    if source.shape != target.shape:
        raise ValueError("point_hh source/target edge arrays must have equal shape")
    return source, target


def _edge_params(
    params: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    n_edge: int,
) -> tuple[torch.Tensor, ...] | None:
    if params.get(_PREPARED_KEY, False):
        if "gS" not in params:
            return None
        return tuple(params[name] for name in _SYNAPSE_PARAM_NAMES)
    present = [
        name in params
        for name in _SYNAPSE_PARAM_NAMES
    ]
    if not any(present):
        return None
    if not all(present):
        missing = [
            name
            for name, exists in zip(
                _SYNAPSE_PARAM_NAMES,
                present,
            )
            if not exists
        ]
        raise KeyError(f"incomplete point_hh synapse parameters: missing {missing}")
    values = tuple(
        _as_param(params, name, reference, n_edge)
        for name in _SYNAPSE_PARAM_NAMES
    )
    if bool((values[0] < 0).any()):
        raise ValueError("point_hh gS must be non-negative")
    if bool((values[3] <= 0).any()):
        raise ValueError("point_hh delta_syn must be positive")
    if bool((values[4] <= 0).any()):
        raise ValueError("point_hh k_minus_syn must be positive")
    return values


def prepare_params(
    params: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    session,
) -> dict[str, torch.Tensor]:
    """Materialize and validate fixed-size tensors once per full forward."""
    if params.get(_PREPARED_KEY, False):
        return params
    n_node = int(session.connectome.n_node)
    prepared = {
        name: _as_param(params, name, reference, n_node)
        for name in _MEMBRANE_PARAM_NAMES
    }
    if bool((prepared["cap"] <= 0).any()):
        raise ValueError("point_hh cap must be positive")
    if bool(
        (prepared["gNa"] < 0).any()
        or (prepared["gK"] < 0).any()
        or (prepared["gLeak"] < 0).any()
    ):
        raise ValueError("point_hh membrane conductances must be non-negative")
    source, target = _edge_indices(session, reference.device)
    syn_params = _edge_params(params, reference, int(source.numel()))
    if syn_params is not None:
        prepared.update(zip(_SYNAPSE_PARAM_NAMES, syn_params))
    prepared["_source_idx"] = source
    prepared["_target_idx"] = target
    prepared[_PREPARED_KEY] = True
    return prepared


def release_steady(
    pre_v: torch.Tensor,
    v_th_syn: torch.Tensor,
    delta_syn: torch.Tensor,
) -> torch.Tensor:
    """Steady ionotropic open probability for each batch and edge."""
    return torch.sigmoid((pre_v - v_th_syn) / delta_syn)


def _release_step(
    s: torch.Tensor,
    pre_v: torch.Tensor,
    v_th_syn: torch.Tensor,
    delta_syn: torch.Tensor,
    k_minus_syn: torch.Tensor,
    dt_ms: float,
) -> torch.Tensor:
    s_inf = release_steady(pre_v, v_th_syn, delta_syn)
    tau_ms = (1.0 - s_inf) / k_minus_syn
    updated = s_inf + (s - s_inf) * torch.exp(
        -float(dt_ms) / torch.clamp(tau_ms, min=torch.finfo(s.dtype).tiny)
    )
    return torch.clamp(updated, 0.0, 1.0)


def _synaptic_terms(
    v: torch.Tensor,
    s: torch.Tensor,
    params: Mapping[str, torch.Tensor],
    session,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return node ``sum(g)``, ``sum(gE)`` and inward synaptic current."""
    source, target = _edge_indices(session, v.device, params)
    n_edge = int(source.numel())
    syn_params = _edge_params(params, v, n_edge)
    zeros = torch.zeros_like(v)
    if syn_params is None or n_edge == 0:
        if s.shape[-1] != n_edge:
            raise ValueError(f"point_hh s has {s.shape[-1]} edges, expected {n_edge}")
        return zeros, zeros, zeros
    g_s, e_syn, _v_th, _delta, _k_minus = syn_params
    if s.shape != v.shape[:-1] + (n_edge,):
        raise ValueError(
            f"point_hh s shape {tuple(s.shape)} != {tuple(v.shape[:-1] + (n_edge,))}"
        )
    edge_g = s * g_s
    sum_g = torch.zeros_like(v)
    sum_g_e = torch.zeros_like(v)
    target_b = target.expand(edge_g.shape)
    sum_g.scatter_add_(-1, target_b, edge_g)
    sum_g_e.scatter_add_(-1, target_b, edge_g * e_syn)
    post_v = v.index_select(-1, target)
    edge_i_in = edge_g * (e_syn - post_v)
    i_syn = torch.zeros_like(v)
    i_syn.scatter_add_(-1, target_b, edge_i_in)
    return sum_g, sum_g_e, i_syn


def initial_state(
    v: torch.Tensor,
    *,
    n_edge: int = 0,
    s: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Construct a state from voltage, using steady membrane gates."""
    m, h, n = steady_gates(v)
    if s is None:
        s = v.new_zeros(v.shape[:-1] + (int(n_edge),))
    return {"v": v, "m": m, "h": h, "n": n, "s": s}


def detach_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach every Point-HH state tensor."""
    return {name: state[name].detach() for name in STATE_KEYS}


def step(
    state: Mapping[str, torch.Tensor],
    params: Mapping[str, torch.Tensor],
    i_sti: torch.Tensor,
    session,
    *,
    delta_ms: float,
    return_component: bool = False,
):
    """Advance one Point-HH model step.

    Gates and release use exact constant-voltage updates.  Voltage uses a
    backward-Euler conductance update with the post-gate/post-release
    conductances frozen over the model step.
    """
    dt = float(delta_ms)
    if dt <= 0.0:
        raise ValueError(f"point_hh delta_ms must be positive; got {dt}")
    v = state["v"]
    params = prepare_params(params, v, session)
    m, h, n, s = (state[name] for name in ("m", "h", "n", "s"))
    n_node = int(v.shape[-1])
    i_sti = torch.as_tensor(i_sti, dtype=v.dtype, device=v.device)
    if i_sti.shape != v.shape:
        i_sti = torch.broadcast_to(i_sti, v.shape)

    cap = _as_param(params, "cap", v, n_node)
    g_na = _as_param(params, "gNa", v, n_node)
    g_k = _as_param(params, "gK", v, n_node)
    g_leak = _as_param(params, "gLeak", v, n_node)
    e_na = _as_param(params, "eNa", v, n_node)
    e_k = _as_param(params, "eK", v, n_node)
    e_leak = _as_param(params, "eLeak", v, n_node)

    am, bm, ah, bh, an, bn = gate_rates(v)
    m_new = _rush_larsen(m, am, bm, dt)
    h_new = _rush_larsen(h, ah, bh, dt)
    n_new = _rush_larsen(n, an, bn, dt)

    source, _target = _edge_indices(session, v.device, params)
    n_edge = int(source.numel())
    syn_params = _edge_params(params, v, n_edge)
    if syn_params is None or n_edge == 0:
        if s.shape[-1] != n_edge:
            raise ValueError(f"point_hh s has {s.shape[-1]} edges, expected {n_edge}")
        s_new = s
    else:
        _g_s, _e_syn, v_th, delta, k_minus = syn_params
        pre_v = v.index_select(-1, source)
        s_new = _release_step(s, pre_v, v_th, delta, k_minus, dt)

    g_na_open = g_na * m_new.pow(3) * h_new
    g_k_open = g_k * n_new.pow(4)
    syn_g, syn_g_e, _ = _synaptic_terms(v, s_new, params, session)
    sum_g = g_na_open + g_k_open + g_leak + syn_g
    sum_g_e = (
        g_na_open * e_na + g_k_open * e_k + g_leak * e_leak + syn_g_e
    )
    dt_over_cap = dt / cap
    v_new = (v + dt_over_cap * (i_sti + sum_g_e)) / (
        1.0 + dt_over_cap * sum_g
    )
    v_min = float(getattr(session, "point_hh_v_min", -120.0))
    v_max = float(getattr(session, "point_hh_v_max", 80.0))
    v_new = torch.clamp(v_new, v_min, v_max)
    new_state = {"v": v_new, "m": m_new, "h": h_new, "n": n_new, "s": s_new}
    if not return_component:
        return new_state

    i_na = g_na_open * (v_new - e_na)
    i_k = g_k_open * (v_new - e_k)
    i_leak = g_leak * (v_new - e_leak)
    _syn_g, _syn_g_e, i_syn = _synaptic_terms(v_new, s_new, params, session)
    component = {
        "i_na": i_na,
        "i_k": i_k,
        "i_leak": i_leak,
        "i_syn": i_syn,
        "i_sti": i_sti,
        "g_na": g_na_open,
        "g_k": g_k_open,
        "g_leak": g_leak.expand_as(v_new),
        "s": s_new,
        "current_residual": (
            cap * (v_new - v) / dt
            - (i_sti - i_na - i_k - i_leak + i_syn)
        ),
    }
    return new_state, component


def _dc_state_from_v(
    v: torch.Tensor,
    params: Mapping[str, torch.Tensor],
    i_sti: torch.Tensor,
    session,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    n_node = int(v.shape[-1])
    cap_dummy = params.get("cap")
    del cap_dummy  # Capacitance does not enter a DC balance.
    g_na = _as_param(params, "gNa", v, n_node)
    g_k = _as_param(params, "gK", v, n_node)
    g_leak = _as_param(params, "gLeak", v, n_node)
    e_na = _as_param(params, "eNa", v, n_node)
    e_k = _as_param(params, "eK", v, n_node)
    e_leak = _as_param(params, "eLeak", v, n_node)
    m, h, n = steady_gates(v)
    source, _target = _edge_indices(session, v.device, params)
    n_edge = int(source.numel())
    syn_params = _edge_params(params, v, n_edge)
    if syn_params is None or n_edge == 0:
        s = v.new_zeros(v.shape[:-1] + (n_edge,))
    else:
        _g_s, _e_syn, v_th, delta, _k_minus = syn_params
        s = release_steady(v.index_select(-1, source), v_th, delta)
    state = {"v": v, "m": m, "h": h, "n": n, "s": s}
    g_na_open = g_na * m.pow(3) * h
    g_k_open = g_k * n.pow(4)
    syn_g, syn_g_e, _ = _synaptic_terms(v, s, params, session)
    sum_g = g_na_open + g_k_open + g_leak + syn_g
    if bool((sum_g <= 0).any()):
        raise ValueError("point_hh DC balance requires positive total conductance")
    sum_g_e = (
        g_na_open * e_na + g_k_open * e_k + g_leak * e_leak + syn_g_e
    )
    return (i_sti + sum_g_e) / sum_g, state


def pre_steady(session, params, n_b, i_sti=None):
    """Build the background-balanced Point-HH state at output sample zero."""
    if i_sti is None:
        raise TypeError("point_hh pre_steady requires i_sti")
    if i_sti.dim() == 2:
        i_sti = i_sti.unsqueeze(1)
    i0 = i_sti[:, 0, :]
    params = prepare_params(params, i0, session)
    n_node = int(session.connectome.n_node)
    if i0.shape != (int(n_b), n_node):
        raise ValueError(
            f"point_hh background current shape {tuple(i0.shape)} "
            f"!= {(int(n_b), n_node)}"
        )
    e_leak = _as_param(params, "eLeak", i0, n_node)
    v = e_leak.expand(int(n_b), n_node).clone()
    mode = str(session.pre_steady)
    if mode not in ("probe", "solve"):
        raise ValueError(f"point_hh pre_steady must be probe|solve; got {mode!r}")
    n_iter = 1 if mode == "probe" else int(session.pre_steady_n_iter)
    damp = 1.0 if mode == "probe" else float(session.pre_steady_damp)
    v_min = float(getattr(session, "point_hh_v_min", -120.0))
    v_max = float(getattr(session, "point_hh_v_max", 80.0))
    state = None
    for _ in range(n_iter):
        target, state = _dc_state_from_v(v, params, i0, session)
        v = torch.clamp(v + damp * (target - v), v_min, v_max)
    _target, state = _dc_state_from_v(v, params, i0, session)
    state["v"] = v
    return state


__all__ = [
    "STATE_KEYS",
    "detach_state",
    "gate_rates",
    "initial_state",
    "prepare_params",
    "pre_steady",
    "release_steady",
    "stable_vtrap",
    "steady_gates",
    "step",
]
