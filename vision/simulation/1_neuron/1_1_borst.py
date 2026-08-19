# -*- coding: utf-8 -*-
"""Borst neuron + i_h (``--model borst``).

Shared model helpers (time/ms conversion, ``syn_strength``, ``expand_euler``)
live here; ``1_2_hp_lp`` imports them.

Dynamics only: ``pre_steady`` / ``step``. Ca
forward lives in ``neuron.forward``. Scalars are injected kwargs
(from ``session`` flat fields), never nested under Physics.

t=0 uses ``session.pre_steady`` (``--pre-steady …``):

* ``probe``: ``g_syn`` from ``v=e_leak``, ``u/u_rev=0``, then ohmic
  ``v = (i_sti + Σ g·e) / Σ g``
* ``solve``: fixed-iter under-relaxed DC map with i_h at ``ss(v)``;
  uses ``session.pre_steady_n_iter`` / ``session.pre_steady_damp``

Euler (``session.euler`` = ``implicit`` | ``explicit``):

    cap dv/dt = i_sti + Σ g_i (E_i − v)

    implicit:  v ← (v + α (i_sti + Σ g·e)) / (1 + α Σ g),  α = Δt/cap
    explicit:  v ← v + α (i_sti + Σ g·e − Σ g · v)

i_h gate kinetics are always explicit Euler, independent of ``euler``.

Conductances: ``g_h = u · h_g_max · a_h``, ``g_h_rev = u_rev · h_g_max · a_h_rev``
with fixed ``h_g_max`` (session) and ``a_h`` in z / ``a_h_rev``.
"""
from __future__ import annotations

import torch

EULER_CLI = {"im": "implicit", "ex": "explicit"}
EULER_MODES = tuple(EULER_CLI.values())


def t_from_ms(ms: float, *, delta_ms: float) -> int:
    """Convert milliseconds to time index count ``t`` (rounded)."""
    return int(round(float(ms) / float(delta_ms)))


def ms_from_t(
    t: float,
    *,
    t_onset: int,
    delta_ms_pre: float,
    delta_ms: float,
) -> float:
    """Absolute ms at sample index ``t`` (piecewise pre / post-onset dt)."""
    t = float(t)
    t0 = int(t_onset)
    if t <= t0:
        return t * float(delta_ms_pre)
    return t0 * float(delta_ms_pre) + (t - t0) * float(delta_ms)


def t_abs_from_ms(
    ms: float,
    *,
    t_onset: int,
    delta_ms_pre: float,
    delta_ms: float,
) -> int:
    """Absolute sample index for ms from t=0 with piecewise pre / post dt."""
    ms = float(ms)
    t0 = int(t_onset)
    dt_pre = float(delta_ms_pre)
    ms_pre = t0 * dt_pre
    if ms <= ms_pre:
        return t_from_ms(ms, delta_ms=dt_pre)
    return t0 + t_from_ms(ms - ms_pre, delta_ms=delta_ms)


def e_h_rev(e_leak, e_h: float):
    """Rev-channel reversal ``2 * e_leak - e_h`` (scalar or per-node)."""
    return 2.0 * e_leak - float(e_h)


def expand_euler(token: str) -> str:
    """Map CLI ``im`` / ``ex`` → ``implicit`` / ``explicit``."""
    key = EULER_CLI.get(str(token))
    if key is None:
        raise ValueError(f"euler {token!r} not in CLI {tuple(EULER_CLI)}")
    return key


def syn_strength(params):
    """Active syn_strength tensor (exactly one of syn_strength_cell / syn_strength_edge)."""
    if "syn_strength_edge" in params:
        return params["syn_strength_edge"]
    return params["syn_strength_cell"]


def _gate_ss(v, v_mid, slope):
    return 1.0 / (1.0 + torch.exp((v_mid - v) * slope))


def _i_h_tau(v, v_mid):
    return (
        1.5 / (torch.exp(-0.1 * (v - v_mid)) + torch.exp(+0.1 * (v - v_mid))) * 1000.0
        + 100.0
    )


def _i_h_gate_step(
    v, u, u_rev, a_h, a_h_rev,
    v_mid_h_g, h_slope, v_mid_h_tau, v_mid_h_g_rev, h_slope_rev, v_mid_h_tau_rev,
    *,
    delta_ms: float,
    h_g_max: float,
):
    """Advance i_h gates and channel conductances for masked hexes only.

    Gate ODE uses explicit Euler regardless of ``euler``.
    """
    i_h_ss = _gate_ss(v, v_mid_h_g, h_slope)
    i_h_ss_rev = _gate_ss(v, v_mid_h_g_rev, -h_slope_rev)
    dt = float(delta_ms)
    dt_over_tau_h = dt / _i_h_tau(v, v_mid_h_tau)
    dt_over_tau_h_rev = dt / _i_h_tau(v, v_mid_h_tau_rev)
    u = dt_over_tau_h * (i_h_ss - u) + u
    u_rev = dt_over_tau_h_rev * (i_h_ss_rev - u_rev) + u_rev
    gmax = float(h_g_max)
    return u, u_rev, u * gmax * a_h, u_rev * gmax * a_h_rev


def update_v(
    v, u, u_rev, a_in, a_out, syn_strength_val, v_th, a_h, a_h_rev,
    v_mid_h_g, h_slope, v_mid_h_tau, v_mid_h_g_rev, h_slope_rev, v_mid_h_tau_rev,
    i_sti, connectome, e_leak, *,
    delta_ms: float,
    cap: float,
    g_leak: float,
    e_exc: float,
    e_inh: float,
    e_h: float,
    h_g_max: float,
    euler: str,
    return_component: bool = False,
):
    """One borst step; reversal / cap scalars are required kwargs."""
    conn = connectome.conn
    i_h_mask = (a_h + a_h_rev) != 0
    g_h = u.new_zeros(u.shape)
    g_h_rev = u_rev.new_zeros(u_rev.shape)
    i_h_kwargs_common = dict(delta_ms=delta_ms, h_g_max=h_g_max)
    if i_h_mask.any():
        i_h_kwargs = dict(
            v_mid_h_g=v_mid_h_g, h_slope=h_slope, v_mid_h_tau=v_mid_h_tau,
            v_mid_h_g_rev=v_mid_h_g_rev, h_slope_rev=h_slope_rev, v_mid_h_tau_rev=v_mid_h_tau_rev,
        )
        if i_h_mask.all():
            u, u_rev, g_h, g_h_rev = _i_h_gate_step(
                v, u, u_rev, a_h, a_h_rev, **i_h_kwargs_common, **i_h_kwargs)
        else:
            idx = i_h_mask
            u_a, u_rev_a, g_a, g_rev_a = _i_h_gate_step(
                v[:, idx], u[:, idx], u_rev[:, idx],
                a_h[idx], a_h_rev[idx],
                **i_h_kwargs_common,
                **{k: val[idx] for k, val in i_h_kwargs.items()},
            )
            u = u.clone()
            u_rev = u_rev.clone()
            u[:, idx] = u_a.to(dtype=u.dtype)
            u_rev[:, idx] = u_rev_a.to(dtype=u_rev.dtype)
            g_h[:, idx] = g_a.to(dtype=g_h.dtype)
            g_h_rev[:, idx] = g_rev_a.to(dtype=g_h_rev.dtype)

    v_out = torch.relu(v - v_th) * a_out
    g_exc, g_inh = conn.exc_inh_g(v_out, syn_strength_val)
    g_exc = g_exc * a_in
    g_inh = g_inh * a_in

    dt = float(delta_ms)
    dt_over_cap = dt / cap
    e_h_rev_val = e_h_rev(e_leak, e_h)
    sum_g_e = (
        g_exc * e_exc + g_inh * e_inh + g_leak * e_leak
        + e_h * g_h + e_h_rev_val * g_h_rev
    )
    sum_g = g_exc + g_inh + g_h + g_h_rev + g_leak
    if euler == "implicit":
        v = (v + dt_over_cap * (i_sti + sum_g_e)) / (1.0 + dt_over_cap * sum_g)
    else:
        v = v + dt_over_cap * (i_sti + sum_g_e - sum_g * v)

    if return_component:
        return v, u, u_rev, g_exc, g_inh, g_h, g_h_rev
    return v, u, u_rev


def v_component_from_g(
    v_pre, g_exc, g_inh, g_h, g_h_rev, i_sti, e_leak, *,
    delta_ms: float,
    cap: float,
    g_leak: float,
    e_exc: float,
    e_inh: float,
    e_h: float,
    euler: str,
):
    """Numerator / denom terms matching ``update_v`` (torch or numpy)."""
    dt = float(delta_ms)
    dt_over_cap = dt / cap
    e_h_rev_val = e_h_rev(e_leak, e_h)
    num_exc = g_exc * e_exc
    num_inh = g_inh * e_inh
    num_leak = g_leak * e_leak
    num_i_h = g_h * e_h
    num_i_h_rev = g_h_rev * e_h_rev_val
    sum_g_e = num_exc + num_inh + num_leak + num_i_h + num_i_h_rev
    sum_g = g_exc + g_inh + g_h + g_h_rev + g_leak
    if euler == "implicit":
        num = v_pre + dt_over_cap * (i_sti + sum_g_e)
        den = 1.0 + dt_over_cap * sum_g
        num_v = v_pre
    else:
        num = v_pre + dt_over_cap * (i_sti + sum_g_e - sum_g * v_pre)
        den = 1.0
        num_v = v_pre * (1.0 - dt_over_cap * sum_g)
    return {
        "num_exc": num_exc,
        "num_inh": num_inh,
        "num_leak": num_leak,
        "num_i_h": num_i_h,
        "num_i_h_rev": num_i_h_rev,
        "num_v": num_v,
        "i_sti": i_sti,
        "sum_g_e": sum_g_e,
        "sum_g": sum_g,
        "dt_over_cap": dt_over_cap,
        "num": num,
        "den": den,
    }


def _i_h_ss(v, a_h, a_h_rev, v_mid_h_g, h_slope, v_mid_h_g_rev, h_slope_rev, *, h_g_max: float):
    """DC i_h gates ``u = ss(v)`` and conductances (no time step)."""
    i_h_ss = _gate_ss(v, v_mid_h_g, h_slope)
    i_h_ss_rev = _gate_ss(v, v_mid_h_g_rev, -h_slope_rev)
    gmax = float(h_g_max)
    return (
        i_h_ss,
        i_h_ss_rev,
        i_h_ss * gmax * a_h,
        i_h_ss_rev * gmax * a_h_rev,
    )


def _ohmic_v(i0, g_exc, g_inh, g_h, g_h_rev, e_leak, session):
    """``v_dc = (i + Σ g·e) / Σ g`` for frozen conductances."""
    e_exc = float(session.e_exc)
    e_inh = float(session.e_inh)
    e_h = float(session.e_h)
    g_leak = float(session.g_leak)
    e_h_rev_val = e_h_rev(e_leak, e_h)
    sum_g_e = (
        g_exc * e_exc + g_inh * e_inh + g_leak * e_leak
        + e_h * g_h + e_h_rev_val * g_h_rev
    )
    sum_g = g_exc + g_inh + g_h + g_h_rev + g_leak
    return (i0 + sum_g_e) / sum_g


def v_dc_from_v(v, params, i0, e_leak, session, *, with_i_h_ss: bool):
    """One DC map step: ohmic ``v_dc`` from ``g`` at ``v`` (+ i_h ``ss(v)`` if asked)."""
    v_out = torch.relu(v - params["v_th"]) * params["a_out"]
    g_exc, g_inh = session.connectome.conn.exc_inh_g(v_out, syn_strength(params))
    g_exc = g_exc * params["a_in"]
    g_inh = g_inh * params["a_in"]
    if with_i_h_ss:
        u, u_rev, g_h, g_h_rev = _i_h_ss(
            v, params["a_h"], params["a_h_rev"],
            params["v_mid_h_g"], params["h_slope"], params["v_mid_h_g_rev"], params["h_slope_rev"],
            h_g_max=session.h_g_max,
        )
    else:
        u = torch.zeros_like(v)
        u_rev = torch.zeros_like(v)
        g_h = torch.zeros_like(v)
        g_h_rev = torch.zeros_like(v)
    return _ohmic_v(i0, g_exc, g_inh, g_h, g_h_rev, e_leak, session), u, u_rev


def pre_steady(session, params, n_b, i_sti=None):
    """``u``, ``u_rev``, ``v`` at t=0 from ``session.pre_steady``."""
    if i_sti is None:
        raise TypeError("borst pre_steady requires i_sti")
    pre_steady_mode = str(session.pre_steady)
    if pre_steady_mode not in ("probe", "solve"):
        raise ValueError(f"borst pre_steady must be probe|solve; got {pre_steady_mode!r}")
    e_leak = params["e_leak"]
    v = e_leak.expand(n_b, session.connectome.n_node).clone()
    i0 = i_sti[:, 0, :]
    if pre_steady_mode == "probe":
        v_dc, u, u_rev = v_dc_from_v(v, params, i0, e_leak, session, with_i_h_ss=False)
        return u, u_rev, v_dc
    damp = float(session.pre_steady_damp)
    for _ in range(int(session.pre_steady_n_iter)):
        v_dc, _, _ = v_dc_from_v(v, params, i0, e_leak, session, with_i_h_ss=True)
        v = v + damp * (v_dc - v)
    _, u, u_rev = v_dc_from_v(v, params, i0, e_leak, session, with_i_h_ss=True)
    return u, u_rev, v


def step(u, u_rev, v, params, i_sti, session, *, delta_ms: float, return_component: bool = False):
    """One borst update; returns ``(u, u_rev, v)`` or + g component tuple."""
    if return_component:
        v, u, u_rev, g_exc, g_inh, g_h, g_h_rev = update_v(
            v, u, u_rev,
            params["a_in"], params["a_out"], syn_strength(params), params["v_th"],
            params["a_h"], params["a_h_rev"],
            params["v_mid_h_g"], params["h_slope"], params["v_mid_h_tau"],
            params["v_mid_h_g_rev"], params["h_slope_rev"], params["v_mid_h_tau_rev"],
            i_sti, session.connectome, params["e_leak"],
            delta_ms=float(delta_ms),
            cap=session.cap,
            g_leak=session.g_leak,
            e_exc=session.e_exc,
            e_inh=session.e_inh,
            e_h=session.e_h,
            h_g_max=session.h_g_max,
            euler=session.euler,
            return_component=True,
        )
        return u, u_rev, v, (g_exc, g_inh, g_h, g_h_rev)
    v, u, u_rev = update_v(
        v, u, u_rev,
        params["a_in"], params["a_out"], syn_strength(params), params["v_th"],
        params["a_h"], params["a_h_rev"],
        params["v_mid_h_g"], params["h_slope"], params["v_mid_h_tau"],
        params["v_mid_h_g_rev"], params["h_slope_rev"], params["v_mid_h_tau_rev"],
        i_sti, session.connectome, params["e_leak"],
        delta_ms=float(delta_ms),
        cap=session.cap,
        g_leak=session.g_leak,
        e_exc=session.e_exc,
        e_inh=session.e_inh,
        e_h=session.e_h,
        h_g_max=session.h_g_max,
        euler=session.euler,
        return_component=False,
    )
    return u, u_rev, v
