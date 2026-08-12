# -*- coding: utf-8 -*-
"""Borst neuron + i_h (``--model borst``).

Dynamics only: ``prepare_i_sti`` / ``pre_steady`` / ``step``. Full-T Ca
forward lives in ``neuron.forward``. Membrane scalars are injected kwargs
(from ``session`` flat fields), never a Physics bag.

t=0 membrane state uses ``session.pre_steady`` (``--pre-steady …``):

* ``probe``: ``g_syn`` from ``v=e_leak``, ``u/u_rev=0``, then ohmic
  ``v = (i_sti + Σ gE) / Σ g``
* ``solve``: fixed-iter under-relaxed DC map with i_h at ``ss(v)``;
  uses ``session.pre_steady_iters`` / ``session.pre_steady_damp``

Membrane Euler (``session.euler`` = ``implicit`` | ``explicit``):

    C dv/dt = i_sti + Σ g_i (E_i − v)

    implicit:  v ← (v + α (i_sti + Σ gE)) / (1 + α Σ g),  α = Δt/C
    explicit:  v ← v + α (i_sti + Σ gE − Σ g · v)

i_h gate kinetics are always explicit Euler, independent of ``euler``.

Conductances: ``g_h = u · h_g_max · a_h``, ``g_h_rev = u_rev · h_g_max · a_h_rev``
with fixed ``h_g_max`` (session) and ``a_h`` in z / ``a_h_rev``.
"""
from __future__ import annotations

import torch

from neuron.param import e_h_rev as calc_e_h_rev, expand_euler, membrane_dt_over_c
from neuron.schema import borst_i_h_rev_kwargs, syn_strength


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
    """Advance i_h gate states and channel conductances for masked hexes only.

    Gate ODE uses explicit Euler regardless of membrane ``euler``.
    """
    i_h_ss = _gate_ss(v, v_mid_h_g, h_slope)
    i_h_ss_rev = _gate_ss(v, v_mid_h_g_rev, -h_slope_rev)
    dt = float(delta_ms)
    u = dt / _i_h_tau(v, v_mid_h_tau) * (i_h_ss - u) + u
    u_rev = dt / _i_h_tau(v, v_mid_h_tau_rev) * (i_h_ss_rev - u_rev) + u_rev
    gmax = float(h_g_max)
    return u, u_rev, u * gmax * a_h, u_rev * gmax * a_h_rev


def _sum_gE_g(g_exc, g_inh, g_h, g_h_rev, e_leak, *, e_exc, e_inh, e_h, g_leak):
    e_hr = calc_e_h_rev(e_leak, e_h)
    sum_gE = (
        g_exc * e_exc + g_inh * e_inh + g_leak * e_leak
        + e_h * g_h + e_hr * g_h_rev
    )
    sum_g = g_exc + g_inh + g_h + g_h_rev + g_leak
    return sum_gE, sum_g


def update_v(
    v, u, u_rev, a_in, a_out, syn_strength, v_th, a_h, a_h_rev,
    v_mid_h_g, h_slope, v_mid_h_tau, v_mid_h_g_rev, h_slope_rev, v_mid_h_tau_rev,
    i_sti, backend, e_leak, *,
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
    """One borst step; membrane / reversal scalars are required kwargs."""
    euler = expand_euler(euler)
    conn = backend.conn
    i_h_mask = (a_h + a_h_rev) != 0
    g_h = u.new_zeros(u.shape)
    g_h_rev = u_rev.new_zeros(u_rev.shape)
    i_h_kw_common = dict(delta_ms=delta_ms, h_g_max=h_g_max)
    if i_h_mask.any():
        i_h_kw = dict(
            v_mid_h_g=v_mid_h_g, h_slope=h_slope, v_mid_h_tau=v_mid_h_tau,
            v_mid_h_g_rev=v_mid_h_g_rev, h_slope_rev=h_slope_rev, v_mid_h_tau_rev=v_mid_h_tau_rev,
        )
        if i_h_mask.all():
            u, u_rev, g_h, g_h_rev = _i_h_gate_step(
                v, u, u_rev, a_h, a_h_rev, **i_h_kw_common, **i_h_kw)
        else:
            idx = i_h_mask
            u_a, u_rev_a, g_a, g_rev_a = _i_h_gate_step(
                v[:, idx], u[:, idx], u_rev[:, idx],
                a_h[idx], a_h_rev[idx],
                **i_h_kw_common,
                **{k: val[idx] for k, val in i_h_kw.items()},
            )
            u = u.clone()
            u_rev = u_rev.clone()
            u[:, idx] = u_a.to(dtype=u.dtype)
            u_rev[:, idx] = u_rev_a.to(dtype=u_rev.dtype)
            g_h[:, idx] = g_a.to(dtype=g_h.dtype)
            g_h_rev[:, idx] = g_rev_a.to(dtype=g_h_rev.dtype)

    v_out = torch.relu(v - v_th) * a_out
    g_exc, g_inh = conn.exc_inh_drive(v_out, syn_strength)
    g_exc = g_exc * a_in
    g_inh = g_inh * a_in

    dt_over_c = membrane_dt_over_c(cap, delta_ms)
    sum_gE, sum_g = _sum_gE_g(
        g_exc, g_inh, g_h, g_h_rev, e_leak,
        e_exc=e_exc, e_inh=e_inh, e_h=e_h, g_leak=g_leak,
    )
    if euler == "implicit":
        v = (v + dt_over_c * (i_sti + sum_gE)) / (1.0 + dt_over_c * sum_g)
    else:
        v = v + dt_over_c * (i_sti + sum_gE - sum_g * v)

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
    euler = expand_euler(euler)
    dt_over_c = membrane_dt_over_c(cap, delta_ms)
    e_hr = calc_e_h_rev(e_leak, e_h)
    num_exc = g_exc * e_exc
    num_inh = g_inh * e_inh
    num_leak = g_leak * e_leak
    num_i_h = g_h * e_h
    num_i_h_rev = g_h_rev * e_hr
    sum_gE = num_exc + num_inh + num_leak + num_i_h + num_i_h_rev
    sum_g = g_exc + g_inh + g_h + g_h_rev + g_leak
    if euler == "implicit":
        num = v_pre + dt_over_c * (i_sti + sum_gE)
        den = 1.0 + dt_over_c * sum_g
        num_v = v_pre
    else:
        num = v_pre + dt_over_c * (i_sti + sum_gE - sum_g * v_pre)
        den = 1.0
        num_v = v_pre * (1.0 - dt_over_c * sum_g)
    return {
        "num_exc": num_exc,
        "num_inh": num_inh,
        "num_leak": num_leak,
        "num_i_h": num_i_h,
        "num_i_h_rev": num_i_h_rev,
        "num_v": num_v,
        "i_sti": i_sti,
        "sum_gE": sum_gE,
        "sum_g": sum_g,
        "dt_over_c": dt_over_c,
        "num": num,
        "den": den,
    }


def prepare_i_sti(session, params, i_sti, pack):
    """Sti current ``(B, T, N)`` as membrane drive (no rescale)."""
    del params, pack
    return i_sti.unsqueeze(0) if i_sti.dim() == 2 else i_sti


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
    """``v★ = (i + Σ gE) / Σ g`` for frozen conductances."""
    sum_gE, sum_g = _sum_gE_g(
        g_exc, g_inh, g_h, g_h_rev, e_leak,
        e_exc=float(session.e_exc),
        e_inh=float(session.e_inh),
        e_h=float(session.e_h),
        g_leak=float(session.g_leak),
    )
    return (i0 + sum_gE) / sum_g


def _syn_g(v, params, backend):
    v_out = torch.relu(v - params["v_th"]) * params["a_out"]
    g_exc, g_inh = backend.conn.exc_inh_drive(v_out, syn_strength(params))
    return g_exc * params["a_in"], g_inh * params["a_in"]


def _dc_v_star(v, params, i0, e_leak, session, *, with_i_h_ss: bool):
    """One DC map step: ohmic ``v★`` from ``g_syn(v)`` (+ i_h ``ss(v)`` if asked)."""
    g_exc, g_inh = _syn_g(v, params, session.backend)
    if with_i_h_ss:
        i_h_rev = (session.train_opts or {})["i_h_rev"]
        a_h_rev, v_mid_h_g_rev, h_slope_rev, _v_mid_h_tau = borst_i_h_rev_kwargs(params, i_h_rev)
        u, u_rev, g_h, g_h_rev = _i_h_ss(
            v, params["a_h"], a_h_rev,
            params["v_mid_h_g"], params["h_slope"], v_mid_h_g_rev, h_slope_rev,
            h_g_max=session.h_g_max,
        )
    else:
        u = torch.zeros_like(v)
        u_rev = torch.zeros_like(v)
        g_h = torch.zeros_like(v)
        g_h_rev = torch.zeros_like(v)
    return _ohmic_v(i0, g_exc, g_inh, g_h, g_h_rev, e_leak, session), u, u_rev


def pre_steady(session, params, B, i_sti=None):
    """``(u, u_rev)``, ``v`` at t=0 from ``session.pre_steady``."""
    if i_sti is None:
        raise TypeError("borst pre_steady requires i_sti")
    pre_steady = str(session.pre_steady)
    if pre_steady not in ("probe", "solve"):
        raise ValueError(f"borst pre_steady must be probe|solve; got {pre_steady!r}")
    e_leak = params["e_leak"]
    v = e_leak.expand(B, session.backend.n_nodes).clone()
    i0 = i_sti[:, 0, :]
    if pre_steady == "probe":
        v_star, u, u_rev = _dc_v_star(v, params, i0, e_leak, session, with_i_h_ss=False)
        return (u, u_rev), v_star
    damp = float(session.pre_steady_damp)
    for _ in range(int(session.pre_steady_iters)):
        v_star, _, _ = _dc_v_star(v, params, i0, e_leak, session, with_i_h_ss=True)
        v = v + damp * (v_star - v)
    _, u, u_rev = _dc_v_star(v, params, i0, e_leak, session, with_i_h_ss=True)
    return (u, u_rev), v


def step(state, v, params, i_sti, session, *, delta_ms: float, return_component: bool = False):
    """One borst update; returns ``((u, u_rev), v)`` or + g component tuple."""
    u, u_rev = state
    i_h_rev = (session.train_opts or {})["i_h_rev"]
    a_h_rev, v_mid_h_g_rev, h_slope_rev, v_mid_h_tau_rev = borst_i_h_rev_kwargs(
        params, i_h_rev,
    )
    out = update_v(
        v, u, u_rev,
        params["a_in"], params["a_out"], syn_strength(params), params["v_th"],
        params["a_h"], a_h_rev,
        params["v_mid_h_g"], params["h_slope"], params["v_mid_h_tau"],
        v_mid_h_g_rev, h_slope_rev, v_mid_h_tau_rev,
        i_sti, session.backend, params["e_leak"],
        delta_ms=float(delta_ms),
        cap=session.cap,
        g_leak=session.g_leak,
        e_exc=session.e_exc,
        e_inh=session.e_inh,
        e_h=session.e_h,
        h_g_max=session.h_g_max,
        euler=session.euler,
        return_component=return_component,
    )
    if return_component:
        v, u, u_rev, g_exc, g_inh, g_h, g_h_rev = out
        return (u, u_rev), v, (g_exc, g_inh, g_h, g_h_rev)
    v, u, u_rev = out
    return (u, u_rev), v
