# -*- coding: utf-8 -*-
"""Borst neuron + i_h (``--model borst``).

Dynamics only: ``prepare_i_sti`` / ``pre_steady`` / ``step``. Full-T Ca
forward lives in ``neuron.forward``. Membrane scalars are injected kwargs
(from ``session`` flat fields), never a Physics bag.

t=0 membrane state uses ``session.pre_steady`` (``--pre-steady …``):

* ``probe``: ``g_syn`` from ``v=e_leak``, ``u_on/u_off=0``, then ohmic
  ``v = (i_sti + Σ gE) / Σ g``
* ``solve``: fixed-iter under-relaxed DC map with i_h at ``ss(v)``;
  uses ``session.pre_steady_iters`` / ``session.pre_steady_damp``

Membrane Euler (``session.euler`` = ``implicit`` | ``explicit``):

    C dv/dt = i_sti + Σ g_i (E_i − v)

    implicit:  v ← (v + α (i_sti + Σ gE)) / (1 + α Σ g),  α = Δt/C
    explicit:  v ← v + α (i_sti + Σ gE − Σ g · v)

i_h gate kinetics are always explicit Euler, independent of ``euler``.
"""
from __future__ import annotations

import torch

from neuron.params import e_h_off as calc_e_h_off, expand_euler, membrane_dt_over_c
from neuron.schema import borst_i_h_off_kwargs, syn_strength


def rectsyn(x, thrld):
    result = x - thrld
    return result * (result > 0)


def _i_h_gate_step(
    v, u_on, u_off, h_g_max, h_g_max_off,
    h_v_mid, h_slope, tau_v_mid, h_v_mid_off, h_slope_off, tau_v_mid_off,
    *,
    delta_ms: float,
    a_h: float,
):
    """Advance i_h gate states and channel conductances for active columns only.

    Gate ODE uses explicit Euler regardless of membrane ``euler``.
    """
    slope_on = h_slope
    slope_off = -h_slope_off
    i_h_ss_on = 1.0 / (1.0 + torch.exp((h_v_mid - v) * slope_on))
    i_h_ss_off = 1.0 / (1.0 + torch.exp((h_v_mid_off - v) * slope_off))
    tau_on = (
        1.5 / (torch.exp(-0.1 * (v - tau_v_mid)) + torch.exp(+0.1 * (v - tau_v_mid))) * 1000.0
        + 100.0
    )
    tau_off = (
        1.5
        / (torch.exp(-0.1 * (v - tau_v_mid_off)) + torch.exp(+0.1 * (v - tau_v_mid_off)))
        * 1000.0
        + 100.0
    )
    dt = float(delta_ms)
    u_on = dt / tau_on * (i_h_ss_on - u_on) + u_on
    u_off = dt / tau_off * (i_h_ss_off - u_off) + u_off
    g_i_h_on = u_on * h_g_max * float(a_h)
    g_i_h_off = u_off * h_g_max_off * float(a_h)
    return u_on, u_off, g_i_h_on, g_i_h_off


def update_v(
    v, u_on, u_off, a_in, a_out, syn_strength, v_th, h_g_max, h_g_max_off,
    h_v_mid, h_slope, tau_v_mid, h_v_mid_off, h_slope_off, tau_v_mid_off,
    i_sti, backend, *,
    delta_ms: float,
    cap: float,
    g_leak: float,
    e_exc: float,
    e_inh: float,
    e_h: float,
    e_leak_rest: float,
    a_h: float,
    euler: str,
    return_component: bool = False,
):
    """One borst step; membrane / reversal scalars are required kwargs."""
    euler = expand_euler(euler)
    e_leak = backend.e_leak
    conn = backend.conn
    i_h_active = (h_g_max + h_g_max_off) != 0
    g_i_h_on = u_on.new_zeros(u_on.shape)
    g_i_h_off = u_off.new_zeros(u_off.shape)
    i_h_kw_common = dict(delta_ms=delta_ms, a_h=a_h)
    if i_h_active.any():
        i_h_kw = dict(
            h_v_mid=h_v_mid, h_slope=h_slope, tau_v_mid=tau_v_mid,
            h_v_mid_off=h_v_mid_off, h_slope_off=h_slope_off, tau_v_mid_off=tau_v_mid_off,
        )
        if i_h_active.all():
            u_on, u_off, g_i_h_on, g_i_h_off = _i_h_gate_step(
                v, u_on, u_off, h_g_max, h_g_max_off, **i_h_kw_common, **i_h_kw)
        else:
            idx = i_h_active
            u_on_a, u_off_a, g_on_a, g_off_a = _i_h_gate_step(
                v[:, idx], u_on[:, idx], u_off[:, idx],
                h_g_max[idx], h_g_max_off[idx],
                **i_h_kw_common,
                **{k: val[idx] for k, val in i_h_kw.items()},
            )
            u_on = u_on.clone()
            u_off = u_off.clone()
            u_on[:, idx] = u_on_a.to(dtype=u_on.dtype)
            u_off[:, idx] = u_off_a.to(dtype=u_off.dtype)
            g_i_h_on[:, idx] = g_on_a.to(dtype=g_i_h_on.dtype)
            g_i_h_off[:, idx] = g_off_a.to(dtype=g_i_h_off.dtype)
    g_i_h = g_i_h_on + g_i_h_off

    g_exc, g_inh = conn.exc_inh_drive(rectsyn(v, v_th) * a_out, syn_strength)
    g_exc = g_exc * a_in
    g_inh = g_inh * a_in

    dt_over_c = membrane_dt_over_c(cap, delta_ms)
    e_h_off = calc_e_h_off(e_leak_rest, e_h)
    sum_gE = (
        g_exc * e_exc + g_inh * e_inh + g_leak * e_leak
        + e_h * g_i_h_on + e_h_off * g_i_h_off
    )
    sum_g = g_exc + g_inh + g_i_h + g_leak
    if euler == "implicit":
        v = (v + dt_over_c * (i_sti + sum_gE)) / (1.0 + dt_over_c * sum_g)
    else:
        v = v + dt_over_c * (i_sti + sum_gE - sum_g * v)

    if return_component:
        return v, u_on, u_off, g_exc, g_inh, g_i_h_on, g_i_h_off
    return v, u_on, u_off


def v_component_from_g(
    v_pre, g_exc, g_inh, g_i_h_on, g_i_h_off, i_sti, e_leak, *,
    delta_ms: float,
    cap: float,
    g_leak: float,
    e_exc: float,
    e_inh: float,
    e_h: float,
    e_leak_rest: float,
    euler: str,
):
    """Numerator / denom terms matching ``update_v`` (torch or numpy)."""
    euler = expand_euler(euler)
    dt_over_c = membrane_dt_over_c(cap, delta_ms)
    e_h_off = calc_e_h_off(e_leak_rest, e_h)
    num_exc = g_exc * e_exc
    num_inh = g_inh * e_inh
    num_leak = g_leak * e_leak
    num_i_h_on = g_i_h_on * e_h
    num_i_h_off = g_i_h_off * e_h_off
    sum_gE = num_exc + num_inh + num_leak + num_i_h_on + num_i_h_off
    sum_g = g_exc + g_inh + g_i_h_on + g_i_h_off + g_leak
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
        "num_i_h_on": num_i_h_on,
        "num_i_h_off": num_i_h_off,
        "num_v": num_v,
        "i_sti": i_sti,
        "sum_gE": sum_gE,
        "sum_g": sum_g,
        "dt_over_c": dt_over_c,
        "num": num,
        "den": den,
    }


def prepare_i_sti(session, p, i_sti, pack):
    """PR current ``(B, T, N)`` as membrane drive (no rescale)."""
    del p, pack
    return i_sti.unsqueeze(0) if i_sti.dim() == 2 else i_sti


def _i_h_ss(v, h_g_max, h_g_max_off, h_v_mid, h_slope, h_v_mid_off, h_slope_off, *, a_h: float):
    """DC i_h gates ``u = ss(v)`` and conductances (no time step)."""
    i_h_ss_on = 1.0 / (1.0 + torch.exp((h_v_mid - v) * h_slope))
    i_h_ss_off = 1.0 / (1.0 + torch.exp((h_v_mid_off - v) * (-h_slope_off)))
    gain = float(a_h)
    return (
        i_h_ss_on,
        i_h_ss_off,
        i_h_ss_on * h_g_max * gain,
        i_h_ss_off * h_g_max_off * gain,
    )


def _ohmic_v(i0, g_exc, g_inh, g_i_h_on, g_i_h_off, e_leak, session):
    """``v★ = (i + Σ gE) / Σ g`` for frozen conductances."""
    g_leak = float(session.g_leak)
    e_h_off = calc_e_h_off(session.e_leak_rest, session.e_h)
    sum_gE = (
        g_exc * float(session.e_exc)
        + g_inh * float(session.e_inh)
        + g_leak * e_leak
        + float(session.e_h) * g_i_h_on
        + e_h_off * g_i_h_off
    )
    sum_g = g_exc + g_inh + g_i_h_on + g_i_h_off + g_leak
    return (i0 + sum_gE) / sum_g


def _syn_g(v, p, backend):
    g_exc, g_inh = backend.conn.exc_inh_drive(
        rectsyn(v, p["v_th"]) * p["a_out"], syn_strength(p),
    )
    return g_exc * p["a_in"], g_inh * p["a_in"]


def _dc_v_star(v, p, i0, e_leak, session, *, with_i_h_ss: bool):
    """One DC map step: ohmic ``v★`` from ``g_syn(v)`` (+ i_h ``ss(v)`` if asked)."""
    g_exc, g_inh = _syn_g(v, p, session.backend)
    if with_i_h_ss:
        i_h_off = (session.train_opts or {})["i_h_off"]
        h_g_max_off, h_v_mid_off, h_slope_off, _tau = borst_i_h_off_kwargs(p, i_h_off)
        u_on, u_off, g_i_h_on, g_i_h_off = _i_h_ss(
            v, p["h_g_max"], h_g_max_off,
            p["h_v_mid"], p["h_slope"], h_v_mid_off, h_slope_off,
            a_h=session.a_h,
        )
    else:
        u_on = torch.zeros_like(v)
        u_off = torch.zeros_like(v)
        g_i_h_on = torch.zeros_like(v)
        g_i_h_off = torch.zeros_like(v)
    v_star = _ohmic_v(i0, g_exc, g_inh, g_i_h_on, g_i_h_off, e_leak, session)
    return v_star, u_on, u_off


def _pre_steady_probe(session, p, B, i_sti):
    """One-shot ohmic: ``g_syn`` from ``v=e_leak``, i_h off; balance membrane."""
    backend = session.backend
    dtype = session.sim_dtype
    n = backend.n_nodes
    dev = backend.conn.node_cell.device
    e_leak = backend.e_leak.to(device=dev, dtype=dtype)
    v_probe = e_leak.expand(B, n).clone()
    i0 = i_sti[:, 0, :]
    v_star, u_on, u_off = _dc_v_star(
        v_probe, p, i0, e_leak, session, with_i_h_ss=False,
    )
    return (u_on, u_off), v_star


def _pre_steady_solve(session, p, B, i_sti, *, iters: int, damp: float):
    """Fixed-iter under-relaxed DC map with i_h at ``ss(v)`` (not time stepping)."""
    backend = session.backend
    dtype = session.sim_dtype
    n = backend.n_nodes
    dev = backend.conn.node_cell.device
    e_leak = backend.e_leak.to(device=dev, dtype=dtype)
    v = e_leak.expand(B, n).clone()
    i0 = i_sti[:, 0, :]
    damp = float(damp)
    for _ in range(int(iters)):
        v_star, _, _ = _dc_v_star(
            v, p, i0, e_leak, session, with_i_h_ss=True,
        )
        v = v + damp * (v_star - v)
    _, u_on, u_off = _dc_v_star(v, p, i0, e_leak, session, with_i_h_ss=True)
    return (u_on, u_off), v


def pre_steady(session, p, B, i_sti=None):
    """``(u_on, u_off)``, ``v`` at t=0 from ``session.pre_steady``."""
    if i_sti is None:
        raise TypeError("borst pre_steady requires i_sti")
    mode = str(session.pre_steady)
    if mode == "probe":
        return _pre_steady_probe(session, p, B, i_sti)
    if mode == "solve":
        return _pre_steady_solve(
            session, p, B, i_sti,
            iters=int(session.pre_steady_iters),
            damp=float(session.pre_steady_damp),
        )
    raise ValueError(f"borst pre_steady must be probe|solve; got {mode!r}")


def _membrane_kwargs(session, delta_ms: float):
    return dict(
        delta_ms=float(delta_ms),
        cap=session.cap,
        g_leak=session.g_leak,
        e_exc=session.e_exc,
        e_inh=session.e_inh,
        e_h=session.e_h,
        e_leak_rest=session.e_leak_rest,
        a_h=session.a_h,
        euler=session.euler,
    )


def step(state, v, p, i_sti, session, *, delta_ms: float, return_component: bool = False):
    """One borst update; returns ``((u_on, u_off), v)`` or + g component tuple."""
    u_on, u_off = state
    i_h_off = (session.train_opts or {})["i_h_off"]
    h_g_max_off, h_v_mid_off, h_slope_off, tau_v_mid_off = borst_i_h_off_kwargs(
        p, i_h_off,
    )
    out = update_v(
        v, u_on, u_off,
        p["a_in"], p["a_out"], syn_strength(p), p["v_th"],
        p["h_g_max"], h_g_max_off,
        p["h_v_mid"], p["h_slope"], p["tau_v_mid"],
        h_v_mid_off, h_slope_off, tau_v_mid_off,
        i_sti, session.backend,
        **_membrane_kwargs(session, delta_ms),
        return_component=return_component,
    )
    if return_component:
        v, u_on, u_off, g_exc, g_inh, g_i_h_on, g_i_h_off = out
        return (u_on, u_off), v, (g_exc, g_inh, g_i_h_on, g_i_h_off)
    v, u_on, u_off = out
    return (u_on, u_off), v
