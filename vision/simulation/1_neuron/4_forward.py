# -*- coding: utf-8 -*-
"""Shared absolute ``v`` / ``ca`` forward for all neuron models.

Per-model modules supply only ``pre_steady`` / ``step``.
This module owns the time loop. Train / plots read absolute ``v``
when ``train_opts['filter']=='none'``; with ``'ca'``, readout is ``ca`` from
``neuron.filter_ca`` on ``v_ca = relu(v − v_th_ca)·a_ca``. Cost compares the
readout to ``a_gt * gts + bias_gt``.
"""
from __future__ import annotations

import torch

import neuron.borst as _borst
import neuron.hp_lp as _hp_lp
from neuron.filter_ca import filter_ca

# Per-model dynamics for ``forward_v`` / ``forward_ca`` (pre_steady / step only).
MODEL_DRIVERS = {
    "borst": _borst,
    "hp_lp": _hp_lp,
}
MODELS = tuple(MODEL_DRIVERS)


def a_sti_radius_effective(params, pack):
    """``a_sti_radius`` after ``pack.a_sti_radius_mask`` (cost scale==0 → 0)."""
    a_sti_radius = params["a_sti_radius"]
    if pack is None:
        return a_sti_radius
    mask = getattr(pack, "a_sti_radius_mask", None)
    if mask is None:
        return a_sti_radius
    return a_sti_radius * mask.to(device=a_sti_radius.device, dtype=a_sti_radius.dtype)


def a_sti_mid_effective(params, pack):
    """Gaussian surround gains ``exp(-0.5 * (mid / σ)²)`` from scalar ``a_sti_mid``=σ."""
    sigma = params["a_sti_mid"]
    mids = getattr(pack, "a_sti_mids", None) if pack is not None else None
    if mids is None:
        raise ValueError("a_sti_mid requires pack.a_sti_mids")
    mids = mids.to(device=sigma.device, dtype=sigma.dtype)
    return torch.exp(-0.5 * (mids / sigma) ** 2)


def _indexed_stimulus_delta(
    values, value_idxs, pack, *, n_b: int, n_node: int,
):
    """Aggregate indexed amplitudes once into a dense ``(B, N)`` delta.

    The old path scattered an ``(n_entry, T)`` tensor.  Scattering scalars here
    preserves duplicate multi-spot/multi-bar sums while leaving time expansion
    to the individual simulation step.
    """
    sti_bs = pack.sti_bs
    sti_nodes = pack.sti_nodes
    if pack.i_sti_pulse is None or sti_bs is None or sti_nodes is None:
        raise ValueError(
            "indexed stimulus set but i_sti_pulse/sti_bs/sti_nodes missing"
        )
    delta = values.new_zeros(int(n_b) * int(n_node))
    if sti_bs.numel():
        delta.index_add_(
            0,
            sti_bs * int(n_node) + sti_nodes,
            values[value_idxs],
        )
    return delta.reshape(int(n_b), int(n_node))


def pack_stimulus_delta(i_sti, params, pack):
    """Return ``(node_delta[B,N], pulse[T])`` for indexed spot/sbar drive."""
    if i_sti.dim() == 2:
        i_sti = i_sti.unsqueeze(0)
    n_b, _n_t, n_node = map(int, i_sti.shape)
    delta = None
    pulse = None

    radius_idxs = (
        getattr(pack, "a_sti_radius_idxs", None) if pack is not None else None
    )
    if radius_idxs is not None and "a_sti_radius" in params:
        delta = _indexed_stimulus_delta(
            a_sti_radius_effective(params, pack), radius_idxs, pack,
            n_b=n_b, n_node=n_node,
        )
        pulse = pack.i_sti_pulse

    mid_idxs = (
        getattr(pack, "a_sti_mid_idxs", None) if pack is not None else None
    )
    if mid_idxs is not None and "a_sti_mid" in params:
        mid_delta = _indexed_stimulus_delta(
            a_sti_mid_effective(params, pack), mid_idxs, pack,
            n_b=n_b, n_node=n_node,
        )
        delta = mid_delta if delta is None else delta + mid_delta
        pulse = pack.i_sti_pulse
    return delta, pulse


def _materialize_stimulus_delta(delta, pulse):
    """Expand ``delta[B,N]`` with shared ``pulse[T]`` or per-b ``pulse[B,T]``."""
    if pulse.dim() == 1:
        return delta[:, None, :] * pulse[None, :, None]
    if pulse.dim() == 2 and int(pulse.shape[0]) == int(delta.shape[0]):
        return delta[:, None, :] * pulse[:, :, None]
    raise ValueError(
        f"indexed stimulus pulse must be (T,) or (B,T); got {tuple(pulse.shape)}"
    )


def inject_a_sti_radius(i_sti, params, pack):
    """Materialize spot indexed drive; simulation uses the step-wise fast path.

    Uses :func:`a_sti_radius_effective` so masked radii are 0 whether indi or fixed.
    """
    a_sti_radius_idxs = getattr(pack, "a_sti_radius_idxs", None) if pack is not None else None
    if a_sti_radius_idxs is None or "a_sti_radius" not in params:
        return i_sti
    if i_sti.dim() == 2:
        i_sti = i_sti.unsqueeze(0)
    n_b, _n_t, n_node = i_sti.shape
    delta = _indexed_stimulus_delta(
        a_sti_radius_effective(params, pack), a_sti_radius_idxs, pack,
        n_b=int(n_b), n_node=int(n_node),
    )
    return i_sti + _materialize_stimulus_delta(delta, pack.i_sti_pulse)


def inject_a_sti_mid(i_sti, params, pack):
    """Materialize sbar Gaussian surround; simulation uses the step-wise fast path."""
    a_sti_mid_idxs = (
        getattr(pack, "a_sti_mid_idxs", None) if pack is not None else None
    )
    if a_sti_mid_idxs is None or "a_sti_mid" not in params:
        return i_sti
    if i_sti.dim() == 2:
        i_sti = i_sti.unsqueeze(0)
    n_b, _n_t, n_node = i_sti.shape
    delta = _indexed_stimulus_delta(
        a_sti_mid_effective(params, pack), a_sti_mid_idxs, pack,
        n_b=int(n_b), n_node=int(n_node),
    )
    return i_sti + _materialize_stimulus_delta(delta, pack.i_sti_pulse)


def pack_t_onset(pack) -> int:
    """Stimulus onset index for ``pack``.

    Prefer explicit ``pack.t_onset`` (spot when ``ms_post`` extends ``i_sti`` past
    ``gts``). Else ``n_t - gts.shape[1]`` (mbar / ``ms_post=0``).
    """
    t = getattr(pack, "t_onset", None)
    if t is not None:
        return int(t)
    i_sti = pack.i_sti
    return int(i_sti.shape[1] if i_sti.dim() == 3 else i_sti.shape[0]) - int(pack.gts.shape[1])


def step_delta_ms(session, t: int, t_onset: int) -> float:
    """``delta_ms`` for the update that produces sample ``t`` (from ``t-1`` → ``t``).

    Pre-onset steps (``t <= t_onset``) use ``session.delta_ms_pre``; later steps
    use ``session.delta_ms``.
    """
    return float(
        session.delta_ms_pre if int(t) <= int(t_onset) else session.delta_ms
    )


def _session_filter(session) -> str:
    """``train_opts['filter']``; default ``none`` (same pattern as ``_session_*`` in cost)."""
    return str((session.train_opts or {}).get("filter", "none"))


def v_ca_from_v(v, params, session):
    """``v_ca = relu(v − v_th_ca)·a_ca`` (per-node tensors in ``params``).

    Callers must run ``train.override_val_from`` (via ``params_from_z``)
    so ``v_th_ca`` / ``a_ca`` already hold ``v_th`` / ``a_out`` when the
    matching ``val_from`` entries are enabled.
    """
    return torch.relu(v - params["v_th_ca"]) * params["a_ca"]


def ca_from_v_ca(v_ca, params, session, *, t_onset: int):
    """Apply ``filter_ca`` over time on pre-computed ``v_ca``; ``ca[0] = v_ca[0]``."""
    tau_ca = torch.clamp(params["tau_ca"], min=float(session.delta_ms))
    ca = v_ca[:, 0]
    trace = ca.new_empty(v_ca.shape)
    trace[:, 0] = ca
    for t in range(1, int(v_ca.shape[1])):
        ca = filter_ca(
            ca, v_ca[:, t],
            delta_ms=step_delta_ms(session, t, t_onset),
            tau_ca=tau_ca,
        )
        trace[:, t] = ca
    return trace


def forward_v(session, params, i_sti, *, pack=None):
    """``v`` trace ``(B, T, N)`` (no Ca filter).

    Time index ``t`` is post-update at sample ``t``. Drive comes from
    ``MODEL_DRIVERS[model].pre_steady`` / ``step``.

    ``session.train_opts['pre_grad']`` (default ``True``): when ``False``, steps
    with ``t < t_onset`` run under ``torch.no_grad()``, then ``v`` and model
    internals (``v_slow`` or ``u``/``u_rev``) are detached before post steps so
    BPTT does not enter pre.
    """
    if session.model not in MODEL_DRIVERS:
        raise ValueError(
            f"no MODEL_DRIVERS entry for model={session.model!r}; "
            f"expected one of {tuple(MODEL_DRIVERS)}"
        )
    drv = MODEL_DRIVERS[session.model]
    pack = pack or session.primary_pack
    if i_sti.dim() == 2:
        i_sti = i_sti.unsqueeze(0)
    n_b = int(i_sti.shape[0])
    t_end = int(i_sti.shape[1])
    t_onset = pack_t_onset(pack)
    sti_delta, sti_pulse = pack_stimulus_delta(i_sti, params, pack)

    def drive_at(t):
        drive = i_sti[:, int(t)]
        if sti_delta is not None:
            pulse_t = (
                sti_pulse[int(t)]
                if sti_pulse.dim() == 1 else sti_pulse[:, int(t), None]
            )
            drive = drive + sti_delta * pulse_t
        return drive

    pre_i_sti = drive_at(0).unsqueeze(1)
    if session.model == "hp_lp":
        v_slow, v = drv.pre_steady(session, params, n_b, i_sti=pre_i_sti)
        u = u_rev = None
    else:
        u, u_rev, v = drv.pre_steady(session, params, n_b, i_sti=pre_i_sti)
        v_slow = None
    trace = v.new_empty(n_b, t_end, v.shape[-1])
    trace[:, 0] = v

    def take(t):
        nonlocal v_slow, u, u_rev, v
        if session.model == "hp_lp":
            v_slow, v = drv.step(v_slow, v, params, drive_at(t - 1), session, delta_ms=step_delta_ms(session, t, t_onset))
        else:
            u, u_rev, v = drv.step(u, u_rev, v, params, drive_at(t - 1), session, delta_ms=step_delta_ms(session, t, t_onset))
        trace[:, t] = v

    if bool((session.train_opts or {})["pre_grad"]) or t_onset <= 0:
        for t in range(1, t_end):
            take(t)
        return trace
    with torch.no_grad():
        for t in range(1, t_onset):
            take(t)
    if session.model == "hp_lp":
        v_slow = v_slow.detach()
    else:
        u = u.detach()
        u_rev = u_rev.detach()
    v = v.detach()
    for t in range(max(t_onset, 1), t_end):
        take(t)
    return trace


def forward_ca(session, params, i_sti, *, pack=None):
    """``ca`` trace ``(B, T, N)``: ``forward_v`` then ``filter_ca``."""
    pack = pack or session.primary_pack
    return ca_from_v_ca(v_ca_from_v(forward_v(session, params, i_sti, pack=pack), params, session), params, session, t_onset=pack_t_onset(pack))


def forward_trace(session, params, i_sti, *, pack=None):
    """Readout trace ``(B, T, N)``: ``forward_ca`` when ``filter=ca``, else ``forward_v``."""
    pack = pack or session.primary_pack
    if _session_filter(session) == "ca":
        return forward_ca(session, params, i_sti, pack=pack)
    return forward_v(session, params, i_sti, pack=pack)


def forward_nodes(session, params, nodes=None, i_sti=None, pack=None):
    """``forward_trace`` then index nodes; squeeze when ``i_sti`` is ``(T, N)``."""
    pack = pack or session.primary_pack
    if nodes is None:
        nodes = pack.entry_nodes
    if i_sti is None:
        i_sti = session.pack_i_sti(pack)
    squeeze = i_sti.dim() == 2
    trace = forward_trace(session, params, i_sti.unsqueeze(0) if squeeze else i_sti, pack=pack)[:, :, nodes]
    if squeeze:
        trace = trace.squeeze(0)
    return trace
