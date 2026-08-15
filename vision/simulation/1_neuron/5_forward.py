# -*- coding: utf-8 -*-
"""Shared full-T absolute ``v`` forward for all neuron models.

Per-model modules supply only ``standardize_i_sti`` / ``pre_steady`` / ``step``.
This module owns the time loop. Train / plots read absolute ``v``
when ``train_opts['filter']=='none'``; with ``'ca'``, readout is ``ca`` from
``neuron.filter_ca`` on ``v_ca = relu(v − v_th_ca)·a_ca``. Cost compares the
readout to ``a_gt * gts + bias_gt``.
"""
from __future__ import annotations

import torch

from neuron import model_borst as _model_borst
from neuron import model_hp_lp as _model_hp_lp
from neuron.filter_ca import filter_ca

# Per-model dynamics for ``forward_full`` (standardize_i_sti / pre_steady / step only).
MODEL_DRIVERS = {
    "borst": _model_borst,
    "hp_lp": _model_hp_lp,
}


def a_sti_radius_effective(params, pack):
    """``a_sti_radius`` after ``pack.a_sti_radius_mask`` (cost scale==0 → 0)."""
    a_sti_radius = params["a_sti_radius"]
    if pack is None:
        return a_sti_radius
    mask = getattr(pack, "a_sti_radius_mask", None)
    if mask is None:
        return a_sti_radius
    return a_sti_radius * mask.to(device=a_sti_radius.device, dtype=a_sti_radius.dtype)


def inject_a_sti_radius(i_sti, params, pack):
    """``i += a_sti_radius[r] * sti_wave`` on spot radius sti contribs; else pass-through.

    Uses :func:`a_sti_radius_effective` so masked radii are 0 whether indi or fixed.
    """
    a_sti_radius_idxs = getattr(pack, "a_sti_radius_idxs", None) if pack is not None else None
    if a_sti_radius_idxs is None or "a_sti_radius" not in params:
        return i_sti
    wave = pack.sti_wave
    sti_bs = pack.sti_bs
    node = pack.sti_nodes
    if wave is None or sti_bs is None or node is None:
        raise ValueError(
            "spot pack a_sti_radius_idxs set but sti_wave/sti_bs/sti_nodes missing"
        )
    if i_sti.dim() == 2:
        i_sti = i_sti.unsqueeze(0)
    a_sti_radius = a_sti_radius_effective(params, pack)
    out = i_sti.clone()
    if sti_bs.numel() == 0:
        return out
    n_b, n_t, n_nodes = out.shape
    add = a_sti_radius[a_sti_radius_idxs][:, None] * wave[None, :]
    flat = out.permute(0, 2, 1).reshape(n_b * n_nodes, n_t)
    flat.index_add_(0, sti_bs * n_nodes + node, add)
    return flat.reshape(n_b, n_nodes, n_t).permute(0, 2, 1).contiguous()


def pack_t_onset(pack) -> int:
    """Stimulus onset index for ``pack``.

    Prefer explicit ``pack.t_onset`` (spot when ``ms_post`` extends ``i_sti`` past
    ``gts``). Else ``n_t - gts.shape[1]`` (moving_bar / ``ms_post=0``).
    """
    t = getattr(pack, "t_onset", None)
    if t is not None:
        return int(t)
    i_sti = pack.i_sti
    n_t = int(i_sti.shape[1] if i_sti.dim() == 3 else i_sti.shape[0])
    return n_t - int(pack.gts.shape[1])


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
    opts = session.train_opts or {}
    return str(opts.get("filter", "none"))


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
    rows = [ca]
    for t in range(1, int(v_ca.shape[1])):
        ca = filter_ca(
            ca, v_ca[:, t],
            delta_ms=step_delta_ms(session, t, t_onset),
            tau_ca=tau_ca,
        )
        rows.append(ca)
    return torch.stack(rows, dim=1)


def forward_v(session, params, i_sti, *, pack=None):
    """Full-T ``v`` ``(B, T, N)`` (no Ca filter)."""
    if session.model not in MODEL_DRIVERS:
        raise ValueError(
            f"no MODEL_DRIVERS entry for model={session.model!r}; "
            f"expected one of {tuple(MODEL_DRIVERS)}"
        )
    drv = MODEL_DRIVERS[session.model]
    pack = pack or session.primary_pack
    i_sti = inject_a_sti_radius(
        drv.standardize_i_sti(session, params, i_sti, pack), params, pack,
    )
    n_b, t_end = int(i_sti.shape[0]), int(i_sti.shape[1])
    t_onset = pack_t_onset(pack)
    pre_grad = bool((session.train_opts or {})["pre_grad"])
    state, v = drv.pre_steady(session, params, n_b, i_sti=i_sti)
    v_rows = [v]

    def take(t):
        nonlocal state, v
        state, v = drv.step(
            state, v, params, i_sti[:, t - 1], session,
            delta_ms=step_delta_ms(session, t, t_onset),
        )
        v_rows.append(v)

    if pre_grad or t_onset <= 0:
        for t in range(1, t_end):
            take(t)
        return torch.stack(v_rows, dim=1)
    with torch.no_grad():
        for t in range(1, t_onset):
            take(t)
    state = tuple(state_tensor.detach() for state_tensor in state)
    v = v.detach()
    for t in range(max(t_onset, 1), t_end):
        take(t)
    return torch.stack(v_rows, dim=1)


def forward_ca(session, params, i_sti, *, pack=None):
    """Full-T ``ca`` ``(B, T, N)``: ``forward_v`` then ``filter_ca``."""
    pack = pack or session.primary_pack
    v = forward_v(session, params, i_sti, pack=pack)
    return ca_from_v_ca(v_ca_from_v(v, params, session), params, session, t_onset=pack_t_onset(pack))


def forward_full(session, params, i_sti, *, pack=None):
    """Shared full-T forward for every ``session.model``.

    Time index ``t`` is post-update at sample ``t``. Drive comes from
    ``MODEL_DRIVERS[model].standardize_i_sti`` / ``pre_steady`` / ``step``.

    ``session.train_opts['pre_grad']`` (default ``True``): when ``False``, steps
    with ``t < t_onset`` run under ``torch.no_grad()``, then ``v`` / ``state``
    are detached before post steps so BPTT does not enter pre.

    ``session.train_opts['filter']``: ``none`` → :func:`forward_v`; ``ca`` →
    :func:`forward_ca`.

    Returns
    -------
    Readout trace ``(B, T, N)`` (``v`` or ``ca``).
    """
    if _session_filter(session) == "ca":
        return forward_ca(session, params, i_sti, pack=pack)
    return forward_v(session, params, i_sti, pack=pack)


def forward_nodes(session, params, node_idx=None, i_sti=None, pack=None):
    """``forward_full`` then index nodes; squeeze when ``i_sti`` is ``(T, N)``."""
    pack = pack or session.primary_pack
    if node_idx is None:
        node_idx = pack.entry_nodes
    if i_sti is None:
        i_sti = session.pack_i_sti(pack)
    squeeze = i_sti.dim() == 2
    i_sti_b = i_sti.unsqueeze(0) if squeeze else i_sti
    out = forward_full(session, params, i_sti_b, pack=pack)
    out = out[:, :, node_idx]
    if squeeze:
        out = out.squeeze(0)
    return out
