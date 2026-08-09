# -*- coding: utf-8 -*-
"""Shared full-T absolute ``v`` forward for all neuron models.

Per-model modules supply only ``prepare_i_sti`` / ``pre_steady`` / ``step``.
This module owns the time loop. Training / plots read absolute membrane ``v``;
cost compares ``v`` to ``a_gt * gt + bias_gt``. The unused Ca filter stays
in ``neuron.filter_ca``.
"""
from __future__ import annotations

import torch

from neuron import model_borst as _model_borst
from neuron import model_hp_lp as _model_hp_lp

# Per-model dynamics for ``forward_full`` (prepare_i_sti / pre_steady / step only).
MODEL_DRIVERS = {
    "borst": _model_borst,
    "hp_lp": _model_hp_lp,
}


def a_sti_radius_effective(p, pack):
    """``a_sti_radius`` after ``pack.sti_radius_gate`` (cost weight==0 → 0)."""
    alpha = p["a_sti_radius"]
    if pack is None:
        return alpha
    gate = getattr(pack, "sti_radius_gate", None)
    if gate is None:
        return alpha
    return alpha * gate.to(device=alpha.device, dtype=alpha.dtype)


def apply_a_sti_radius(i_sti, p, pack):
    """``i += a_sti_radius[r] * sti_wave`` on spot radius PR contribs; else pass-through.

    Uses :func:`a_sti_radius_effective` so gated slots are 0 whether indi or fixed.
    """
    sti_radius = getattr(pack, "sti_radius", None) if pack is not None else None
    if sti_radius is None or "a_sti_radius" not in p:
        return i_sti
    wave = pack.sti_wave
    batch = pack.sti_batch
    node = pack.sti_node
    if wave is None or batch is None or node is None:
        raise ValueError(
            "spot pack sti_radius set but sti_wave/sti_batch/sti_node missing"
        )
    if i_sti.dim() == 2:
        i_sti = i_sti.unsqueeze(0)
    alpha = a_sti_radius_effective(p, pack)
    out = i_sti.clone()
    if batch.numel() == 0:
        return out
    B, T, N = out.shape
    add = alpha[sti_radius][:, None] * wave[None, :]
    flat = out.permute(0, 2, 1).reshape(B * N, T)
    flat.index_add_(0, batch * N + node, add)
    return flat.reshape(B, N, T).permute(0, 2, 1).contiguous()


def pack_t_onset(pack) -> int:
    """Stimulus onset index for ``pack``.

    Prefer explicit ``pack.t_onset`` (spot when ``ms_post`` extends ``i_sti`` past
    ``gt``). Else ``n_t - gt.shape[1]`` (moving_bar / ``ms_post=0``).
    """
    t = getattr(pack, "t_onset", None)
    if t is not None:
        return int(t)
    i_sti = pack.i_sti
    n_t = int(i_sti.shape[1] if i_sti.dim() == 3 else i_sti.shape[0])
    return n_t - int(pack.gt.shape[1])


def step_delta_ms(session, t: int, t_onset: int) -> float:
    """``delta_ms`` for the update that produces sample ``t`` (from ``t-1`` → ``t``).

    Pre-onset steps (``t <= t_onset``) use ``session.delta_ms_pre``; later steps
    use ``session.delta_ms``.
    """
    return float(
        session.delta_ms_pre if int(t) <= int(t_onset) else session.delta_ms
    )


def forward_full(session, p, i_sti, *, pack=None):
    """Shared full-T forward for every ``session.model``.

    Time index ``t`` is post-update at step ``t``. Membrane drive comes from
    ``MODEL_DRIVERS[model].prepare_i_sti`` / ``pre_steady`` / ``step``.

    ``session.train_opts['pre_grad']`` (default ``True``): when ``False``, steps
    with ``t < t_onset`` run under ``torch.no_grad()``, then ``v`` / ``state``
    are detached before post steps so BPTT does not enter pre.

    Returns
    -------
    Absolute ``v`` ``(B, T, N)``.
    """
    if session.model not in MODEL_DRIVERS:
        raise ValueError(
            f"no MODEL_DRIVERS entry for model={session.model!r}; "
            f"expected one of {tuple(MODEL_DRIVERS)}"
        )
    drv = MODEL_DRIVERS[session.model]
    pack = pack or session.primary_readout
    i_sti = apply_a_sti_radius(
        drv.prepare_i_sti(session, p, i_sti, pack), p, pack,
    )
    B, t_end = int(i_sti.shape[0]), int(i_sti.shape[1])
    t_onset = pack_t_onset(pack)
    pre_grad = bool((session.train_opts or {})["pre_grad"])
    state, v = drv.pre_steady(session, p, B, i_sti=i_sti)
    v_rows = [v]

    def take(t):
        nonlocal state, v
        state, v = drv.step(
            state, v, p, i_sti[:, t - 1], session,
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
    state = tuple(s.detach() for s in state)
    v = v.detach()
    for t in range(max(t_onset, 1), t_end):
        take(t)
    return torch.stack(v_rows, dim=1)


def forward_nodes(session, p, node_index=None, i_sti=None, pack=None):
    """``forward_full`` then index nodes; squeeze when ``i_sti`` is ``(T, N)``."""
    pack = pack or session.primary_readout
    if node_index is None:
        node_index = pack.readout_node
    if i_sti is None:
        i_sti = session.pack_i_sti(pack)
    squeeze = i_sti.dim() == 2
    i_sti_b = i_sti.unsqueeze(0) if squeeze else i_sti
    out = forward_full(session, p, i_sti_b, pack=pack)
    out = out[:, :, node_index]
    if squeeze:
        out = out.squeeze(0)
    return out
