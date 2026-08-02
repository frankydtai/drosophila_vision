# -*- coding: utf-8 -*-
"""Shared full-T absolute ``v`` forward for all neuron models.

Per-model modules supply only ``prepare_i_sti`` / ``init_state`` / ``step``.
This module owns the time loop. Training / plots read absolute membrane ``v``;
cost compares ``v`` to ``gt_scale * gt + gt_bias``. The unused Ca filter stays
in ``neuron.filter_ca``.
"""
from __future__ import annotations

import torch

from neuron import model_borst as _model_borst
from neuron import model_hp_lp as _model_hp_lp

# Per-model dynamics for ``forward_full`` (prepare_i_sti / init_state / step only).
MODEL_DRIVERS = {
    "borst": _model_borst,
    "hp_lp": _model_hp_lp,
}


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


def _detach_state(state):
    """Detach every tensor in a model ``state`` tuple."""
    return tuple(s.detach() for s in state)


def forward_full(session, p, i_sti, *, pack=None):
    """Shared full-T forward for every ``session.model``.

    Time index ``t`` is post-update at step ``t``. Membrane drive comes from
    ``MODEL_DRIVERS[model].prepare_i_sti`` / ``init_state`` / ``step``.

    ``session.train_opts['pre_grad']`` (default ``True``): when ``False``, steps
    with ``t < t_onset`` run under ``torch.no_grad()``, then ``v`` / ``state``
    are detached before post steps so BPTT does not enter pre.

    Returns
    -------
    Absolute ``v`` ``(B, T, N)``.
    """
    try:
        drv = MODEL_DRIVERS[session.model]
    except KeyError as exc:
        raise ValueError(
            f"no MODEL_DRIVERS entry for model={session.model!r}; "
            f"expected one of {tuple(MODEL_DRIVERS)}"
        ) from exc

    pack = pack or session.primary_readout
    i_sti = drv.prepare_i_sti(session, p, i_sti, pack)
    B, t_end, _n = int(i_sti.shape[0]), int(i_sti.shape[1]), int(i_sti.shape[2])
    t_onset = pack_t_onset(pack)
    pre_grad = bool((session.train_opts or {})["pre_grad"])
    state, v = drv.init_state(session, p, B)
    v_rows = [v]
    if pre_grad or t_onset <= 0:
        for t in range(1, t_end):
            state, v = drv.step(state, v, p, i_sti[:, t - 1], session)
            v_rows.append(v)
        return torch.stack(v_rows, dim=1)
    with torch.no_grad():
        for t in range(1, t_onset):
            state, v = drv.step(state, v, p, i_sti[:, t - 1], session)
            v_rows.append(v)
    state = _detach_state(state)
    v = v.detach()
    for t in range(max(t_onset, 1), t_end):
        state, v = drv.step(state, v, p, i_sti[:, t - 1], session)
        v_rows.append(v)
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
