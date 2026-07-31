# -*- coding: utf-8 -*-
"""Shared full-T ``v`` forward for all neuron models.

Per-model modules supply only ``prepare_signal`` / ``init_state`` / ``step``.
This module owns the time loop and ``t_onset`` reference. Training / plots read
``v_delta = v - v_onset``; the unused Ca filter stays in ``neuron.filter_ca``.
"""
from __future__ import annotations

import torch

from neuron import model_borst as _model_borst
from neuron import model_hp_lp as _model_hp_lp

# Per-model dynamics for ``run_full`` (prepare_signal / init_state / step only).
MODEL_DRIVERS = {
    "borst": _model_borst,
    "hp_lp": _model_hp_lp,
}


def _detach_state(state):
    """Detach every tensor in a model ``state`` tuple."""
    return tuple(s.detach() for s in state)


def run_full(session, p, sig, *, return_v_onset=False, pack=None):
    """Shared full-T forward for every ``session.model``.

    Time index ``t`` is post-update at step ``t``. Membrane drive comes from
    ``MODEL_DRIVERS[model].prepare_signal`` / ``init_state`` / ``step``.
    ``v_onset`` is ``v`` at ``t_onset - 1``.

    ``session.train_opts['pre_grad']`` (default ``True``): when ``False``, steps
    with ``t < t_onset`` run under ``torch.no_grad()``, then ``v`` / ``state`` /
    ``v_onset`` are detached before post steps so BPTT does not enter pre.

    Returns
    -------
    ``v_delta`` ``(B, T, N)``, or with ``return_v_onset``: ``(v_delta, v_onset, v_full)``.
    """
    try:
        drv = MODEL_DRIVERS[session.model]
    except KeyError as exc:
        raise ValueError(
            f"no MODEL_DRIVERS entry for model={session.model!r}; "
            f"expected one of {tuple(MODEL_DRIVERS)}"
        ) from exc

    pack = pack or session.primary_pack
    x = drv.prepare_signal(session, p, sig, pack)
    B, t_end, _n = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    t_onset = int(pack.signal.shape[1] - pack.data.shape[1])
    pre_grad = bool((session.train_opts or {})["pre_grad"])
    state, v = drv.init_state(session, p, B)
    v_rows = [v]
    if pre_grad or t_onset <= 0:
        for t in range(1, t_end):
            state, v = drv.step(state, v, p, x[:, t - 1], session)
            v_rows.append(v)
        v_full = torch.stack(v_rows, dim=1)
        v_onset = v_full[:, t_onset - 1, :].clone()
    else:
        with torch.no_grad():
            for t in range(1, t_onset):
                state, v = drv.step(state, v, p, x[:, t - 1], session)
                v_rows.append(v)
        state = _detach_state(state)
        v = v.detach()
        for t in range(max(t_onset, 1), t_end):
            state, v = drv.step(state, v, p, x[:, t - 1], session)
            v_rows.append(v)
        v_full = torch.stack(v_rows, dim=1)
        v_onset = v_full[:, t_onset - 1, :].detach()
    v_delta = v_full - v_onset.unsqueeze(1)

    if return_v_onset:
        return v_delta, v_onset, v_full
    return v_delta


def run_units(
    session, p, neuron_index=None, return_v_onset=False, sig=None, pack=None,
):
    """``run_full`` then index units; squeeze when ``sig`` is ``(T, N)``."""
    pack = pack or session.primary_pack
    if neuron_index is None:
        neuron_index = pack.readout_unit
    if sig is None:
        sig = session.pack_signal(pack)
    squeeze = sig.dim() == 2
    sig_b = sig.unsqueeze(0) if squeeze else sig
    out, v_onset, _v_full = run_full(
        session, p, sig_b, return_v_onset=True, pack=pack,
    )
    out = out[:, :, neuron_index]
    v_onset = v_onset[:, neuron_index]
    if squeeze:
        out = out.squeeze(0)
        v_onset = v_onset.squeeze(0)
    if return_v_onset:
        return out, v_onset
    return out
