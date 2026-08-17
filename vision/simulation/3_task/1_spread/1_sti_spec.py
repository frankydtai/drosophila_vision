# -*- coding: utf-8 -*-
"""Spread sti spec: timing, ``sti_mask``, and bright/dark ``contrast`` (shared by spot / spread / moving_bar)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from network import path  # noqa: F401 -- FAFBv783 on sys.path
from neuron.borst import t_from_ms

CONTRASTS = ("bright", "dark")


@dataclass(frozen=True)
class StiTiming:
    ms_pre: float
    ms_response: float
    ms_post: float
    ms_sti: Optional[float]
    delta_ms: float
    delta_ms_pre: float
    t_onset: int
    n_t: int
    n_t_gt: int


def standardize_sti_timing(sti_opts: dict) -> dict:
    ms_sti = sti_opts.get("ms_sti")
    ms_response = sti_opts.get("ms_response")
    if ms_sti is not None and ms_response is not None:
        sti_opts["ms_response"] = max(float(ms_response), float(ms_sti))
    return sti_opts


def override_sti_timing(
    sti_opts: dict,
    *,
    ms_pre=None,
    ms_response=None,
    ms_post=None,
    ms_sti=None,
    delta_ms=None,
    delta_ms_pre=None,
) -> dict:
    pairs = (
        ("ms_pre", ms_pre),
        ("ms_post", ms_post),
        ("delta_ms", delta_ms),
        ("delta_ms_pre", delta_ms_pre),
        ("ms_response", ms_response),
        ("ms_sti", ms_sti),
    )
    val = lambda ms: None if ms is None else float(ms)
    before = {key: val(sti_opts.get(key)) for key, _ in pairs}
    for key, ms in pairs:
        if ms is not None:
            sti_opts[key] = float(ms)
    standardize_sti_timing(sti_opts)
    sti_opts.pop("n_t", None)
    return {key: sti_opts[key] for key, _ in pairs if val(sti_opts.get(key)) != before[key]}


def resolve_sti_timing(opts) -> StiTiming:
    for key in ("ms_pre", "ms_response", "delta_ms", "delta_ms_pre"):
        if opts.get(key) is None:
            raise ValueError(f"spread sti opts require {key}")
    ms_pre = float(opts["ms_pre"])
    ms_response = float(opts["ms_response"])
    delta_ms = float(opts["delta_ms"])
    delta_ms_pre = float(opts["delta_ms_pre"])
    ms_post = float(opts.get("ms_post", 0.0))
    ms_sti = opts.get("ms_sti")
    t_onset = t_from_ms(ms_pre, delta_ms=delta_ms_pre)
    n_t = (
        t_onset
        + t_from_ms(ms_response, delta_ms=delta_ms)
        + t_from_ms(ms_post, delta_ms=delta_ms)
        + 1
    )
    n_t_gt = t_onset + t_from_ms(ms_response, delta_ms=delta_ms) + 1
    return StiTiming(
        ms_pre=ms_pre,
        ms_response=ms_response,
        ms_post=ms_post,
        ms_sti=ms_sti,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        t_onset=int(t_onset),
        n_t=int(n_t),
        n_t_gt=int(n_t_gt),
    )


def t_sti_end(t_onset, n_t, ms_sti=None, *, delta_ms: float) -> int:
    t0 = int(t_onset)
    mt = int(n_t)
    if mt <= 0:
        raise ValueError(f"n_t must be positive, got {n_t}")
    if ms_sti is None:
        return mt - 1
    w = max(1, t_from_ms(float(ms_sti), delta_ms=float(delta_ms)))
    return min(mt - 1, t0 + w - 1)


def sti_mask(t_onset, n_t, ms_sti=None, *, delta_ms: float) -> np.ndarray:
    t_onset = int(t_onset)
    n_t = int(n_t)
    mask = np.zeros(n_t)
    if ms_sti is None:
        mask[t_onset:] = 1.0
    else:
        n_t_on = max(1, t_from_ms(ms_sti, delta_ms=delta_ms))
        mask[t_onset:min(n_t, t_onset + n_t_on)] = 1.0
    return mask
