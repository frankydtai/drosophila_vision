# -*- coding: utf-8 -*-
"""Spread sti spec: ``sti_mask`` and bright/dark ``contrast`` (shared by spot / spread / mbar)."""
from __future__ import annotations

import numpy as np

from network import path  # noqa: F401 -- FAFBv783 on sys.path
from neuron.borst import t_from_ms

CONTRASTS = ("bright", "dark")


def i_baseline_from_i_sti(i_sti: dict) -> float:
    """Midpoint of bright/dark sti currents."""
    return 0.5 * (float(i_sti["bright"]) + float(i_sti["dark"]))


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
