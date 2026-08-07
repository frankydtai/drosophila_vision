# -*- coding: utf-8 -*-
"""Neuron parameter formulas — no numeric bindings.

Numeric literals live in ``param_defaults`` and are injected by the
caller (session fields / kwargs). Schema box numbers live in
``param_defaults.PARAM_BOXES``.
"""
from __future__ import annotations

from typing import Tuple


def ms_to_t(ms: float, *, delta_ms: float) -> int:
    """Convert milliseconds to time index count ``t`` (rounded)."""
    return int(round(float(ms) / float(delta_ms)))


def t_to_ms(
    t: float,
    *,
    t_onset: int,
    delta_ms_pre: float,
    delta_ms: float,
) -> float:
    """Absolute ms at sample index ``t`` (piecewise pre / post-onset dt)."""
    t = float(t)
    t0 = int(t_onset)
    dt_pre = float(delta_ms_pre)
    dt = float(delta_ms)
    if t <= t0:
        return t * dt_pre
    return float(t0) * dt_pre + (t - float(t0)) * dt


def ms_to_t_abs(
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
    ms_pre = float(t0) * dt_pre
    if ms <= ms_pre:
        return ms_to_t(ms, delta_ms=dt_pre)
    return t0 + ms_to_t(ms - ms_pre, delta_ms=delta_ms)


def membrane_dt_over_c(cap: float, delta_ms: float) -> float:
    """``delta_ms / cap``."""
    return float(delta_ms) / float(cap)


def e_h_rev(e_leak, e_h: float):
    """Rev-channel reversal ``2 * e_leak - e_h`` (scalar or per-node)."""
    return 2.0 * e_leak - float(e_h)


# Non-numeric vocabularies (names / modes), not run defaults.
I_H_REV_MODES = ("on", "off", "mirrored")
I_H_DIR_REVERSE_CELLS: Tuple[int, ...] = ()
KNOWN_MODELS = ("borst", "hp_lp")
EULER_MODES = ("implicit", "explicit")
EULER_CLI = {"im": "implicit", "ex": "explicit"}


def expand_euler(token: str) -> str:
    """Map CLI ``im``/``ex`` (or already-expanded name) → ``implicit``/``explicit``."""
    key = str(token)
    if key in EULER_MODES:
        return key
    if key in EULER_CLI:
        return EULER_CLI[key]
    raise ValueError(
        f"euler {token!r} not in CLI {tuple(EULER_CLI)} or modes {EULER_MODES}"
    )
