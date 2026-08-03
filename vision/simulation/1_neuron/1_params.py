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


def membrane_dt_over_c(capac: float, delta_ms: float) -> float:
    """``delta_ms / capac``."""
    return float(delta_ms) / float(capac)


def e_ih_off(e_leak_rest: float, e_ih: float) -> float:
    """OFF-channel reversal ``2 * E_LEAK_REST - E_Ih``."""
    return 2.0 * float(e_leak_rest) - float(e_ih)


# Non-numeric vocabularies (names / modes), not run defaults.
IH_OFF_MODES = ("on", "off", "mirrored")
IH_DIR_REVERSE_CELLS: Tuple[int, ...] = ()
LEAK_DEPOL_CELLS = ["L1", "L2", "L3"]
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
