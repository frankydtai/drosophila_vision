# -*- coding: utf-8 -*-
"""Neuron parameter formulas — no numeric bindings.

Numeric literals live in ``training.defaults`` and are injected by the
caller (session fields / kwargs). Schema box numbers live in
``training.defaults.PARAM_BOXES``.
"""
from __future__ import annotations

from typing import Tuple


def ms_to_t(ms: float, *, delta_ms: float) -> int:
    """Convert milliseconds to time index count ``t`` (rounded)."""
    return int(round(float(ms) / float(delta_ms)))


def membrane_cdt(capac: float, delta_ms: float) -> float:
    """``capac / delta_ms``."""
    return float(capac) / float(delta_ms)


def e_ih_off(e_leak_rest: float, e_ih: float) -> float:
    """OFF-channel reversal ``2 * E_LEAK_REST - E_Ih``."""
    return 2.0 * float(e_leak_rest) - float(e_ih)


# Non-numeric vocabularies (names / modes), not run defaults.
IH_OFF_MODES = ("on", "off", "mirrored")
IH_DIR_REVERSE_CELLS: Tuple[int, ...] = ()
LEAK_DEPOL_CELLS = ["L1", "L2", "L3"]
KNOWN_MODELS = ("borst", "hp_lp")
