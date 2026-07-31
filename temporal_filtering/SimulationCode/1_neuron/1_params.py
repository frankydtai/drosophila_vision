# -*- coding: utf-8 -*-
"""Neuron physics *types* and formulas — no numeric bindings.

Numeric values are assembled by the caller (``training.defaults``) into
:class:`Physics` and passed into dynamics. Schema box numbers live in
``training.defaults.PARAM_BOXES``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Physics:
    """Membrane / synapse / Ih / Ca constants for one run (all fields required)."""

    delta_ms: float
    capac: float
    g_leak: float
    E_exc: float
    E_inh: float
    E_Ih: float
    E_LEAK_REST: float
    E_LEAK_DEPOL: float
    Ih_gain: float
    Ca_tau: float
    DATA_AMP: float
    STATE_CLAMP: float
    exc_synweight: float
    inh_synweight: float

    @property
    def cdt(self) -> float:
        """``capac / delta_ms``."""
        return self.capac / self.delta_ms

    @property
    def E_IH_OFF(self) -> float:
        """OFF-channel reversal ``2 * E_LEAK_REST - E_Ih``."""
        return 2.0 * self.E_LEAK_REST - self.E_Ih


def ms_to_t(ms: float, *, delta_ms: float) -> int:
    """Convert milliseconds to time index count ``t`` (rounded)."""
    return int(round(float(ms) / float(delta_ms)))


# Non-numeric vocabularies (names / modes), not run defaults.
IH_OFF_MODES = ("on", "off", "mirrored")
IH_DIR_REVERSE_CELLS: Tuple[int, ...] = ()
LEAK_DEPOL_TYPES = ["L1", "L2", "L3"]
KNOWN_MODELS = ("borst", "hp_lp")
