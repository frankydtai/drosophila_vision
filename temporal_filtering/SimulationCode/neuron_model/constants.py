# -*- coding: utf-8 -*-
"""Shared biophysics / neuron-model constants (used by conductance and hp_lp)."""
from __future__ import annotations

from typing import Tuple

from training_config import DELTAT_MS

# simulation step
deltat = DELTAT_MS  # [ms]

# conductance membrane
g_leak = 1.0  # nS
E_exc = +10.0  # mV
E_inh = -70.0  # mV
capac = +40.0  # pF → 50 ms τ_m when g_leak = 1 nS
cdt = capac / deltat

Ca_tau = 50.0  # ms Ca readout

E_LEAK_REST = -50.0
E_LEAK_DEPOL = -20.0

exc_synweight = 0.001
inh_synweight = 0.001

# Ih
E_Ih = +50.0  # mV, ON-channel reversal
E_IH_OFF = -150.0  # OFF-channel reversal (2*E_LEAK_REST - E_Ih)
Ih_gain = 1.0

IH_OFF_MODES = ("on", "off", "mirrored")
IH_OFF_DEFAULT = "on"
IH_OFF_SCALAR_SEGMENTS = frozenset({"Ih_midv_off", "Ih_slope_off", "tau_midv_off"})
IH_OFF_GMAX_SEGMENT = "Ih_gmax_off"
IH_DIR_REVERSE_CELLS: Tuple[int, ...] = ()

# hp_lp / explicit-Euler state clamp
STATE_CLAMP = 1.0e6

# registered model names
KNOWN_MODELS = ("conductance", "hp_lp")
