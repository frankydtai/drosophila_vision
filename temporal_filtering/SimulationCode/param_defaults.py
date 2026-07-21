# -*- coding: utf-8 -*-
"""Trainable parameter numeric defaults (lo / hi / init / jit / fixed_val).

Schema segment *structure* (name, kind, count, indi/shared/fixed partitions)
stays in ``FiveCol_MedSim_Pytorch.build_*_schema``; this module is the single
place to edit box bounds and initialisation numbers.

``fixed_val`` is used for units in the fixed partition when present (else ``init``).
"""
from __future__ import annotations

from training_config import DELTAT_MS

# Shared gain box (in_gain / out_gain / out_scale upper end; syn_strength hi).
GAIN_LO = 0.1
GAIN_HI = 100.0

P = {
    # --- conductance + adaptive shared gains / readout ---
    "in_gain": dict(lo=GAIN_LO, hi=GAIN_HI, init=1, jit=0.1),
    "out_gain": dict(lo=GAIN_LO, hi=GAIN_HI, init=1, jit=01),
    "out_scale": dict(lo=GAIN_LO, hi=GAIN_HI, init=1., jit=0.1),
    # --- conductance type→type α (network ScatterConn) ---
    "syn_strength": dict(lo=0.0, hi=GAIN_HI, init=1.0, jit=0.1),
    # --- conductance Ih ---
    "Ih_gmax": dict(lo=0.0, hi=100.0, init=50.0, jit=10.0, fixed_val=0.0),
    "Ih_gmax_off": dict(lo=0.0, hi=100.0, init=50.0, jit=10.0, fixed_val=0.0),
    "Ih_midv": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=5.0),
    "Ih_slope": dict(lo=-0.40, hi=-0.20, init=-0.25, jit=0.02),
    "tau_midv": dict(lo=-70.0, hi=-40.0, init=-50.0, jit=5.0),
    "Ih_midv_off": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=5.0),
    "Ih_slope_off": dict(lo=-0.40, hi=-0.20, init=-0.25, jit=0.02),
    "tau_midv_off": dict(lo=-70.0, hi=-40.0, init=-50.0, jit=5.0),
    # --- adaptive ---
    "tau_m": dict(lo=DELTAT_MS, hi=1000.0, init=50.0, jit=10.0),
    "bias": dict(lo=-2.0, hi=2.0, init=0.0, jit=0.1),
    "adapt_gain": dict(lo=-2.0, hi=2.0, init=0.0, jit=0.1, fixed_val=0.0),
    "tau_adapt": dict(lo=DELTAT_MS, hi=2000.0, init=100.0, jit=20.0, fixed_val=DELTAT_MS),
}
