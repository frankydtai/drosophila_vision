# -*- coding: utf-8 -*-
"""Numeric source for membrane constants, schema boxes, stimulus, and CLI values.

Literals / constant bags only — no functions. Only ``4_training`` / figures /
run scripts import this module. Layers ``1_neuron`` / ``2_network`` /
``3_task`` take numbers by injection only (Gruntman paradigm ms/geometry
constants may live in ``task.moving_bar``).

Constants follow original definition order across numbered cores
``1.1`` … ``4.7`` (empty cores omitted).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# 1.1 neuron.params (flat; no Physics bag)
# ---------------------------------------------------------------------------

DELTA_MS = 10.0
CAPAC = 40.0
G_LEAK = 1.0
G_IN = 1.0  # nS; hp_lp converts i_sti (pA) → mV via i_sti / g_in
E_EXC = 10.0
E_INH = -70.0
E_IH = 50.0
E_LEAK_REST = -50.0
E_LEAK_DEPOL = -20.0
IH_GAIN = 1.0
CA_TAU = 50.0
DATA_AMP = 20.0
STATE_CLAMP = 1.0e6
SYN_SCALE_EXC = 0.001
SYN_SCALE_INH = 0.001

IH_OFF = "on"

GAIN_LO = 0.5
GAIN_HI = 5.0
IH_GMAX_INDI_NAMES = ("L1", "L2", "L4", "L5")

PARAM_BOXES: Dict[str, dict] = {
    "in_gain": dict(lo=GAIN_LO, hi=GAIN_HI, init=1.0, jit=0.1),
    "out_gain": dict(lo=GAIN_LO, hi=GAIN_HI, init=1.0, jit=0.1),
    "out_scale": dict(lo=GAIN_LO, hi=GAIN_HI, init=1.0, jit=0.1),
    "syn_strength_cell": dict(lo=GAIN_LO, hi=GAIN_HI, init=1.0, jit=0.1),
    "syn_strength_edge": dict(lo=GAIN_LO, hi=GAIN_HI, init=1.0, jit=0.1),
    "v_th": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=0.0, fixed_val=-50.0),
    "Ih_gmax": dict(lo=0.0, hi=100.0, init=50.0, jit=10.0, fixed_val=0.0),
    "Ih_gmax_off": dict(lo=0.0, hi=100.0, init=50.0, jit=10.0, fixed_val=0.0),
    "Ih_midv": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=5.0),
    "Ih_slope": dict(lo=-0.40, hi=-0.20, init=-0.25, jit=0.02),
    "tau_midv": dict(lo=-70.0, hi=-40.0, init=-50.0, jit=5.0),
    "Ih_midv_off": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=5.0),
    "Ih_slope_off": dict(lo=-0.40, hi=-0.20, init=-0.25, jit=0.02),
    "tau_midv_off": dict(lo=-70.0, hi=-40.0, init=-50.0, jit=5.0),
    "tau_lp": dict(lo=10.0, hi=100.0, init=50.0, jit=5.0),
    "v_rest": dict(lo=-20.0, hi=20.0, init=0.0, jit=1.0),
    "tau_hp": dict(lo=100.0, hi=10000.0, init=200.0, jit=20.0, fixed_val=10000.0),
    "hp_gain": dict(lo=GAIN_LO, hi=GAIN_HI, init=1.0, jit=0.1, fixed_val=1.0),
}

MODEL = "hp_lp"

# ---------------------------------------------------------------------------
# 1.2 neuron.schema
# ---------------------------------------------------------------------------

SYN_MODE = "per_cell"

# ---------------------------------------------------------------------------
# 1.2 neuron.forward
# ---------------------------------------------------------------------------
PRE_GRAD = True

# ---------------------------------------------------------------------------
# 2.1 network.path
# ---------------------------------------------------------------------------

NETWORK = "right_min_neuron1_extent10"

# ---------------------------------------------------------------------------
# 2.3 network.construction
# ---------------------------------------------------------------------------

I_BASELINE = 20.0
I_BRIGHT = 40.0
I_DARK = 0.0

# ---------------------------------------------------------------------------
# 3.1 task.spot.input
# ---------------------------------------------------------------------------

PRE_MS = 500.0
RESPONSE_MS = 500.0
PULSE_MS = 100.0
SPOT_EXTENT = 1.0
SPOT_EXTENTS: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
FULLY_INSIDE = True
MULTI_SPOT = True
SHIFT_EXTENT = 1

# ---------------------------------------------------------------------------
# 3.2 task.spot.data
# ---------------------------------------------------------------------------

SPOT_COST_RADII: Tuple[float, ...] = (0.0, 1.0, math.sqrt(3), 2.0)
SPOT_COST_RADIUS_WEIGHT: Dict[float, float] = {
    0.0: 1.0,
    1.0: 1.0 / 6.0,
    2.0: 1.0 / 6.0,
}
SPOT_COST_RADIUS_WEIGHT_EXTENT1: Dict[float, float] = {
    0.0: 1.0,
    1.0: 1.0 / 6.0,
}
SPOT_COST_RADIUS_KEY_ALIASES: Dict[str, float] = {
    "sqrt3": math.sqrt(3),
}

# ---------------------------------------------------------------------------
# task.moving_bar.input
# ---------------------------------------------------------------------------

MULTI_BAR = True

# ---------------------------------------------------------------------------
# 4.1 training.config
# ---------------------------------------------------------------------------

TASK = "spot_bright"

# ---------------------------------------------------------------------------
# 4.4 training.cost
# ---------------------------------------------------------------------------

NOFRUNS = 1
NOFSTEPS_CPU = 50
NOFSTEPS_GPU = 200
LRS = "0.1"
CHECKPOINT_INTERVAL = 1000

# ---------------------------------------------------------------------------
# 4.5 training.session
# ---------------------------------------------------------------------------

FP = 64
SEQUENTIAL = False

# ---------------------------------------------------------------------------
# 6 analyze.cell_dynamics
# ---------------------------------------------------------------------------

T_REL_START = -10
T_REL_STOP = 10