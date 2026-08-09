# -*- coding: utf-8 -*-
"""Numeric source for membrane constants, schema boxes, stimulus, and CLI values.

Literals / constant bags only — no functions. Only ``4_training`` / figures /
analyze / run scripts may import this module. Layers ``1_neuron`` /
``2_network`` / ``3_task`` take numbers by injection only (Gruntman paradigm
ms/geometry constants may live in ``task.moving_bar``).

Constants follow original definition order across numbered cores
``1.1`` … ``4.7`` (empty cores omitted).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

DEFAULT_RUN_NAME = """
29008568-run-nofsteps-500-a-h-init.L1,L2,L4,L5-0.5-e-leak-init.L1,L2,L3-0-task-spot
""".strip()
DEFAULT_RUN_PATH = "hp_lp/" + DEFAULT_RUN_NAME

# ---------------------------------------------------------------------------
# 1.1 neuron.params (flat; no Physics bag)
# ---------------------------------------------------------------------------

DELTA_MS = 1.0
DELTA_MS_PRE = 1.0  # pre-onset (t < t_onset) step; post-onset uses DELTA_MS
CAP = 40.0
G_LEAK = 1.0  # nS; borst leak conductance; hp_lp converts i_sti (pA) → mV via i_sti / g_leak
E_EXC = 10.0
E_INH = -70.0
E_H = 50.0
H_G_MAX = 100.0
CA_TAU = 50.0
DATA_AMP = 20.0
STATE_CLAMP = 1.0e6
SYN_SCALE_EXC = 0.001
SYN_SCALE_INH = 0.001

I_H_REV = "off"
EULER = "im"  # CLI token; expand to implicit|explicit via neuron.params.expand_euler

GAIN_LO = 0.1
GAIN_HI = 10.0
H_CELLS = ("L1", "L2", "L4", "L5")

# train_mode: indi | shared | fixed | indi_named
#   indi_named → indi=H_CELLS, fixed=remainder with init_override=0
# Fixed nodes always use init / init_override.
PARAM_BOXES: Dict[str, dict] = {
    "a_gt": dict(lo=0.5, hi=2.0, init=1.0, jit=0.1, train_mode="indi"),
    "bias_gt": dict(lo=-50.0, hi=50.0, init=0.0, jit=1.0, train_mode="indi"),
    "syn_strength_cell": dict(lo=GAIN_LO, hi=GAIN_HI, init=1.0, jit=0.1, train_mode="indi"),
    "syn_strength_edge": dict(lo=GAIN_LO, hi=GAIN_HI, init=1.0, jit=0.1, train_mode="indi"),
    "a_in": dict(lo=0.01, hi=100, init=1.0, jit=0.1, train_mode="shared"),
    "a_out": dict(lo=GAIN_LO, hi=GAIN_HI, init=1.0, jit=0.1, train_mode="indi"),
    "e_leak": dict(lo=-50.0, hi=50.0, init=0.0, jit=1.0, train_mode="indi"),
    "v_th": dict(lo=-100.0, hi=-100.0, init=-50.0, jit=0.0, train_mode="indi"),
    "tau_lp": dict(lo=10.0, hi=100.0, init=10.0, jit=2.0, train_mode="fixed"),
    "tau_hp": dict(lo=100.0, hi=500.0, init=200.0, jit=20.0, train_mode="shared"),
    "a_h": dict(lo=0.0, hi=1.0, init=0, jit=0.1, train_mode="indi_named"),
    "v_mid_h_g": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=5.0, train_mode="shared"),
    "v_mid_h_tau": dict(lo=-70.0, hi=-40.0, init=-50.0, jit=5.0, train_mode="shared"),
    "h_slope": dict(lo=-0.40, hi=-0.20, init=-0.25, jit=0.02, train_mode="shared"),
    "a_h_rev": dict(lo=0.0, hi=1.0, init=0.5, jit=0.1, train_mode="indi_named"),
    "v_mid_h_g_rev": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=5.0, train_mode="shared"),
    "v_mid_h_tau_rev": dict(lo=-70.0, hi=-40.0, init=-50.0, jit=5.0, train_mode="shared"),
    "h_slope_rev": dict(lo=-0.40, hi=-0.20, init=-0.25, jit=0.02, train_mode="shared"),
    # Slots from SPOT_STI_RADII; cost-radius weight==0 gates slot to 0 in forward.
    "a_sti_radius": dict(lo=0.0, hi=1.0, init=0.0, jit=0.05, train_mode="indi"),
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

MS_PRE = 10.0
MS_SPOT = 50.0
MS_RESPONSE = 100.0
MS_POST = 0.0
SPOT_EXTENT = 1.0
SPOT_EXTENTS: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
FULLY_INSIDE = True
MULTI_SPOT = True
SHIFT_EXTENT = 1.0

# ---------------------------------------------------------------------------
# 3.2 task.spot.gt (RecF/ImpR literals live in task.spot.gt)
# 3.3 task.spot.readout
# ---------------------------------------------------------------------------

SPOT_COST_RADII: Tuple[float, ...] = (0.0, 1.0, math.sqrt(3), 2.0)
# a_sti_radius: center r=0 baked @1; all SPOT_STI_RADII are slots.
# Cost-radius weight==0 → sti_radius_gate forces that slot to 0 in forward.
SPOT_STI_RADII: Tuple[float, ...] = (1.0, math.sqrt(3), 2.0)
SPOT_COST_RADIUS_WEIGHT: Dict[float, float] = {
    0.0: 1.0,
    1.0: 1.0 / 3.0,
    2.0: 1.0 / 3.0,
}
SPOT_COST_RADIUS_WEIGHT_EXTENT1: Dict[float, float] = {
    0.0: 1.0,
    1.0: 2,
}
# name → float for cost CLI and a_sti_radius node_names (reverse of SPOT_*_RADII).
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

COST_NORM = "a_gt2"  # gt_power | a_gt2; see training.config.COST_NORMS
# Spot: post-onset cost sample spacing (ms); 0, interval, 2*interval, ...
COST_INTERVAL_MS = 10.0
# Spot: per-radius explicit post-onset ms (overwrites COST_INTERVAL_MS for that r).
COST_MS: Dict[float, Tuple[float, ...]] = {
    1.0: (0.0, MS_SPOT),
}
# Cost/plot affine bias = v at t_onset (not schema bias_gt), clamped to
# PARAM_BOXES["bias_gt"] lo/hi.
BIAS_GT_FROM_V_ONSET = True
# With bias_gt_from_v_onset: True keeps onset in graph; False detaches.
BIAS_GT_FROM_V_ONSET_GRAD = True

# Membrane t=0 pre steady (``--pre-steady MODE``). Not param init.
# Shared by borst / hp_lp: probe (ohmic one-shot) | solve (fixed-iter DC).
PRE_STEADY = "solve"
PRE_STEADY_ITERS = 50  # solve only
PRE_STEADY_DAMP = 0.1  # solve under-relaxation
NOFRUNS = 1
NOFSTEPS_CPU = 0
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
