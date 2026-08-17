# -*- coding: utf-8 -*-
"""Numeric source for model constants, schema param, sti, and CLI values.

Literals and constants only. Only ``4_train`` / figures / analyze / run scripts may
import this module. Layers ``1_neuron`` / ``2_network`` / ``3_task`` take
numbers by injection only (Gruntman moving_bar ms/geometry constants may live
in ``task.moving_bar``).

Enum **allowed-token sets** (``("a", "b", …)``) do **not** live here — only the
default scalar string, with allowed tokens named in a comment pointing at
``train.config`` (same pattern as ``cost_norm`` → ``COST_NORMS``).

Constants follow original definition order across numbered cores
``1.1`` … ``4.7`` (empty cores omitted).
"""
from __future__ import annotations

from typing import Dict, Tuple

RUN_NAME = """
29104256-run-n-iter-300-param-a_h.mode=fixed-param-a_h.mode.h_cells=indi-param-a_h.val.L1=0.5
""".strip()
RUN_PATH = "hp_lp/" + RUN_NAME

# ---------------------------------------------------------------------------
# 1.1.1   neuron.borst / MODEL (session scalars; not packed into z)
# ---------------------------------------------------------------------------

MODEL: Dict[str, object] = {
    "delta_ms": 2.0,
    "delta_ms_pre": 2.0,  # pre-onset (t < t_onset); post-onset uses delta_ms
    "cap": 40.0,
    "g_leak": 1.0,  # nS; borst leak conductance; hp_lp converts i_sti (pA) → mV via i_sti / g_leak
    "e_exc": 10.0,
    "e_inh": -70.0,
    "e_h": 50.0,
    "h_g_max": 100.0,
    "gt_amp": 20.0,
    "v_clamp": 1.0e6,
    "a_syn_exc": 0.001,
    "a_syn_inh": 0.001,
    "euler": "im",  # CLI token; expand to implicit|explicit via neuron.borst.expand_euler
}

# ---------------------------------------------------------------------------
# 1.1.2   neuron.hp_lp
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1.2   neuron.filter_ca
# ---------------------------------------------------------------------------

# Readout filter default: ``NEURON_SCHEMA['filter']`` (``none`` | ``ca``); see train.config.expand_filter.

# ---------------------------------------------------------------------------
# 1.3   neuron.schema
# ---------------------------------------------------------------------------

NEURON_SCHEMA: Dict[str, object] = {
    "model": "hp_lp",  # borst | hp_lp — see train.config.MODELS
    "filter": "none",  # none | ca — see train.config.expand_filter
    "a_lo": 0.1,
    "a_hi": 10.0,
    "h_cells": ("L1", "L2", "L4", "L5"),
    # mode: indi | shared | fixed | frozen
    # exception: optional per-node init/mode tokens (later wins); see neuron.schema
    "params": {
        "a_gt": dict(lo=0.5, hi=2.0, init=1.0, jit=0.1, mode="indi"),
        "bias_gt": dict(lo=-200.0, hi=200.0, init=0.0, jit=1.0, mode="indi"),
        "syn_strength_cell": dict(lo=0.1, hi=10.0, init=1.0, jit=0.1, mode="indi"),
        "syn_strength_edge": dict(lo=0.1, hi=10.0, init=1.0, jit=0.1, mode="indi"),
        "a_in": dict(lo=0.01, hi=100, init=1.0, jit=0.1, mode="shared"),
        "a_out": dict(lo=0.1, hi=10.0, init=1.0, jit=0.1, mode="indi"),
        "e_leak": dict(lo=-50.0, hi=50.0, init=0.0, jit=1.0, mode="indi"),
        "v_th": dict(lo=-100.0, hi=100.0, init=-50.0, jit=0.0, mode="indi"),
        "tau_lp": dict(lo=10.0, hi=100.0, init=10.0, jit=2.0, mode="indi"),
        "tau_hp_rise": dict(lo=100.0, hi=500.0, init=200.0, jit=20.0, mode="indi"),
        "tau_hp_fall": dict(lo=100.0, hi=500.0, init=200.0, jit=20.0, mode="indi"),
        "a_h": dict(lo=0.0, hi=1.0, init=0, jit=0.1, mode="indi",
            exception="init.L1,L2,L4,L5=0.5",
        ),
        "v_mid_h_g": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=5.0, mode="shared"),
        "v_mid_h_tau": dict(lo=-70.0, hi=-40.0, init=-50.0, jit=5.0, mode="shared"),
        "h_slope": dict(lo=-0.40, hi=-0.20, init=-0.25, jit=0.02, mode="shared"),
        "a_h_rev": dict(lo=0.0, hi=1.0, init=0.0, jit=0.1, mode="fixed",
            exception="mode.h_cells=indi",
        ),
        "v_mid_h_g_rev": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=5.0, mode="shared"),
        "v_mid_h_tau_rev": dict(lo=-70.0, hi=-40.0, init=-50.0, jit=5.0, mode="shared"),
        "h_slope_rev": dict(lo=-0.40, hi=-0.20, init=-0.25, jit=0.02, mode="shared"),
        "v_th_ca": dict(lo=-100.0, hi=100.0, init=-50.0, jit=0.0, mode="indi"),
        "a_ca": dict(lo=0.1, hi=10.0, init=1.0, jit=0.1, mode="indi"),
        "tau_ca": dict(lo=100.0, hi=1000.0, init=350.0, jit=10.0, mode="indi"),
        "a_sti_radius": dict(lo=0.0, hi=1.0, init=0.0, jit=0.05, mode="indi"),
    },
    "syn_mode": "per_cell",
}

# ---------------------------------------------------------------------------
# 1.4   neuron.forward
# ---------------------------------------------------------------------------

NEURON_FORWARD: Dict[str, object] = {
    "pre_grad": True,
}

# ---------------------------------------------------------------------------
# 2.1 network.path
# ---------------------------------------------------------------------------

NETWORK_PATH: Dict[str, object] = {
    "network": "right_min_neuron1_r10",
}

# ---------------------------------------------------------------------------
# 3.1 task.spot.sti_geo
# ---------------------------------------------------------------------------

SPOT_INPUT_GEO: Dict[str, object] = {
    "spot_radius": 1.0,
    "fully_inside": True,
    "multi_spot": True,
    "shift_radius": 1.0,
}

# ---------------------------------------------------------------------------
# 3.1.2 task.spot.sti_spec
# ---------------------------------------------------------------------------

SPOT_INPUT_SPEC: Dict[str, object] = {
    "i_bright": 40.0,
    "i_dark": 0.0,
    "contrasts": ("bright", "dark"),
    "ms_pre": 20.0,
    "ms_sti": 160.0,
    "ms_response": 300.0,
    "ms_post": 0.0,
}

# ---------------------------------------------------------------------------
# 3.2 task.spot.gt (rf/ir literals live in task.spot.gt)
# 3.3 task.spot.pack
# ---------------------------------------------------------------------------

SPOT_PACK: Dict[str, object] = {
    # Spot cost GT mode default (``--spot-gt-mode``). Allowed tokens in comment only:
    # all | positive — see train.config.SPOT_GT_MODES (never define SPOT_GT_MODES here).
    "spot_gt_mode": "pos",
    # Hex-lattice radii (``build_hex.hex_radius`` / ``shell_hexes``); radius 2 = full shell 12.
    "spot_cost_radii": (0, 1, 2),
    # a_sti_radius: center radius=0 baked @1; a_sti_radii are non-center radii.
    # Cost-radius scale==0 → a_sti_radius_mask forces that radius to 0 in forward.
    "a_sti_radii": (1, 2),
    "spot_cost_radius_scale": {
        0: 1.0,
        1: 1.0 / 3.0,
        2: 1.0 / 3.0,
    },
    "spot_cost_radius_scale_radius1": {
        0: 1.0,
        1: 2.0,
    },
}

# ---------------------------------------------------------------------------
# task.moving_bar.sti_geo / sti_spec
# task.moving_bar.gt (fig1 Vm + motion preference; no literals here)
# task.moving_bar.pack (no literals yet)
# ---------------------------------------------------------------------------

MOVING_BAR_INPUT_GEO: Dict[str, object] = {
    "multi_bar": True,
    "bar_radius": 2,
}

MOVING_BAR_INPUT_SPEC: Dict[str, object] = {
    "i_bright": SPOT_INPUT_SPEC["i_bright"],
    "i_dark": SPOT_INPUT_SPEC["i_dark"],
    "ms_pre": SPOT_INPUT_SPEC["ms_pre"],
}

# ---------------------------------------------------------------------------
# 4.1 train.config
# ---------------------------------------------------------------------------

TRAIN_CONFIG: Dict[str, object] = {
    "task": "spot",
}

# ---------------------------------------------------------------------------
# 4.4 train.param (val_from)
# ---------------------------------------------------------------------------

# Param targets copied from another source at materialize time (``--val-from``).
VAL_FROM: Dict[str, object] = {
    "bias_gt": {"source": "v_onset", "enabled": True},
    "v_th_ca": {"source": "v_th", "enabled": False},
    "a_ca": {"source": "a_out", "enabled": False},
    "a_h_rev": {"source": "a_h", "enabled": False},
    "v_mid_h_g_rev": {"source": "v_mid_h_g", "enabled": False},
    "h_slope_rev": {"source": "h_slope", "enabled": False},
    "v_mid_h_tau_rev": {"source": "v_mid_h_tau", "enabled": False},
}

# ---------------------------------------------------------------------------
# 4.4 train.cost
# 4.5 train.optimization
# ---------------------------------------------------------------------------

TRAIN_OPTIMIZATION: Dict[str, object] = {
    "part_cost_scales": {},
    "cost_norm": "a_gt2",  # gt_power | a_gt2; see train.config.COST_NORMS
    # Spot cost sampling (train_opts; consumed at spot pack build): post-onset ms grid.
    "cost_interval_ms": 10.0,
    # Per-radius explicit post-onset ms (overwrites cost_interval_ms for that radius).
    "cost_ms": {
        1: (0.0, SPOT_INPUT_SPEC["ms_sti"]),
    },
    # t=0 pre steady (``--pre-steady``). Not param init.
    # Shared by borst / hp_lp: probe (ohmic one-shot) | solve (fixed-iter DC).
    "pre_steady": "solve",
    "pre_steady_n_iter": 50,  # solve only
    "pre_steady_damp": 0.1,  # solve under-relaxation
    "n_run": 1,
    "n_iter_cpu": 0,
    "n_iter_gpu": 200,
    "lrs": "0.1",
    "checkpoint_interval": 1000,
}

# ---------------------------------------------------------------------------
# 4.5 train.session
# ---------------------------------------------------------------------------

TRAIN_SESSION: Dict[str, object] = {
    "fp": 32,
    "sequential": False,
}


# ---------------------------------------------------------------------------
# 6 analyze.cell_dynamics
# ---------------------------------------------------------------------------

ANALYZE_CELL_DYNAMICS: Dict[str, object] = {
    "t_rel_start": -10,
    "t_rel_stop": 10,
}

# ---------------------------------------------------------------------------
# 6 analyze.trace
# ---------------------------------------------------------------------------

ANALYZE_SYN_SIGN: Dict[str, object] = {
    "bins": 20,
}

ANALYZE_TRACE: Dict[str, object] = {
    "trace_osc_min_f": 0.5,
    "trace_osc_max_f": 20.0,
    "trace_osc_peak_threshold": 0.5,
    "trace_osc_z_threshold": 2.0,
    "trace_osc_snr_min": 2.0,
    "trace_drift_min_slope_mv_over_s": 1.0,
    "trace_drift_min_r": 0.5,
    "trace_baseline_ms": 200.0,
    "trace_flat_max_abs": 0.5,
    "trace_flat_v_peak_to_peak_max": 1.0,
    "trace_flat_abs_mean": 0.2,
}
