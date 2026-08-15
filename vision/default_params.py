# -*- coding: utf-8 -*-
"""Numeric source for membrane constants, schema optimizable, sti, and CLI values.

Literals and constants only. Only ``4_train`` / figures / analyze / run scripts may
import this module. Layers ``1_neuron`` / ``2_network`` / ``3_task`` take
numbers by injection only (Gruntman paradigm ms/geometry constants may live
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
# 1.1 neuron.param (flat; not nested under Physics)
# ---------------------------------------------------------------------------

NEURON_PARAM: Dict[str, object] = {
    "delta_ms": {"v": 2.0, "ca": 2.0},
    "delta_ms_pre": {"v": 2.0, "ca": 2.0},  # pre-onset (t < t_onset); post-onset uses delta_ms
    "cap": 40.0,
    "g_leak": 1.0,  # nS; borst leak conductance; hp_lp converts i_sti (pA) → mV via i_sti / g_leak
    "e_exc": 10.0,
    "e_inh": -70.0,
    "e_h": 50.0,
    "h_g_max": 100.0,
    "gt_amp": {"v": 20.0, "ca": 2.0},
    "state_clamp": 1.0e6,
    "a_syn_exc": 0.001,
    "a_syn_inh": 0.001,
    "euler": "im",  # CLI token; expand to implicit|explicit via neuron.param.expand_euler
}

# ---------------------------------------------------------------------------
# 1.2 neuron.schema
# ---------------------------------------------------------------------------

NEURON_SCHEMA: Dict[str, object] = {
    "a_lo": 0.1,
    "a_hi": 10.0,
    "h_cells": ("L1", "L2", "L4", "L5"),
    # mode: indi | shared | fixed | frozen
    # param: optional per-node val/mode tokens (later wins); see neuron.schema
    "optimizable": {
        "a_gt": dict(lo=0.5, hi=2.0, val=1.0, jit=0.1, mode="indi"),
        "bias_gt": dict(lo=-200.0, hi=200.0, val=0.0, jit=1.0, mode="indi"),
        "syn_strength_cell": dict(lo=0.1, hi=10.0, val=1.0, jit=0.1, mode="indi"),
        "syn_strength_edge": dict(lo=0.1, hi=10.0, val=1.0, jit=0.1, mode="indi"),
        "a_in": dict(lo=0.01, hi=100, val=1.0, jit=0.1, mode="shared"),
        "a_out": dict(lo=0.1, hi=10.0, val=1.0, jit=0.1, mode="indi"),
        "e_leak": dict(lo=-50.0, hi=50.0, val=0.0, jit=1.0, mode="indi"),
        "v_th": dict(lo=-100.0, hi=100.0, val=-50.0, jit=0.0, mode="indi"),
        "tau_lp": dict(lo=10.0, hi=100.0, val=10.0, jit=2.0, mode="indi"),
        "tau_hp_rise": dict(lo=100.0, hi=500.0, val=200.0, jit=20.0, mode="indi"),
        "tau_hp_fall": dict(lo=100.0, hi=500.0, val=200.0, jit=20.0, mode="indi"),
        "a_h": dict(lo=0.0, hi=1.0, val=0, jit=0.1, mode="indi",
            param="val.L1,L2,L4,L5=0.5",
        ),
        "v_mid_h_g": dict(lo=-70.0, hi=-30.0, val=-50.0, jit=5.0, mode="shared"),
        "v_mid_h_tau": dict(lo=-70.0, hi=-40.0, val=-50.0, jit=5.0, mode="shared"),
        "h_slope": dict(lo=-0.40, hi=-0.20, val=-0.25, jit=0.02, mode="shared"),
        "a_h_rev": dict(lo=0.0, hi=1.0, val=0.0, jit=0.1, mode="fixed",
            param="mode.h_cells=indi",
        ),
        "v_mid_h_g_rev": dict(lo=-70.0, hi=-30.0, val=-50.0, jit=5.0, mode="shared"),
        "v_mid_h_tau_rev": dict(lo=-70.0, hi=-40.0, val=-50.0, jit=5.0, mode="shared"),
        "h_slope_rev": dict(lo=-0.40, hi=-0.20, val=-0.25, jit=0.02, mode="shared"),
        "v_th_ca": dict(lo=-100.0, hi=100.0, val=-50.0, jit=0.0, mode="indi"),
        "a_ca": dict(lo=0.1, hi=10.0, val=1.0, jit=0.1, mode="indi"),
        "tau_ca": dict(lo=100.0, hi=1000.0, val=350.0, jit=10.0, mode="indi"),
        "a_sti_radius": dict(lo=0.0, hi=1.0, val=0.0, jit=0.05, mode="indi"),
    },
    "syn_mode": "per_cell",
}

MODEL: Dict[str, object] = {
    "model": "hp_lp",
}

# ---------------------------------------------------------------------------
# 1.2 neuron.forward
# ---------------------------------------------------------------------------

NEURON_FORWARD: Dict[str, object] = {
    "pre_grad": True,
}

# ---------------------------------------------------------------------------
# 1.3 neuron.filter
# ---------------------------------------------------------------------------

NEURON_FILTER: Dict[str, object] = {
    # Readout filter: none = v (schema skips v_th_ca/a_ca/tau_ca); ca = ca.
    # Allowed tokens in comment only: none | ca — see train.config.expand_filter.
    "filter": "none",
}

# ---------------------------------------------------------------------------
# 2.1 network.path
# ---------------------------------------------------------------------------

NETWORK_PATH: Dict[str, object] = {
    "network": "right_min_neuron1_r10",
}

# ---------------------------------------------------------------------------
# 2.3 network.construction
# ---------------------------------------------------------------------------

NETWORK_CONSTRUCTION: Dict[str, object] = {
    "i_baseline": 20.0,
    "i_bright": 40.0,
    "i_dark": 0.0,
}

# ---------------------------------------------------------------------------
# Shared step-sti timing (spot, static_bar, …)
# ---------------------------------------------------------------------------

STI_TIMING: Dict[str, object] = {
    "ms_pre": {"v": 20.0, "ca": 20.0},
    "ms_sti": {"v": 160.0, "ca": 25.0},
    "ms_response": {"v": 300.0, "ca": 400.0},
    "ms_post": 0.0,
}

# ---------------------------------------------------------------------------
# 3.1 task.spot.sti_geo
# ---------------------------------------------------------------------------

SPOT_INPUT: Dict[str, object] = {
    "spot_radius": 1.0,
    "fully_inside": True,
    "multi_spot": True,
    "shift_radius": 1.0,
}

# ---------------------------------------------------------------------------
# 3.2 task.spot.gt (rf/ir literals live in task.spot.gt)
# 3.3 task.spot.pack
# ---------------------------------------------------------------------------

SPOT_PACK: Dict[str, object] = {
    # Spot cost GT mode default (``--spot-gt-mode``). Allowed tokens in comment only:
    # all | positive — see train.config.SPOT_GT_MODES (never define SPOT_GT_MODES here).
    "spot_gt_mode": "pos",
    # Hex-lattice radii (``build_hex.hex_radius`` / ``members_at_shell``); radius 2 = full shell 12.
    "spot_cost_radii": (0, 1, 2),
    # a_sti_radius: center r=0 baked @1; all a_sti_radii are slots.
    # Cost-radius scale==0 → a_sti_radius_mask forces that slot to 0 in forward.
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

MOVING_BAR_INPUT: Dict[str, object] = {
    "multi_bar": True,
    "bar_radius": 2,
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
    "cost_norm": "a_gt2",  # gt_power | a_gt2; see train.config.COST_NORMS
    # Spot cost sampling (train_opts; consumed at spot pack build): post-onset ms grid.
    "cost_interval_ms": 10.0,
    # Per-radius explicit post-onset ms (overwrites cost_interval_ms for that radius).
    "cost_ms": {
        # Allow per-branch override: second ms value follows ``STI_TIMING["ms_sti"]``
        # and will be resolved by ``open_session``'s ``resolve_filter_branches``.
        1: (0.0, STI_TIMING["ms_sti"]),
    },
    # Membrane t=0 pre steady (``--pre-steady``). Not param init.
    # Shared by borst / hp_lp: probe (ohmic one-shot) | solve (fixed-iter DC).
    "pre_steady": "solve",
    "pre_steady_iters": 50,  # solve only
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

SPOT_STI_TIMING_OPTS: Dict[str, object] = {
    "ms_pre": STI_TIMING["ms_pre"],
    "ms_response": STI_TIMING["ms_response"],
    "ms_post": STI_TIMING["ms_post"],
    "delta_ms": NEURON_PARAM["delta_ms"],
    "delta_ms_pre": NEURON_PARAM["delta_ms_pre"],
    "ms_sti": STI_TIMING["ms_sti"],
}

MOVING_BAR_STI_TIMING_OPTS: Dict[str, object] = {
    "ms_pre": STI_TIMING["ms_pre"],
    "delta_ms": NEURON_PARAM["delta_ms"],
    "delta_ms_pre": NEURON_PARAM["delta_ms_pre"],
}

SPOT_STI_SHARED_OPTS: Dict[str, object] = {
    **SPOT_STI_TIMING_OPTS,
    "shift_radius": SPOT_INPUT["shift_radius"],
    "spot_radius": SPOT_INPUT["spot_radius"],
    "multi_spot": SPOT_INPUT["multi_spot"],
    "fully_inside": SPOT_INPUT["fully_inside"],
}

SPOT_BRIGHT_STI_OPTS: Dict[str, object] = {
    **SPOT_STI_SHARED_OPTS,
    "i_baseline_spot": NETWORK_CONSTRUCTION["i_baseline"],
    "i_bright_spot": NETWORK_CONSTRUCTION["i_bright"],
}

SPOT_DARK_STI_OPTS: Dict[str, object] = {
    **SPOT_STI_SHARED_OPTS,
    "i_baseline_spot": NETWORK_CONSTRUCTION["i_baseline"],
    "i_dark_spot": NETWORK_CONSTRUCTION["i_dark"],
}

MOVING_BAR_BRIGHT_STI_OPTS: Dict[str, object] = {
    **MOVING_BAR_STI_TIMING_OPTS,
    "i_baseline_moving_bar": NETWORK_CONSTRUCTION["i_baseline"],
    "i_bright_moving_bar": NETWORK_CONSTRUCTION["i_bright"],
    "multi_bar": MOVING_BAR_INPUT["multi_bar"],
}

MOVING_BAR_DARK_STI_OPTS: Dict[str, object] = {
    **MOVING_BAR_STI_TIMING_OPTS,
    "i_baseline_moving_bar": NETWORK_CONSTRUCTION["i_baseline"],
    "i_dark_moving_bar": NETWORK_CONSTRUCTION["i_dark"],
    "multi_bar": MOVING_BAR_INPUT["multi_bar"],
}

TRAIN_OPTS: Dict[str, object] = {
    "backend": "network",
    "tasks": (TRAIN_CONFIG["task"],),
    "part_cost_scales": {},
    "cost_norm": TRAIN_OPTIMIZATION["cost_norm"],
    "cost_interval_ms": TRAIN_OPTIMIZATION["cost_interval_ms"],
    "cost_ms": TRAIN_OPTIMIZATION["cost_ms"],
    "pre_steady": TRAIN_OPTIMIZATION["pre_steady"],
    "pre_steady_iters": TRAIN_OPTIMIZATION["pre_steady_iters"],
    "pre_steady_damp": TRAIN_OPTIMIZATION["pre_steady_damp"],
    "sequential": TRAIN_SESSION["sequential"],
    "spot_bright_sti_opts": SPOT_BRIGHT_STI_OPTS,
    "spot_dark_sti_opts": SPOT_DARK_STI_OPTS,
    "moving_bar_bright_sti_opts": MOVING_BAR_BRIGHT_STI_OPTS,
    "moving_bar_dark_sti_opts": MOVING_BAR_DARK_STI_OPTS,
    "packs": None,
    "param_modes": None,
    "euler": NEURON_PARAM["euler"],
    "syn_mode": NEURON_SCHEMA["syn_mode"],
    "pre_grad": NEURON_FORWARD["pre_grad"],
    "val_from": {k: dict(v) for k, v in VAL_FROM.items()},
    "filter": NEURON_FILTER["filter"],
    "spot_gt_mode": SPOT_PACK["spot_gt_mode"],
    "fp": TRAIN_SESSION["fp"],
    "network": None,
    "network_json": None,
    "dev": None,
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
    "trace_drift_min_slope_mv_per_s": 1.0,
    "trace_drift_min_r": 0.5,
    "trace_baseline_ms": 200.0,
    "trace_flat_max_abs": 0.5,
    "trace_flat_v_peak_to_peak_max": 1.0,
    "trace_flat_abs_mean": 0.2,
}
