# -*- coding: utf-8 -*-
"""Neuron models: ``model=borst``, ``model=hp_lp``, ``model=point_hh``, ``model=shiu``.

Dynamics live in ``neuron.borst`` / ``neuron.hp_lp`` /
``neuron.point_hh`` / ``neuron.shiu``.
Shared ``v`` / ``ca`` forward is ``neuron.forward``; the unused Ca filter
lives in ``neuron.filter_ca``; pack readout selection in
``neuron.readout``. The ``train`` package owns session, cost, and the
CLI driver. Numeric defaults live in ``config``.
"""
from __future__ import annotations

import neuron.borst as _borst
from neuron.point_hh import (
    detach_state as point_hh_detach_state,
    gate_rates as point_hh_gate_rates,
    initial_state as point_hh_initial_state,
    pre_steady as point_hh_pre_steady,
    release_steady as point_hh_release_steady,
    stable_vtrap as point_hh_stable_vtrap,
    steady_gates as point_hh_steady_gates,
    step as point_hh_step,
)
from neuron.borst import (
    EULER_CLI,
    EULER_MODES,
    e_h_rev,
    expand_euler,
    syn_strength,
    t_from_ms,
    t_abs_from_ms,
    ms_from_t,
)
from neuron.filter_ca import filter_ca
from neuron.schema import (
    I_H_SHAPE_PARAMS,
    SYN_MODES,
    build_borst_schema,
    build_hp_lp_schema,
    build_point_hh_schema,
    build_shiu_schema,
    build_schema,
)
from neuron.forward import (
    MODEL_DRIVERS,
    MODELS,
    ca_from_v_ca,
    forward_ca,
    forward_nodes,
    forward_trace,
    forward_v,
    pack_t_onset,
    step_delta_ms,
    v_ca_from_v,
)
from neuron.readout import (
    pack_cost_window_t_min,
    pack_traces,
    window_time_traces,
)

# --- borst ---
update_v = _borst.update_v
v_component_from_g = _borst.v_component_from_g


__all__ = [
    "I_H_SHAPE_PARAMS",
    "SYN_MODES",
    "MODEL_DRIVERS",
    "MODELS",
    "e_h_rev",
    "t_from_ms",
    "t_abs_from_ms",
    "ms_from_t",
    "build_schema",
    "build_borst_schema",
    "build_hp_lp_schema",
    "build_point_hh_schema",
    "build_shiu_schema",
    "syn_strength",
    "update_v",
    "v_component_from_g",
    "filter_ca",
    "point_hh_detach_state",
    "point_hh_gate_rates",
    "point_hh_initial_state",
    "point_hh_pre_steady",
    "point_hh_release_steady",
    "point_hh_stable_vtrap",
    "point_hh_steady_gates",
    "point_hh_step",
    "forward_nodes",
    "forward_v",
    "forward_ca",
    "forward_trace",
    "ca_from_v_ca",
    "v_ca_from_v",
    "pack_cost_window_t_min",
    "pack_t_onset",
    "step_delta_ms",
    "pack_traces",
    "window_time_traces",
    "EULER_MODES",
    "EULER_CLI",
    "expand_euler",
]
