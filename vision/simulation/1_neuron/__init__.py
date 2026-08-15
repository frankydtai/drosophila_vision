# -*- coding: utf-8 -*-
"""Neuron models: ``--model borst``, hp_lp.

Dynamics live in ``neuron.model_borst`` / ``neuron.model_hp_lp``.
Shared full-T ``v`` forward is ``neuron.forward``; the unused Ca filter
lives in ``neuron.filter_ca``; pack readout selection in
``neuron.readout``. The ``train`` package owns session, cost, and the
CLI driver. Numeric defaults live in ``const_default``.
"""
from __future__ import annotations

from neuron.param import (
    EULER_CLI,
    EULER_MODES,
    I_H_DIR_REVERSE_CELLS,
    KNOWN_MODELS,
    e_h_rev,
    expand_euler,
    t_from_ms,
    t_abs_from_ms,
    ms_from_t,
)
from neuron.schema import (
    I_H_SHAPE_PARAMS,
    SYN_MODES,
    build_borst_schema,
    build_hp_lp_schema,
    build_schema,
    syn_strength,
)
from neuron import model_borst as _model_borst
from neuron.filter_ca import filter_ca
from neuron.forward import (
    MODEL_DRIVERS,
    ca_from_v_ca,
    forward_ca,
    forward_full,
    forward_nodes,
    forward_v,
    pack_t_onset,
    step_delta_ms,
    v_ca_from_v,
)
from neuron.readout import (
    CA_PACK_COST_TRACES,
    pack_cost_traces,
    pack_traces,
    window_time_traces,
)

# --- borst ---
update_v = _model_borst.update_v
v_component_from_g = _model_borst.v_component_from_g


__all__ = [
    "KNOWN_MODELS",
    "I_H_SHAPE_PARAMS",
    "SYN_MODES",
    "MODEL_DRIVERS",
    "CA_PACK_COST_TRACES",
    "e_h_rev",
    "t_from_ms",
    "t_abs_from_ms",
    "ms_from_t",
    "build_schema",
    "build_borst_schema",
    "build_hp_lp_schema",
    "syn_strength",
    "update_v",
    "v_component_from_g",
    "filter_ca",
    "forward_full",
    "forward_nodes",
    "forward_v",
    "forward_ca",
    "ca_from_v_ca",
    "v_ca_from_v",
    "pack_t_onset",
    "step_delta_ms",
    "pack_cost_traces",
    "pack_traces",
    "window_time_traces",
    "I_H_DIR_REVERSE_CELLS",
    "EULER_MODES",
    "EULER_CLI",
    "expand_euler",
]
