# -*- coding: utf-8 -*-
"""Neuron models: ``--model borst``, hp_lp.

Dynamics live in ``neuron.model_borst`` / ``neuron.model_hp_lp``.
Shared full-T ``v`` forward is ``neuron.forward``; the unused Ca filter
lives in ``neuron.filter_ca``; pack readout selection in
``neuron.readout``. The ``training`` package owns session, cost, and the
CLI driver. Numeric defaults live in ``param_defaults``.
"""
from __future__ import annotations

from neuron.params import (
    EULER_CLI,
    EULER_MODES,
    I_H_DIR_REVERSE_CELLS,
    I_H_REV_MODES,
    KNOWN_MODELS,
    e_h_rev,
    expand_euler,
    membrane_dt_over_c,
    ms_to_t,
    ms_to_t_abs,
    t_to_ms,
)
from neuron.schema import (
    ALL_PARAM_NAMES,
    I_H_SHAPE_PARAM_NAMES,
    SYN_MODES,
    build_borst_schema,
    build_hp_lp_schema,
    borst_i_h_rev_kwargs,
    default_schema,
    normalize_syn_mode,
    syn_strength,
)
from neuron import model_borst as _model_borst
from neuron import model_hp_lp as _model_hp_lp
from neuron.filter_ca import ca_alpha, ca_filter
from neuron.forward import (
    MODEL_DRIVERS,
    forward_full,
    forward_nodes,
    pack_t_onset,
    step_delta_ms,
)
from neuron.readout import (
    CA_PACK_READOUTS,
    pack_readout,
    readout_pack_traces,
    window_time_traces,
)

# --- borst ---
rectsyn = _model_borst.rectsyn
update_v = _model_borst.update_v
v_component_from_g = _model_borst.v_component_from_g

# --- hp_lp ---
update_state_hp_lp = _model_hp_lp.update_state_hp_lp


__all__ = [
    "KNOWN_MODELS",
    "ALL_PARAM_NAMES",
    "I_H_SHAPE_PARAM_NAMES",
    "SYN_MODES",
    "MODEL_DRIVERS",
    "CA_PACK_READOUTS",
    "e_h_rev",
    "membrane_dt_over_c",
    "ms_to_t",
    "ms_to_t_abs",
    "t_to_ms",
    "default_schema",
    "build_borst_schema",
    "build_hp_lp_schema",
    "borst_i_h_rev_kwargs",
    "normalize_syn_mode",
    "syn_strength",
    "rectsyn",
    "update_v",
    "v_component_from_g",
    "update_state_hp_lp",
    "ca_filter",
    "ca_alpha",
    "forward_full",
    "forward_nodes",
    "pack_t_onset",
    "step_delta_ms",
    "pack_readout",
    "readout_pack_traces",
    "window_time_traces",
    "I_H_REV_MODES",
    "I_H_DIR_REVERSE_CELLS",
    "EULER_MODES",
    "EULER_CLI",
    "expand_euler",
]
