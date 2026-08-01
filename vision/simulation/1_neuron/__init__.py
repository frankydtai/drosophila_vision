# -*- coding: utf-8 -*-
"""Neuron models: ``--model borst``, hp_lp.

Dynamics live in ``neuron.model_borst`` / ``neuron.model_hp_lp``.
Shared full-T ``v`` forward is ``neuron.forward``; the unused Ca filter
lives in ``neuron.filter_ca``; pack readout selection in
``neuron.readout``. The ``training`` package owns session, cost, and the
CLI driver. Numeric defaults live in ``training.defaults``.
"""
from __future__ import annotations

from neuron.params import (
    IH_DIR_REVERSE_CELLS,
    IH_OFF_MODES,
    KNOWN_MODELS,
    LEAK_DEPOL_CELLS,
    e_ih_off,
    membrane_cdt,
    ms_to_t,
)
from neuron.schema import (
    ALL_PARAM_NAMES,
    IH_SHAPE_PARAM_NAMES,
    SYN_MODES,
    build_borst_schema,
    build_hp_lp_schema,
    borst_ih_off_kwargs,
    default_schema,
    normalize_syn_mode,
    synaptic_scale,
)
from neuron import model_borst as _model_borst
from neuron import model_hp_lp as _model_hp_lp
from neuron.filter_ca import ca_alpha, ca_filter
from neuron.forward import (
    MODEL_DRIVERS,
    run_full,
    run_nodes,
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
v_budget_from_g = _model_borst.v_budget_from_g

# --- hp_lp ---
update_state_hp_lp = _model_hp_lp.update_state_hp_lp


__all__ = [
    "KNOWN_MODELS",
    "ALL_PARAM_NAMES",
    "IH_SHAPE_PARAM_NAMES",
    "SYN_MODES",
    "MODEL_DRIVERS",
    "CA_PACK_READOUTS",
    "e_ih_off",
    "membrane_cdt",
    "ms_to_t",
    "default_schema",
    "build_borst_schema",
    "build_hp_lp_schema",
    "borst_ih_off_kwargs",
    "normalize_syn_mode",
    "synaptic_scale",
    "rectsyn",
    "update_v",
    "v_budget_from_g",
    "update_state_hp_lp",
    "ca_filter",
    "ca_alpha",
    "run_full",
    "run_nodes",
    "pack_readout",
    "readout_pack_traces",
    "window_time_traces",
    "LEAK_DEPOL_CELLS",
    "IH_OFF_MODES",
    "IH_DIR_REVERSE_CELLS",
]
