# -*- coding: utf-8 -*-
"""Neuron models: conductance, hp_lp.

Dynamics / schemas live in per-model modules. Shared full-T Ca forward is
``neuron_model.forward``; the Ca readout filter and its inverse live in
``neuron_model.ca_filter``; pack readout selection in ``neuron_model.readout``.
The ``training`` package owns session, cost, and the CLI driver.
"""
from __future__ import annotations

from neuron_model.constants import (
    IH_DIR_REVERSE_CELLS,
    IH_OFF_DEFAULT,
    IH_OFF_GMAX_SEGMENT,
    IH_OFF_MODES,
    IH_OFF_SCALAR_SEGMENTS,
    KNOWN_MODELS,
    STATE_CLAMP,
    Ca_tau,
    E_IH_OFF,
    E_Ih,
    E_LEAK_DEPOL,
    E_LEAK_REST,
    E_exc,
    E_inh,
    Ih_gain,
    capac,
    cdt,
    deltat,
    exc_synweight,
    g_leak,
    inh_synweight,
)
from neuron_model.schema import (
    ALL_PARAM_NAMES,
    IH_SHAPE_PARAM_NAMES,
    SYN_MODE_DEFAULT,
    SYN_MODES,
    apply_ih_off_mode,
    build_conductance_schema,
    build_hp_lp_schema,
    conductance_ih_off_kwargs,
    conductance_schema,
    default_schema,
    normalize_syn_mode,
    synaptic_scale,
)
from neuron_model import conductance as _conductance
from neuron_model import hp_lp as _hp_lp
from neuron_model.ca_filter import CA_ALPHA, ca_readout_step, ca_to_v_delta
from neuron_model.forward import (
    MODEL_DRIVERS,
    run_full,
    run_units,
)
from neuron_model.readout import (
    CA_PACK_READOUTS,
    pack_readout,
    readout_pack_traces,
    window_time_traces,
)

# --- conductance ---
rectsyn = _conductance.rectsyn
update_v = _conductance.update_v
v_budget_from_g = _conductance.v_budget_from_g

# --- hp_lp ---
update_state_hp_lp = _hp_lp.update_state_hp_lp


__all__ = [
    "KNOWN_MODELS",
    "ALL_PARAM_NAMES",
    "IH_SHAPE_PARAM_NAMES",
    "SYN_MODES",
    "SYN_MODE_DEFAULT",
    "MODEL_DRIVERS",
    "CA_PACK_READOUTS",
    "default_schema",
    "build_conductance_schema",
    "build_hp_lp_schema",
    "conductance_schema",
    "apply_ih_off_mode",
    "conductance_ih_off_kwargs",
    "normalize_syn_mode",
    "synaptic_scale",
    "rectsyn",
    "update_v",
    "v_budget_from_g",
    "update_state_hp_lp",
    "ca_readout_step",
    "ca_to_v_delta",
    "CA_ALPHA",
    "run_full",
    "run_units",
    "pack_readout",
    "readout_pack_traces",
    "window_time_traces",
    "STATE_CLAMP",
    "deltat",
    "Ca_tau",
    "g_leak",
    "cdt",
    "E_exc",
    "E_inh",
    "E_Ih",
    "E_IH_OFF",
    "E_LEAK_REST",
    "E_LEAK_DEPOL",
    "Ih_gain",
    "capac",
    "exc_synweight",
    "inh_synweight",
    "IH_OFF_MODES",
    "IH_OFF_DEFAULT",
    "IH_OFF_SCALAR_SEGMENTS",
    "IH_OFF_GMAX_SEGMENT",
    "IH_DIR_REVERSE_CELLS",
]
