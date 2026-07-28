# -*- coding: utf-8 -*-
"""Neuron models: conductance, hp_lp.

Dynamics / schemas / pack readouts live here. ``FiveCol_MedSim_Pytorch`` owns
session, cost, and training drivers and re-exports these symbols for callers.
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
    apply_ih_off_mode,
    build_conductance_schema,
    build_hp_lp_schema,
    conductance_ih_off_kwargs,
    conductance_schema,
    default_schema,
)
from neuron_model import conductance as _conductance
from neuron_model import hp_lp as _hp_lp

# --- conductance ---
rectsyn = _conductance.rectsyn
update_Vm = _conductance.update_Vm
vm_budget_from_g = _conductance.vm_budget_from_g
run_conductance = _conductance.run_conductance
run_conductance_full = _conductance.run_conductance_full

# --- hp_lp ---
update_state_hp_lp = _hp_lp.update_state_hp_lp
run_hp_lp = _hp_lp.run_hp_lp

# Register pack readouts here only — batching stays in FiveCol ``_pack_cost``.
MODEL_PACK_READOUTS = {
    "conductance": _conductance.pack_readout,
    "hp_lp": _hp_lp.pack_readout,
}


def params_from_z(z, session):
    """Unpack z → param dict for ``session.model``."""
    import FiveCol_MedSim_Pytorch as fc

    return fc.assign_params(z, list(session.schema), session.backend)


__all__ = [
    "KNOWN_MODELS",
    "ALL_PARAM_NAMES",
    "IH_SHAPE_PARAM_NAMES",
    "MODEL_PACK_READOUTS",
    "default_schema",
    "build_conductance_schema",
    "build_hp_lp_schema",
    "conductance_schema",
    "apply_ih_off_mode",
    "conductance_ih_off_kwargs",
    "rectsyn",
    "update_Vm",
    "vm_budget_from_g",
    "run_conductance",
    "run_conductance_full",
    "update_state_hp_lp",
    "run_hp_lp",
    "params_from_z",
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
