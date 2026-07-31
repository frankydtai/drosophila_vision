# -*- coding: utf-8 -*-
"""Training package: session assembly, cost, parameter schema, and CLI driver.

Public facade re-exporting the names callers use as ``fc.*`` (``import training
as fc``). Engine internals are split across :mod:`training.target_pack`,
:mod:`training.params`, :mod:`training.cost`, and :mod:`training.session`; the
shared vocabulary lives in :mod:`training.config`. Numeric defaults live in
:mod:`training.defaults`. Lower layers (``neuron``, ``task``, ``network``)
never import this package at load time.
"""
from __future__ import annotations

from neuron import (
    ALL_PARAM_NAMES,
    IH_OFF_MODES,
    IH_SHAPE_PARAM_NAMES,
    KNOWN_MODELS,
    Physics,
    SYN_MODES,
    default_schema,
    ms_to_t,
    normalize_syn_mode,
    run_full,
    run_units,
    update_v,
    v_budget_from_g,
)
from training.defaults import (
    CA_TAU,
    CAPAC,
    DATA_AMP,
    IH_GMAX_INDI_NAMES,
    DELTA_MS,
    E_EXC,
    E_IH,
    E_INH,
    E_LEAK_DEPOL,
    E_LEAK_REST,
    EXC_SYNWEIGHT,
    G_LEAK,
    IH_GAIN,
    IH_OFF,
    INH_SYNWEIGHT,
    PARAM_BOXES,
    PRE_MS,
    RESPONSE_MS,
    STATE_CLAMP,
    SYN_MODE,
    PHYSICS,
)

# Derived physics (formulas live on :class:`~neuron.params.Physics`).
E_IH_OFF = PHYSICS.E_IH_OFF
cdt = PHYSICS.cdt

# Historical ``fc.*`` attribute aliases.
E_Ih = E_IH
Ca_tau = CA_TAU
capac = CAPAC
delta_ms = DELTA_MS
exc_synweight = EXC_SYNWEIGHT
g_leak = G_LEAK
inh_synweight = INH_SYNWEIGHT
Ih_gain = IH_GAIN
E_exc = E_EXC
E_inh = E_INH

from training.config import (
    CLI_TARGET_NAMES,
    MOVING_BAR_TARGETS,
    PD_ND_LABELS,
    SPOT_TARGETS,
    TARGET_ALIASES,
    TRAIN_OPTS_FILE,
    VALID_TARGETS,
    cost_part_keys_for_target,
    expand_cost_weight_dict,
    expand_target_list,
    moving_bar_cost_part_key,
    normalize_target_list,
    resolve_cost_extent_by_target,
    session_cost_part_keys,
)
from training.target_pack import (
    FusedForward,
    ModelBackend,
    TargetPack,
    TrainingResult,
    TrainSession,
    active_device,
)
from training.params import (
    PAIR_SEP,
    PARTITION_BUCKETS,
    apply_partitions,
    assign_params,
    attach_param_carry,
    build_e_leak,
    build_ih_dir,
    calc_multi_col_params,
    edge_unit_names,
    guess_initial_params,
    pair_unit_names,
    params_from_z,
    parse_partition_text,
    remap_named_unit_values,
    schema_bounds,
    schema_guess,
    schema_nparams,
    schema_partitions_record,
    seg_ntrain,
    type_unit_names,
    unit_names_for_segment,
    unit_values_to_z,
    validate_edge_weight_partition,
    z_to_unit_values,
)
from training.cost import (
    ca_cost,
    calc_cost,
    calc_cost_parts,
    do_many_runs,
    gradient_network,
    out_scale_for_units,
    train_staged,
)
from training.session import (
    NETWORK_TARGET_BUILDERS,
    _cost_extent_column_coltag,
    apply_pack_override,
    build_i_cli_by_target,
    extend_target_pack_mirror_fit,
    load_network_backend,
    make_train_opts,
    open_session,
    open_session_from_opts,
    open_session_from_outdir,
    resolve_type_indices,
)

from task.spot.data import make_spot_stimulus_opts
from task.moving_bar.data import (
    make_moving_bar_stimulus_opts,
    session_moving_bar_i_baseline,
)
