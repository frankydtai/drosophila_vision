# -*- coding: utf-8 -*-
"""Training package: session assembly, cost, parameter schema, and CLI driver.

Public facade re-exporting the names callers use as ``fc.*`` (``import training
as fc``). Engine internals are split across :mod:`training.target_pack`,
:mod:`training.params`, :mod:`training.cost`, and :mod:`training.session`; the
shared vocabulary lives in :mod:`training.config`. Lower layers
(``neuron_model``, ``stimulus``, ``network``) never import this package at load
time.
"""
from __future__ import annotations

from neuron_model import (
    ALL_PARAM_NAMES,
    IH_OFF_DEFAULT,
    IH_OFF_MODES,
    IH_SHAPE_PARAM_NAMES,
    KNOWN_MODELS,
    SYN_MODE_DEFAULT,
    SYN_MODES,
    Ca_tau,
    E_IH_OFF,
    E_Ih,
    E_LEAK_DEPOL,
    E_LEAK_REST,
    E_exc,
    E_inh,
    Ih_gain,
    apply_ih_off_mode,
    ca_to_v_delta,
    capac,
    cdt,
    conductance_schema,
    default_schema,
    deltat,
    g_leak,
    normalize_syn_mode,
    run_full,
    run_units,
    update_v,
    v_budget_from_g,
)

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
    FusedConductanceForward,
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
    _pack_signal_scale,
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

from stimulus.spot.data import make_spot_stimulus_opts
from stimulus.moving_bar.data import (
    make_moving_bar_stimulus_opts,
    session_moving_bar_i_baseline,
)
