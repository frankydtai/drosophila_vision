# -*- coding: utf-8 -*-
"""Train package: session assembly, cost, optimization, parameter schema, implementation, and cli.

Public facade re-exporting train API names (``import train``). Engine internals are split across
:mod:`train.param`, :mod:`train.session`, :mod:`train.cost`, :mod:`train.optimization`,
:mod:`train.implementation`, :mod:`train.cli`; the shared vocabulary lives in
:mod:`train.config`. Numeric defaults live in :mod:`default_params`. Lower layers
(``neuron``, ``task``, ``network``) never import this package at load time.
"""
from __future__ import annotations

from default_params import (
    ANALYZE_CELL_DYNAMICS,
    MODEL,
    MOVING_BAR_INPUT,
    NEURON_FORWARD,
    NEURON_PARAM,
    NEURON_SCHEMA,
    NETWORK_CONSTRUCTION,
    NETWORK_PATH,
    SPOT_INPUT,
    SPOT_PACK,
    TRAIN_CONFIG,
    TRAIN_OPTIMIZATION,
    TRAIN_OPTS,
    TRAIN_SESSION,
)

from neuron import (
    PARAM_NAMES,
    EULER_CLI,
    EULER_MODES,
    I_H_REV_MODES,
    I_H_SHAPE_PARAM_NAMES,
    KNOWN_MODELS,
    SYN_MODES,
    default_schema,
    e_h_rev,
    expand_euler,
    membrane_dt_over_c,
    t_from_ms,
    t_abs_from_ms,
    normalize_syn_mode,
    ca_from_v_ca,
    forward_ca,
    forward_full,
    forward_nodes,
    forward_v,
    pack_t_onset,
    step_delta_ms,
    ms_from_t,
    update_state_hp_lp,
    update_v,
    v_ca_from_v,
    v_component_from_g,
)

from train.config import (
    CLI_TASK_NAMES,
    COST_NORMS,
    SPOT_GT_MODES,
    MOVING_BAR_TASKS,
    PD_ND_LABELS,
    SPOT_TASKS,
    TASK_ALIASES,
    TRAIN_OPTS_FILE,
    VALID_TASKS,
    cost_part_keys_for_task,
    expand_cost_norm,
    expand_part_cost_scale_dict,
    expand_filter,
    expand_spot_gt_mode,
    expand_gt_dict,
    expand_pre_steady,
    expand_tasks,
    moving_bar_cost_part_key,
    normalize_tasks,
    resolve_cost_radius_by_task,
    session_cost_part_keys,
)
from train.param import (
    PAIR_SEP,
    TRAIN_MODES,
    ModelBackend,
    SIM_DTYPE,
    active_device,
    apply_train_modes,
    assign_params,
    attach_param_carry,
    seed_fixed_from_named,
    build_i_h_dirs,
    calc_multi_col_params,
    edge_node_names,
    guess_initial_params,
    z_moments_from_named,
    pair_node_names,
    params_from_z,
    parse_train_mode_text,
    remap_named_moments,
    remap_named_node_values,
    materialize_from_opts,
    bias_gt_from_onset_trace,
    schema_bounds,
    schema_guess,
    schema_nparams,
    schema_train_modes_record,
    seg_n_z,
    cell_node_names,
    node_names_for_segment,
    z_from_node_values,
    validate_syn_strength_edge_train_mode,
    named_moments_from_z,
    node_values_from_z,
    sim_dtype_from_fp,
)
from train.session import (
    NETWORK_TASK_BUILDERS,
    Pack,
    TrainSession,
    _cost_radius_hex_coltag,
    apply_pack_override,
    build_i_cli_by_task,
    extend_pack_mirror_fit,
    load_network_backend,
    build_train_opts,
    open_session,
    open_session_from_opts,
    open_session_from_outdir,
    resolve_gt_cells_by_task,
    resolve_cell_indices,
)
from train.cost import (
    FusedForward,
    calc_cost,
    calc_cost_parts,
    gt_affine_for_nodes,
    pack_cost_abs_time_idx,
)
from train.optimization import (
    TrainResult,
    adam_moments_from_state_dict,
    do_many_runs,
    gradient_network,
    optimize_staged,
)

from task.spot.pack import build_spot_sti_opts
from task.moving_bar.gt import (
    build_moving_bar_sti_opts,
    session_moving_bar_i_baseline,
)
