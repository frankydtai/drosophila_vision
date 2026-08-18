# -*- coding: utf-8 -*-
"""Train package: session assembly, cost, optimization, parameter schema, implementation.

Public facade re-exporting train API names (``import train``). Engine internals are split across
:mod:`train.param`, :mod:`train.session`, :mod:`train.cost`, :mod:`train.optimization`,
:mod:`train.implementation`. Contrast tokens live in
:mod:`task.spread.sti_spec`. Numeric defaults live in :mod:`config`. Lower layers
(``neuron``, ``task``, ``network``) never import this package at load time.
"""
from __future__ import annotations

from config import (
    ANALYZE_CELL_DYNAMICS,
    MODEL,
    MOVING_BAR_INPUT_GEO,
    MOVING_BAR_INPUT_SPEC,
    NEURON_FORWARD,
    NEURON_SCHEMA,
    NETWORK_PATH,
    SPOT_INPUT_GEO,
    SPREAD_INPUT_SPEC,
    SPREAD_GT,
    TRAIN_CONFIG,
    TRAIN_OPTIMIZATION,
    TRAIN_SESSION,
)

from neuron import (
    I_H_SHAPE_PARAMS,
    EULER_CLI,
    EULER_MODES,
    MODELS,
    SYN_MODES,
    build_schema,
    e_h_rev,
    expand_euler,
    t_from_ms,
    t_abs_from_ms,
    ca_from_v_ca,
    forward_ca,
    forward_nodes,
    forward_trace,
    forward_v,
    pack_t_onset,
    step_delta_ms,
    ms_from_t,
    update_v,
    v_ca_from_v,
    v_component_from_g,
)

from task.spread.sti_spec import CONTRASTS
from task.moving_bar.sti_spec import PD_ND_LABELS
from train.param import (
    PAIR_SEP,
    PARAM_MODES,
    SIM_DTYPE,
    active_device,
    assign_params,
    schema_with_param_carry,
    inits_from_node_vals,
    schema_copy,
    edges_from_connectome,
    z_adams_from_node_vals,
    pairs_from_connectome,
    params_from_z,
    override_params,
    remap_node_vals_adams,
    remap_node_vals,
    override_val_from,
    bias_gt_from_onset_trace,
    parse_val_from_tokens,
    resolve_val_from,
    val_from_enabled,
    schema_clamps,
    schema_n_z,
    z_init_from_schema,
    param_n_z,
    cells_from_connectome,
    z_from_node_vals,
    adams_from_z,
    node_vals_from_z,
    sim_dtype_from_fp,
)
from train.session import (
    Pack,
    TrainSession,
    TASKS,
    _cost_hex_label,
    load_train_connectome,
    resolve_train_opts,
    open_session,
    resolve_session,
    session_from_run_dir,
    resolve_gt_cells_by_task,
    resolve_cell_idxs,
    run_data_dir,
)
from train.cost import (
    COST_NORMS,
    FusedPacks,
    calc_cost,
    calc_cost_parts,
    forward_pack,
    gt_affine_from_cell,
    gt_affine_from_nodes,
    moving_bar_cell_cost_part_key,
    moving_bar_cost_part_key,
    pack_cost_abs_ts,
    session_cost_part_keys,
    spread_cost_part_key,
    spot_cost_part_key,
)
from train.optimization import (
    TrainResult,
    adams_from_optimizer,
    do_many_runs,
    gradient_network,
    optimize_staged,
)

from task.spot.pack import build_spot_sti_opts
from task.moving_bar.pack import build_moving_bar_sti_opts
from task.moving_bar.sti_spec import i_baseline_from_i_sti
