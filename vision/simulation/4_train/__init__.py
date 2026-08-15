# -*- coding: utf-8 -*-
"""Train package: session assembly, cost, optimization, parameter schema, implementation, and cli.

Public facade re-exporting train API names (``import train``). Engine internals are split across
:mod:`train.param`, :mod:`train.session`, :mod:`train.cost`, :mod:`train.optimization`,
:mod:`train.implementation`, :mod:`train.cli`; the shared vocabulary lives in
:mod:`train.config`. Numeric defaults live in :mod:`const_default`. Lower layers
(``neuron``, ``task``, ``network``) never import this package at load time.
"""
from __future__ import annotations

from const_default import (
    ANALYZE_CELL_DYNAMICS,
    MODEL,
    MOVING_BAR_INPUT,
    NEURON_FORWARD,
    NEURON_CONST,
    NEURON_SCHEMA,
    NETWORK_CONSTRUCTION,
    NETWORK_PATH,
    SPOT_INPUT,
    SPOT_PACK,
    STI_TIMING,
    TRAIN_CONFIG,
    TRAIN_OPTIMIZATION,
    TRAIN_OPTS,
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
    forward_v,
    pack_t_onset,
    step_delta_ms,
    ms_from_t,
    update_v,
    v_ca_from_v,
    v_component_from_g,
)

from train.config import (
    COST_NORMS,
    CONTRASTS,
    SPOT_GT_MODES,
    PD_ND_LABELS,
    TASKS,
    cost_part_keys_from_task,
    cost_radius_by_task,
    expand_cost_norm,
    expand_part_cost_scale,
    expand_filter,
    expand_spot_gt_mode,
    expand_gt,
    expand_pre_steady,
    moving_bar_cost_part_key,
    parse_contrasts,
    parse_tasks,
    session_cost_part_keys,
    spot_cost_part_key,
)
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
    parse_param_cli,
    parse_param_init_val_tokens,
    override_params,
    remap_node_vals_adams,
    remap_node_vals,
    override_val_from,
    bias_gt_from_onset_trace,
    parse_val_from_tokens,
    resolve_val_from,
    val_from_enabled,
    resolve_param_modes,
    schema_clamps,
    schema_n_z,
    z_init_from_schema,
    param_modes_from_schema,
    param_n_z,
    cells_from_connectome,
    z_from_node_vals,
    validate_syn_strength_edge_param_mode,
    adams_from_z,
    node_vals_from_z,
    sim_dtype_from_fp,
)
from train.session import (
    Pack,
    TrainSession,
    _cost_hex_label,
    load_train_connectome,
    resolve_train_opts,
    open_session,
    resolve_session,
    session_from_outdir,
    resolve_gt_cells_by_task,
    resolve_cell_idxs,
)
from train.cost import (
    FusedPacks,
    calc_cost,
    calc_cost_parts,
    gt_affine_from_nodes,
    pack_cost_abs_ts,
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
