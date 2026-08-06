# -*- coding: utf-8 -*-
"""Training package: session assembly, cost, parameter schema, and implement.

Public facade re-exporting training API names (``import training``). Engine internals are split across :mod:`training.readout_pack`,
:mod:`training.params`, :mod:`training.cost`, and :mod:`training.session`; the
shared vocabulary lives in :mod:`training.config`. Numeric defaults live in
:mod:`param_defaults`. Lower layers (``neuron``, ``task``, ``network``)
never import this package at load time.
"""
from __future__ import annotations

from neuron import (
    ALL_PARAM_NAMES,
    EULER_CLI,
    EULER_MODES,
    IH_OFF_MODES,
    IH_SHAPE_PARAM_NAMES,
    KNOWN_MODELS,
    SYN_MODES,
    default_schema,
    e_ih_off,
    expand_euler,
    membrane_dt_over_c,
    ms_to_t,
    normalize_syn_mode,
    forward_full,
    forward_nodes,
    pack_t_onset,
    update_state_hp_lp,
    update_v,
    v_component_from_g,
)
from param_defaults import (
    CA_TAU,
    CAPAC,
    DATA_AMP,
    DELTA_MS,
    E_EXC,
    E_IH,
    E_INH,
    E_LEAK_DEPOL,
    E_LEAK_REST,
    SYN_SCALE_EXC,
    G_LEAK,
    G_IN,
    IH_GAIN,
    IH_GMAX_INDI_NAMES,
    IH_OFF,
    SYN_SCALE_INH,
    PARAM_BOXES,
    MS_PRE,
    MS_POST,
    MS_PULSE,
    MS_RESPONSE,
    STATE_CLAMP,
    SYN_MODE,
    T_REL_START,
    T_REL_STOP,
)

from training.config import (
    CLI_TASK_NAMES,
    COST_NORMS,
    MOVING_BAR_TASKS,
    PD_ND_LABELS,
    SPOT_TASKS,
    PRE_STEADY_MODES,
    TASK_ALIASES,
    TRAIN_OPTS_FILE,
    VALID_TASKS,
    cost_part_keys_for_readout,
    expand_cost_norm,
    expand_cost_weight_dict,
    expand_gt_dict,
    expand_pre_steady_mode,
    expand_tasks,
    moving_bar_cost_part_key,
    normalize_tasks,
    resolve_cost_extent_by_task,
    session_cost_part_keys,
)
from training.readout_pack import (
    FusedForward,
    ModelBackend,
    ReadoutPack,
    TrainingResult,
    TrainSession,
    active_device,
)
from training.params import (
    PAIR_SEP,
    TRAIN_MODES,
    apply_train_modes,
    assign_params,
    attach_param_carry,
    build_e_leak,
    build_ih_dir,
    calc_multi_col_params,
    edge_node_names,
    guess_initial_params,
    named_moments_to_z,
    pair_node_names,
    params_from_z,
    parse_train_mode_text,
    remap_named_moments,
    remap_named_node_values,
    schema_bounds,
    schema_guess,
    schema_nparams,
    schema_train_modes_record,
    seg_ntrain,
    cell_node_names,
    node_names_for_segment,
    node_values_to_z,
    validate_syn_strength_edge_train_mode,
    z_moments_to_named,
    z_to_node_values,
)
from training.cost import (
    adam_moments_from_state_dict,
    calc_cost,
    calc_cost_parts,
    do_many_runs,
    gradient_network,
    gt_affine_for_nodes,
    train_staged,
)
from training.session import (
    NETWORK_TASK_BUILDERS,
    _cost_extent_hex_coltag,
    apply_pack_override,
    build_i_cli_by_task,
    extend_readout_pack_mirror_fit,
    load_network_backend,
    make_train_opts,
    open_session,
    open_session_from_opts,
    open_session_from_outdir,
    resolve_gt_cells_by_task,
    resolve_cell_indices,
)

from task.spot.gt import make_spot_stimulus_opts
from task.moving_bar.gt import (
    make_moving_bar_stimulus_opts,
    session_moving_bar_i_baseline,
)
