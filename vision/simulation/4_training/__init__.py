# -*- coding: utf-8 -*-
"""Training package: session assembly, cost, parameter schema, and CLI driver.

Public facade re-exporting training API names (``import training``). Engine internals are split across :mod:`training.readout_pack`,
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
    SYN_MODES,
    default_schema,
    e_ih_off,
    membrane_cdt,
    ms_to_t,
    normalize_syn_mode,
    run_full,
    run_nodes,
    update_state_hp_lp,
    update_v,
    v_component_from_g,
)
from training.defaults import (
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
    PRE_MS,
    PULSE_MS,
    RESPONSE_MS,
    STATE_CLAMP,
    SYN_MODE,
    T_REL_START,
    T_REL_STOP,
)

from training.config import (
    CLI_TASK_NAMES,
    MOVING_BAR_TASKS,
    PD_ND_LABELS,
    SPOT_TASKS,
    TASK_ALIASES,
    TRAIN_OPTS_FILE,
    VALID_TASKS,
    cost_part_keys_for_readout,
    expand_cost_weight_dict,
    expand_gt_dict,
    expand_task_list,
    moving_bar_cost_part_key,
    normalize_task_list,
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
    PARTITION_BUCKETS,
    apply_partitions,
    assign_params,
    attach_param_carry,
    build_e_leak,
    build_ih_dir,
    calc_multi_col_params,
    edge_node_names,
    guess_initial_params,
    pair_node_names,
    params_from_z,
    parse_partition_text,
    remap_named_node_values,
    schema_bounds,
    schema_guess,
    schema_nparams,
    schema_partitions_record,
    seg_ntrain,
    cell_node_names,
    node_names_for_segment,
    node_values_to_z,
    validate_syn_strength_edge_partition,
    z_to_node_values,
)
from training.cost import (
    ca_cost,
    calc_cost,
    calc_cost_parts,
    do_many_runs,
    gradient_network,
    out_scale_for_nodes,
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

from task.spot.data import make_spot_stimulus_opts
from task.moving_bar.data import (
    make_moving_bar_stimulus_opts,
    session_moving_bar_i_baseline,
)
