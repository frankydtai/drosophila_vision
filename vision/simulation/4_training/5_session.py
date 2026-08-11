# -*- coding: utf-8 -*-
"""Session assembly: build a :class:`TrainSession` from canonical training opts.

Owns stimulus-opts finalisation (CLI overrides -> per-task sidecar dicts),
network backend construction, and the per-task ``ReadoutPack`` builders. The
builders wrap the neutral gt dataclasses from ``task`` (which sit below
``training`` in the import graph) and stamp the cross-cutting readout controls:

* spot: sparse ``cost_time_idx`` / optional ``cost_time_mask`` (#4; ``cost_ms``
  overwrites interval per radius), ``ms_spot`` (#1) already baked into
  the stimulus, ``waveform_mse=True``;
* moving bar: ``waveform_mse`` from cost weights (True when a cost window is
  built).

Model traces are absolute ``v`` (``filter=none``) or ``ca`` (``filter=ca``);
cost compares the readout to ``a_gt * gt + bias_gt``. When
``bias_gt_from_v_onset``, ``bias_gt`` is written from ``v`` at ``t_onset`` (or ``ca`` when
``filter=ca``) — same value appears in ``param.csv``. Spot ImpR uses Arenz digitized
when ``filter=ca``; RecF ``a_radius`` scaling is unchanged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neuron.params import t_from_ms
from neuron import (
    default_schema,
    expand_euler,
    normalize_syn_mode,
)
from param_defaults import (
    CAP,
    GT_CA_AMP,
    GT_V_AMP,
    DELTA_MS,
    DELTA_MS_PRE,
    E_EXC,
    E_H,
    E_INH,
    EULER,
    SYN_SCALE_EXC,
    FP,
    FULLY_INSIDE,
    G_LEAK,
    H_G_MAX,
    SYN_SCALE_INH,
    MULTI_BAR,
    MULTI_SPOT,
    PRE_GRAD,
    BIAS_GT_FROM_V_ONSET,
    BIAS_GT_FROM_V_ONSET_GRAD,
    V_TH_CA_FROM_V_TH,
    A_CA_FROM_A_OUT,
    FILTER,
    SHIFT_RADIUS,
    SPOT_COST_RADII,
    SPOT_COST_RADIUS_WEIGHT,
    SPOT_COST_RADIUS_WEIGHT_RADIUS1,
    SPOT_COST_RADIUS_KEY_ALIASES,
    SPOT_STI_RADII,
    SPOT_RADIUS,
    STATE_CLAMP,
    H_CELLS,
    I_H_REV,
    I_BASELINE,
    I_BRIGHT,
    I_DARK,
    PARAM_BOXES,
    MS_PRE,
    MS_POST,
    MS_SPOT,
    MS_SPOT_CA,
    MS_RESPONSE,
    SYN_MODE,
    COST_NORM,
    PRE_STEADY,
    PRE_STEADY_DAMP,
    PRE_STEADY_ITERS,
)

from training.config import (
    CLI_TASK_NAMES,
    I_CLI_BRIGHT_TASKS,
    I_CLI_DARK_TASKS,
    I_CLI_SIDECAR_FIELD,
    PD_ND_LABELS,
    SPOT_TASKS,
    TASK_ALIASES,
    TASK_I_FIELDS,
    TRAIN_OPTS_FILE,
    VALID_TASKS,
    _MOVING_BAR_BASELINE_KEY,
    _MOVING_BAR_I_KEY,
    _SPOT_BASELINE_KEY,
    _SPOT_I_KEY,
    expand_cost_norm,
    expand_cost_weight_dict,
    expand_filter,
    expand_gt_dict,
    expand_pre_steady,
    moving_bar_cost_part_key,
    normalize_tasks,
    run_data_dir,
)
from training.readout_pack import (
    ModelBackend,
    ReadoutPack,
    TrainSession,
    active_device,
    SIM_DTYPE,
    sim_dtype_from_fp,
)
from training.params import (
    apply_train_modes,
    attach_param_carry,
    build_i_h_dir,
    schema_nparams,
    schema_train_modes_record,
    node_names_for_segment,
)
from training.cost import _build_cost_subpacks, _build_fused_forward

from task.spot.gt import expand_gt_cells as expand_spot_gt_cells
from task.spot.readout import (
    SPOT_POLARITIES,
    build_spot_gt,
    default_spot_cost_radius_weight,
    expand_cost_ms_dict,
    expand_spot_cost_r_w_dict,
    build_spot_stimulus_opts,
    spot_sti_radius_gate,
)
from task.spot.input import (
    build_spot_a_sti_radius_drive,
    normalize_spot_timing,
    spot_radius_half_steps,
    spot_from_opts,
    spot_gt_n_t_from_opts,
    spot_stimulus_batches,
    spot_timing_t_from_opts,
)
from task.moving_bar.gt import (
    _enrich_moving_bar_stimulus_opts,
    build_moving_bar_gt,
    expand_gt_cells as expand_moving_bar_gt_cells,
    build_moving_bar_stimulus_opts,
)
from network.construction import (
    load_network, gt_cells_from_opts, node_cell_names, normalize_cost_radius,
)


def resolve_cell_indices(cells, backend: ModelBackend):
    """Map cell names to indices in the network vocabulary."""
    if backend.network is None:
        raise ValueError("resolve_cell_indices requires backend.network")
    names = [str(n) for n in cells]
    tn = list(backend.network.cells)
    return [tn.index(n) for n in names if n in tn]


def extend_readout_pack_mirror_fit(pack, mirror_types, mirror_fit, mirror_sign=-1.0, backend=None):
    """Extend a :class:`ReadoutPack`: mirror *mirror_fit* gts onto *mirror_types*."""
    if backend is None or backend.network is None:
        raise ValueError("extend_readout_pack_mirror_fit requires backend.network")
    connectome = backend.network
    names = node_cell_names(connectome)
    entry_node_idx = pack.readout_node.cpu().numpy()
    b_arr = pack.readout_batch.cpu().numpy()
    w_arr = pack.cost_weight.cpu().numpy()
    r_arr = (
        pack.spot_cost_radius.cpu().numpy()
        if pack.spot_cost_radius is not None else None
    )
    network_node_u = connectome.u.detach().cpu().numpy() if hasattr(connectome.u, "detach") else np.asarray(connectome.u)
    network_node_v = connectome.v.detach().cpu().numpy() if hasattr(connectome.v, "detach") else np.asarray(connectome.v)
    extra_batch, extra_node_idx, extra_entries, extra_cost_weight, extra_radius, extra_pd_nd = [], [], [], [], [], []
    for entry_i in range(len(entry_node_idx)):
        node_idx = int(entry_node_idx[entry_i])
        if str(names[node_idx]) != mirror_fit:
            continue
        batch = int(b_arr[entry_i])
        mirror_fit_node_u, mirror_fit_node_v = int(network_node_u[node_idx]), int(network_node_v[node_idx])
        mirror_readout = float(mirror_sign) * pack.gt[entry_i:entry_i + 1]
        cost_weight = float(w_arr[entry_i])
        radius = float(r_arr[entry_i]) if r_arr is not None else None
        for mtype in mirror_types:
            candidates = np.where(
                (network_node_u == mirror_fit_node_u)
                & (network_node_v == mirror_fit_node_v)
                & (names == str(mtype))
            )[0]
            for candidate_node_idx in candidates:
                extra_batch.append(batch)
                extra_node_idx.append(int(candidate_node_idx))
                extra_entries.append(mirror_readout)
                extra_cost_weight.append(cost_weight)
                if radius is not None:
                    extra_radius.append(radius)
                if pack.cost_pd_nd is not None:
                    extra_pd_nd.append(int(pack.cost_pd_nd[entry_i].item()))
    return _append_mirror_pack_entries(
        pack, extra_node_idx, extra_entries,
        readout_batch=extra_batch, cost_weight=extra_cost_weight,
        spot_cost_radius=extra_radius if extra_radius else None,
        cost_pd_nd=extra_pd_nd if extra_pd_nd else None,
    )


def _append_mirror_pack_entries(
    pack, extra_nodes, extra_entries, readout_batch=None, cost_weight=None,
    spot_cost_radius=None, cost_pd_nd=None,
):
    extra_nodes_t = torch.tensor(extra_nodes, dtype=torch.long, device=active_device())
    extra_gt_t = torch.cat(extra_entries, dim=0)
    n_all = int(pack.readout_node.shape[0]) + len(extra_nodes)
    n_extra = len(extra_nodes)
    if readout_batch is None:
        readout_batch = torch.zeros(n_extra, dtype=torch.long, device=active_device())
    else:
        readout_batch = torch.tensor(readout_batch, dtype=torch.long, device=active_device())
    if cost_weight is None:
        w_dtype = pack.cost_weight.dtype
        cost_weight = torch.ones(n_all, dtype=w_dtype, device=active_device())
    else:
        base_w = pack.cost_weight
        w_dtype = base_w.dtype
        cost_weight = torch.cat([
            base_w,
            torch.tensor(cost_weight, dtype=w_dtype, device=active_device()),
        ])
    spot_cost_radius_out = pack.spot_cost_radius
    if spot_cost_radius is not None:
        base_r = pack.spot_cost_radius
        r_dtype = base_r.dtype if base_r is not None else SIM_DTYPE
        extra_r_t = torch.tensor(spot_cost_radius, dtype=r_dtype, device=active_device())
        spot_cost_radius_out = (
            torch.cat([base_r, extra_r_t])
            if base_r is not None else extra_r_t
        )
    all_gt = torch.cat([pack.gt, extra_gt_t], dim=0)
    cost_pd_nd_out = pack.cost_pd_nd
    if cost_pd_nd is not None:
        extra_pd_t = torch.tensor(cost_pd_nd, dtype=torch.long, device=active_device())
        cost_pd_nd_out = (
            torch.cat([pack.cost_pd_nd, extra_pd_t])
            if pack.cost_pd_nd is not None else extra_pd_t
        )
    return ReadoutPack(
        name=pack.name,
        i_sti=pack.i_sti,
        gt=all_gt,
        power=pack.power + torch.sum(extra_gt_t ** 2),
        cost_weight=cost_weight,
        readout_batch=torch.cat([pack.readout_batch, readout_batch]),
        readout_node=torch.cat([pack.readout_node, extra_nodes_t]),
        cost_t0=pack.cost_t0,
        cost_radius=pack.cost_radius,
        spot_cost_radius=spot_cost_radius_out,
        cost_pd_nd=cost_pd_nd_out,
        dsi_pos_entries=pack.dsi_pos_entries,
        dsi_neg_entries=pack.dsi_neg_entries,
        dsi_pos_ptr=pack.dsi_pos_ptr,
        dsi_neg_ptr=pack.dsi_neg_ptr,
        dsi_gt=pack.dsi_gt,
        dsi_weight=pack.dsi_weight,
        dsi_power=pack.dsi_power,
        cost_time_idx=pack.cost_time_idx,
        cost_time_mask=pack.cost_time_mask,
        waveform_mse=pack.waveform_mse,
        t_onset=pack.t_onset,
    )


def apply_pack_override(pack, override, backend: ModelBackend):
    """Apply one serializable pack override dict (saved in ``train_opts.json``)."""
    specs = override.get("mirror_fits")
    if specs is None and "mirror_fit" in override:
        specs = [override["mirror_fit"]]
    if not specs:
        raise ValueError(f"unknown pack override {override!r}")
    for spec in specs:
        if "mirror_types" not in spec:
            raise ValueError(f"mirror_fit spec needs mirror_types: {spec!r}")
        pack = extend_readout_pack_mirror_fit(
            pack,
            mirror_types=[str(t) for t in spec["mirror_types"]],
            mirror_fit=spec["mirror_fit"],
            mirror_sign=float(spec.get("mirror_sign", -1.0)),
            backend=backend,
        )
    return pack


def _network_backend_from_connectome(
    connectome, *, sim_dtype=SIM_DTYPE,
) -> ModelBackend:
    """Build a :class:`ModelBackend` from an already-loaded connectome graph."""
    conn = connectome.conn
    return ModelBackend(
        conn=conn,
        i_h_dir=build_i_h_dir(conn, dtype=sim_dtype),
        n_cells=connectome.n_cells,
        n_hexes=1,
        network=connectome,
    )


def load_network_backend(
    network_json,
    dev: Optional[str] = None,
    *,
    syn_scale_exc: float,
    syn_scale_inh: float,
    sim_dtype=SIM_DTYPE,
    syn_mode=SYN_MODE,
    param_boxes=PARAM_BOXES,
    h_cells=H_CELLS,
) -> ModelBackend:
    """Load connectome network into a :class:`ModelBackend`."""
    dev = dev or active_device()
    mode = normalize_syn_mode(syn_mode)
    connectome = load_network(
        network_json, device=dev,
        syn_scale_exc=syn_scale_exc, syn_scale_inh=syn_scale_inh,
        dtype=sim_dtype, syn_mode=mode,
    )
    backend = _network_backend_from_connectome(connectome, sim_dtype=sim_dtype)
    print(f"network: {network_json}")
    print(f"  n_nodes={backend.n_nodes}, n_cells={backend.n_cells}, "
          f"n_pairs={backend.conn.n_pairs}, n_edges={backend.conn.n_edges}, "
          f"syn_mode={mode}, "
          f"nparams={schema_nparams(default_schema('borst', backend, syn_mode=mode, param_boxes=param_boxes, h_cells=h_cells, sti_radii=SPOT_STI_RADII, radius_key_aliases=SPOT_COST_RADIUS_KEY_ALIASES))}")
    return backend


@dataclass
class _TrainBindCtx:
    """Per-task builder context during :func:`open_session`."""

    model_backend: ModelBackend
    dev: str
    sim_dtype: torch.dtype = SIM_DTYPE
    cost_weights: Optional[Dict[str, float]] = None
    spot_bright_stimulus_opts: Optional[dict] = None
    spot_dark_stimulus_opts: Optional[dict] = None
    moving_bar_bright_stimulus_opts: Optional[dict] = None
    moving_bar_dark_stimulus_opts: Optional[dict] = None
    filter: str = "none"


def _moving_bar_waveform_mse_enabled(cost_weights: Optional[dict], pack_name: str) -> bool:
    """True if PD or ND waveform MSE weight is non-zero for ``pack_name``."""
    w = expand_cost_weight_dict(cost_weights or {})
    return any(
        float(w.get(moving_bar_cost_part_key(pack_name, lab), 1.0)) != 0.0
        for lab in PD_ND_LABELS
    )


def _moving_bar_polarity_opts(ctx: _TrainBindCtx, polarity: str) -> dict:
    if polarity == "bright":
        raw = ctx.moving_bar_bright_stimulus_opts
    elif polarity == "dark":
        raw = ctx.moving_bar_dark_stimulus_opts
    else:
        raise ValueError(f"unknown moving-bar polarity {polarity!r}")
    if raw:
        return dict(raw)
    return build_moving_bar_stimulus_opts(
        polarity,
        i_baseline_moving_bar=I_BASELINE,
        i_moving_bar=I_BRIGHT if polarity == "bright" else I_DARK,
        ms_pre=MS_PRE,
        delta_ms=DELTA_MS,
        delta_ms_pre=DELTA_MS_PRE,
        multi_bar=MULTI_BAR,
    )



def _cost_radius_hex_coltag(cost_radius, n_cost_hexes) -> str:
    radius_tag = "all hexes" if cost_radius is None else f"radius={int(cost_radius)}"
    if isinstance(n_cost_hexes, dict):
        hex_labels = ", ".join(
            f"b{int(batch)}={int(n_hex)}"
            for batch, n_hex in sorted(n_cost_hexes.items())
        )
        return f"cost hexes per batch [{hex_labels}], {radius_tag}"
    return f"{int(n_cost_hexes)} cost hexes, {radius_tag}"


def _build_network_moving_bar_readout(ctx: _TrainBindCtx, connectome, *, pack_name: str, polarity: str):
    dev = ctx.dev or active_device()
    opts = _moving_bar_polarity_opts(ctx, polarity)
    if "cost_radius" in opts:
        cost_radius = normalize_cost_radius(opts["cost_radius"])
    else:
        network_radius = int(connectome.meta.get("radius", -1))
        default_radius = -1 if network_radius <= 0 else network_radius - 1
        cost_radius = normalize_cost_radius(default_radius)
    build_kw = dict(
        connectome=connectome,
        device=dev,
        sim_dtype=ctx.sim_dtype,
        t_onset=t_from_ms(
            float(opts["ms_pre"]),
            delta_ms=float(opts["delta_ms_pre"]),
        ),
        delta_ms=float(opts["delta_ms"]),
        cost_radius=cost_radius,
        i_baseline_moving_bar=opts[_MOVING_BAR_BASELINE_KEY],
        contrasts=(polarity,),
        gt_cells=gt_cells_from_opts(opts),
        multi_bar=bool(opts.get("multi_bar", MULTI_BAR)),
        waveform_mse=_moving_bar_waveform_mse_enabled(ctx.cost_weights, pack_name),
    )
    peak_key = _MOVING_BAR_I_KEY[polarity]
    build_kw[peak_key] = opts[peak_key]
    T = build_moving_bar_gt(**build_kw)
    stim = _enrich_moving_bar_stimulus_opts(opts, T.info, cost_radius=cost_radius)
    pack = ReadoutPack(
        name=pack_name,
        i_sti=T.i_sti,
        gt=T.gt,
        power=T.power,
        cost_weight=T.cost_weight,
        readout_batch=T.readout_batch,
        readout_node=T.readout_node,
        cost_t0=T.cost_t0,
        cost_radius=cost_radius,
        cost_pd_nd=T.cost_pd_nd,
        dsi_pos_entries=T.dsi_pos_entries,
        dsi_neg_entries=T.dsi_neg_entries,
        dsi_pos_ptr=T.dsi_pos_ptr,
        dsi_neg_ptr=T.dsi_neg_ptr,
        dsi_gt=T.dsi_gt,
        dsi_weight=T.dsi_weight,
        dsi_power=T.dsi_power,
        waveform_mse=bool(T.info["waveform_mse"]),
    )
    coltag = _cost_radius_hex_coltag(cost_radius, T.info["n_cost_hexes"])
    tag = (
        f"moving-bar {polarity} (B={T.n_batch} stimuli, "
        f"{T.info['n_cost']} cost nodes, {coltag})"
    )
    return pack, stim, tag


def _spot_cost_time_idx_and_mask(opts, spot_cost_radius, *, device, sim_dtype):
    """Union ``cost_time_idx``; ``cost_time_mask`` when radii differ.

    ``cost_ms[r]`` overwrites ``cost_interval_ms`` grid for that radius.
    """
    n = int(spot_cost_radius.shape[0])
    if n == 0:
        return None, None
    delta_ms = float(opts["delta_ms"])
    t_onset, _ = spot_timing_t_from_opts(opts)
    post = spot_gt_n_t_from_opts(opts) - t_onset
    overrides = expand_cost_ms_dict(
        stimulus_opts=opts, aliases=SPOT_COST_RADIUS_KEY_ALIASES,
    )
    rad = np.round(spot_cost_radius.detach().cpu().numpy().astype(float), 6)
    radii = {float(r) for r in rad.tolist()}
    grid = None
    if any(r not in overrides for r in radii):
        interval_ms = opts.get("cost_interval_ms")
        if interval_ms is None:
            raise ValueError("spot cost needs cost_ms or cost_interval_ms")
        interval_ms = float(interval_ms)
        if interval_ms <= 0:
            raise ValueError("cost_interval_ms must be > 0")
        if post <= 0:
            raise ValueError("spot post-onset window must be > 0 for cost_interval_ms")
        step = max(1, int(round(interval_ms / delta_ms)))
        grid = [t * delta_ms for t in range(0, post, step)]
    radius_ts = {}
    for r in radii:
        ms_list = overrides[r] if r in overrides else grid
        ts = set()
        for ms in ms_list:
            t = int(round(float(ms) / delta_ms))
            if t < 0 or t >= post:
                raise ValueError(
                    f"cost time {ms} ms post-onset t out of range [0,{post})"
                )
            ts.add(t)
        radius_ts[r] = ts
    union = sorted({t for ts in radius_ts.values() for t in ts})
    cost_time_idx = torch.tensor(union, dtype=torch.long, device=device)
    union_set = set(union)
    if all(ts == union_set for ts in radius_ts.values()):
        return cost_time_idx, None
    col_from_t = {t: j for j, t in enumerate(union)}
    mask = torch.zeros(n, len(union), dtype=sim_dtype, device=device)
    for i, r in enumerate(rad.tolist()):
        for t in radius_ts[float(r)]:
            mask[i, col_from_t[t]] = 1.0
    return cost_time_idx, mask


def _build_network_spot_task(
    ctx: _TrainBindCtx, connectome, *, polarity: str,
) -> Tuple[ReadoutPack, dict, str]:
    if polarity not in SPOT_POLARITIES:
        raise ValueError(f"spot polarity must be 'bright' or 'dark', got {polarity!r}")
    pack_name = f"spot_{polarity}"
    peak_key = _SPOT_I_KEY[polarity]
    ctx_opts = (
        ctx.spot_bright_stimulus_opts if polarity == "bright" else ctx.spot_dark_stimulus_opts
    )
    if not ctx_opts:
        raise ValueError(f"{pack_name} requires stimulus opts (from build_train_opts / CLI)")
    opts = normalize_spot_timing(dict(ctx_opts))
    cost_radius = normalize_cost_radius(opts.get("cost_radius"))
    shift_radius = int(opts["shift_radius"])
    spot_radius = float(opts["spot_radius"])
    multi_spot = bool(opts["multi_spot"])
    fully_inside = bool(opts["fully_inside"])
    delta_ms = float(opts["delta_ms"])
    dev = ctx.dev or active_device()
    t_onset, n_t = spot_timing_t_from_opts(opts)
    i_spot = float(opts[peak_key])
    default_w = default_spot_cost_radius_weight(
        spot_radius,
        weights=SPOT_COST_RADIUS_WEIGHT,
        weights_radius1=SPOT_COST_RADIUS_WEIGHT_RADIUS1,
    )
    gt_amp = GT_CA_AMP if ctx.filter == "ca" else GT_V_AMP
    T = build_spot_gt(
        connectome,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        shift_radius=shift_radius,
        device=dev,
        sim_dtype=ctx.sim_dtype,
        n_t=n_t,
        t_onset=t_onset,
        cost_radius=cost_radius,
        spot_cost_radius_weight=expand_spot_cost_r_w_dict(
            stimulus_opts=opts, aliases=SPOT_COST_RADIUS_KEY_ALIASES,
        ),
        i_baseline_spot=float(opts[_SPOT_BASELINE_KEY]),
        i_bright_spot=i_spot if polarity == "bright" else float(opts.get("i_bright_spot", I_BRIGHT)),
        i_dark_spot=i_spot if polarity == "dark" else float(opts.get("i_dark_spot", I_DARK)),
        polarity=polarity,
        ms_spot=float(opts.get("ms_spot", MS_SPOT)),
        ms_response=float(opts["ms_response"]),
        gt_amp=gt_amp,
        delta_ms=delta_ms,
        default_cost_weights=default_w,
        spot_cost_radii=SPOT_COST_RADII,
        gt_cells=gt_cells_from_opts(opts),
        filter=str(ctx.filter),
    )
    cost_time_idx, cost_time_mask = _spot_cost_time_idx_and_mask(
        opts, T.spot_cost_radius, device=dev, sim_dtype=ctx.sim_dtype,
    )
    stim = dict(opts)
    if "present_gts" in T.info:
        stim["gt_cells"] = list(T.info["present_gts"])
    # Replace center-only bake from build_spot_gt: center @1 in i_sti + a_sti_radius radii.
    i_baseline = float(opts[_SPOT_BASELINE_KEY])
    spot = spot_from_opts(connectome, stimulus_opts=opts)
    batches = spot_stimulus_batches(spot)
    cost_r_w = expand_spot_cost_r_w_dict(
        stimulus_opts=opts, aliases=SPOT_COST_RADIUS_KEY_ALIASES,
    )
    gate = spot_sti_radius_gate(
        cost_r_w,
        default_weights=default_w,
        spot_sti_radii=SPOT_STI_RADII,
    )
    i_sti, sti_wave, sti_batch, sti_node, sti_radius = build_spot_a_sti_radius_drive(
        connectome,
        batches,
        sti_radii=SPOT_STI_RADII,
        t_onset=int(t_onset),
        n_t=int(n_t),
        ms_spot=float(opts.get("ms_spot", MS_SPOT)),
        delta_ms=delta_ms,
        i_baseline=i_baseline,
        i_peak=i_spot,
        sim_dtype=ctx.sim_dtype,
        device=dev,
    )
    pack = ReadoutPack(
        name=pack_name,
        i_sti=i_sti,
        gt=T.gt,
        power=T.power,
        cost_weight=T.cost_weight,
        readout_batch=T.readout_batch,
        readout_node=T.readout_node,
        cost_t0=None,
        cost_stim_u=T.readout_stim_u,
        cost_stim_v=T.readout_stim_v,
        cost_radius=cost_radius,
        spot_cost_radius=T.spot_cost_radius,
        cost_time_idx=cost_time_idx,
        cost_time_mask=cost_time_mask,
        waveform_mse=True,
        t_onset=int(t_onset),
        sti_wave=sti_wave,
        sti_batch=sti_batch,
        sti_node=sti_node,
        sti_radius=sti_radius,
        sti_radius_gate=torch.as_tensor(gate, dtype=ctx.sim_dtype, device=dev),
    )
    coltag = _cost_radius_hex_coltag(cost_radius, T.info["n_cost_hexes"])
    shifttag = f"{T.info['n_shifts']} shifts"
    tag = (
        f"{pack_name} (B={T.n_batch} stimuli [{T.info['n_centers']} centers simultaneous "
        f"x {shifttag}], {T.info['n_cost']} cost nodes, {coltag})"
    )
    return pack, stim, tag


NETWORK_TASK_BUILDERS = {
    "spot_bright": lambda ctx, connectome: _build_network_spot_task(ctx, connectome, polarity="bright"),
    "spot_dark": lambda ctx, connectome: _build_network_spot_task(ctx, connectome, polarity="dark"),
    "moving_bar_bright": lambda ctx, connectome: _build_network_moving_bar_readout(
        ctx, connectome, pack_name="moving_bar_bright", polarity="bright",
    ),
    "moving_bar_dark": lambda ctx, connectome: _build_network_moving_bar_readout(
        ctx, connectome, pack_name="moving_bar_dark", polarity="dark",
    ),
}


def _i_cli_task_names(cli_field, name):
    """Resolve CLI task token for one ``--i_*`` flag."""
    if name not in CLI_TASK_NAMES:
        raise ValueError(
            f"unknown task {name!r} in --{cli_field} "
            f"(expected {'|'.join(CLI_TASK_NAMES)})",
        )
    if cli_field == "i_baseline":
        if name in TASK_ALIASES:
            return TASK_ALIASES[name]
        return [name]
    if cli_field == "i_bright":
        if name not in I_CLI_BRIGHT_TASKS:
            raise ValueError(
                f"--i-bright does not accept task {name!r} "
                f"(expected spot|spot_bright|moving_bar|moving_bar_bright)",
            )
        return list(I_CLI_BRIGHT_TASKS[name])
    if name not in I_CLI_DARK_TASKS:
        raise ValueError(
            f"--i-dark does not accept task {name!r} "
            f"(expected spot|spot_dark|moving_bar|moving_bar_dark)",
        )
    return list(I_CLI_DARK_TASKS[name])


def build_i_cli_by_task(kv_by_field):
    """Merge per-flag comma KV dicts into ``{'by_task': {task: {field: val}}}``."""
    by_task = {}
    for cli_field, kv in kv_by_field.items():
        if not kv:
            continue
        for name, val in kv.items():
            for t in _i_cli_task_names(cli_field, name):
                sidecar_field = I_CLI_SIDECAR_FIELD[(cli_field, t)]
                by_task.setdefault(t, {})[sidecar_field] = float(val)
    return {"by_task": by_task} if by_task else None


def resolve_gt_cells_by_task(by_task_kv) -> Dict[str, List[str]]:
    """Map concrete tasks to final gt cell lists (task + cell aliases expanded)."""
    expanded = expand_gt_dict(by_task_kv or {})
    bad = [k for k in expanded if k not in VALID_TASKS]
    if bad:
        raise ValueError(
            f"unknown task(s) in --gt: {bad} "
            f"(expected {'|'.join(CLI_TASK_NAMES)})",
        )
    out: Dict[str, List[str]] = {}
    for tname, cells in expanded.items():
        if tname in SPOT_TASKS:
            out[tname] = list(expand_spot_gt_cells(cells))
        else:
            out[tname] = list(expand_moving_bar_gt_cells(cells))
    return out


def apply_i_cli(opts, task_name, i_cli):
    """Merge per-task CLI ``--i_*`` overrides into stimulus opts."""
    if not i_cli:
        return opts
    overrides = (i_cli.get("by_task") or {}).get(task_name)
    if not overrides:
        return opts
    out = dict(opts or {})
    allowed = TASK_I_FIELDS[task_name]
    for key, val in overrides.items():
        if key not in allowed:
            raise ValueError(f"{key!r} not valid for task {task_name!r}")
        out[key] = float(val)
    return out


_STIMULUS_TRAIN_OPT_SPECS = (
    ("spot_bright", "spot_bright_stimulus_opts"),
    ("spot_dark", "spot_dark_stimulus_opts"),
    ("moving_bar_bright", "moving_bar_bright_stimulus_opts"),
    ("moving_bar_dark", "moving_bar_dark_stimulus_opts"),
)


def _finalize_stimulus_opts(
    opts,
    task_name,
    *,
    cost_radius_by_task,
    shift_radius,
    spot_radius,
    multi_spot,
    fully_inside,
    spot_cost_radius_weight,
    i_cli,
):
    raw = dict(opts or {})
    if task_name in SPOT_TASKS:
        polarity = "bright" if task_name == "spot_bright" else "dark"
        peak_key = _SPOT_I_KEY[polarity]
        out = build_spot_stimulus_opts(
            polarity,
            i_baseline_spot=float(raw.get(_SPOT_BASELINE_KEY, I_BASELINE)),
            i_spot=float(raw.get(
                peak_key, I_BRIGHT if polarity == "bright" else I_DARK,
            )),
            ms_pre=float(raw.get("ms_pre", MS_PRE)),
            ms_response=float(raw.get("ms_response", MS_RESPONSE)),
            ms_post=float(raw.get("ms_post", MS_POST)),
            delta_ms=float(raw.get("delta_ms", DELTA_MS)),
            delta_ms_pre=float(raw.get("delta_ms_pre", DELTA_MS_PRE)),
            shift_radius=int(raw.get(
                "shift_radius",
                shift_radius if shift_radius is not None else SHIFT_RADIUS,
            )),
            spot_radius=float(raw.get(
                "spot_radius",
                spot_radius if spot_radius is not None else SPOT_RADIUS,
            )),
            multi_spot=bool(raw.get(
                "multi_spot",
                multi_spot if multi_spot is not None else MULTI_SPOT,
            )),
            fully_inside=bool(raw.get(
                "fully_inside",
                fully_inside if fully_inside is not None else FULLY_INSIDE,
            )),
            ms_spot=float(raw.get("ms_spot", MS_SPOT)),
            cost_interval_ms=raw.get("cost_interval_ms"),
            cost_ms=(
                expand_cost_ms_dict(
                    stimulus_opts=raw, aliases=SPOT_COST_RADIUS_KEY_ALIASES,
                )
                if "cost_ms" in raw else None
            ),
            gt_cells=raw.get("gt_cells"),
        )
    elif task_name in ("moving_bar_bright", "moving_bar_dark"):
        polarity = "bright" if task_name == "moving_bar_bright" else "dark"
        peak_key = _MOVING_BAR_I_KEY[polarity]
        out = build_moving_bar_stimulus_opts(
            polarity,
            i_baseline_moving_bar=float(raw.get(_MOVING_BAR_BASELINE_KEY, I_BASELINE)),
            i_moving_bar=float(raw.get(
                peak_key, I_BRIGHT if polarity == "bright" else I_DARK,
            )),
            ms_pre=float(raw.get("ms_pre", MS_PRE)),
            delta_ms=float(raw.get("delta_ms", DELTA_MS)),
            delta_ms_pre=float(raw.get("delta_ms_pre", DELTA_MS_PRE)),
            multi_bar=bool(raw.get("multi_bar", MULTI_BAR)),
            gt_cells=raw.get("gt_cells"),
        )
    else:
        out = raw
    if cost_radius_by_task and task_name in cost_radius_by_task:
        out["cost_radius"] = int(cost_radius_by_task[task_name])
    elif "cost_radius" in out:
        if out["cost_radius"] is None:
            out.pop("cost_radius", None)
        else:
            out["cost_radius"] = int(out["cost_radius"])
    if task_name in SPOT_TASKS:
        spot_radius_half_steps(spot_radius)
        out["shift_radius"] = int(shift_radius)
        out["spot_radius"] = float(spot_radius)
        out["multi_spot"] = bool(multi_spot)
        out["fully_inside"] = bool(fully_inside)
        if spot_cost_radius_weight is not None:
            out["spot_cost_radius_weight"] = {
                str(k): float(v) for k, v in spot_cost_radius_weight.items()
            }
    return apply_i_cli(out, task_name, i_cli)


def build_train_opts(
    backend="network",
    tasks=None,
    cost_weights=None,
    pack_overrides=None,
    sequential=None,
    cost_radius_by_task=None,
    shift_radius=None,
    spot_radius=None,
    multi_spot=MULTI_SPOT,
    fully_inside=FULLY_INSIDE,
    spot_cost_radius_weight=None,
    i_cli=None,
    moving_bar_bright_stimulus_opts=None,
    moving_bar_dark_stimulus_opts=None,
    spot_bright_stimulus_opts=None,
    spot_dark_stimulus_opts=None,
    network_json=None,
    network=None,
    train_modes=None,
    syn_mode=SYN_MODE,
    cost_norm=COST_NORM,
    dev=None,
    packs=None,
    i_h_rev=I_H_REV,
    euler=EULER,
    pre_steady=None,
    pre_steady_iters=PRE_STEADY_ITERS,
    pre_steady_damp=PRE_STEADY_DAMP,
    fp=FP,
    pre_grad=PRE_GRAD,
    bias_gt_from_v_onset=BIAS_GT_FROM_V_ONSET,
    bias_gt_from_v_onset_grad=BIAS_GT_FROM_V_ONSET_GRAD,
    v_th_ca_from_v_th=V_TH_CA_FROM_V_TH,
    a_ca_from_a_out=A_CA_FROM_A_OUT,
    filter=FILTER,
):
    """Canonical training opts for :func:`open_session` (network backend)."""
    if backend != "network":
        raise ValueError(f"backend must be 'network', got {backend!r}")
    if network is None and network_json is None:
        raise ValueError("build_train_opts requires network or network_json")
    fp = int(fp)
    if fp not in (16, 32, 64):
        raise ValueError(f"fp must be 16, 32, or 64; got {fp!r}")
    cost_norm = expand_cost_norm(cost_norm)
    filter = expand_filter(filter)
    pre_steady = expand_pre_steady(
        PRE_STEADY if pre_steady is None else pre_steady
    )
    pre_steady_iters = int(pre_steady_iters)
    pre_steady_damp = float(pre_steady_damp)
    if pre_steady_iters < 1:
        raise ValueError(f"pre_steady_iters must be >= 1; got {pre_steady_iters}")
    if not (0.0 < pre_steady_damp <= 1.0):
        raise ValueError(
            f"pre_steady_damp must be in (0, 1]; got {pre_steady_damp}"
        )
    bias_gt_from_v_onset = bool(bias_gt_from_v_onset)
    bias_gt_from_v_onset_grad = bool(bias_gt_from_v_onset_grad)
    if not bias_gt_from_v_onset:
        bias_gt_from_v_onset_grad = False
    if bias_gt_from_v_onset:
        train_modes = dict(train_modes or {})
        train_modes["bias_gt"] = {
            "indi": [], "shared": [], "fixed": [], "frozen": ["all"],
        }
    v_th_ca_from_v_th = bool(v_th_ca_from_v_th)
    if v_th_ca_from_v_th:
        train_modes = dict(train_modes or {})
        train_modes["v_th_ca"] = {
            "indi": [], "shared": [], "fixed": [], "frozen": ["all"],
        }
    a_ca_from_a_out = bool(a_ca_from_a_out)
    if a_ca_from_a_out:
        train_modes = dict(train_modes or {})
        train_modes["a_ca"] = {
            "indi": [], "shared": [], "fixed": [], "frozen": ["all"],
        }
    tl = normalize_tasks(tasks)
    if spot_radius is None:
        spot_radius = SPOT_RADIUS
    if shift_radius is None:
        shift_radius = SHIFT_RADIUS
    raw_by_name = {
        "spot_bright": spot_bright_stimulus_opts,
        "spot_dark": spot_dark_stimulus_opts,
        "moving_bar_bright": moving_bar_bright_stimulus_opts,
        "moving_bar_dark": moving_bar_dark_stimulus_opts,
    }
    finalize_kw = dict(
        cost_radius_by_task=cost_radius_by_task,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_weight=spot_cost_radius_weight,
        i_cli=i_cli,
    )
    stimulus_opts = {}
    for tname, opts_key in _STIMULUS_TRAIN_OPT_SPECS:
        raw = raw_by_name[tname]
        if tname not in tl and raw is None:
            stimulus_opts[opts_key] = None
            continue
        stimulus_opts[opts_key] = _finalize_stimulus_opts(
            raw,
            tname,
            **finalize_kw,
        )
    opts = {
        "backend": "network",
        "tasks": tl,
        "cost_weights": expand_cost_weight_dict(cost_weights or {}),
        "cost_norm": cost_norm,
        "pre_steady": pre_steady,
        "pre_steady_iters": pre_steady_iters,
        "pre_steady_damp": pre_steady_damp,
        "sequential": sequential,
        **stimulus_opts,
    }
    if pack_overrides is not None:
        opts["pack_overrides"] = pack_overrides
    if packs is not None:
        opts["packs"] = packs
    if train_modes is not None:
        opts["train_modes"] = train_modes
    opts["i_h_rev"] = str(i_h_rev)
    opts["euler"] = expand_euler(euler)
    opts["syn_mode"] = normalize_syn_mode(syn_mode)
    opts["pre_grad"] = bool(pre_grad)
    opts["bias_gt_from_v_onset"] = bias_gt_from_v_onset
    opts["bias_gt_from_v_onset_grad"] = bias_gt_from_v_onset_grad
    opts["v_th_ca_from_v_th"] = v_th_ca_from_v_th
    opts["a_ca_from_a_out"] = a_ca_from_a_out
    opts["filter"] = filter
    if filter == "ca":
        for key in ("spot_bright_stimulus_opts", "spot_dark_stimulus_opts"):
            so = opts.get(key)
            if so is not None:
                so["ms_spot"] = float(MS_SPOT_CA)
    opts["fp"] = fp
    opts.update({
        "network": network,
        "network_json": str(network_json) if network_json is not None else None,
        "dev": dev,
    })
    return opts


def _train_opts_for_sidecar(opts, tasks, resolved_stim, sequential_bool) -> dict:
    """Build JSON-serializable train_opts record (network backend only)."""
    def _stim(key):
        got = resolved_stim.get(key)
        return got if got is not None else opts.get(key)

    record = {
        "backend": "network",
        "tasks": list(tasks),
        "cost_weights": {
            str(k): float(v) for k, v in (opts.get("cost_weights") or {}).items()
        },
        "cost_norm": expand_cost_norm(opts.get("cost_norm", COST_NORM)),
        "pre_steady": expand_pre_steady(opts.get("pre_steady", PRE_STEADY)),
        "pre_steady_iters": int(opts.get("pre_steady_iters", PRE_STEADY_ITERS)),
        "pre_steady_damp": float(opts.get("pre_steady_damp", PRE_STEADY_DAMP)),
        "sequential": bool(sequential_bool),
        "network_json": str(opts["network_json"]),
        "spot_bright_stimulus_opts": _stim("spot_bright_stimulus_opts"),
        "spot_dark_stimulus_opts": _stim("spot_dark_stimulus_opts"),
        "moving_bar_bright_stimulus_opts": _stim("moving_bar_bright_stimulus_opts"),
        "moving_bar_dark_stimulus_opts": _stim("moving_bar_dark_stimulus_opts"),
    }
    overrides = opts.get("pack_overrides")
    if overrides:
        record["pack_overrides"] = overrides
    if opts.get("train_modes"):
        record["train_modes"] = opts["train_modes"]
    if "i_h_rev" in opts:
        record["i_h_rev"] = str(opts["i_h_rev"])
    if "euler" not in opts:
        raise ValueError("train opts require euler (implicit|explicit)")
    record["euler"] = expand_euler(opts["euler"])
    record["syn_mode"] = normalize_syn_mode(opts.get("syn_mode", SYN_MODE))
    record["pre_grad"] = bool(opts.get("pre_grad", True))
    record["bias_gt_from_v_onset"] = bool(
        opts.get("bias_gt_from_v_onset", BIAS_GT_FROM_V_ONSET)
    )
    record["bias_gt_from_v_onset_grad"] = bool(
        opts.get("bias_gt_from_v_onset_grad", BIAS_GT_FROM_V_ONSET_GRAD)
    )
    record["v_th_ca_from_v_th"] = bool(
        opts.get("v_th_ca_from_v_th", V_TH_CA_FROM_V_TH)
    )
    record["a_ca_from_a_out"] = bool(
        opts.get("a_ca_from_a_out", A_CA_FROM_A_OUT)
    )
    record["filter"] = expand_filter(opts.get("filter", FILTER))
    record["fp"] = int(opts.get("fp", FP))
    return record


def _schema_from_opts(model, model_backend, schema, train_opts_record, *, i_h_rev=None):
    if schema is not None:
        return list(schema)
    syn_mode = SYN_MODE
    if train_opts_record:
        syn_mode = normalize_syn_mode(train_opts_record.get("syn_mode", SYN_MODE))
    kw = dict(
        syn_mode=syn_mode,
        param_boxes=PARAM_BOXES,
        h_cells=H_CELLS,
        sti_radii=SPOT_STI_RADII,
        radius_key_aliases=SPOT_COST_RADIUS_KEY_ALIASES,
    )
    if model == "borst":
        kw["i_h_rev"] = str(i_h_rev if i_h_rev is not None else I_H_REV)
    base = default_schema(model, model_backend, **kw)
    if not train_opts_record:
        return base
    modes = train_opts_record.get("train_modes")
    if modes:
        base = apply_train_modes(
            base, modes, lambda seg: node_names_for_segment(seg, model_backend),
        )
    return base


def _build_session(
    model_backend: ModelBackend,
    model: str,
    tasks: List[str],
    packs: Dict[str, ReadoutPack],
    *,
    delta_ms: float,
    delta_ms_pre: float,
    cost_weights=None,
    sequential=None,
    dev=None,
    train_opts_record=None,
    schema: Optional[list] = None,
    sim_dtype=SIM_DTYPE,
) -> TrainSession:
    dev_ref = dev or active_device()
    seq = False if sequential is None else bool(sequential)
    if train_opts_record is not None:
        train_opts_record["model"] = model
        train_opts_record["sequential"] = bool(seq)
    i_h_rev = I_H_REV
    if train_opts_record is not None and "i_h_rev" in train_opts_record:
        i_h_rev = str(train_opts_record["i_h_rev"])
    if train_opts_record is None or "euler" not in train_opts_record:
        raise ValueError("train opts require euler (implicit|explicit)")
    euler = expand_euler(train_opts_record["euler"])
    pre_steady = expand_pre_steady(
        train_opts_record.get("pre_steady", PRE_STEADY),
    )
    train_opts_record["pre_steady"] = pre_steady
    pre_steady_iters = int(
        train_opts_record.get("pre_steady_iters", PRE_STEADY_ITERS)
    )
    pre_steady_damp = float(
        train_opts_record.get("pre_steady_damp", PRE_STEADY_DAMP)
    )
    train_opts_record["pre_steady_iters"] = pre_steady_iters
    train_opts_record["pre_steady_damp"] = pre_steady_damp
    sch = _schema_from_opts(
        model, model_backend, schema, train_opts_record, i_h_rev=i_h_rev,
    )
    train_opts_record["train_modes"] = schema_train_modes_record(
        sch, lambda seg: node_names_for_segment(seg, model_backend),
    )
    sch = attach_param_carry(sch)
    session = TrainSession(
        backend=model_backend,
        model=model,
        schema=tuple(sch),
        readouts=dict(packs),
        tasks=tuple(tasks),
        cost_weights=expand_cost_weight_dict(cost_weights),
        sequential=bool(seq),
        device=dev_ref,
        delta_ms=float(delta_ms),
        delta_ms_pre=float(delta_ms_pre),
        cap=CAP,
        g_leak=G_LEAK,
        e_exc=E_EXC,
        e_inh=E_INH,
        e_h=E_H,
        h_g_max=H_G_MAX,
        GT_V_AMP=GT_V_AMP,
        GT_CA_AMP=GT_CA_AMP,
        STATE_CLAMP=STATE_CLAMP,
        syn_scale_exc=SYN_SCALE_EXC,
        syn_scale_inh=SYN_SCALE_INH,
        euler=euler,
        pre_steady=pre_steady,
        pre_steady_iters=pre_steady_iters,
        pre_steady_damp=pre_steady_damp,
        sim_dtype=sim_dtype,
        train_opts=train_opts_record,
    )
    cost_subpacks = _build_cost_subpacks(session)
    fused_forward = _build_fused_forward(session, cost_subpacks)
    return replace(session, cost_subpacks=cost_subpacks, fused_forward=fused_forward)


def open_session(
    opts: dict,
    model: str,
    *,
    schema: Optional[list] = None,
    model_backend: Optional[ModelBackend] = None,
) -> TrainSession:
    """Build a :class:`TrainSession` from canonical training opts."""
    backend_name = str(opts.get("backend", "network"))
    if backend_name != "network":
        raise ValueError(f"backend must be 'network', got {backend_name!r}")
    tasks = normalize_tasks(opts.get("tasks"))
    bad = [t for t in tasks if t not in VALID_TASKS]
    if bad:
        raise ValueError(f"unknown task(s) {bad!r} (expected {'|'.join(CLI_TASK_NAMES)})")
    dev = opts.get("dev") or active_device()
    sim_dtype = sim_dtype_from_fp(int(opts.get("fp", FP)))
    delta_ms = _stimulus_delta_ms(opts, "delta_ms")
    delta_ms_pre = _stimulus_delta_ms(opts, "delta_ms_pre")

    connectome = opts.get("network")
    syn_mode = normalize_syn_mode(opts.get("syn_mode", SYN_MODE))
    if connectome is None:
        nj = opts.get("network_json")
        if not nj:
            raise ValueError("open_session(network) requires opts['network'] or network_json")
        connectome = load_network(
            nj, device=dev,
            syn_scale_exc=SYN_SCALE_EXC, syn_scale_inh=SYN_SCALE_INH,
            dtype=sim_dtype, syn_mode=syn_mode,
        )
    if model_backend is None:
        model_backend = _network_backend_from_connectome(
            connectome, sim_dtype=sim_dtype,
        )
    elif model_backend.network is not connectome:
        raise ValueError("model_backend.network must be opts['network']")
    ctx = _TrainBindCtx(
        model_backend=model_backend,
        dev=dev,
        sim_dtype=sim_dtype,
        cost_weights=opts.get("cost_weights"),
        spot_bright_stimulus_opts=opts.get("spot_bright_stimulus_opts"),
        spot_dark_stimulus_opts=opts.get("spot_dark_stimulus_opts"),
        moving_bar_bright_stimulus_opts=opts.get("moving_bar_bright_stimulus_opts"),
        moving_bar_dark_stimulus_opts=opts.get("moving_bar_dark_stimulus_opts"),
        filter=expand_filter(opts.get("filter", FILTER)),
    )
    packs = {}
    pack_overrides = opts.get("pack_overrides") or {}
    resolved_stim = {}
    for tname in tasks:
        pack, stim, _tag = NETWORK_TASK_BUILDERS[tname](ctx, connectome)
        if tname in pack_overrides:
            pack = apply_pack_override(pack, pack_overrides[tname], model_backend)
        packs[tname] = pack
        resolved_stim[f"{tname}_stimulus_opts"] = stim
    record = _train_opts_for_sidecar(
        opts, tasks, resolved_stim, bool(opts.get("sequential")),
    )
    return _build_session(
        model_backend, model, tasks, packs,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        cost_weights=opts.get("cost_weights"),
        sequential=opts.get("sequential"),
        dev=dev,
        train_opts_record=record,
        schema=schema,
        sim_dtype=sim_dtype,
    )


def _stimulus_delta_ms(opts: dict, key: str) -> float:
    """``delta_ms`` / ``delta_ms_pre`` from stimulus opts (required)."""
    for _tname, opts_key in _STIMULUS_TRAIN_OPT_SPECS:
        so = opts.get(opts_key)
        if isinstance(so, dict) and so.get(key) is not None:
            dt = float(so[key])
            if dt <= 0:
                raise ValueError(f"stimulus opts {key} must be > 0, got {dt}")
            return dt
    raise ValueError(
        f"train opts require {key} in a stimulus opts dict "
        f"(one of {[k for _, k in _STIMULUS_TRAIN_OPT_SPECS]})"
    )


def open_session_from_opts(opts: dict, model: str | None = None, **kwargs) -> TrainSession:
    """Restore a session from a saved ``train_opts.json`` dict."""
    opts = dict(opts)
    if model is None:
        model = opts.get("model")
        if not model:
            raise ValueError("train_opts requires model")
    opts["packs"] = None
    backend = str(opts.get("backend", "network"))
    if backend != "network":
        raise ValueError(f"backend must be 'network', got {backend!r}")
    nj = opts.get("network_json")
    if not nj:
        raise ValueError("train_opts requires network_json")
    if not opts.get("tasks"):
        raise ValueError("train_opts requires tasks")
    sim_dtype = sim_dtype_from_fp(int(opts.get("fp", FP)))
    syn_mode = normalize_syn_mode(opts.get("syn_mode", SYN_MODE))
    mb = load_network_backend(
        nj, dev=opts.get("dev") or active_device(), sim_dtype=sim_dtype,
        syn_mode=syn_mode,
        syn_scale_exc=SYN_SCALE_EXC, syn_scale_inh=SYN_SCALE_INH,
    )
    opts["network"] = mb.network
    opts["syn_mode"] = syn_mode
    kwargs.setdefault("model_backend", mb)
    return open_session({**opts, "backend": "network"}, model, **kwargs)


def open_session_from_outdir(
    outdir: str,
    model: str | None = None,
) -> TrainSession:
    """Load ``train_opts.json`` from a run folder and return a ready session."""
    opts_path = os.path.join(run_data_dir(os.path.abspath(outdir)), TRAIN_OPTS_FILE)
    if not os.path.isfile(opts_path):
        raise FileNotFoundError(f"missing {opts_path}")
    with open(opts_path) as f:
        opts = json.load(f)
    return open_session_from_opts(opts, model)
