# -*- coding: utf-8 -*-
"""Session types and assembly: ``Pack`` / ``TrainSession`` + open helpers.

Owns sti-opts finalisation (CLI overrides -> per-task sidecar dicts),
network backend construction, and the per-task ``Pack`` builders. The
builders wrap the neutral gt dataclasses from ``task`` (which sit below
``train`` in the import graph) and stamp the cross-cutting pack cost controls:

* spot: sparse ``cost_time_indices`` / optional ``cost_time_mask`` (#4; ``cost_ms``
  overwrites interval per radius), ``ms_sti`` (#1) already baked into
  the sti, ``waveform_mse=True``;
* moving bar: ``waveform_mse`` from cost scales (True when a cost window is
  built).

Model traces are absolute ``v`` (``filter=none``) or ``ca`` (``filter=ca``);
cost compares the pack to ``a_gt * gts + bias_gt``. Schema includes
``v_th_ca``/``a_ca``/``tau_ca`` only when ``filter=ca``. When
``val_from`` (``--val-from``), ``bias_gt`` is written from ``v`` at ``t_onset`` (or ``ca`` when
``filter=ca``) — same value appears in ``param.csv``. Spot ir uses Arenz digitized
when ``filter=ca``. ``spot_gt_mode`` (``all`` | ``pos``) gates cost GT via
:func:`task.spot.gt.spot_gt_active`; dark multiplies by :func:`task.spot.gt.contrast_sign`.
Gt cells are :func:`network.construction.active_gt_cells`; cost pack applies
:func:`task.spot.gt.spot_gt_active` per cell.

Branch-resolution contract (must keep):

* Do not add parameter-specific helper parsers/casters in pre-session stages.
* All ``{v,ca}`` branch selection must be unified through
  :func:`resolve_filter_branches`.
* Before :func:`open_session`, keep branch-capable values raw; only perform
  ``float``/``int``/``bool`` casting after branch resolution for those
  branch-capable values.
"""
from __future__ import annotations

from default_params import (
    MOVING_BAR_BRIGHT_STI_OPTS,
    MOVING_BAR_DARK_STI_OPTS,
    MOVING_BAR_INPUT,
    NETWORK_CONSTRUCTION,
    NEURON_FILTER,
    NEURON_FORWARD,
    NEURON_PARAM,
    NEURON_SCHEMA,
    SPOT_BRIGHT_STI_OPTS,
    SPOT_DARK_STI_OPTS,
    SPOT_INPUT,
    SPOT_PACK,
    STI_TIMING,
    TRAIN_OPTIMIZATION,
    TRAIN_OPTS,
    TRAIN_SESSION,
    VAL_FROM,
)

import copy
import json
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neuron.param import membrane_dt_over_c, t_from_ms
from neuron import (
    build_schema,
    expand_euler,
    normalize_syn_mode,
)

from train.config import (
    CLI_TASK_NAMES,
    I_STI_KEYS,
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
    expand_part_cost_scale_dict,
    expand_filter,
    expand_spot_gt_mode,
    expand_gt_dict,
    expand_pre_steady,
    moving_bar_cost_part_key,
    resolve_tasks,
    run_data_dir,
)
from train.param import (
    ModelBackend,
    SIM_DTYPE,
    active_device,
    apply_param_init,
    apply_param_modes,
    attach_param_carry,
    resolve_param_modes,
    build_i_h_dirs,
    resolve_val_from,
    node_names_for_segment,
    schema_nparams,
    schema_param_modes_record,
    sim_dtype_from_fp,
    val_from_enabled,
)

from task.spot.gt import expand_gt_cells as expand_spot_gt_cells
from task.spot.pack import (
    SPOT_CONTRASTS,
    build_spot_gt,
    resolve_spot_cost_radius_scale_defaults,
    expand_cost_ms_dict,
    expand_spot_cost_r_s_dict,
    build_spot_sti_opts,
    build_a_sti_radius_mask,
)
from task.spot.sti_geo import (
    resolve_spot,
    spot_radius_half_steps,
    spot_sti_batches,
)
from task.spot.sti_spec import (
    build_spot_a_sti_radius_drive,
    normalize_sti_timing,
    resolve_sti_timing,
)
from task.moving_bar.gt import expand_gt_cells as expand_moving_bar_gt_cells
from task.moving_bar.pack import (
    build_moving_bar_gt,
    build_moving_bar_sti_opts,
    enrich_moving_bar_sti_opts,
)
from network.construction import (
    load_network, gt_cells_from_opts, node_cell_names, normalize_cost_radius,
)


@dataclass(frozen=True)
class Pack:
    """One train pack: task drive + entry indices + gts.

    Spot ``i_sti`` / ``gts`` time dims follow ``neuron`` / task
    timing. Moving bar uses ``COST_WINDOW`` and per-task ``n_t``.
    """

    name: str
    i_sti: torch.Tensor  # (B, T, N)
    gts: torch.Tensor  # (n_cost, T')
    power: torch.Tensor  # scalar
    cost_scales: torch.Tensor  # (n_cost,)
    entry_batches: torch.Tensor  # (n_cost,)
    entry_nodes: torch.Tensor  # (n_cost,)
    cost_t0s: Optional[torch.Tensor] = None  # (n_cost,) absolute step for windowed readouts
    cost_radius: Optional[int] = None  # hex-disc ring count for cost readouts
    entry_radii: Optional[torch.Tensor] = None  # (n_cost,) Euclidean radius per spot entry
    cost_sti_us: Optional[torch.Tensor] = None  # (n_cost,) sti anchor u per spot cost entry
    cost_sti_vs: Optional[torch.Tensor] = None  # (n_cost,) sti anchor v per spot cost entry
    cost_pd_nds: Optional[torch.Tensor] = None  # (n_cost,) long; 0=PD, 1=ND (moving_bar)
    dsi_pos_entries: Optional[torch.Tensor] = None  # flat cost-entry idx (right|up)
    dsi_neg_entries: Optional[torch.Tensor] = None  # flat cost-entry idx (left|down)
    dsi_pos_ptr: Optional[torch.Tensor] = None  # (n_dsi+1,) CSR
    dsi_neg_ptr: Optional[torch.Tensor] = None  # (n_dsi+1,) CSR
    dsi_gts: Optional[torch.Tensor] = None  # (n_dsi,)
    dsi_scales: Optional[torch.Tensor] = None  # (n_dsi,)
    dsi_power: Optional[torch.Tensor] = None  # scalar
    cost_time_indices: Optional[torch.Tensor] = None  # (n_sample,) sparse post-onset t idx
    cost_time_mask: Optional[torch.Tensor] = None  # (n_cost, n_sample) 0/1 per-radius
    waveform_mse: bool = True  # spot: True; moving bar: set at build
    t_onset: Optional[int] = None  # explicit onset; spot when ms_post extends i_sti past gt
    # Spot a_sti_radius: i = i_sti + a_sti_radius[r] * sti_wave on (sti_batches, sti_nodes).
    sti_wave: Optional[torch.Tensor] = None  # (T,) (i_peak - i_baseline) * u(t)
    sti_batches: Optional[torch.Tensor] = None  # (n_contrib,) long
    sti_nodes: Optional[torch.Tensor] = None  # (n_contrib,) long
    a_sti_radius_indices: Optional[torch.Tensor] = None  # (n_contrib,) long → a_sti_radius index
    # (n_sti_slots,) 0/1: cost-radius scale==0 forces a_sti_radius slot to 0 in forward.
    a_sti_radius_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TrainSession:
    """Immutable runtime context for one train / plotting run.

    Membrane / synapse scalars are flat fields (injected from
    ``default_params`` at session open). ``delta_ms`` / ``delta_ms_pre`` come
    only from sti opts — never a separate Physics bag.
    """

    backend: ModelBackend
    model: str
    schema: tuple
    packs: Dict[str, Pack]
    tasks: Tuple[str, ...]
    part_cost_scales: Dict[str, float]
    sequential: bool
    device: str
    delta_ms: float
    delta_ms_pre: float
    cap: float
    g_leak: float
    e_exc: float
    e_inh: float
    e_h: float
    h_g_max: float
    gt_amp: float
    state_clamp: float
    a_syn_exc: float
    a_syn_inh: float
    euler: str
    pre_steady: str = "solve"
    pre_steady_iters: int = 60
    pre_steady_damp: float = 1.0
    sim_dtype: torch.dtype = SIM_DTYPE
    train_opts: Optional[dict] = None

    def with_schema(self, schema) -> "TrainSession":
        return replace(self, schema=tuple(schema))

    @property
    def dt_over_c(self) -> float:
        return membrane_dt_over_c(self.cap, self.delta_ms)

    @property
    def primary_pack(self) -> Pack:
        return self.packs[self.tasks[0]]

    @property
    def n_t(self) -> int:
        i_sti = self.primary_pack.i_sti
        return int(i_sti.shape[1] if i_sti.dim() == 3 else i_sti.shape[0])

    def pack_i_sti(self, pack: Optional[Pack] = None) -> torch.Tensor:
        pack = pack or self.primary_pack
        i_sti = pack.i_sti
        if pack.name in SPOT_TASKS and i_sti.dim() == 3 and int(i_sti.shape[0]) == 1:
            i_sti = i_sti.squeeze(0)
        return i_sti

    def pack_for(self, name: str) -> Pack:
        if name not in self.packs:
            raise KeyError(f"pack {name!r} not in session")
        return self.packs[name]


def resolve_cell_indices(cells, backend: ModelBackend):
    """Map cell names to indices in the network vocabulary."""
    if backend.network is None:
        raise ValueError("resolve_cell_indices requires backend.network")
    names = [str(n) for n in cells]
    tn = list(backend.network.cells)
    return [tn.index(n) for n in names if n in tn]


def extend_pack_mirror_fit(pack, mirror_types, mirror_fit, mirror_sign=-1.0, backend=None):
    """Extend a :class:`Pack`: mirror *mirror_fit* gts onto *mirror_types*."""
    if backend is None or backend.network is None:
        raise ValueError("extend_pack_mirror_fit requires backend.network")
    connectome = backend.network
    names = node_cell_names(connectome)
    entry_nodes_idx = pack.entry_nodes.cpu().numpy()
    b_arr = pack.entry_batches.cpu().numpy()
    w_arr = pack.cost_scales.cpu().numpy()
    r_arr = (
        pack.entry_radii.cpu().numpy()
        if pack.entry_radii is not None else None
    )
    network_node_u = connectome.us.detach().cpu().numpy() if hasattr(connectome.us, "detach") else np.asarray(connectome.us)
    network_node_v = connectome.vs.detach().cpu().numpy() if hasattr(connectome.vs, "detach") else np.asarray(connectome.vs)
    extra_batch, extra_node_idx, extra_entries, extra_cost_scales, extra_radius, extra_pd_nd = [], [], [], [], [], []
    for entry_i in range(len(entry_nodes_idx)):
        node_idx = int(entry_nodes_idx[entry_i])
        if str(names[node_idx]) != mirror_fit:
            continue
        batch = int(b_arr[entry_i])
        mirror_fit_node_u, mirror_fit_node_v = int(network_node_u[node_idx]), int(network_node_v[node_idx])
        mirror_gt = float(mirror_sign) * pack.gts[entry_i:entry_i + 1]
        cost_scale = float(w_arr[entry_i])
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
                extra_entries.append(mirror_gt)
                extra_cost_scales.append(cost_scale)
                if radius is not None:
                    extra_radius.append(radius)
                if pack.cost_pd_nds is not None:
                    extra_pd_nd.append(int(pack.cost_pd_nds[entry_i].item()))
    return _append_mirror_pack_entries(
        pack, extra_node_idx, extra_entries,
        entry_batches=extra_batch, cost_scales=extra_cost_scales,
        entry_radii=extra_radius if extra_radius else None,
        cost_pd_nds=extra_pd_nd if extra_pd_nd else None,
    )


def _append_mirror_pack_entries(
    pack, extra_nodes, extra_entries, entry_batches=None, cost_scales=None,
    entry_radii=None, cost_pd_nds=None,
):
    extra_nodes_t = torch.tensor(extra_nodes, dtype=torch.long, device=active_device())
    extra_gts = torch.cat(extra_entries, dim=0)
    n_all = int(pack.entry_nodes.shape[0]) + len(extra_nodes)
    n_extra = len(extra_nodes)
    if entry_batches is None:
        entry_batches = torch.zeros(n_extra, dtype=torch.long, device=active_device())
    else:
        entry_batches = torch.tensor(entry_batches, dtype=torch.long, device=active_device())
    if cost_scales is None:
        w_dtype = pack.cost_scales.dtype
        cost_scales = torch.ones(n_all, dtype=w_dtype, device=active_device())
    else:
        base_w = pack.cost_scales
        w_dtype = base_w.dtype
        cost_scales = torch.cat([
            base_w,
            torch.tensor(cost_scales, dtype=w_dtype, device=active_device()),
        ])
    entry_radii_out = pack.entry_radii
    if entry_radii is not None:
        base_r = pack.entry_radii
        r_dtype = base_r.dtype if base_r is not None else SIM_DTYPE
        extra_r_t = torch.tensor(entry_radii, dtype=r_dtype, device=active_device())
        entry_radii_out = (
            torch.cat([base_r, extra_r_t])
            if base_r is not None else extra_r_t
        )
    gts = torch.cat([pack.gts, extra_gts], dim=0)
    cost_pd_nd_out = pack.cost_pd_nds
    if cost_pd_nds is not None:
        extra_pd_t = torch.tensor(cost_pd_nds, dtype=torch.long, device=active_device())
        cost_pd_nd_out = (
            torch.cat([pack.cost_pd_nds, extra_pd_t])
            if pack.cost_pd_nds is not None else extra_pd_t
        )
    return Pack(
        name=pack.name,
        i_sti=pack.i_sti,
        gts=gts,
        power=pack.power + torch.sum(extra_gts ** 2),
        cost_scales=cost_scales,
        entry_batches=torch.cat([pack.entry_batches, entry_batches]),
        entry_nodes=torch.cat([pack.entry_nodes, extra_nodes_t]),
        cost_t0s=pack.cost_t0s,
        cost_radius=pack.cost_radius,
        entry_radii=entry_radii_out,
        cost_pd_nds=cost_pd_nd_out,
        dsi_pos_entries=pack.dsi_pos_entries,
        dsi_neg_entries=pack.dsi_neg_entries,
        dsi_pos_ptr=pack.dsi_pos_ptr,
        dsi_neg_ptr=pack.dsi_neg_ptr,
        dsi_gts=pack.dsi_gts,
        dsi_scales=pack.dsi_scales,
        dsi_power=pack.dsi_power,
        cost_time_indices=pack.cost_time_indices,
        cost_time_mask=pack.cost_time_mask,
        waveform_mse=pack.waveform_mse,
        t_onset=pack.t_onset,
    )


def apply_pack_override(pack, override, backend: ModelBackend):
    """Apply one serializable pack override dict (saved in ``train_opts.json``)."""
    specs = override.get("mirror_fit")
    if specs is None:
        raise ValueError(f"unknown pack override {override!r}")
    if isinstance(specs, dict):
        specs = [specs]
    for spec in specs:
        if "mirror_types" not in spec:
            raise ValueError(f"mirror_fit spec needs mirror_types: {spec!r}")
        pack = extend_pack_mirror_fit(
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
        i_h_dirs=build_i_h_dirs(conn, dtype=sim_dtype),
        n_cells=connectome.n_cells,
        n_hexes=1,
        network=connectome,
    )


def load_network_backend(
    network_json,
    dev: Optional[str] = None,
    *,
    a_syn_exc: float,
    a_syn_inh: float,
    sim_dtype=SIM_DTYPE,
    syn_mode=NEURON_SCHEMA['syn_mode'],
    optimizable=NEURON_SCHEMA['optimizable'],
    h_cells=NEURON_SCHEMA['h_cells'],
) -> ModelBackend:
    """Load connectome network into a :class:`ModelBackend`."""
    dev = dev or active_device()
    mode = normalize_syn_mode(syn_mode)
    connectome = load_network(
        network_json, device=dev,
        a_syn_exc=a_syn_exc, a_syn_inh=a_syn_inh,
        dtype=sim_dtype, syn_mode=mode,
    )
    backend = _network_backend_from_connectome(connectome, sim_dtype=sim_dtype)
    print(f"network: {network_json}")
    print(f"  n_nodes={backend.n_nodes}, n_cells={backend.n_cells}, "
          f"n_pairs={backend.conn.n_pairs}, n_edges={backend.conn.n_edges}, "
          f"syn_mode={mode}, "
          f"nparams={schema_nparams(build_schema('borst', backend, syn_mode=mode, optimizable=optimizable, h_cells=h_cells, a_sti_radii=SPOT_PACK['a_sti_radii'], radius_key_aliases=SPOT_PACK['spot_cost_radius_key_aliases']))}")
    return backend


@dataclass
class _TrainBindCtx:
    """Per-task builder context during :func:`open_session`."""

    model_backend: ModelBackend
    dev: str
    delta_ms: float
    delta_ms_pre: float
    gt_amp: float
    sim_dtype: torch.dtype = SIM_DTYPE
    part_cost_scales: Optional[Dict[str, float]] = None
    spot_bright_sti_opts: Optional[dict] = None
    spot_dark_sti_opts: Optional[dict] = None
    moving_bar_bright_sti_opts: Optional[dict] = None
    moving_bar_dark_sti_opts: Optional[dict] = None
    filter: str = "none"
    spot_gt_mode: str = "all"
    cost_interval_ms: float = TRAIN_OPTIMIZATION['cost_interval_ms']
    cost_ms: Optional[dict] = None


def _moving_bar_waveform_mse_enabled(part_cost_scales: Optional[dict], pack_name: str) -> bool:
    """True if PD or ND waveform MSE scale is non-zero for ``pack_name``."""
    w = expand_part_cost_scale_dict(part_cost_scales or {})
    return any(
        float(w.get(moving_bar_cost_part_key(pack_name, lab), 1.0)) != 0.0
        for lab in PD_ND_LABELS
    )


def _moving_bar_contrast_opts(ctx: _TrainBindCtx, contrast: str) -> dict:
    if contrast == "bright":
        raw = ctx.moving_bar_bright_sti_opts
    elif contrast == "dark":
        raw = ctx.moving_bar_dark_sti_opts
    else:
        raise ValueError(f"unknown moving-bar contrast {contrast!r}")
    if raw:
        return dict(raw)
    _nc = resolve_filter_branches(NETWORK_CONSTRUCTION, filter=ctx.filter)
    _si = resolve_filter_branches(STI_TIMING, filter=ctx.filter)
    return build_moving_bar_sti_opts(
        contrast,
        i_baseline_moving_bar=_nc['i_baseline'],
        i_moving_bar=_nc['i_bright'] if contrast == "bright" else _nc['i_dark'],
        ms_pre=_si['ms_pre'],
        delta_ms=ctx.delta_ms,
        delta_ms_pre=ctx.delta_ms_pre,
        multi_bar=MOVING_BAR_INPUT['multi_bar'],
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


def _build_network_moving_bar_pack(
    ctx: _TrainBindCtx,
    connectome,
    *,
    pack_name: str,
    contrast: str,
):
    dev = ctx.dev or active_device()
    opts = _moving_bar_contrast_opts(ctx, contrast)
    if "cost_radius" in opts:
        cost_radius = normalize_cost_radius(opts["cost_radius"])
    else:
        network_radius = int(connectome.meta.get("radius", -1))
        network_cost_radius = -1 if network_radius <= 0 else network_radius - 1
        cost_radius = normalize_cost_radius(network_cost_radius)
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
        contrasts=(contrast,),
        gt_cells=gt_cells_from_opts(opts),
        multi_bar=bool(opts.get("multi_bar", MOVING_BAR_INPUT['multi_bar'])),
        waveform_mse=_moving_bar_waveform_mse_enabled(ctx.part_cost_scales, pack_name),
    )
    peak_key = _MOVING_BAR_I_KEY[contrast]
    build_kw[peak_key] = opts[peak_key]
    T = build_moving_bar_gt(**build_kw)
    sti_opts = enrich_moving_bar_sti_opts(opts, T.info, cost_radius=cost_radius)
    pack = Pack(
        name=pack_name,
        i_sti=T.i_sti,
        gts=T.gts,
        power=T.power,
        cost_scales=T.cost_scales,
        entry_batches=T.entry_batches,
        entry_nodes=T.entry_nodes,
        cost_t0s=T.cost_t0s,
        cost_radius=cost_radius,
        cost_pd_nds=T.cost_pd_nds,
        dsi_pos_entries=T.dsi_pos_entries,
        dsi_neg_entries=T.dsi_neg_entries,
        dsi_pos_ptr=T.dsi_pos_ptr,
        dsi_neg_ptr=T.dsi_neg_ptr,
        dsi_gts=T.dsi_gts,
        dsi_scales=T.dsi_scales,
        dsi_power=T.dsi_power,
        waveform_mse=bool(T.info["waveform_mse"]),
    )
    coltag = _cost_radius_hex_coltag(cost_radius, T.info["n_cost_hexes"])
    tag = (
        f"moving-bar {contrast} (B={T.n_batch} stis, "
        f"{T.info['n_cost']} cost nodes, {coltag})"
    )
    return pack, sti_opts, tag


def _spot_cost_time_idx_and_mask(opts, entry_radii, *, ctx: _TrainBindCtx, device, sim_dtype):
    """Union ``cost_time_indices``; ``cost_time_mask`` when radii differ.

    ``ctx.cost_ms[r]`` overwrites ``ctx.cost_interval_ms`` grid for that radius.
    """
    n = int(entry_radii.shape[0])
    if n == 0:
        return None, None
    timing = resolve_sti_timing(opts)
    delta_ms = timing.delta_ms
    post = timing.n_t_gt - timing.t_onset
    overrides = expand_cost_ms_dict(
        cost_ms=ctx.cost_ms, aliases=SPOT_PACK['spot_cost_radius_key_aliases'],
    )
    rad = np.round(entry_radii.detach().cpu().numpy().astype(float), 6)
    radii = {float(r) for r in rad.tolist()}
    grid = None
    if any(r not in overrides for r in radii):
        interval_ms = float(ctx.cost_interval_ms)
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
    cost_time_indices = torch.tensor(union, dtype=torch.long, device=device)
    union_set = set(union)
    if all(ts == union_set for ts in radius_ts.values()):
        return cost_time_indices, None
    slot_from_t = {t: j for j, t in enumerate(union)}
    mask = torch.zeros(n, len(union), dtype=sim_dtype, device=device)
    for i, r in enumerate(rad.tolist()):
        for t in radius_ts[float(r)]:
            mask[i, slot_from_t[t]] = 1.0
    return cost_time_indices, mask


def _build_network_spot_task(
    ctx: _TrainBindCtx,
    connectome,
    *,
    contrast: str,
    gt_amp: float,
) -> Tuple[Pack, dict, str]:
    if contrast not in SPOT_CONTRASTS:
        raise ValueError(f"spot contrast must be 'bright' or 'dark', got {contrast!r}")
    pack_name = f"spot_{contrast}"
    peak_key = _SPOT_I_KEY[contrast]
    ctx_opts = (
        ctx.spot_bright_sti_opts if contrast == "bright" else ctx.spot_dark_sti_opts
    )
    if not ctx_opts:
        raise ValueError(f"{pack_name} requires sti opts (from resolve_train_opts / CLI)")
    opts = normalize_sti_timing(dict(ctx_opts))
    timing = resolve_sti_timing(opts)
    cost_radius = normalize_cost_radius(opts.get("cost_radius"))
    shift_radius = int(opts["shift_radius"])
    spot_radius = float(opts["spot_radius"])
    multi_spot = bool(opts["multi_spot"])
    fully_inside = bool(opts["fully_inside"])
    dev = ctx.dev or active_device()
    t_onset = timing.t_onset
    n_t = timing.n_t
    i_spot = float(opts[peak_key])
    cost_radius_scales = resolve_spot_cost_radius_scale_defaults(
        spot_radius,
        scales=SPOT_PACK['spot_cost_radius_scale'],
        scales_radius1=SPOT_PACK['spot_cost_radius_scale_radius1'],
    )
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
        spot_cost_radius_scale=expand_spot_cost_r_s_dict(
            sti_opts=opts, aliases=SPOT_PACK['spot_cost_radius_key_aliases'],
        ),
        i_baseline_spot=float(opts[_SPOT_BASELINE_KEY]),
        i_bright_spot=i_spot if contrast == "bright" else float(opts.get("i_bright_spot", resolve_filter_branches(NETWORK_CONSTRUCTION['i_bright'], filter=ctx.filter))),
        i_dark_spot=i_spot if contrast == "dark" else float(opts.get("i_dark_spot", resolve_filter_branches(NETWORK_CONSTRUCTION['i_dark'], filter=ctx.filter))),
        contrast=contrast,
        ms_sti=timing.ms_sti,
        ms_response=timing.ms_response,
        gt_amp=gt_amp,
        delta_ms=timing.delta_ms,
        cost_radius_scales=cost_radius_scales,
        spot_cost_radii=SPOT_PACK['spot_cost_radii'],
        gt_cells=gt_cells_from_opts(opts),
        filter=str(ctx.filter),
        spot_gt_mode=str(ctx.spot_gt_mode),
    )
    cost_time_indices, cost_time_mask = _spot_cost_time_idx_and_mask(
        opts, T.entry_radii, ctx=ctx, device=dev, sim_dtype=ctx.sim_dtype,
    )
    sti_opts = dict(opts)
    # Replace center-only bake from build_spot_gt: center @1 in i_sti + a_sti_radius radii.
    i_baseline = float(opts[_SPOT_BASELINE_KEY])
    spot = resolve_spot(connectome, sti_opts=opts)
    batches = spot_sti_batches(spot)
    cost_r_s = expand_spot_cost_r_s_dict(
        sti_opts=opts, aliases=SPOT_PACK['spot_cost_radius_key_aliases'],
    )
    a_sti_radius_mask = build_a_sti_radius_mask(
        cost_r_s,
        cost_radius_scales=cost_radius_scales,
        a_sti_radii=SPOT_PACK['a_sti_radii'],
    )
    i_sti, sti_wave, sti_batches, sti_nodes, a_sti_radius_indices = build_spot_a_sti_radius_drive(
        connectome,
        batches,
        a_sti_radii=SPOT_PACK['a_sti_radii'],
        t_onset=int(t_onset),
        n_t=int(n_t),
        ms_sti=timing.ms_sti,
        delta_ms=timing.delta_ms,
        i_baseline=i_baseline,
        i_peak=i_spot,
        sim_dtype=ctx.sim_dtype,
        device=dev,
    )
    pack = Pack(
        name=pack_name,
        i_sti=i_sti,
        gts=T.gts,
        power=T.power,
        cost_scales=T.cost_scales,
        entry_batches=T.entry_batches,
        entry_nodes=T.entry_nodes,
        cost_t0s=None,
        cost_sti_us=T.entry_sti_us,
        cost_sti_vs=T.entry_sti_vs,
        cost_radius=cost_radius,
        entry_radii=T.entry_radii,
        cost_time_indices=cost_time_indices,
        cost_time_mask=cost_time_mask,
        waveform_mse=True,
        t_onset=int(t_onset),
        sti_wave=sti_wave,
        sti_batches=sti_batches,
        sti_nodes=sti_nodes,
        a_sti_radius_indices=a_sti_radius_indices,
        a_sti_radius_mask=torch.as_tensor(
            a_sti_radius_mask, dtype=ctx.sim_dtype, device=dev,
        ),
    )
    coltag = _cost_radius_hex_coltag(cost_radius, T.info["n_cost_hexes"])
    shifttag = f"{T.info['n_shifts']} shifts"
    tag = (
        f"{pack_name} (B={T.n_batch} stis [{T.info['n_centers']} centers simultaneous "
        f"x {shifttag}], {T.info['n_cost']} cost nodes, {coltag})"
    )
    return pack, sti_opts, tag


NETWORK_TASK_BUILDERS = {
    "spot_bright": lambda ctx, connectome, gt_amp: _build_network_spot_task(
        ctx, connectome, contrast="bright", gt_amp=gt_amp,
    ),
    "spot_dark": lambda ctx, connectome, gt_amp: _build_network_spot_task(
        ctx, connectome, contrast="dark", gt_amp=gt_amp,
    ),
    "moving_bar_bright": lambda ctx, connectome, gt_amp: _build_network_moving_bar_pack(
        ctx, connectome, pack_name="moving_bar_bright", contrast="bright",
    ),
    "moving_bar_dark": lambda ctx, connectome, gt_amp: _build_network_moving_bar_pack(
        ctx, connectome, pack_name="moving_bar_dark", contrast="dark",
    ),
}


def resolve_i_sti_paradigm(name):
    """Map one ``--i-sti`` task token to ``spot`` or ``moving_bar``."""
    if name not in CLI_TASK_NAMES:
        raise ValueError(
            f"unknown task {name!r} in --i-sti "
            f"(expected {'|'.join(CLI_TASK_NAMES)})",
        )
    if name in TASK_ALIASES:
        return name
    if name in SPOT_TASKS:
        return "spot"
    return "moving_bar"


def _i_sti_sidecar_keys(task_name):
    if task_name in SPOT_TASKS:
        baseline_key = _SPOT_BASELINE_KEY
        peak_keys = _SPOT_I_KEY
    else:
        baseline_key = _MOVING_BAR_BASELINE_KEY
        peak_keys = _MOVING_BAR_I_KEY
    contrast = "bright" if task_name.endswith("_bright") else "dark"
    return baseline_key, peak_keys[contrast], contrast


def apply_i_sti(opts, task_name, i_sti):
    """Merge per-paradigm CLI ``--i-sti`` overrides into sti opts."""
    if not i_sti:
        return opts
    paradigm = resolve_i_sti_paradigm(task_name)
    overrides = i_sti.get(paradigm)
    if not overrides:
        return opts
    out = dict(opts or {})
    baseline_key, peak_key, contrast = _i_sti_sidecar_keys(task_name)
    allowed = TASK_I_FIELDS[task_name]
    merged = {}
    if "baseline" in overrides:
        merged[baseline_key] = overrides["baseline"]
    if contrast in overrides:
        merged[peak_key] = overrides[contrast]
    for key, val in merged.items():
        if key not in allowed:
            raise ValueError(f"{key!r} not valid for task {task_name!r}")
        out[key] = val
    return out


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


_STI_TRAIN_OPT_SPECS = (
    ("spot_bright", "spot_bright_sti_opts"),
    ("spot_dark", "spot_dark_sti_opts"),
    ("moving_bar_bright", "moving_bar_bright_sti_opts"),
    ("moving_bar_dark", "moving_bar_dark_sti_opts"),
)

_STI_OPTS_BY_TASK = {
    "spot_bright": SPOT_BRIGHT_STI_OPTS,
    "spot_dark": SPOT_DARK_STI_OPTS,
    "moving_bar_bright": MOVING_BAR_BRIGHT_STI_OPTS,
    "moving_bar_dark": MOVING_BAR_DARK_STI_OPTS,
}


def _resolve_sti_opts(
    opts,
    task_name,
    *,
    cost_radius_by_task,
    shift_radius,
    spot_radius,
    multi_spot,
    fully_inside,
    spot_cost_radius_scale,
    i_sti,
):
    raw = dict(_STI_OPTS_BY_TASK.get(task_name, {}))
    raw.update(opts or {})
    if task_name in SPOT_TASKS:
        contrast = "bright" if task_name == "spot_bright" else "dark"
        peak_key = _SPOT_I_KEY[contrast]
        out = build_spot_sti_opts(
            contrast,
            i_baseline_spot=raw[_SPOT_BASELINE_KEY],
            i_spot=raw[peak_key],
            ms_pre=raw["ms_pre"],
            ms_response=raw["ms_response"],
            ms_post=raw["ms_post"],
            delta_ms=raw["delta_ms"],
            delta_ms_pre=raw["delta_ms_pre"],
            shift_radius=(
                shift_radius if shift_radius is not None else raw["shift_radius"]
            ),
            spot_radius=(
                spot_radius if spot_radius is not None else raw["spot_radius"]
            ),
            multi_spot=(
                multi_spot if multi_spot is not None else raw["multi_spot"]
            ),
            fully_inside=(
                fully_inside if fully_inside is not None else raw["fully_inside"]
            ),
            ms_sti=raw["ms_sti"],
            gt_cells=raw.get("gt_cells"),
        )
    elif task_name in ("moving_bar_bright", "moving_bar_dark"):
        contrast = "bright" if task_name == "moving_bar_bright" else "dark"
        peak_key = _MOVING_BAR_I_KEY[contrast]
        out = build_moving_bar_sti_opts(
            contrast,
            i_baseline_moving_bar=raw[_MOVING_BAR_BASELINE_KEY],
            i_moving_bar=raw[peak_key],
            ms_pre=raw["ms_pre"],
            delta_ms=raw["delta_ms"],
            delta_ms_pre=raw["delta_ms_pre"],
            multi_bar=raw["multi_bar"],
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
        out["shift_radius"] = shift_radius
        out["spot_radius"] = spot_radius
        out["multi_spot"] = multi_spot
        out["fully_inside"] = fully_inside
        if spot_cost_radius_scale is not None:
            out["spot_cost_radius_scale"] = {
                str(k): float(v) for k, v in spot_cost_radius_scale.items()
            }
    return apply_i_sti(out, task_name, i_sti)


def resolve_train_opts(
    backend="network",
    tasks=None,
    part_cost_scales=None,
    pack_overrides=None,
    sequential=None,
    cost_radius_by_task=None,
    shift_radius=None,
    spot_radius=None,
    multi_spot=SPOT_INPUT['multi_spot'],
    fully_inside=SPOT_INPUT['fully_inside'],
    spot_cost_radius_scale=None,
    i_sti=None,
    cost_norm=TRAIN_OPTIMIZATION['cost_norm'],
    cost_interval_ms=TRAIN_OPTIMIZATION['cost_interval_ms'],
    cost_ms=None,
    moving_bar_bright_sti_opts=None,
    moving_bar_dark_sti_opts=None,
    spot_bright_sti_opts=None,
    spot_dark_sti_opts=None,
    network_json=None,
    network=None,
    param_modes=None,
    param_init=None,
    syn_mode=NEURON_SCHEMA['syn_mode'],
    dev=None,
    packs=None,
    euler=NEURON_PARAM['euler'],
    pre_steady=None,
    pre_steady_iters=TRAIN_OPTIMIZATION['pre_steady_iters'],
    pre_steady_damp=TRAIN_OPTIMIZATION['pre_steady_damp'],
    fp=TRAIN_SESSION['fp'],
    pre_grad=NEURON_FORWARD['pre_grad'],
    val_from=None,
    filter=NEURON_FILTER['filter'],
    spot_gt_mode=SPOT_PACK['spot_gt_mode'],
):
    """Canonical train opts for :func:`open_session` (network backend)."""
    if backend != "network":
        raise ValueError(f"backend must be 'network', got {backend!r}")
    if network is None and network_json is None:
        raise ValueError("resolve_train_opts requires network or network_json")
    fp = int(fp)
    if fp not in (16, 32, 64):
        raise ValueError(f"fp must be 16, 32, or 64; got {fp!r}")
    filter = expand_filter(filter)
    if pre_steady is None:
        pre_steady = TRAIN_OPTIMIZATION['pre_steady']
    pre_steady_iters = int(pre_steady_iters)
    pre_steady_damp = float(pre_steady_damp)
    if pre_steady_iters < 1:
        raise ValueError(f"pre_steady_iters must be >= 1; got {pre_steady_iters}")
    if not (0.0 < pre_steady_damp <= 1.0):
        raise ValueError(
            f"pre_steady_damp must be in (0, 1]; got {pre_steady_damp}"
        )
    val_from = resolve_val_from(val_from=val_from)
    val_from_opts = {"val_from": val_from}
    if filter != "ca":
        if val_from_enabled(val_from_opts, "v_th_ca") or val_from_enabled(val_from_opts, "a_ca"):
            raise ValueError(
                "--val-from v_th_ca / a_ca require --filter ca "
                f"(got filter={filter!r})"
            )
        if param_modes:
            param_modes = {
                k: v for k, v in param_modes.items()
                if k not in ("v_th_ca", "a_ca", "tau_ca")
            } or None
    param_modes = resolve_param_modes(param_modes, val_from_opts)
    tl = resolve_tasks(tasks)
    if spot_radius is None:
        spot_radius = SPOT_INPUT['spot_radius']
    if shift_radius is None:
        shift_radius = SPOT_INPUT['shift_radius']
    raw_by_name = {
        "spot_bright": spot_bright_sti_opts,
        "spot_dark": spot_dark_sti_opts,
        "moving_bar_bright": moving_bar_bright_sti_opts,
        "moving_bar_dark": moving_bar_dark_sti_opts,
    }
    finalize_kw = dict(
        cost_radius_by_task=cost_radius_by_task,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_scale=spot_cost_radius_scale,
        i_sti=i_sti,
    )
    sti_opts = {}
    for tname, opts_key in _STI_TRAIN_OPT_SPECS:
        raw = raw_by_name[tname]
        if tname not in tl and raw is None:
            sti_opts[opts_key] = None
            continue
        sti_opts[opts_key] = _resolve_sti_opts(
            raw,
            tname,
            **finalize_kw,
        )
    opts = copy.deepcopy(TRAIN_OPTS)
    opts.update({
        "backend": "network",
        "tasks": tl,
        "part_cost_scales": expand_part_cost_scale_dict(part_cost_scales or {}),
        "cost_norm": cost_norm,
        "cost_interval_ms": cost_interval_ms,
        "cost_ms": copy.deepcopy(
            cost_ms if cost_ms is not None else TRAIN_OPTIMIZATION['cost_ms']
        ),
        "pre_steady": pre_steady,
        "pre_steady_iters": pre_steady_iters,
        "pre_steady_damp": pre_steady_damp,
        "sequential": sequential,
        **sti_opts,
        "euler": euler,
        "syn_mode": syn_mode,
        "pre_grad": pre_grad,
        "val_from": copy.deepcopy(val_from),
        "filter": filter,
        "spot_gt_mode": spot_gt_mode,
        "fp": fp,
        "network": network,
        "network_json": str(network_json) if network_json is not None else None,
        "dev": dev,
    })
    if pack_overrides is not None:
        opts["pack_overrides"] = pack_overrides
    if packs is not None:
        opts["packs"] = packs
    if param_modes is not None:
        opts["param_modes"] = param_modes
    if param_init:
        opts["param_init"] = [
            [name, node, float(val)] for name, node, val in param_init
        ]
    return opts


def _cost_ms_sidecar(cost_ms) -> dict:
    """JSON sidecar: radius keys as strings, ms lists as floats."""
    out: dict = {}
    for key, vals in (cost_ms or {}).items():
        ms_list = list(vals) if isinstance(vals, (list, tuple)) else [vals]
        out[str(float(key))] = [float(x) for x in ms_list]
    return out


def _train_opts_for_sidecar(opts, tasks, resolved_sti, sequential_bool) -> dict:
    """Build JSON-serializable train_opts record (network backend only)."""
    def _sti(key):
        got = resolved_sti.get(key)
        return got if got is not None else opts.get(key)

    record = {
        "backend": "network",
        "tasks": list(tasks),
        "part_cost_scales": {
            str(k): float(v) for k, v in (opts.get("part_cost_scales") or {}).items()
        },
        "cost_norm": opts.get("cost_norm", TRAIN_OPTIMIZATION['cost_norm']),
        "cost_interval_ms": float(
            opts.get("cost_interval_ms", TRAIN_OPTIMIZATION['cost_interval_ms'])
        ),
        "cost_ms": _cost_ms_sidecar(
            opts.get("cost_ms", TRAIN_OPTIMIZATION['cost_ms'])
        ),
        "pre_steady": opts.get("pre_steady", TRAIN_OPTIMIZATION['pre_steady']),
        "pre_steady_iters": int(opts.get("pre_steady_iters", TRAIN_OPTIMIZATION['pre_steady_iters'])),
        "pre_steady_damp": float(opts.get("pre_steady_damp", TRAIN_OPTIMIZATION['pre_steady_damp'])),
        "sequential": sequential_bool,
        "network_json": str(opts["network_json"]),
        "spot_bright_sti_opts": _sti("spot_bright_sti_opts"),
        "spot_dark_sti_opts": _sti("spot_dark_sti_opts"),
        "moving_bar_bright_sti_opts": _sti("moving_bar_bright_sti_opts"),
        "moving_bar_dark_sti_opts": _sti("moving_bar_dark_sti_opts"),
    }
    overrides = opts.get("pack_overrides")
    if overrides:
        record["pack_overrides"] = overrides
    if opts.get("param_modes"):
        record["param_modes"] = opts["param_modes"]
    if opts.get("param_init"):
        record["param_init"] = opts["param_init"]
    if "euler" not in opts:
        raise ValueError("train opts require euler (implicit|explicit)")
    record["euler"] = opts["euler"]
    record["syn_mode"] = opts.get("syn_mode", NEURON_SCHEMA['syn_mode'])
    record["pre_grad"] = opts.get("pre_grad", NEURON_FORWARD['pre_grad'])
    record["val_from"] = copy.deepcopy(opts.get("val_from", VAL_FROM))
    record["filter"] = expand_filter(opts.get("filter", NEURON_FILTER['filter']))
    record["spot_gt_mode"] = opts.get("spot_gt_mode", SPOT_PACK['spot_gt_mode'])
    record["fp"] = int(opts.get("fp", TRAIN_SESSION['fp']))
    return record


def resolve_schema(model, model_backend, schema, train_opts_record):
    if schema is not None:
        return list(schema)
    filter = NEURON_FILTER['filter']
    if train_opts_record:
        filter = expand_filter(train_opts_record.get("filter", NEURON_FILTER['filter']))
    syn_mode = normalize_syn_mode(resolve_filter_branches(
        (train_opts_record or {}).get("syn_mode", NEURON_SCHEMA['syn_mode']),
        filter=filter,
    ))
    kw = dict(
        syn_mode=syn_mode,
        optimizable=NEURON_SCHEMA['optimizable'],
        h_cells=NEURON_SCHEMA['h_cells'],
        filter=filter,
        a_sti_radii=SPOT_PACK['a_sti_radii'],
        radius_key_aliases=SPOT_PACK['spot_cost_radius_key_aliases'],
    )
    base = build_schema(model, model_backend, **kw)
    if not train_opts_record:
        return base
    modes = train_opts_record.get("param_modes")
    if modes:
        base = apply_param_modes(
            base, modes, lambda segment: node_names_for_segment(segment, model_backend),
        )
    return base


def _build_session(
    model_backend: ModelBackend,
    model: str,
    tasks: List[str],
    packs: Dict[str, Pack],
    *,
    delta_ms: float,
    delta_ms_pre: float,
    gt_amp: float,
    part_cost_scales=None,
    sequential=None,
    dev=None,
    train_opts_record=None,
    schema: Optional[list] = None,
    sim_dtype=SIM_DTYPE,
) -> TrainSession:
    dev_ref = dev or active_device()
    seq = False if sequential is None else bool(sequential)
    filter = expand_filter(
        (train_opts_record or {}).get("filter", NEURON_FILTER['filter'])
    )
    _np = resolve_filter_branches(NEURON_PARAM, filter=filter)
    if train_opts_record is not None:
        train_opts_record["model"] = model
        train_opts_record["sequential"] = bool(seq)
    if train_opts_record is None or "euler" not in train_opts_record:
        raise ValueError("train opts require euler (implicit|explicit)")
    euler = expand_euler(resolve_filter_branches(train_opts_record["euler"], filter=filter))
    pre_steady = expand_pre_steady(resolve_filter_branches(
        train_opts_record.get("pre_steady", TRAIN_OPTIMIZATION['pre_steady']),
        filter=filter,
    ))
    train_opts_record["pre_steady"] = pre_steady
    pre_steady_iters = int(
        train_opts_record.get("pre_steady_iters", TRAIN_OPTIMIZATION['pre_steady_iters'])
    )
    pre_steady_damp = float(
        train_opts_record.get("pre_steady_damp", TRAIN_OPTIMIZATION['pre_steady_damp'])
    )
    train_opts_record["pre_steady_iters"] = pre_steady_iters
    train_opts_record["pre_steady_damp"] = pre_steady_damp
    sch = resolve_schema(
        model, model_backend, schema, train_opts_record,
    )
    param_init = (train_opts_record or {}).get("param_init")
    if param_init:
        edits = [(row[0], row[1], float(row[2])) for row in param_init]
        sch = apply_param_init(sch, model_backend, edits)
    train_opts_record["param_modes"] = schema_param_modes_record(
        sch, lambda segment: node_names_for_segment(segment, model_backend),
    )
    sch = attach_param_carry(sch)
    session = TrainSession(
        backend=model_backend,
        model=model,
        schema=tuple(sch),
        packs=dict(packs),
        tasks=tuple(tasks),
        part_cost_scales=expand_part_cost_scale_dict(part_cost_scales),
        sequential=bool(seq),
        device=dev_ref,
        delta_ms=float(delta_ms),
        delta_ms_pre=float(delta_ms_pre),
        cap=float(_np['cap']),
        g_leak=float(_np['g_leak']),
        e_exc=float(_np['e_exc']),
        e_inh=float(_np['e_inh']),
        e_h=float(_np['e_h']),
        h_g_max=float(_np['h_g_max']),
        gt_amp=float(gt_amp),
        state_clamp=float(_np['state_clamp']),
        a_syn_exc=float(_np['a_syn_exc']),
        a_syn_inh=float(_np['a_syn_inh']),
        euler=euler,
        pre_steady=pre_steady,
        pre_steady_iters=pre_steady_iters,
        pre_steady_damp=pre_steady_damp,
        sim_dtype=sim_dtype,
        train_opts=train_opts_record,
    )
    return session


def open_session(
    opts: dict,
    model: str,
    *,
    schema: Optional[list] = None,
    model_backend: Optional[ModelBackend] = None,
) -> TrainSession:
    """Build a :class:`TrainSession` from canonical train opts."""
    opts = dict(opts)
    filter = expand_filter(opts.get("filter", NEURON_FILTER['filter']))
    resolved = resolve_filter_branches(
        {"opts": opts, "gt_amp": NEURON_PARAM['gt_amp'], "_np": NEURON_PARAM},
        filter=filter,
    )
    opts = resolved["opts"]
    gt_amp = float(resolved["gt_amp"])
    _np = resolved["_np"]
    backend_name = str(opts.get("backend", "network"))
    if backend_name != "network":
        raise ValueError(f"backend must be 'network', got {backend_name!r}")
    tasks = resolve_tasks(opts.get("tasks"))
    bad = [t for t in tasks if t not in VALID_TASKS]
    if bad:
        raise ValueError(f"unknown task(s) {bad!r} (expected {'|'.join(CLI_TASK_NAMES)})")
    dev = opts.get("dev") or active_device()
    sim_dtype = sim_dtype_from_fp(int(opts.get("fp", TRAIN_SESSION['fp'])))
    delta_ms = _sti_delta_ms(opts, "delta_ms")
    delta_ms_pre = _sti_delta_ms(opts, "delta_ms_pre")

    connectome = opts.get("network")
    syn_mode = normalize_syn_mode(opts.get("syn_mode", NEURON_SCHEMA['syn_mode']))
    if connectome is None:
        nj = opts.get("network_json")
        if not nj:
            raise ValueError("open_session(network) requires opts['network'] or network_json")
        connectome = load_network(
            nj, device=dev,
            a_syn_exc=float(_np['a_syn_exc']),
            a_syn_inh=float(_np['a_syn_inh']),
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
        part_cost_scales=opts.get("part_cost_scales"),
        spot_bright_sti_opts=opts.get("spot_bright_sti_opts"),
        spot_dark_sti_opts=opts.get("spot_dark_sti_opts"),
        moving_bar_bright_sti_opts=opts.get("moving_bar_bright_sti_opts"),
        moving_bar_dark_sti_opts=opts.get("moving_bar_dark_sti_opts"),
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        gt_amp=gt_amp,
        filter=filter,
        spot_gt_mode=expand_spot_gt_mode(opts.get("spot_gt_mode", SPOT_PACK['spot_gt_mode'])),
        cost_interval_ms=float(
            opts.get("cost_interval_ms", TRAIN_OPTIMIZATION['cost_interval_ms'])
        ),
        cost_ms=dict(opts.get("cost_ms") or {}),
    )
    packs = {}
    pack_overrides = opts.get("pack_overrides") or {}
    resolved_sti = {}
    for tname in tasks:
        pack, sti_opts, _tag = NETWORK_TASK_BUILDERS[tname](
            ctx, connectome, gt_amp=gt_amp,
        )
        if tname in pack_overrides:
            pack = apply_pack_override(pack, pack_overrides[tname], model_backend)
        packs[tname] = pack
        resolved_sti[f"{tname}_sti_opts"] = sti_opts
    record = _train_opts_for_sidecar(
        opts, tasks, resolved_sti, bool(opts.get("sequential")),
    )
    return _build_session(
        model_backend, model, tasks, packs,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        gt_amp=gt_amp,
        part_cost_scales=opts.get("part_cost_scales"),
        sequential=opts.get("sequential"),
        dev=dev,
        train_opts_record=record,
        schema=schema,
        sim_dtype=sim_dtype,
    )


def resolve_filter_branches(val, *, filter: str):
    branch = "ca" if str(filter) == "ca" else "v"
    if isinstance(val, dict):
        keys = set(val)
        if keys and keys <= {"v", "ca"}:
            if branch not in val:
                raise KeyError(
                    f"filter-branch dict missing {branch!r} (keys={sorted(val)!r})"
                )
            return resolve_filter_branches(val[branch], filter=filter)
        return {
            key: resolve_filter_branches(got, filter=filter)
            for key, got in val.items()
        }
    if isinstance(val, list):
        return [resolve_filter_branches(got, filter=filter) for got in val]
    if isinstance(val, tuple):
        return tuple(resolve_filter_branches(got, filter=filter) for got in val)
    return val


def _sti_delta_ms(opts: dict, key: str) -> float:
    """``delta_ms`` / ``delta_ms_pre`` from sti opts (required)."""
    for _tname, opts_key in _STI_TRAIN_OPT_SPECS:
        so = opts.get(opts_key)
        if isinstance(so, dict) and so.get(key) is not None:
            dt = float(so[key])
            if dt <= 0:
                raise ValueError(f"sti opts {key} must be > 0, got {dt}")
            return dt
    raise ValueError(
        f"train opts require {key} in a sti opts dict "
        f"(one of {[k for _, k in _STI_TRAIN_OPT_SPECS]})"
    )


def resolve_session(opts: dict, model: str | None = None, **kwargs) -> TrainSession:
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
    sim_dtype = sim_dtype_from_fp(int(opts.get("fp", TRAIN_SESSION['fp'])))
    _filter = expand_filter(opts.get("filter", NEURON_FILTER['filter']))
    _np = resolve_filter_branches(NEURON_PARAM, filter=_filter)
    syn_mode = normalize_syn_mode(resolve_filter_branches(
        opts.get("syn_mode", NEURON_SCHEMA['syn_mode']), filter=_filter,
    ))
    mb = load_network_backend(
        nj, dev=opts.get("dev") or active_device(), sim_dtype=sim_dtype,
        syn_mode=syn_mode,
        a_syn_exc=float(_np['a_syn_exc']),
        a_syn_inh=float(_np['a_syn_inh']),
    )
    opts["network"] = mb.network
    opts["syn_mode"] = syn_mode
    kwargs.setdefault("model_backend", mb)
    return open_session({**opts, "backend": "network"}, model, **kwargs)


def session_from_outdir(
    outdir: str,
    model: str | None = None,
) -> TrainSession:
    """Load ``train_opts.json`` from a run folder and return a ready session."""
    opts_path = os.path.join(run_data_dir(os.path.abspath(outdir)), TRAIN_OPTS_FILE)
    if not os.path.isfile(opts_path):
        raise FileNotFoundError(f"missing {opts_path}")
    with open(opts_path) as f:
        opts = json.load(f)
    return resolve_session(opts, model)
