# -*- coding: utf-8 -*-
"""Session types and assembly: ``Pack`` / ``TrainSession`` + open helpers.

Owns sti-opts finalisation (CLI tokens -> per-task sidecar dicts),
connectome loading, and the per-task×contrast ``Pack`` builders. The
builders wrap the neutral gt dataclasses from ``task`` (which sit below
``train`` in the import graph) and stamp the cross-cutting pack cost controls:

* spot: sparse ``cost_ts`` / optional ``cost_time_mask`` (#4; ``cost_ms``
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

``task`` ∈ {spot, moving_bar} and ``contrast`` ∈ {bright, dark} are independent;
packs are ``packs[task][contrast]``.

Branch-resolution contract (must keep):

* Do not add parameter-specific helper parsers/casters in pre-session stages.
* All ``{v,ca}`` branch selection must be unified through
  :func:`resolve_filter_branches`.
* Before :func:`open_session`, keep branch-capable values raw; only perform
  ``float``/``int``/``bool`` casting after branch resolution for those
  branch-capable values.
"""
from __future__ import annotations

from const_default import (
    I_STI,
    MOVING_BAR_INPUT,
    MOVING_BAR_STI_OPTS,
    NEURON_FILTER,
    NEURON_FORWARD,
    NEURON_CONST,
    NEURON_SCHEMA,
    SPOT_INPUT,
    SPOT_PACK,
    SPOT_STI_OPTS,
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
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from neuron.param import t_from_ms
from neuron import (
    build_schema,
    expand_euler,
)

from train.config import (
    CONTRASTS,
    PD_ND_LABELS,
    TASKS,
    cost_part_keys_from_pack,
    expand_part_cost_scale,
    expand_filter,
    expand_spot_gt_mode,
    expand_gt,
    expand_pre_steady,
    moving_bar_cost_part_key,
    parse_contrasts,
    parse_tasks,
    run_data_dir,
)
from train.param import (
    SIM_DTYPE,
    active_device,
    schema_with_params,
    schema_copy,
    schema_with_param_carry,
    resolve_param_modes,
    resolve_val_from,
    schema_n_z,
    param_modes_from_schema,
    sim_dtype_from_fp,
    val_from_enabled,
)

from task.spot.gt import expand_gt_cells as expand_spot_gt_cells
from task.spot.pack import (
    SPOT_CONTRASTS,
    build_spot_gt,
    expand_cost_ms,
    expand_spot_cost_radius_scale,
    build_spot_sti_opts,
    build_a_sti_radius_mask,
)
from task.spot.sti_geo import (
    resolve_spot,
    spot_sti_bs,
)
from task.spot.sti_spec import (
    build_spot_a_sti_radius_drive,
    standardize_sti_timing,
    resolve_sti_timing,
)
from task.moving_bar.gt import expand_gt_cells as expand_moving_bar_gt_cells
from task.moving_bar.pack import (
    build_moving_bar_gt,
    build_moving_bar_sti_opts,
)
from task.moving_bar.sti_spec import i_baseline_from_i_sti
from network.construction import (
    load_network, gt_cells_from_opts, standardize_cost_radius, Network,
)


@dataclass(frozen=True)
class Pack:
    """One train pack: task×contrast drive + entries + gts.

    Spot ``i_sti`` / ``gts`` time dims follow ``neuron`` / task
    timing. Moving bar uses ``COST_WINDOW`` and per-task ``n_t``.
    """

    task: str
    contrast: str
    i_sti: torch.Tensor  # (B, T, N)
    gts: torch.Tensor  # (n_cost, T')
    power: torch.Tensor  # scalar
    cost_scales: torch.Tensor  # (n_cost,)
    entry_bs: torch.Tensor  # (n_cost,)
    entry_nodes: torch.Tensor  # (n_cost,)
    cost_t0s: Optional[torch.Tensor] = None  # (n_cost,) absolute step for windowed readouts
    cost_radius: Optional[int] = None  # hex-disc radius for cost readouts
    entry_radii: Optional[torch.Tensor] = None  # (n_cost,) long hex-lattice radius per spot entry
    cost_sti_us: Optional[torch.Tensor] = None  # (n_cost,) sti anchor u per spot cost entry
    cost_sti_vs: Optional[torch.Tensor] = None  # (n_cost,) sti anchor v per spot cost entry
    cost_pd_nds: Optional[torch.Tensor] = None  # (n_cost,) long; 0=PD, 1=ND (moving_bar)
    dsi_pos_entries: Optional[torch.Tensor] = None  # flat cost entry (right|up)
    dsi_neg_entries: Optional[torch.Tensor] = None  # flat cost entry (left|down)
    dsi_pos_ptr: Optional[torch.Tensor] = None  # (n_dsi+1,) CSR
    dsi_neg_ptr: Optional[torch.Tensor] = None  # (n_dsi+1,) CSR
    dsi_gts: Optional[torch.Tensor] = None  # (n_dsi,)
    dsi_scales: Optional[torch.Tensor] = None  # (n_dsi,)
    dsi_power: Optional[torch.Tensor] = None  # scalar
    cost_ts: Optional[torch.Tensor] = None  # (n_sample,) sparse post-onset t
    cost_time_mask: Optional[torch.Tensor] = None  # (n_cost, n_sample) 0/1 per-radius
    waveform_mse: bool = True  # spot: True; moving bar: set at build
    t_onset: Optional[int] = None  # explicit onset; spot when ms_post extends i_sti past gt
    # Spot a_sti_radius: i_sti += a_sti_radius[radius] * sti_pulse on (sti_bs, sti_nodes).
    sti_pulse: Optional[torch.Tensor] = None  # (T,) (i_sti - i_baseline) * pulse(t)
    sti_bs: Optional[torch.Tensor] = None  # (n_contrib,) long
    sti_nodes: Optional[torch.Tensor] = None  # (n_contrib,) long
    a_sti_radius_idxs: Optional[torch.Tensor] = None  # (n_contrib,) long → a_sti_radius index
    # (n_a_sti_radius,) 0/1: cost-radius scale==0 forces that a_sti_radius to 0 in forward.
    a_sti_radius_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TrainSession:
    """Immutable runtime context for one train / plotting run.

    Model / synapse scalars are flat fields (injected from
    ``const_default`` at session open). ``delta_ms`` / ``delta_ms_pre`` come
    only from sti opts — never nested under Physics.
    """

    connectome: Network
    model: str
    schema: dict
    packs: Dict[str, Dict[str, Pack]]
    tasks: Tuple[str, ...]
    contrasts: Tuple[str, ...]
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
    v_clamp: float
    a_syn_exc: float
    a_syn_inh: float
    euler: str
    pre_steady: str = "solve"
    pre_steady_n_iter: int = 60
    pre_steady_damp: float = 1.0
    sim_dtype: torch.dtype = SIM_DTYPE
    train_opts: Optional[dict] = None

    def with_schema(self, schema) -> "TrainSession":
        return replace(self, schema=schema_copy(schema))

    def iter_packs(self) -> Iterator[Pack]:
        for task in self.tasks:
            for contrast in self.contrasts:
                yield self.packs[task][contrast]

    @property
    def primary_pack(self) -> Pack:
        return self.packs[self.tasks[0]][self.contrasts[0]]

    @property
    def n_t(self) -> int:
        i_sti = self.primary_pack.i_sti
        return int(i_sti.shape[1] if i_sti.dim() == 3 else i_sti.shape[0])

    def pack_i_sti(self, pack: Optional[Pack] = None) -> torch.Tensor:
        pack = pack or self.primary_pack
        i_sti = pack.i_sti
        if pack.task == "spot" and i_sti.dim() == 3 and int(i_sti.shape[0]) == 1:
            i_sti = i_sti.squeeze(0)
        return i_sti


def resolve_cell_idxs(cells, connectome: Network):
    """Map cells to idxs in the network vocabulary."""
    if connectome is None:
        raise ValueError("resolve_cell_idxs requires connectome")
    wanted = [str(n) for n in cells]
    vocab = list(connectome.cells)
    return [vocab.index(n) for n in wanted if n in vocab]


def load_train_connectome(
    network_json,
    device: Optional[str] = None,
    *,
    a_syn_exc: float,
    a_syn_inh: float,
    sim_dtype=SIM_DTYPE,
    syn_mode=NEURON_SCHEMA['syn_mode'],
    params=NEURON_SCHEMA['params'],
    h_cells=NEURON_SCHEMA['h_cells'],
) -> Network:
    """Load connectome ``Network`` for train (print summary)."""
    device = device or active_device()
    mode = syn_mode
    connectome = load_network(
        network_json, device=device,
        a_syn_exc=a_syn_exc, a_syn_inh=a_syn_inh,
        dtype=sim_dtype, syn_mode=mode,
    )
    print(f"network: {network_json}")
    print(f"  n_node={connectome.n_node}, n_cell={connectome.n_cell}, "
          f"n_pair={connectome.conn.n_pair}, n_edge={connectome.conn.n_edge}, "
          f"syn_mode={mode}, "
          f"n_z={schema_n_z(build_schema('borst', connectome, syn_mode=mode, params=params, h_cells=h_cells, a_sti_radii=SPOT_PACK['a_sti_radii']))}")
    return connectome


def _moving_bar_waveform_mse_enabled(
    part_cost_scales: Optional[dict], task: str, contrast: str,
) -> bool:
    """True if PD or ND waveform MSE scale is non-zero for ``task``×``contrast``."""
    scales = expand_part_cost_scale(part_cost_scales or {})
    return any(
        float(scales.get(moving_bar_cost_part_key(task, contrast, part), 1.0)) != 0.0
        for part in PD_ND_LABELS
    )


def _moving_bar_sti_opts(
    moving_bar_sti_opts: Optional[dict],
    *,
    filter: str,
    delta_ms: float,
    delta_ms_pre: float,
) -> dict:
    """Build moving-bar sti opts; CLI override takes precedence over filter default."""
    if moving_bar_sti_opts:
        return dict(moving_bar_sti_opts)
    sti_timing = resolve_filter_branches(STI_TIMING, filter=filter)
    return build_moving_bar_sti_opts(
        ms_pre=sti_timing['ms_pre'],
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        multi_bar=MOVING_BAR_INPUT['multi_bar'],
    )


def _cost_hex_label(cost_radius, n_cost_hex) -> str:
    radius_label = "all hexes" if cost_radius is None else f"radius={int(cost_radius)}"
    if isinstance(n_cost_hex, dict):
        hex_labels = ", ".join(
            f"b{int(b)}={int(n_hex)}"
            for b, n_hex in sorted(n_cost_hex.items())
        )
        return f"cost hexes per b [{hex_labels}], {radius_label}"
    return f"{int(n_cost_hex)} cost hexes, {radius_label}"


def _build_moving_bar_pack(
    connectome,
    *,
    contrast: str,
    gt_amp: float,
    device: str,
    sim_dtype: torch.dtype,
    i_sti: Dict[str, Dict[str, float]],
    part_cost_scales: Optional[Dict[str, float]],
    moving_bar_sti_opts: Optional[dict],
    filter: str,
    delta_ms: float,
    delta_ms_pre: float,
):
    task = "moving_bar"
    device = device or active_device()
    opts = _moving_bar_sti_opts(
        moving_bar_sti_opts,
        filter=filter,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
    )
    if "cost_radius" in opts:
        cost_radius = standardize_cost_radius(opts["cost_radius"])
    else:
        network_radius = int(connectome.meta.get("radius", -1))
        network_cost_radius = -1 if network_radius <= 0 else network_radius - 1
        cost_radius = standardize_cost_radius(network_cost_radius)
    T = build_moving_bar_gt(
        connectome=connectome,
        device=device,
        sim_dtype=sim_dtype,
        t_onset=t_from_ms(
            float(opts["ms_pre"]),
            delta_ms=float(opts["delta_ms_pre"]),
        ),
        delta_ms=float(opts["delta_ms"]),
        cost_radius=cost_radius,
        i_baseline=i_baseline_from_i_sti(i_sti, task),
        i_sti=float(i_sti[task][contrast]),
        contrasts=(contrast,),
        gt_cells=gt_cells_from_opts(opts),
        multi_bar=bool(opts.get("multi_bar", MOVING_BAR_INPUT['multi_bar'])),
        waveform_mse=_moving_bar_waveform_mse_enabled(
            part_cost_scales, task, contrast,
        ),
    )
    sti_opts = dict(opts)
    sti_opts["n_t"] = int(T.n_t)
    sti_opts["spec_tokens"] = list(T.spec_tokens)
    if cost_radius is not None:
        sti_opts["cost_radius"] = int(cost_radius)
    sti_opts["gt_cells"] = list(T.active_gts)
    pack = Pack(
        task=task,
        contrast=contrast,
        i_sti=T.i_sti,
        gts=T.gts,
        power=T.power,
        cost_scales=T.cost_scales,
        entry_bs=T.entry_bs,
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
        waveform_mse=bool(T.waveform_mse),
    )
    hex_label = _cost_hex_label(cost_radius, T.n_cost_hex)
    label = (
        f"moving-bar {contrast} (B={T.n_b} stis, "
        f"{int(T.entry_bs.shape[0])} cost nodes, {hex_label})"
    )
    return pack, sti_opts, label


def _spot_cost_ts_and_mask(
    opts, entry_radii, *,
    cost_ms: Optional[dict],
    cost_interval_ms: float,
    device: str,
    sim_dtype: torch.dtype,
):
    """Union ``cost_ts``; ``cost_time_mask`` when radii differ.

    ``cost_ms[radius]`` overwrites ``cost_interval_ms`` grid for that radius.
    """
    n = int(entry_radii.shape[0])
    if n == 0:
        return None, None
    timing = resolve_sti_timing(opts)
    delta_ms = timing.delta_ms
    post = timing.n_t_gt - timing.t_onset
    cost_ms_by_radius = expand_cost_ms(cost_ms=cost_ms)
    entry_radii = entry_radii.detach().cpu().numpy().astype(np.int64, copy=False)
    radii = {int(radius) for radius in entry_radii.tolist()}
    grid = None
    if any(radius not in cost_ms_by_radius for radius in radii):
        if cost_interval_ms <= 0:
            raise ValueError("cost_interval_ms must be > 0")
        if post <= 0:
            raise ValueError("spot post-onset window must be > 0 for cost_interval_ms")
        step = max(1, int(round(cost_interval_ms / delta_ms)))
        grid = [t * delta_ms for t in range(0, post, step)]
    radius_ts = {}
    for radius in radii:
        mss = cost_ms_by_radius[radius] if radius in cost_ms_by_radius else grid
        ts = set()
        for ms in mss:
            t = int(round(float(ms) / delta_ms))
            if t < 0 or t >= post:
                raise ValueError(
                    f"cost time {ms} ms post-onset t out of range [0,{post})"
                )
            ts.add(t)
        radius_ts[radius] = ts
    union = sorted({t for ts in radius_ts.values() for t in ts})
    cost_ts = torch.tensor(union, dtype=torch.long, device=device)
    union_set = set(union)
    if all(ts == union_set for ts in radius_ts.values()):
        return cost_ts, None
    union_t = {t: pos for pos, t in enumerate(union)}
    mask = torch.zeros(n, len(union), dtype=sim_dtype, device=device)
    for entry, radius in enumerate(entry_radii.tolist()):
        for t in radius_ts[int(radius)]:
            mask[entry, union_t[t]] = 1.0
    return cost_ts, mask


def _build_spot_pack(
    connectome,
    *,
    contrast: str,
    gt_amp: float,
    device: str,
    sim_dtype: torch.dtype,
    i_sti: Dict[str, Dict[str, float]],
    spot_sti_opts: Optional[dict],
    filter: str,
    spot_gt_mode: str,
    cost_ms: Optional[dict],
    cost_interval_ms: float,
) -> Tuple[Pack, dict, str]:
    if contrast not in SPOT_CONTRASTS:
        raise ValueError(f"spot contrast must be 'bright' or 'dark', got {contrast!r}")
    task = "spot"
    if not spot_sti_opts:
        raise ValueError("spot requires sti opts (from resolve_train_opts / CLI)")
    opts = standardize_sti_timing(dict(spot_sti_opts))
    timing = resolve_sti_timing(opts)
    cost_radius = standardize_cost_radius(opts.get("cost_radius"))
    shift_radius = int(opts["shift_radius"])
    spot_radius = float(opts["spot_radius"])
    multi_spot = bool(opts["multi_spot"])
    fully_inside = bool(opts["fully_inside"])
    device = device or active_device()
    t_onset = timing.t_onset
    n_t = timing.n_t
    i_baseline = i_baseline_from_i_sti(i_sti, task)
    # spot_radius == 1: fold radius=2 into radius=1 scales
    cost_radius_scales = dict(
        SPOT_PACK['spot_cost_radius_scale_radius1']
        if float(spot_radius) == 1
        else SPOT_PACK['spot_cost_radius_scale']
    )
    T = build_spot_gt(
        connectome,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        shift_radius=shift_radius,
        device=device,
        sim_dtype=sim_dtype,
        n_t=n_t,
        t_onset=t_onset,
        cost_radius=cost_radius,
        spot_cost_radius_scale=expand_spot_cost_radius_scale(sti_opts=opts),
        i_baseline=i_baseline,
        i_sti=float(i_sti[task][contrast]),
        contrast=contrast,
        ms_sti=timing.ms_sti,
        ms_response=timing.ms_response,
        gt_amp=gt_amp,
        delta_ms=timing.delta_ms,
        cost_radius_scales=cost_radius_scales,
        spot_cost_radii=SPOT_PACK['spot_cost_radii'],
        gt_cells=gt_cells_from_opts(opts),
        filter=str(filter),
        spot_gt_mode=str(spot_gt_mode),
    )
    cost_ts, cost_time_mask = _spot_cost_ts_and_mask(
        opts, T.entry_radii,
        cost_ms=cost_ms,
        cost_interval_ms=cost_interval_ms,
        device=device,
        sim_dtype=sim_dtype,
    )
    sti_opts = dict(opts)
    # Replace center-only bake from build_spot_gt: center @1 in i_sti + a_sti_radius radii.
    spot = resolve_spot(connectome, sti_opts=opts)
    spot_bs = spot_sti_bs(spot)
    spot_cost_radius_scale = expand_spot_cost_radius_scale(sti_opts=opts)
    a_sti_radius_mask = build_a_sti_radius_mask(
        spot_cost_radius_scale,
        cost_radius_scales=cost_radius_scales,
        a_sti_radii=SPOT_PACK['a_sti_radii'],
    )
    i_sti, sti_pulse, sti_bs, sti_nodes, a_sti_radius_idxs = build_spot_a_sti_radius_drive(
        connectome,
        spot_bs,
        a_sti_radii=SPOT_PACK['a_sti_radii'],
        t_onset=int(t_onset),
        n_t=int(n_t),
        ms_sti=timing.ms_sti,
        delta_ms=timing.delta_ms,
        i_baseline=i_baseline,
        i_sti=float(i_sti[task][contrast]),
        sim_dtype=sim_dtype,
        device=device,
    )
    pack = Pack(
        task=task,
        contrast=contrast,
        i_sti=i_sti,
        gts=T.gts,
        power=T.power,
        cost_scales=T.cost_scales,
        entry_bs=T.entry_bs,
        entry_nodes=T.entry_nodes,
        cost_t0s=None,
        cost_sti_us=T.entry_sti_us,
        cost_sti_vs=T.entry_sti_vs,
        cost_radius=cost_radius,
        entry_radii=T.entry_radii,
        cost_ts=cost_ts,
        cost_time_mask=cost_time_mask,
        waveform_mse=True,
        t_onset=int(t_onset),
        sti_pulse=sti_pulse,
        sti_bs=sti_bs,
        sti_nodes=sti_nodes,
        a_sti_radius_idxs=a_sti_radius_idxs,
        a_sti_radius_mask=torch.as_tensor(
            a_sti_radius_mask, dtype=sim_dtype, device=device,
        ),
    )
    hex_label = _cost_hex_label(cost_radius, T.n_cost_hex)
    shifts_label = f"{T.n_shift} shifts"
    label = (
        f"spot {contrast} (B={T.n_b} stis [{T.n_center} centers simultaneous "
        f"x {shifts_label}], {int(T.gts.shape[0])} cost nodes, {hex_label})"
    )
    return pack, sti_opts, label


_GT_CELLS_EXPAND = {
    "spot": expand_spot_gt_cells,
    "moving_bar": expand_moving_bar_gt_cells,
}


def resolve_gt_cells_by_task(by_task) -> Dict[str, List[str]]:
    """Map concrete tasks to final gt cell lists (task + cell aliases expanded)."""
    expanded = expand_gt(by_task or {})
    bad = [task for task in expanded if task not in TASKS]
    if bad:
        raise ValueError(
            f"unknown task(s) in --gt: {bad} "
            f"(expected {'|'.join(TASKS)})",
        )
    return {
        task: list(_GT_CELLS_EXPAND[task](cells))
        for task, cells in expanded.items()
    }


_STI_TRAIN_OPT_KEYS = (
    ("spot", "spot_sti_opts"),
    ("moving_bar", "moving_bar_sti_opts"),
)

_STI_OPTS_BY_TASK = {
    "spot": SPOT_STI_OPTS,
    "moving_bar": MOVING_BAR_STI_OPTS,
}


def _merge_i_sti(cli_i_sti=None) -> Dict[str, Dict[str, float]]:
    """Deep-copy ``I_STI`` defaults; overlay CLI ``i_sti[task][contrast]``."""
    out = {task: dict(vals) for task, vals in I_STI.items()}
    if not cli_i_sti:
        return out
    for task, by_contrast in cli_i_sti.items():
        task = str(task)
        if task not in out:
            raise ValueError(
                f"unknown task {task!r} in --i-sti "
                f"(expected {'|'.join(TASKS)})",
            )
        for contrast, val in dict(by_contrast).items():
            contrast = str(contrast)
            if contrast not in CONTRASTS:
                raise ValueError(
                    f"unknown contrast {contrast!r} in --i-sti for task {task!r}"
                )
            out[task][contrast] = float(val)
    return out


def _spot_resolve_sti_opts(raw, *, shift_radius, spot_radius, multi_spot, fully_inside, spot_cost_radius_scale, **_):
    out = build_spot_sti_opts(
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
    out["shift_radius"] = shift_radius
    out["spot_radius"] = spot_radius
    out["multi_spot"] = multi_spot
    out["fully_inside"] = fully_inside
    if spot_cost_radius_scale is not None:
        out["spot_cost_radius_scale"] = {
            str(radius): float(scale)
            for radius, scale in spot_cost_radius_scale.items()
        }
    return out


def _moving_bar_resolve_sti_opts(raw, **_):
    return build_moving_bar_sti_opts(
        ms_pre=raw["ms_pre"],
        delta_ms=raw["delta_ms"],
        delta_ms_pre=raw["delta_ms_pre"],
        multi_bar=raw["multi_bar"],
        gt_cells=raw.get("gt_cells"),
    )


_RESOLVE_STI_OPTS = {
    "spot": _spot_resolve_sti_opts,
    "moving_bar": _moving_bar_resolve_sti_opts,
}


def _resolve_sti_opts(
    opts,
    task,
    *,
    cost_radius_by_task,
    shift_radius,
    spot_radius,
    multi_spot,
    fully_inside,
    spot_cost_radius_scale,
):
    raw = dict(_STI_OPTS_BY_TASK.get(task, {}))
    raw.update(opts or {})
    out = _RESOLVE_STI_OPTS[task](
        raw,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_scale=spot_cost_radius_scale,
    )
    if cost_radius_by_task and task in cost_radius_by_task:
        out["cost_radius"] = int(cost_radius_by_task[task])
    elif "cost_radius" in out:
        if out["cost_radius"] is None:
            out.pop("cost_radius", None)
        else:
            out["cost_radius"] = int(out["cost_radius"])
    return out


def resolve_train_opts(
    backend="network",
    tasks=None,
    contrasts=None,
    part_cost_scales=None,
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
    moving_bar_sti_opts=None,
    spot_sti_opts=None,
    network_json=None,
    network=None,
    param_modes=None,
    param_init=None,
    param_clamps=None,
    param_jits=None,
    syn_mode=NEURON_SCHEMA['syn_mode'],
    device=None,
    packs=None,
    euler=NEURON_CONST['euler'],
    pre_steady=None,
    pre_steady_n_iter=TRAIN_OPTIMIZATION['pre_steady_n_iter'],
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
    pre_steady_n_iter = int(pre_steady_n_iter)
    pre_steady_damp = float(pre_steady_damp)
    if pre_steady_n_iter < 1:
        raise ValueError(f"pre_steady_n_iter must be >= 1; got {pre_steady_n_iter}")
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
                param: modes for param, modes in param_modes.items()
                if param not in ("v_th_ca", "a_ca", "tau_ca")
            } or None
    param_modes = resolve_param_modes(param_modes, val_from_opts)
    tasks = parse_tasks(tasks if tasks is not None else TRAIN_OPTS["tasks"])
    contrasts = parse_contrasts(
        contrasts if contrasts is not None else TRAIN_OPTS["contrasts"]
    )
    if spot_radius is None:
        spot_radius = SPOT_INPUT['spot_radius']
    if shift_radius is None:
        shift_radius = SPOT_INPUT['shift_radius']
    merged_i_sti = _merge_i_sti(i_sti)
    finalize_kwargs = dict(
        cost_radius_by_task=cost_radius_by_task,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_scale=spot_cost_radius_scale,
    )
    raw_by_task = {
        "spot": spot_sti_opts,
        "moving_bar": moving_bar_sti_opts,
    }
    sti_opts = {}
    for task, sti_opts_key in _STI_TRAIN_OPT_KEYS:
        raw = raw_by_task[task]
        if task not in tasks and raw is None:
            sti_opts[sti_opts_key] = None
            continue
        sti_opts[sti_opts_key] = _resolve_sti_opts(
            raw,
            task,
            **finalize_kwargs,
        )
    opts = copy.deepcopy(TRAIN_OPTS)
    opts.update({
        "backend": "network",
        "tasks": tasks,
        "contrasts": contrasts,
        "i_sti": merged_i_sti,
        "part_cost_scales": expand_part_cost_scale(part_cost_scales or {}),
        "cost_norm": cost_norm,
        "cost_interval_ms": cost_interval_ms,
        "cost_ms": copy.deepcopy(
            cost_ms if cost_ms is not None else TRAIN_OPTIMIZATION['cost_ms']
        ),
        "pre_steady": pre_steady,
        "pre_steady_n_iter": pre_steady_n_iter,
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
        "device": device,
    })
    if packs is not None:
        opts["packs"] = packs
    if param_modes is not None:
        opts["param_modes"] = param_modes
    if param_init:
        opts["param_init"] = [
            [param, node, float(val)] for param, node, val in param_init
        ]
    if param_clamps:
        opts["param_clamps"] = [
            [param, key, node, float(val)]
            for param, key, node, val in param_clamps
        ]
    if param_jits:
        opts["param_jits"] = [
            [param, node, float(val)] for param, node, val in param_jits
        ]
    return opts


def _cost_ms_sidecar(cost_ms) -> dict:
    """JSON sidecar: radii as strings, mss as floats."""
    out: dict = {}
    for radius, vals in (cost_ms or {}).items():
        mss = list(vals) if isinstance(vals, (list, tuple)) else [vals]
        out[str(int(radius))] = [float(x) for x in mss]
    return out


def _sidecar_train_opts(opts, tasks, contrasts, resolved_sti, sequential_bool) -> dict:
    """Build JSON-serializable train_opts (network backend only)."""
    def _sti(sti_opts_key):
        got = resolved_sti.get(sti_opts_key)
        return got if got is not None else opts.get(sti_opts_key)

    train_opts = {
        "backend": "network",
        "tasks": list(tasks),
        "contrasts": list(contrasts),
        "i_sti": {
            task: {contrast: float(val) for contrast, val in by_contrast.items()}
            for task, by_contrast in (opts.get("i_sti") or {}).items()
        },
        "part_cost_scales": {
            str(part_key): float(scale)
            for part_key, scale in (opts.get("part_cost_scales") or {}).items()
        },
        "cost_norm": opts.get("cost_norm", TRAIN_OPTIMIZATION['cost_norm']),
        "cost_interval_ms": float(
            opts.get("cost_interval_ms", TRAIN_OPTIMIZATION['cost_interval_ms'])
        ),
        "cost_ms": _cost_ms_sidecar(
            opts.get("cost_ms", TRAIN_OPTIMIZATION['cost_ms'])
        ),
        "pre_steady": opts.get("pre_steady", TRAIN_OPTIMIZATION['pre_steady']),
        "pre_steady_n_iter": int(opts.get("pre_steady_n_iter", TRAIN_OPTIMIZATION['pre_steady_n_iter'])),
        "pre_steady_damp": float(opts.get("pre_steady_damp", TRAIN_OPTIMIZATION['pre_steady_damp'])),
        "sequential": sequential_bool,
        "network_json": str(opts["network_json"]),
        "spot_sti_opts": _sti("spot_sti_opts"),
        "moving_bar_sti_opts": _sti("moving_bar_sti_opts"),
    }
    if opts.get("param_modes"):
        train_opts["param_modes"] = opts["param_modes"]
    if opts.get("param_init"):
        train_opts["param_init"] = opts["param_init"]
    if opts.get("param_clamps"):
        train_opts["param_clamps"] = opts["param_clamps"]
    if opts.get("param_jits"):
        train_opts["param_jits"] = opts["param_jits"]
    if "euler" not in opts:
        raise ValueError("train opts require euler (implicit|explicit)")
    train_opts["euler"] = opts["euler"]
    train_opts["syn_mode"] = opts.get("syn_mode", NEURON_SCHEMA['syn_mode'])
    train_opts["pre_grad"] = opts.get("pre_grad", NEURON_FORWARD['pre_grad'])
    train_opts["val_from"] = copy.deepcopy(opts.get("val_from", VAL_FROM))
    train_opts["filter"] = expand_filter(opts.get("filter", NEURON_FILTER['filter']))
    train_opts["spot_gt_mode"] = opts.get("spot_gt_mode", SPOT_PACK['spot_gt_mode'])
    train_opts["fp"] = int(opts.get("fp", TRAIN_SESSION['fp']))
    return train_opts


def resolve_schema(model, connectome, schema, train_opts):
    """Build the train schema: defaults + CLI / sidecar ``param_key`` stamps."""
    if schema is not None:
        return schema_copy(schema)
    filter = NEURON_FILTER['filter']
    if train_opts:
        filter = expand_filter(train_opts.get("filter", NEURON_FILTER['filter']))
    syn_mode = resolve_filter_branches(
        (train_opts or {}).get("syn_mode", NEURON_SCHEMA['syn_mode']),
        filter=filter,
    )
    base = build_schema(
        model,
        connectome,
        syn_mode=syn_mode,
        params=NEURON_SCHEMA['params'],
        h_cells=NEURON_SCHEMA['h_cells'],
        filter=filter,
        a_sti_radii=SPOT_PACK['a_sti_radii'],
    )
    if not train_opts:
        return base
    param_init = train_opts.get("param_init")
    param_inits = (
        [(row[0], row[1], float(row[2])) for row in param_init]
        if param_init else None
    )
    param_clamps = train_opts.get("param_clamps")
    clamps = (
        [(row[0], row[1], row[2], float(row[3])) for row in param_clamps]
        if param_clamps else None
    )
    param_jits = train_opts.get("param_jits")
    jits = (
        [(row[0], row[1], float(row[2])) for row in param_jits]
        if param_jits else None
    )
    return schema_with_params(
        base, connectome,
        param_modes=train_opts.get("param_modes"),
        param_inits=param_inits,
        param_clamps=clamps,
        param_jits=jits,
    )


def _build_session(
    connectome: Network,
    model: str,
    tasks: List[str],
    contrasts: List[str],
    packs: Dict[str, Dict[str, Pack]],
    *,
    delta_ms: float,
    delta_ms_pre: float,
    gt_amp: float,
    part_cost_scales=None,
    sequential=None,
    device=None,
    train_opts=None,
    schema: Optional[dict] = None,
    sim_dtype=SIM_DTYPE,
) -> TrainSession:
    device = device or active_device()
    seq = False if sequential is None else bool(sequential)
    filter = expand_filter(
        (train_opts or {}).get("filter", NEURON_FILTER['filter'])
    )
    neuron_const = resolve_filter_branches(NEURON_CONST, filter=filter)
    if train_opts is not None:
        train_opts["model"] = model
        train_opts["sequential"] = bool(seq)
    if train_opts is None or "euler" not in train_opts:
        raise ValueError("train opts require euler (implicit|explicit)")
    euler = expand_euler(resolve_filter_branches(train_opts["euler"], filter=filter))
    pre_steady = expand_pre_steady(resolve_filter_branches(
        train_opts.get("pre_steady", TRAIN_OPTIMIZATION['pre_steady']),
        filter=filter,
    ))
    train_opts["pre_steady"] = pre_steady
    pre_steady_n_iter = int(
        train_opts.get("pre_steady_n_iter", TRAIN_OPTIMIZATION['pre_steady_n_iter'])
    )
    pre_steady_damp = float(
        train_opts.get("pre_steady_damp", TRAIN_OPTIMIZATION['pre_steady_damp'])
    )
    train_opts["pre_steady_n_iter"] = pre_steady_n_iter
    train_opts["pre_steady_damp"] = pre_steady_damp
    sch = resolve_schema(
        model, connectome, schema, train_opts,
    )
    train_opts["param_modes"] = param_modes_from_schema(
        sch, connectome,
    )
    sch = schema_with_param_carry(sch)
    cli_scales = expand_part_cost_scale(part_cost_scales)
    part_cost_scales_filled = dict(cli_scales)
    for task in tasks:
        for contrast in contrasts:
            cost_part_keys_from_pack(
                packs[task][contrast],
                connectome,
                cli=cli_scales,
                scales=part_cost_scales_filled,
            )
    session = TrainSession(
        connectome=connectome,
        model=model,
        schema=sch,
        packs=dict(packs),
        tasks=tuple(tasks),
        contrasts=tuple(contrasts),
        part_cost_scales=part_cost_scales_filled,
        sequential=bool(seq),
        device=device,
        delta_ms=float(delta_ms),
        delta_ms_pre=float(delta_ms_pre),
        cap=float(neuron_const['cap']),
        g_leak=float(neuron_const['g_leak']),
        e_exc=float(neuron_const['e_exc']),
        e_inh=float(neuron_const['e_inh']),
        e_h=float(neuron_const['e_h']),
        h_g_max=float(neuron_const['h_g_max']),
        gt_amp=float(gt_amp),
        v_clamp=float(neuron_const['v_clamp']),
        a_syn_exc=float(neuron_const['a_syn_exc']),
        a_syn_inh=float(neuron_const['a_syn_inh']),
        euler=euler,
        pre_steady=pre_steady,
        pre_steady_n_iter=pre_steady_n_iter,
        pre_steady_damp=pre_steady_damp,
        sim_dtype=sim_dtype,
        train_opts=train_opts,
    )
    return session


def open_session(
    opts: dict,
    model: str,
    *,
    schema: Optional[dict] = None,
    connectome: Optional[Network] = None,
) -> TrainSession:
    """Build a :class:`TrainSession` from canonical train opts."""
    opts = dict(opts)
    filter = expand_filter(opts.get("filter", NEURON_FILTER['filter']))
    resolved = resolve_filter_branches(
        {"opts": opts, "gt_amp": NEURON_CONST['gt_amp'], "neuron_const": NEURON_CONST},
        filter=filter,
    )
    opts = resolved["opts"]
    gt_amp = float(resolved["gt_amp"])
    neuron_const = resolved["neuron_const"]
    backend_token = str(opts.get("backend", "network"))
    if backend_token != "network":
        raise ValueError(f"backend must be 'network', got {backend_token!r}")
    tasks = parse_tasks(opts.get("tasks"))
    contrasts = parse_contrasts(opts.get("contrasts"))
    i_sti = _merge_i_sti(opts.get("i_sti"))
    opts["i_sti"] = i_sti
    bad = [task for task in tasks if task not in TASKS]
    if bad:
        raise ValueError(f"unknown task(s) {bad!r} (expected {'|'.join(TASKS)})")
    device = opts.get("device") or active_device()
    sim_dtype = sim_dtype_from_fp(int(opts.get("fp", TRAIN_SESSION['fp'])))
    delta_ms = _sti_delta_ms(opts, "delta_ms")
    delta_ms_pre = _sti_delta_ms(opts, "delta_ms_pre")

    net = opts.get("network")
    syn_mode = opts.get("syn_mode", NEURON_SCHEMA['syn_mode'])
    if net is None:
        nj = opts.get("network_json")
        if not nj:
            raise ValueError("open_session(network) requires opts['network'] or network_json")
        net = load_network(
            nj, device=device,
            a_syn_exc=float(neuron_const['a_syn_exc']),
            a_syn_inh=float(neuron_const['a_syn_inh']),
            dtype=sim_dtype, syn_mode=syn_mode,
        )
    if connectome is None:
        connectome = net
    elif connectome is not net:
        raise ValueError("connectome must be opts['network']")
    pack_kwargs = dict(
        connectome=connectome,
        device=device,
        sim_dtype=sim_dtype,
        i_sti=i_sti,
        gt_amp=gt_amp,
        filter=filter,
    )
    packs: Dict[str, Dict[str, Pack]] = {}
    resolved_sti = {}
    for task in tasks:
        packs[task] = {}
        for contrast in contrasts:
            if task == "moving_bar":
                pack, sti_opts, _label = _build_moving_bar_pack(
                    part_cost_scales=opts.get("part_cost_scales"),
                    moving_bar_sti_opts=opts.get("moving_bar_sti_opts"),
                    delta_ms=delta_ms,
                    delta_ms_pre=delta_ms_pre,
                    contrast=contrast, **pack_kwargs,
                )
            elif task == "spot":
                pack, sti_opts, _label = _build_spot_pack(
                    spot_sti_opts=opts.get("spot_sti_opts"),
                    spot_gt_mode=expand_spot_gt_mode(
                        opts.get("spot_gt_mode", SPOT_PACK['spot_gt_mode']),
                    ),
                    cost_ms=dict(opts.get("cost_ms") or {}),
                    cost_interval_ms=float(
                        opts.get("cost_interval_ms", TRAIN_OPTIMIZATION['cost_interval_ms'])
                    ),
                    contrast=contrast, **pack_kwargs,
                )
            else:
                raise ValueError(f"unknown task {task!r}")
            packs[task][contrast] = pack
            resolved_sti[f"{task}_sti_opts"] = sti_opts
    train_opts = _sidecar_train_opts(
        opts, tasks, contrasts, resolved_sti, bool(opts.get("sequential")),
    )
    return _build_session(
        connectome, model, tasks, contrasts, packs,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        gt_amp=gt_amp,
        part_cost_scales=opts.get("part_cost_scales"),
        sequential=opts.get("sequential"),
        device=device,
        train_opts=train_opts,
        schema=schema,
        sim_dtype=sim_dtype,
    )


def resolve_filter_branches(val, *, filter: str):
    branch = "ca" if str(filter) == "ca" else "v"
    if isinstance(val, dict):
        branches = set(val)
        if branches and branches <= {"v", "ca"}:
            if branch not in val:
                raise KeyError(
                    f"filter-branch dict missing {branch!r} (branches={sorted(val)!r})"
                )
            return resolve_filter_branches(val[branch], filter=filter)
        return {
            nested_key: resolve_filter_branches(got, filter=filter)
            for nested_key, got in val.items()
        }
    if isinstance(val, list):
        return [resolve_filter_branches(got, filter=filter) for got in val]
    if isinstance(val, tuple):
        return tuple(resolve_filter_branches(got, filter=filter) for got in val)
    return val


def _sti_delta_ms(opts: dict, sti_timing_key: str) -> float:
    """``delta_ms`` / ``delta_ms_pre`` from sti opts (required)."""
    for _task, sti_opts_key in _STI_TRAIN_OPT_KEYS:
        so = opts.get(sti_opts_key)
        if isinstance(so, dict) and so.get(sti_timing_key) is not None:
            dt = float(so[sti_timing_key])
            if dt <= 0:
                raise ValueError(f"sti opts {sti_timing_key} must be > 0, got {dt}")
            return dt
    raise ValueError(
        f"train opts require {sti_timing_key} in a sti opts dict "
        f"(one of {[key for _, key in _STI_TRAIN_OPT_KEYS]})"
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
    if not opts.get("contrasts"):
        raise ValueError("train_opts requires contrasts")
    sim_dtype = sim_dtype_from_fp(int(opts.get("fp", TRAIN_SESSION['fp'])))
    _filter = expand_filter(opts.get("filter", NEURON_FILTER['filter']))
    neuron_const = resolve_filter_branches(NEURON_CONST, filter=_filter)
    syn_mode = resolve_filter_branches(
        opts.get("syn_mode", NEURON_SCHEMA['syn_mode']), filter=_filter,
    )
    mb = load_train_connectome(
        nj, device=opts.get("device") or active_device(), sim_dtype=sim_dtype,
        syn_mode=syn_mode,
        a_syn_exc=float(neuron_const['a_syn_exc']),
        a_syn_inh=float(neuron_const['a_syn_inh']),
    )
    opts["network"] = mb
    opts["syn_mode"] = syn_mode
    kwargs.setdefault("connectome", mb)
    return open_session({**opts, "backend": "network"}, model, **kwargs)


def session_from_outdir(
    outdir: str,
    model: str | None = None,
) -> TrainSession:
    """Load ``train_opts.json`` from a run folder and return a ready session."""
    opts_path = os.path.join(run_data_dir(os.path.abspath(outdir)), "train_opts.json")
    if not os.path.isfile(opts_path):
        raise FileNotFoundError(f"missing {opts_path}")
    with open(opts_path) as f:
        opts = json.load(f)
    return resolve_session(opts, model)
