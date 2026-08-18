# -*- coding: utf-8 -*-
"""Session types and assembly: ``Pack`` / ``TrainSession`` + open helpers.

Owns sti-opts finalisation (CLI tokens -> per-task sidecar dicts),
connectome loading, and the per-task×contrast ``Pack`` builders. The
builders wrap the neutral gt dataclasses from ``task`` (which sit below
``train`` in the import graph) and stamp the cross-cutting pack cost controls:

* spot: sparse ``cost_ts`` / optional ``cost_time_mask`` (#4; ``cost_ms``
  overwrites interval per radius), ``ms_sti`` (#1) already baked into
  the sti, ``waveform_mse=True``;
* spread: sparse ``cost_ts`` from ``cost_interval_ms`` only;
* moving bar: ``waveform_mse=True``.

Model traces are absolute ``v`` (``filter=none``) or ``ca`` (``filter=ca``);
cost compares the pack to ``a_gt * gts + bias_gt``. Schema includes
``v_th_ca``/``a_ca``/``tau_ca`` only when ``filter=ca``. When
``val_from``, ``bias_gt`` is written from ``v`` at ``t_onset`` (or ``ca`` when
``filter=ca``) — same value appears in ``param.csv``. Spot ir uses Arenz digitized
when ``filter=ca``. ``spread_gt_mode`` (``all`` | ``pos``) gates cost GT via
:func:`task.spread.gt.spread_gt_active`; dark multiplies by :func:`task.spread.gt.contrast_sign`.
Gt cells are :func:`network.construction.active_gt_cells`; cost pack applies
:func:`task.spot.gt.spot_gt_active` per cell (spot; delegates to spread).

``task`` ∈ {spread, spot, moving_bar} and ``contrast`` ∈ {bright, dark} are independent;
packs are ``packs[task][contrast]``.
"""
from __future__ import annotations

from config import (
    MOVING_BAR_INPUT_GEO,
    MOVING_BAR_INPUT_SPEC,
    NEURON_FORWARD,
    MODEL,
    NEURON_SCHEMA,
    SPREAD_INPUT_SPEC,
    SPREAD_GT,
    SPOT_INPUT_GEO,
    SPOT_PACK,
    TRAIN_CONFIG,
    TRAIN_OPTIMIZATION,
    TRAIN_SESSION,
    VAL_FROM,
)

import copy
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from import_bootstrap import parse_comma_list
from neuron.borst import t_from_ms
from neuron import (
    build_schema,
    expand_euler,
)
from train.param import (
    PARAM_MODES,
    SIM_DTYPE,
    active_device,
    schema_copy,
    schema_with_param_carry,
    resolve_val_from,
    schema_n_z,
    sim_dtype_from_fp,
    val_from_enabled,
)

from task.spread.gt import expand_gt_cells as expand_spread_gt_cells
from task.spread.pack import build_spread_gt
from task.spot.gt import expand_gt_cells as expand_spot_gt_cells
from task.spot.pack import (
    build_spot_gt,
    expand_cost_ms,
    build_spot_sti_opts,
    build_a_sti_radius_mask,
)
from task.spot.sti_geo import (
    resolve_spot,
    spot_sti_bs,
)
from task.spot.sti_spec import (
    build_spot_a_sti_radius_drive,
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

TASKS = ("spread", "spot", "moving_bar")
RUN_DATA_SUBDIR = "data"


def run_data_dir(run_dir) -> str:
    return str(Path(run_dir) / RUN_DATA_SUBDIR)


def _tokens(values) -> List[str]:
    if isinstance(values, str):
        return parse_comma_list(values)
    return [str(token) for token in values]


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
    # Spot a_sti_radius: i_sti += a_sti_radius[radius] * i_sti_pulse on (sti_bs, sti_nodes).
    i_sti_pulse: Optional[torch.Tensor] = None  # (T,) (i_peak - i_baseline) * sti_mask(t)
    sti_bs: Optional[torch.Tensor] = None  # (n_contrib,) long
    sti_nodes: Optional[torch.Tensor] = None  # (n_contrib,) long
    a_sti_radius_idxs: Optional[torch.Tensor] = None  # (n_contrib,) long → a_sti_radius index
    # (n_a_sti_radius,) 0/1: cost-radius scale==0 forces that a_sti_radius to 0 in forward.
    a_sti_radius_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TrainSession:
    """Immutable runtime context for one train / plotting run.

    Model / synapse scalars are flat fields (injected from
    ``config`` at session open). ``delta_ms`` / ``delta_ms_pre`` come
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
          f"n_z={schema_n_z(build_schema('borst', connectome, syn_mode=mode, params=params, a_sti_radii=SPOT_PACK['a_sti_radii']))}")
    return connectome


def _moving_bar_sti_opts(
    moving_bar_sti_opts: Optional[dict],
    *,
    delta_ms: float,
    delta_ms_pre: float,
) -> dict:
    """Build moving-bar sti opts; CLI override takes precedence over defaults."""
    if moving_bar_sti_opts:
        return dict(moving_bar_sti_opts)
    return build_moving_bar_sti_opts(
        ms_pre=MOVING_BAR_INPUT_SPEC["ms_pre"],
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        multi_bar=MOVING_BAR_INPUT_GEO['multi_bar'],
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
    i_sti: Dict[str, float],
    moving_bar_sti_opts: Optional[dict],
    filter: str,
    delta_ms: float,
    delta_ms_pre: float,
):
    task = "moving_bar"
    device = device or active_device()
    opts = _moving_bar_sti_opts(
        moving_bar_sti_opts,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
    )
    cost_radius = standardize_cost_radius(opts.get("cost_radius"))
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
        i_baseline=i_baseline_from_i_sti(i_sti),
        i_sti=float(i_sti[contrast]),
        contrasts=(contrast,),
        gt_cells=gt_cells_from_opts(opts),
        multi_bar=bool(opts.get("multi_bar", MOVING_BAR_INPUT_GEO['multi_bar'])),
        waveform_mse=True,
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
    ms_response = float(opts["ms_response"])
    if opts.get("ms_sti") is not None:
        ms_response = max(ms_response, float(opts["ms_sti"]))
    delta_ms = float(opts["delta_ms"])
    post = int(t_from_ms(ms_response, delta_ms=delta_ms)) + 1
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


def _spread_cost_ts(opts, *, cost_interval_ms, device):
    ms_response = float(opts["ms_response"])
    if opts.get("ms_sti") is not None:
        ms_response = max(ms_response, float(opts["ms_sti"]))
    delta_ms = float(opts["delta_ms"])
    post = int(t_from_ms(ms_response, delta_ms=delta_ms)) + 1
    if cost_interval_ms <= 0:
        raise ValueError("cost_interval_ms must be > 0")
    if post <= 0:
        raise ValueError("spread post-onset window must be > 0 for cost_interval_ms")
    step = max(1, int(round(cost_interval_ms / delta_ms)))
    mss = [t * delta_ms for t in range(0, post, step)]
    ts = set()
    for ms in mss:
        t = int(round(float(ms) / delta_ms))
        if t < 0 or t >= post:
            raise ValueError(
                f"cost time {ms} ms post-onset t out of range [0,{post})"
            )
        ts.add(t)
    return torch.tensor(sorted(ts), dtype=torch.long, device=device)


def _build_spread_pack(
    connectome,
    *,
    contrast: str,
    gt_amp: float,
    device: str,
    sim_dtype: torch.dtype,
    i_sti: Dict[str, float],
    spread_sti_opts: Optional[dict],
    filter: str,
    spread_gt_mode: str,
    cost_interval_ms: float,
) -> Tuple[Pack, dict, str]:
    task = "spread"
    if not spread_sti_opts:
        raise ValueError("spread requires sti opts (from resolve_train_opts / CLI)")
    opts = dict(spread_sti_opts)
    ms_sti = opts.get("ms_sti")
    ms_response = opts.get("ms_response")
    if ms_sti is not None and ms_response is not None:
        opts["ms_response"] = max(float(ms_response), float(ms_sti))
    for key in ("ms_pre", "ms_response", "delta_ms", "delta_ms_pre"):
        if opts.get(key) is None:
            raise ValueError(f"spread sti opts require {key}")
    ms_pre = float(opts["ms_pre"])
    ms_response = float(opts["ms_response"])
    delta_ms = float(opts["delta_ms"])
    delta_ms_pre = float(opts["delta_ms_pre"])
    ms_post = float(opts.get("ms_post", 0.0))
    ms_sti = opts.get("ms_sti")
    device = device or active_device()
    t_onset = int(t_from_ms(ms_pre, delta_ms=delta_ms_pre))
    n_t = int(
        t_onset
        + t_from_ms(ms_response, delta_ms=delta_ms)
        + t_from_ms(ms_post, delta_ms=delta_ms)
        + 1
    )
    i_baseline = i_baseline_from_i_sti(i_sti)
    T = build_spread_gt(
        connectome,
        n_t=n_t,
        t_onset=t_onset,
        i_baseline=i_baseline,
        i_sti=float(i_sti[contrast]),
        contrast=contrast,
        gt_amp=gt_amp,
        delta_ms=delta_ms,
        device=device,
        sim_dtype=sim_dtype,
        ms_sti=ms_sti,
        ms_response=ms_response,
        gt_cells=gt_cells_from_opts(opts),
        filter=str(filter),
        spread_gt_mode=str(spread_gt_mode),
    )
    cost_ts = _spread_cost_ts(
        opts,
        cost_interval_ms=cost_interval_ms,
        device=device,
    )
    pack = Pack(
        task=task,
        contrast=contrast,
        i_sti=T.i_sti,
        gts=T.gts,
        power=torch.zeros((), dtype=sim_dtype, device=T.gts.device),
        cost_scales=torch.ones(T.gts.shape[0], dtype=sim_dtype, device=T.gts.device),
        entry_bs=T.entry_bs,
        entry_nodes=T.entry_nodes,
        cost_t0s=None,
        cost_ts=cost_ts,
        waveform_mse=True,
        t_onset=int(t_onset),
    )
    label = f"spread {contrast} ({int(T.gts.shape[0])} cost nodes)"
    return pack, dict(opts), label


def _build_spot_pack(
    connectome,
    *,
    contrast: str,
    gt_amp: float,
    device: str,
    sim_dtype: torch.dtype,
    i_sti: Dict[str, float],
    spot_sti_opts: Optional[dict],
    filter: str,
    spread_gt_mode: str,
    cost_ms: Optional[dict],
    cost_interval_ms: float,
) -> Tuple[Pack, dict, str]:
    task = "spot"
    if not spot_sti_opts:
        raise ValueError("spot requires sti opts (from resolve_train_opts / CLI)")
    opts = dict(spot_sti_opts)
    ms_sti = opts.get("ms_sti")
    ms_response = opts.get("ms_response")
    if ms_sti is not None and ms_response is not None:
        opts["ms_response"] = max(float(ms_response), float(ms_sti))
    for key in ("ms_pre", "ms_response", "delta_ms", "delta_ms_pre"):
        if opts.get(key) is None:
            raise ValueError(f"spot sti opts require {key}")
    ms_pre = float(opts["ms_pre"])
    ms_response = float(opts["ms_response"])
    delta_ms = float(opts["delta_ms"])
    delta_ms_pre = float(opts["delta_ms_pre"])
    ms_post = float(opts.get("ms_post", 0.0))
    ms_sti = opts.get("ms_sti")
    cost_radius = standardize_cost_radius(opts.get("cost_radius"))
    shift_radius = int(opts["shift_radius"])
    spot_radius = float(opts["spot_radius"])
    multi_spot = bool(opts["multi_spot"])
    fully_inside = bool(opts["fully_inside"])
    device = device or active_device()
    t_onset = int(t_from_ms(ms_pre, delta_ms=delta_ms_pre))
    n_t = int(
        t_onset
        + t_from_ms(ms_response, delta_ms=delta_ms)
        + t_from_ms(ms_post, delta_ms=delta_ms)
        + 1
    )
    i_baseline = i_baseline_from_i_sti(i_sti)
    cost_radius_scales = dict(SPOT_PACK['spot_cost_radius_scale'])
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
        i_baseline=i_baseline,
        i_sti=float(i_sti[contrast]),
        contrast=contrast,
        ms_sti=ms_sti,
        ms_response=ms_response,
        gt_amp=gt_amp,
        delta_ms=delta_ms,
        cost_radius_scales=cost_radius_scales,
        spot_cost_radii=SPOT_PACK['spot_cost_radii'],
        gt_cells=gt_cells_from_opts(opts),
        filter=str(filter),
        spread_gt_mode=str(spread_gt_mode),
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
    a_sti_radius_mask = build_a_sti_radius_mask(
        cost_radius_scales=cost_radius_scales,
        a_sti_radii=SPOT_PACK['a_sti_radii'],
    )
    i_sti, i_sti_pulse, sti_bs, sti_nodes, a_sti_radius_idxs = build_spot_a_sti_radius_drive(
        connectome,
        spot_bs,
        a_sti_radii=SPOT_PACK['a_sti_radii'],
        t_onset=int(t_onset),
        n_t=int(n_t),
        ms_sti=ms_sti,
        delta_ms=delta_ms,
        i_baseline=i_baseline,
        i_sti=float(i_sti[contrast]),
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
        i_sti_pulse=i_sti_pulse,
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
    "spread": expand_spread_gt_cells,
    "spot": expand_spot_gt_cells,
    "moving_bar": expand_moving_bar_gt_cells,
}


def resolve_gt_cells_by_task(by_task) -> Dict[str, List[str]]:
    """Map concrete tasks to final gt cell lists (task + cell aliases expanded)."""
    return {
        str(task): list(_GT_CELLS_EXPAND[task]([str(cell) for cell in cells]))
        for task, cells in (by_task or {}).items()
    }


_STI_TRAIN_OPT_KEYS = (
    ("spread", "spread_sti_opts"),
    ("spot", "spot_sti_opts"),
    ("moving_bar", "moving_bar_sti_opts"),
)

_STI_OPTS_BY_TASK = {
    "spread": {
        "ms_pre": SPREAD_INPUT_SPEC["ms_pre"],
        "ms_response": SPREAD_INPUT_SPEC["ms_response"],
        "ms_post": SPREAD_INPUT_SPEC["ms_post"],
        "ms_sti": SPREAD_INPUT_SPEC["ms_sti"],
        "delta_ms": MODEL["delta_ms"],
        "delta_ms_pre": MODEL["delta_ms_pre"],
    },
    "spot": {
        "ms_pre": SPREAD_INPUT_SPEC["ms_pre"],
        "ms_response": SPREAD_INPUT_SPEC["ms_response"],
        "ms_post": SPREAD_INPUT_SPEC["ms_post"],
        "ms_sti": SPREAD_INPUT_SPEC["ms_sti"],
        "delta_ms": MODEL["delta_ms"],
        "delta_ms_pre": MODEL["delta_ms_pre"],
        "shift_radius": SPOT_INPUT_GEO["shift_radius"],
        "spot_radius": SPOT_INPUT_GEO["spot_radius"],
        "multi_spot": SPOT_INPUT_GEO["multi_spot"],
        "fully_inside": SPOT_INPUT_GEO["fully_inside"],
    },
    "moving_bar": {
        "ms_pre": MOVING_BAR_INPUT_SPEC["ms_pre"],
        "delta_ms": MODEL["delta_ms"],
        "delta_ms_pre": MODEL["delta_ms_pre"],
        "multi_bar": MOVING_BAR_INPUT_GEO["multi_bar"],
    },
}


def _i_sti(i_sti=None) -> Dict[str, float]:
    """Bright/dark currents from ``SPREAD_INPUT_SPEC``; optional contrast stamps."""
    i_sti_by_contrast = {
        "bright": float(SPREAD_INPUT_SPEC["i_bright"]),
        "dark": float(SPREAD_INPUT_SPEC["i_dark"]),
    }
    if not i_sti:
        return i_sti_by_contrast
    for contrast, val in i_sti.items():
        i_sti_by_contrast[str(contrast)] = float(val)
    return i_sti_by_contrast


def _spread_resolve_sti_opts(opts, **_):
    ms_sti = opts["ms_sti"]
    gt_cells = opts.get("gt_cells")
    opts = {
        "ms_pre": opts["ms_pre"],
        "ms_response": opts["ms_response"],
        "ms_post": opts["ms_post"],
        "delta_ms": opts["delta_ms"],
        "delta_ms_pre": opts["delta_ms_pre"],
    }
    if ms_sti is not None:
        opts["ms_sti"] = ms_sti
    if gt_cells is not None:
        opts["gt_cells"] = list(gt_cells)
    ms_response = opts.get("ms_response")
    if ms_sti is not None and ms_response is not None:
        opts["ms_response"] = max(float(ms_response), float(ms_sti))
    return opts


def _spot_resolve_sti_opts(opts, *, shift_radius, spot_radius, multi_spot, fully_inside, **_):
    sti_opts = build_spot_sti_opts(
        ms_pre=opts["ms_pre"],
        ms_response=opts["ms_response"],
        ms_post=opts["ms_post"],
        delta_ms=opts["delta_ms"],
        delta_ms_pre=opts["delta_ms_pre"],
        shift_radius=(
            shift_radius if shift_radius is not None else opts["shift_radius"]
        ),
        spot_radius=(
            spot_radius if spot_radius is not None else opts["spot_radius"]
        ),
        multi_spot=(
            multi_spot if multi_spot is not None else opts["multi_spot"]
        ),
        fully_inside=(
            fully_inside if fully_inside is not None else opts["fully_inside"]
        ),
        ms_sti=opts["ms_sti"],
        gt_cells=opts.get("gt_cells"),
    )
    sti_opts["shift_radius"] = shift_radius
    sti_opts["spot_radius"] = spot_radius
    sti_opts["multi_spot"] = multi_spot
    sti_opts["fully_inside"] = fully_inside
    return sti_opts


def _moving_bar_resolve_sti_opts(opts, **_):
    return build_moving_bar_sti_opts(
        ms_pre=opts["ms_pre"],
        delta_ms=opts["delta_ms"],
        delta_ms_pre=opts["delta_ms_pre"],
        multi_bar=opts["multi_bar"],
        gt_cells=opts.get("gt_cells"),
    )


_RESOLVE_STI_OPTS = {
    "spread": _spread_resolve_sti_opts,
    "spot": _spot_resolve_sti_opts,
    "moving_bar": _moving_bar_resolve_sti_opts,
}


def _resolve_sti_opts(
    opts,
    task,
    *,
    cost_radius,
    shift_radius,
    spot_radius,
    multi_spot,
    fully_inside,
):
    opts = {**_STI_OPTS_BY_TASK.get(task, {}), **(opts or {})}
    sti_opts = _RESOLVE_STI_OPTS[task](
        opts,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
    )
    if cost_radius is not None:
        sti_opts["cost_radius"] = int(cost_radius)
    else:
        sti_opts.pop("cost_radius", None)
    return sti_opts


def resolve_train_opts(
    tasks=None,
    contrasts=None,
    part_cost_scales=None,
    sequential=None,
    cost_radius=None,
    shift_radius=None,
    spot_radius=None,
    multi_spot=SPOT_INPUT_GEO['multi_spot'],
    fully_inside=SPOT_INPUT_GEO['fully_inside'],
    i_sti=None,
    cost_norm=TRAIN_OPTIMIZATION['cost_norm'],
    cost_interval_ms=TRAIN_OPTIMIZATION['cost_interval_ms'],
    cost_ms=None,
    moving_bar_sti_opts=None,
    spread_sti_opts=None,
    spot_sti_opts=None,
    network_json=None,
    network=None,
    syn_mode=NEURON_SCHEMA['syn_mode'],
    device=None,
    packs=None,
    euler=MODEL['euler'],
    pre_steady=None,
    pre_steady_n_iter=TRAIN_OPTIMIZATION['pre_steady_n_iter'],
    pre_steady_damp=TRAIN_OPTIMIZATION['pre_steady_damp'],
    fp=TRAIN_SESSION['fp'],
    pre_grad=NEURON_FORWARD['pre_grad'],
    val_from=None,
    filter=NEURON_SCHEMA['filter'],
    spread_gt_mode=SPREAD_GT['spread_gt_mode'],
):
    """Canonical train opts for :func:`open_session`."""
    if network is None and network_json is None:
        raise ValueError("resolve_train_opts requires network or network_json")
    fp = int(fp)
    if fp not in (16, 32, 64):
        raise ValueError(f"fp must be 16, 32, or 64; got {fp!r}")
    filter = str(filter)
    if pre_steady is None:
        pre_steady = TRAIN_OPTIMIZATION['pre_steady']
    if sequential is None:
        sequential = TRAIN_SESSION['sequential']
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
                "val_from v_th_ca / a_ca require filter ca "
                f"(got filter={filter!r})"
            )
    tasks = _tokens(tasks if tasks is not None else TRAIN_CONFIG["tasks"])
    contrasts = _tokens(
        contrasts if contrasts is not None else SPREAD_INPUT_SPEC["contrasts"]
    )
    if spot_radius is None:
        spot_radius = SPOT_INPUT_GEO['spot_radius']
    if shift_radius is None:
        shift_radius = SPOT_INPUT_GEO['shift_radius']
    merged_i_sti = _i_sti(i_sti)
    finalize_kwargs = dict(
        cost_radius=cost_radius,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
    )
    sti_opts_by_task = {
        "spread": spread_sti_opts,
        "spot": spot_sti_opts,
        "moving_bar": moving_bar_sti_opts,
    }
    sti_opts = {}
    for task, sti_opts_key in _STI_TRAIN_OPT_KEYS:
        if task not in tasks and sti_opts_by_task[task] is None:
            sti_opts[sti_opts_key] = None
            continue
        sti_opts[sti_opts_key] = _resolve_sti_opts(
            sti_opts_by_task[task],
            task,
            **finalize_kwargs,
        )
    opts = {
        "tasks": tasks,
        "contrasts": contrasts,
        "i_sti": merged_i_sti,
        "part_cost_scales": {
            str(part_key): float(scale)
            for part_key, scale in (
                part_cost_scales
                if part_cost_scales is not None
                else TRAIN_OPTIMIZATION['part_cost_scales']
                or {}
            ).items()
        },
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
        "spread_gt_mode": spread_gt_mode,
        "fp": fp,
        "packs": None,
        "params": copy.deepcopy(NEURON_SCHEMA["params"]),
        "network": network,
        "network_json": str(network_json) if network_json is not None else None,
        "device": device,
    }
    if packs is not None:
        opts["packs"] = packs
    return opts


def _cost_ms_sidecar(cost_ms) -> dict:
    """JSON sidecar: radii as strings, mss as floats."""
    cost_ms_json: dict = {}
    for radius, vals in (cost_ms or {}).items():
        mss = list(vals) if isinstance(vals, (list, tuple)) else [vals]
        cost_ms_json[str(int(radius))] = [float(x) for x in mss]
    return cost_ms_json


def _sidecar_train_opts(opts, tasks, contrasts, resolved_sti, sequential_bool) -> dict:
    """Build JSON-serializable train_opts."""
    def _sti(sti_opts_key):
        got = resolved_sti.get(sti_opts_key)
        return got if got is not None else opts.get(sti_opts_key)

    train_opts = {
        "tasks": list(tasks),
        "contrasts": list(contrasts),
        "i_sti": {
            contrast: float(val)
            for contrast, val in (opts.get("i_sti") or {}).items()
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
        "spread_sti_opts": _sti("spread_sti_opts"),
        "spot_sti_opts": _sti("spot_sti_opts"),
        "moving_bar_sti_opts": _sti("moving_bar_sti_opts"),
    }
    if opts.get("params"):
        train_opts["params"] = copy.deepcopy(opts["params"])
    if "euler" not in opts:
        raise ValueError("train opts require euler (implicit|explicit)")
    train_opts["euler"] = opts["euler"]
    train_opts["syn_mode"] = opts.get("syn_mode", NEURON_SCHEMA['syn_mode'])
    train_opts["pre_grad"] = opts.get("pre_grad", NEURON_FORWARD['pre_grad'])
    train_opts["val_from"] = copy.deepcopy(opts.get("val_from", VAL_FROM))
    train_opts["filter"] = str(opts.get("filter", NEURON_SCHEMA['filter']))
    train_opts["spread_gt_mode"] = opts.get("spread_gt_mode", SPREAD_GT['spread_gt_mode'])
    train_opts["fp"] = int(opts.get("fp", TRAIN_SESSION['fp']))
    return train_opts


def resolve_schema(model, connectome, schema, train_opts):
    """Build the train schema from sidecar / YAML ``params``."""
    if schema is not None:
        return schema_copy(schema)
    filter = NEURON_SCHEMA['filter']
    if train_opts:
        filter = str(train_opts.get("filter", NEURON_SCHEMA['filter']))
    syn_mode = (train_opts or {}).get("syn_mode", NEURON_SCHEMA['syn_mode'])
    params = (train_opts or {}).get("params") or NEURON_SCHEMA["params"]
    schema = build_schema(
        model,
        connectome,
        syn_mode=syn_mode,
        params=params,
        filter=filter,
        a_sti_radii=SPOT_PACK['a_sti_radii'],
    )
    val_from = (train_opts or {}).get("val_from") or {}
    schema = schema_copy(schema)
    for target, entry in val_from.items():
        if not entry.get("enabled") or target not in schema:
            continue
        spec = schema[target]
        n_node = spec['n_node']
        for mode in PARAM_MODES:
            spec[mode] = []
        spec["frozen"] = list(range(n_node))
    return schema


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
    neuron_const = MODEL
    if train_opts is not None:
        train_opts["model"] = model
        train_opts["sequential"] = bool(seq)
    if train_opts is None or "euler" not in train_opts:
        raise ValueError("train opts require euler (implicit|explicit)")
    euler = expand_euler(train_opts["euler"])
    pre_steady = str(
        train_opts.get("pre_steady", TRAIN_OPTIMIZATION['pre_steady']),
    )
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
    sch = schema_with_param_carry(sch)
    cli_scales = {
        str(part_key): float(scale)
        for part_key, scale in (part_cost_scales or {}).items()
    }
    session = TrainSession(
        connectome=connectome,
        model=model,
        schema=sch,
        packs=dict(packs),
        tasks=tuple(tasks),
        contrasts=tuple(contrasts),
        part_cost_scales=cli_scales,
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
    opts.pop("backend", None)
    filter = str(opts.get("filter", NEURON_SCHEMA['filter']))
    gt_amp = float(MODEL['gt_amp'])
    neuron_const = MODEL
    tasks = _tokens(opts.get("tasks"))
    contrasts = _tokens(opts.get("contrasts"))
    i_sti = _i_sti(opts.get("i_sti"))
    opts["i_sti"] = i_sti
    device = opts.get("device") or active_device()
    sim_dtype = sim_dtype_from_fp(int(opts.get("fp", TRAIN_SESSION['fp'])))
    delta_ms, delta_ms_pre = _sti_delta_ms(opts)

    net = opts.get("network")
    syn_mode = opts.get("syn_mode", NEURON_SCHEMA['syn_mode'])
    if net is None:
        nj = opts.get("network_json")
        if not nj:
            raise ValueError("open_session requires opts['network'] or network_json")
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
            if task == "spread":
                pack, sti_opts, _label = _build_spread_pack(
                    spread_sti_opts=opts.get("spread_sti_opts"),
                    spread_gt_mode=str(
                        opts.get("spread_gt_mode", SPREAD_GT['spread_gt_mode']),
                    ),
                    cost_interval_ms=float(
                        opts.get("cost_interval_ms", TRAIN_OPTIMIZATION['cost_interval_ms'])
                    ),
                    contrast=contrast, **pack_kwargs,
                )
            elif task == "moving_bar":
                pack, sti_opts, _label = _build_moving_bar_pack(
                    moving_bar_sti_opts=opts.get("moving_bar_sti_opts"),
                    delta_ms=delta_ms,
                    delta_ms_pre=delta_ms_pre,
                    contrast=contrast, **pack_kwargs,
                )
            elif task == "spot":
                pack, sti_opts, _label = _build_spot_pack(
                    spot_sti_opts=opts.get("spot_sti_opts"),
                    spread_gt_mode=str(
                        opts.get("spread_gt_mode", SPREAD_GT['spread_gt_mode']),
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


def _sti_delta_ms(opts: dict) -> tuple[float, float]:
    """``delta_ms`` / ``delta_ms_pre`` from sti opts (required)."""
    delta_ms = None
    delta_ms_pre = None
    for _task, sti_opts_key in _STI_TRAIN_OPT_KEYS:
        so = opts.get(sti_opts_key)
        if not isinstance(so, dict):
            continue
        if delta_ms is None and so.get("delta_ms") is not None:
            delta_ms = float(so["delta_ms"])
        if delta_ms_pre is None and so.get("delta_ms_pre") is not None:
            delta_ms_pre = float(so["delta_ms_pre"])
    missing = [
        name for name, val in (("delta_ms", delta_ms), ("delta_ms_pre", delta_ms_pre))
        if val is None
    ]
    if missing:
        raise ValueError(
            f"train opts require {', '.join(missing)} in a sti opts dict "
            f"(one of {[key for _, key in _STI_TRAIN_OPT_KEYS]})"
        )
    if delta_ms <= 0 or delta_ms_pre <= 0:
        raise ValueError(
            f"sti opts delta_ms / delta_ms_pre must be > 0, "
            f"got {delta_ms}, {delta_ms_pre}"
        )
    return delta_ms, delta_ms_pre


def resolve_session(opts: dict, model: str | None = None, **kwargs) -> TrainSession:
    """Restore a session from a saved ``train_opts.json`` dict."""
    opts = dict(opts)
    opts.pop("backend", None)
    if model is None:
        model = opts.get("model")
        if not model:
            raise ValueError("train_opts requires model")
    opts["packs"] = None
    nj = opts.get("network_json")
    if not nj:
        raise ValueError("train_opts requires network_json")
    if not opts.get("tasks"):
        raise ValueError("train_opts requires tasks")
    if not opts.get("contrasts"):
        raise ValueError("train_opts requires contrasts")
    sim_dtype = sim_dtype_from_fp(int(opts.get("fp", TRAIN_SESSION['fp'])))
    neuron_const = MODEL
    syn_mode = opts.get("syn_mode", NEURON_SCHEMA['syn_mode'])
    mb = load_train_connectome(
        nj, device=opts.get("device") or active_device(), sim_dtype=sim_dtype,
        syn_mode=syn_mode,
        a_syn_exc=float(neuron_const['a_syn_exc']),
        a_syn_inh=float(neuron_const['a_syn_inh']),
    )
    opts["network"] = mb
    opts["syn_mode"] = syn_mode
    kwargs.setdefault("connectome", mb)
    return open_session(opts, model, **kwargs)


def session_from_run_dir(
    run_dir: str,
    model: str | None = None,
) -> TrainSession:
    """Load ``train_opts.json`` from a run folder and return a ready session."""
    opts_path = os.path.join(run_data_dir(os.path.abspath(run_dir)), "train_opts.json")
    if not os.path.isfile(opts_path):
        raise FileNotFoundError(f"missing {opts_path}")
    with open(opts_path) as f:
        opts = json.load(f)
    return resolve_session(opts, model)
