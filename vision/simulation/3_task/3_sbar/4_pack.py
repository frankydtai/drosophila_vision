# -*- coding: utf-8 -*-
"""Static-bar pack: bind GT numbers to the network for cost.

Cost hexes, sti ``i_sti``, and :class:`SbarGt` packing.
GT traces come from :mod:`task.sbar.gt`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from neuron.borst import t_from_ms
from network.construction import (
    active_gt_cells,
    gt_cells_from_opts,
    node_cells,
    standardize_cost_radius,
)
from task.sbar.gt import GT_CELLS, load_gt
from task.sbar.sti_geo import sbar_line_mids, sti_hexes
from task.sbar.sti_spec import (
    build_sbar_a_sti_mid_drive,
    build_sbar_signals,
    gruntman_sbar_specs,
)
from task.spread.pack import (
    CostPartPlotSpec,
    cost_hex_label,
    cost_sti_hexes,
)
from task.spread.pack import build_cost_ts, post_onset_n_t
from task.spread.sti_spec import i_baseline_from_i_sti


@dataclass
class SbarGt:
    i_sti: torch.Tensor
    gts: torch.Tensor
    power: torch.Tensor
    cost_scales: torch.Tensor
    entry_bs: torch.Tensor
    entry_nodes: torch.Tensor
    n_b: int
    n_t: int
    n_cost_hex: int
    active_gts: List[str]
    spec_tokens: List[str]
    cost_t0s: Optional[torch.Tensor] = None
    entry_part_keys: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SbarPack:
    """One sbar train pack: task×contrast drive + entries + gts."""

    task: str
    contrast: str
    i_sti: torch.Tensor
    gts: torch.Tensor
    cost_scales: torch.Tensor
    entry_bs: torch.Tensor
    entry_nodes: torch.Tensor
    cost_t0s: Optional[torch.Tensor] = None
    cost_ts: Optional[torch.Tensor] = None
    cost_radius: Optional[int] = None
    t_onset: Optional[int] = None
    i_sti_pulse: Optional[torch.Tensor] = None
    sti_bs: Optional[torch.Tensor] = None
    sti_nodes: Optional[torch.Tensor] = None
    a_sti_mid_idxs: Optional[torch.Tensor] = None
    entry_part_keys: Tuple[str, ...] = ()
    cost_part_plot_specs: Optional[Dict[str, CostPartPlotSpec]] = None


def sbar_a_sti_mids(sti_opts=None) -> tuple[float, ...]:
    """Configured positive absolute distances controlled by ``a_sti_mid``."""
    from config import SBAR_PACK

    values = (sti_opts or {}).get("a_sti_mids", SBAR_PACK["a_sti_mids"])
    mids = tuple(float(mid) for mid in values)
    if any(not np.isfinite(mid) or mid <= 0.0 for mid in mids):
        raise ValueError("a_sti_mids must contain finite positive distances; omit mid=0")
    if len({round(mid, 9) for mid in mids}) != len(mids):
        raise ValueError("a_sti_mids must contain unique absolute distances")
    return mids


def part_key(contrast: str, cell: str, mid) -> str:
    return f"sbar_{contrast}_{cell}_mid{(f'{int(mid):+d}' if float(mid).is_integer() else f'{float(mid):+.1f}')}"


def sbar_direction_active(cell: str, direction: str) -> bool:
    """Whether *direction* lies on the T4/T5 subtype's motion axis."""
    subtype = str(cell)[-1:]
    if subtype in ("a", "b"):
        return str(direction) in ("right", "left")
    if subtype in ("c", "d"):
        return str(direction) in ("up", "down")
    raise ValueError(f"unknown sbar gt cell subtype: {cell!r}")


def sbar_pd_axis_sign(cell: str) -> int:
    """Global x/y sign of increasing position along the subtype's PD axis."""
    subtype = str(cell)[-1:]
    if subtype in ("a", "c"):
        return 1
    if subtype in ("b", "d"):
        return -1
    raise ValueError(f"unknown sbar gt cell subtype: {cell!r}")


def _sbar_gt_key(cell: str, contrast: str, position: float) -> str:
    pathway = (
        "PC" if (cell.startswith("T4") and contrast == "bright")
        or (cell.startswith("T5") and contrast == "dark") else "NC"
    )
    position_label = (
        f"{int(position):+d}"
        if float(position).is_integer()
        else f"{float(position):+.1f}"
    )
    return f"{cell[:2]}_{pathway}_pos{position_label}_w1"


def _sbar_cost_part_plot_specs(
    entry_part_keys: Sequence[str],
    entry_nodes,
    cost_scales,
    connectome,
    contrast: str,
) -> Dict[str, CostPartPlotSpec]:
    specs: Dict[str, CostPartPlotSpec] = {}
    cost_scales = np.asarray(cost_scales, dtype=np.float64)
    entry_nodes = np.asarray(entry_nodes, dtype=np.int64)
    node_cells = connectome.node_cells[entry_nodes]
    if torch.is_tensor(node_cells):
        node_cells = node_cells.detach().cpu().numpy()
    cells = connectome.cells
    for entry, entry_part_key in enumerate(entry_part_keys):
        if float(cost_scales[entry]) <= 0.0:
            continue
        if entry_part_key in specs:
            continue
        mid = float(entry_part_key.rsplit("_mid", 1)[1])
        cell = str(cells[int(node_cells[entry])])
        specs[entry_part_key] = CostPartPlotSpec(
            entry_part_key,
            cell,
            ("mid", mid),
            f"mid={(f'{int(mid):+d}' if float(mid).is_integer() else f'{float(mid):+.1f}')} ({contrast})",
        )
    return specs


def build_sbar_gt(
    connectome,
    *,
    ms_pre: float,
    ms_response: float,
    ms_post: float = 0.0,
    ms_sti=None,
    delta_ms: float,
    delta_ms_pre: float,
    bar_dist: int,
    bar_directions: Sequence[str],
    shift_radius: int = 0,
    multi_bar: bool = True,
    cost_radius: Optional[int] = None,
    i_baseline: float,
    i_sti: float,
    contrasts: Sequence[str],
    gt_cells: Optional[Sequence[str]] = None,
) -> SbarGt:
    """Build sbar cost GT.

    CRITICAL GT FACTS (from ``3_gt.py`` docstring — read there first):
    - CSV has exactly 9 distinct positions: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2 degrees.
      The 0.5-step positions (e.g. pos+0.5, pos-1.5) ARE REQUIRED — do not skip them.
    - ``target_width_led == 1`` filter selects width-1 traces only (2.25° wide).
      The same CSV also has ``target_width_led == 2`` (width-2, 4.5°) — those are a
      different task variant and MUST be excluded by the ``load_gt`` call.
    - Trace ID format: ``{cell_prefix}_{PC|NC}_pos{SIGN}_w1``, e.g. ``T4_PC_pos+0.5_w1``.
      The float position in the trace_id (e.g. ``+0.5``) MUST match the CSV ``position`` field.
    - Each simultaneous bar line is a separate spatial replicate, matching how
      spot expands every simultaneous spot center. ``mid`` is the cost-node
      position relative to that bar line, not the node's absolute axis value.
      Thus multi-bar produces more ``(bar, node)`` cost entries than single-bar.
    - Subtypes a/b use only the right/left axis; subtypes c/d use only the
      up/down axis. In particular, a/b ``mid`` values are integer ``x`` values
      and must never receive half-step ``y`` entries from up/down conditions.

    When a node lies within the GT position range of multiple simultaneous
    bars, its target is the sum of those single-bar GT traces. It remains one
    sample for each bar-relative position, as with overlapping spot centers.
    """
    specs = gruntman_sbar_specs(
        contrasts=tuple(contrasts),
        bar_directions=bar_directions,
        shift_radius=int(shift_radius),
    )
    i_baseline = float(i_baseline)
    sti = build_sbar_signals(
        connectome,
        specs=specs,
        ms_pre=ms_pre,
        ms_response=ms_response,
        ms_post=ms_post,
        ms_sti=ms_sti,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        bar_dist=int(bar_dist),
        multi_bar=bool(multi_bar),
        i_baseline=i_baseline,
        i_sti=float(i_sti),
    )
    gts = load_gt(
        t_onset=sti.t_onset,
        ms_response=ms_response,
        ms_sti=ms_sti,
        delta_ms=float(delta_ms),
        ms_post=float(ms_post),
    )
    active = active_gt_cells(
        gt_cells, GT_CELLS, connectome.cells, context="sbar",
    )
    cells = node_cells(connectome)
    hexes = sti_hexes(connectome)
    cost_hexes = cost_sti_hexes(connectome, cost_radius=cost_radius)
    bar_dist = int(bar_dist)
    t0 = int(sti.t_onset)
    n_t_cost = int(t_from_ms(float(ms_response), delta_ms=float(delta_ms)) + 1)

    entry_bs, entry_nodes, entry_gts, entry_cost_scales, entry_cost_t0s = (
        [], [], [], [], [],
    )
    entry_part_keys: List[str] = []
    cost_bar_hexes = set()

    for b, spec in enumerate(sti.specs):
        bar_mids = sbar_line_mids(
            hexes, spec.direction, bar_dist, multi_bar=bool(multi_bar),
            shift_mid=spec.shift_mid,
        )
        axis = "x" if spec.direction in ("right", "left") else "y"
        for hex in cost_hexes:
            cell_mid = float(hex.x if axis == "x" else hex.y)
            for cell in active:
                if not sbar_direction_active(cell, spec.direction):
                    continue
                bar_samples = []
                for bar_mid in bar_mids:
                    relative_mid = float(
                        sbar_pd_axis_sign(cell) * (bar_mid - cell_mid)
                    )
                    gt_key = _sbar_gt_key(cell, spec.contrast, relative_mid)
                    if gt_key in gts:
                        bar_samples.append((float(bar_mid), relative_mid, gt_key))
                if not bar_samples:
                    continue
                nodes = connectome.nodes_at_uv(hex.u, hex.v, cell, cells=cells)
                if len(nodes) == 0:
                    continue
                gt = sum(
                    (gts[gt_key][t0:t0 + n_t_cost] for _, _, gt_key in bar_samples),
                    np.zeros(n_t_cost, dtype=np.float64),
                )
                for bar_mid, relative_mid, _ in bar_samples:
                    cost_bar_hexes.add((axis, bar_mid, int(hex.u), int(hex.v)))
                    for node in nodes:
                        entry_bs.append(b)
                        entry_nodes.append(int(node))
                        entry_gts.append(gt)
                        entry_cost_scales.append(1.0)
                        entry_cost_t0s.append(t0)
                        entry_part_keys.append(
                            part_key(spec.contrast, cell, relative_mid)
                        )

    if not entry_bs:
        raise ValueError("no static-bar cost nodes (check subtypes and sti hexes)")

    gts = np.asarray(entry_gts, dtype=np.float64)
    cost_scales = np.asarray(entry_cost_scales, dtype=np.float64)
    entry_bs = np.asarray(entry_bs, dtype=np.int64)
    entry_nodes = np.asarray(entry_nodes, dtype=np.int64)
    cost_t0s = np.asarray(entry_cost_t0s, dtype=np.int64)
    power = float(np.sum(cost_scales[:, None] * gts ** 2))
    if power == 0.0:
        power = 1.0

    return SbarGt(
        i_sti=sti.i_sti,
        gts=gts,
        power=power,
        cost_scales=cost_scales,
        entry_bs=entry_bs,
        entry_nodes=entry_nodes,
        cost_t0s=cost_t0s,
        n_b=sti.n_b,
        n_t=int(sti.n_t),
        n_cost_hex=len(cost_bar_hexes),
        active_gts=list(active),
        spec_tokens=[spec.token for spec in sti.specs],
        entry_part_keys=tuple(entry_part_keys),
    )


def build_sbar_sti_opts(
    *,
    ms_pre,
    ms_response,
    ms_post=0.0,
    ms_sti=None,
    delta_ms,
    delta_ms_pre,
    bar_dist: int,
    bar_directions: Sequence[str],
    multi_bar: bool,
    shift_mid: int = 0,
    a_sti_mids=None,
    gt_cells=None,
):
    if a_sti_mids is None:
        a_sti_mids = sbar_a_sti_mids()
    sti_opts = {
        "ms_pre": ms_pre,
        "ms_response": ms_response,
        "ms_post": ms_post,
        "delta_ms": delta_ms,
        "delta_ms_pre": delta_ms_pre,
        "bar_dist": int(bar_dist),
        "bar_directions": list(bar_directions),
        "multi_bar": bool(multi_bar),
        "shift_mid": int(shift_mid),
        "a_sti_mids": [float(mid) for mid in a_sti_mids],
    }
    if ms_sti is not None:
        sti_opts["ms_sti"] = ms_sti
    if gt_cells is not None:
        sti_opts["gt_cells"] = list(gt_cells)
    if ms_sti is not None and sti_opts.get("ms_response") is not None:
        sti_opts["ms_response"] = max(float(sti_opts["ms_response"]), float(ms_sti))
    return sti_opts


def resolve_sbar_sti_opts(opts, **_):
    return build_sbar_sti_opts(
        ms_pre=opts["ms_pre"],
        ms_response=opts["ms_response"],
        ms_post=opts.get("ms_post", 0.0),
        ms_sti=opts.get("ms_sti"),
        delta_ms=opts["delta_ms"],
        delta_ms_pre=opts["delta_ms_pre"],
        bar_dist=opts["bar_dist"],
        bar_directions=opts["bar_directions"],
        multi_bar=opts["multi_bar"],
        shift_mid=opts.get("shift_mid", opts.get("shift_radius", 0)),
        a_sti_mids=opts.get("a_sti_mids", sbar_a_sti_mids()),
        gt_cells=opts.get("gt_cells"),
    )


def sbar_sti_opts(
    sbar_sti_opts: Optional[dict],
    *,
    ms_pre,
    ms_response,
    ms_post=0.0,
    ms_sti=None,
    delta_ms: float,
    delta_ms_pre: float,
    bar_dist: int,
    bar_directions: Sequence[str],
    multi_bar: bool,
    shift_mid: int = 0,
    a_sti_mids=None,
) -> dict:
    if sbar_sti_opts:
        return dict(sbar_sti_opts)
    return build_sbar_sti_opts(
        ms_pre=ms_pre,
        ms_response=ms_response,
        ms_post=ms_post,
        ms_sti=ms_sti,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        bar_dist=bar_dist,
        bar_directions=bar_directions,
        multi_bar=multi_bar,
        shift_mid=shift_mid,
        a_sti_mids=(
            sbar_a_sti_mids() if a_sti_mids is None else a_sti_mids
        ),
    )


def build_sbar_pack(
    connectome,
    *,
    contrast: str,
    gt_amp: float,
    i_sti: Dict[str, float],
    sti_opts: Optional[dict],
    opts: dict,
):
    del gt_amp
    if not sti_opts:
        raise ValueError("sbar requires sti opts (from resolve_train_opts / CLI)")
    sti_opts = dict(sti_opts)
    # ``shift_radius`` is accepted only for loading older saved run options.
    shift_mid = int(sti_opts.get("shift_mid", sti_opts.get("shift_radius", 0)))
    cost_radius = standardize_cost_radius(sti_opts.get("cost_radius"))
    sbar_gt = build_sbar_gt(
        connectome=connectome,
        ms_pre=float(sti_opts["ms_pre"]),
        ms_response=float(sti_opts["ms_response"]),
        ms_post=float(sti_opts.get("ms_post", 0.0)),
        ms_sti=sti_opts.get("ms_sti"),
        delta_ms=float(sti_opts["delta_ms"]),
        delta_ms_pre=float(sti_opts["delta_ms_pre"]),
        cost_radius=cost_radius,
        i_baseline=i_baseline_from_i_sti(i_sti),
        i_sti=float(i_sti[contrast]),
        contrasts=(contrast,),
        gt_cells=gt_cells_from_opts(sti_opts),
        bar_dist=int(sti_opts["bar_dist"]),
        bar_directions=sti_opts["bar_directions"],
        shift_radius=shift_mid,
        multi_bar=bool(sti_opts["multi_bar"]),
    )
    a_sti_mids = sbar_a_sti_mids(sti_opts)
    t_onset = int(t_from_ms(
        float(sti_opts["ms_pre"]), delta_ms=float(sti_opts["delta_ms_pre"]),
    ))
    specs = gruntman_sbar_specs(
        contrasts=(contrast,), bar_directions=sti_opts["bar_directions"],
        shift_radius=shift_mid,
    )
    pack_i_sti, i_sti_pulse, sti_bs, sti_nodes, a_sti_mid_idxs = (
        build_sbar_a_sti_mid_drive(
            connectome,
            specs,
            sbar_gt.i_sti,
            a_sti_mids=a_sti_mids,
            bar_dist=int(sti_opts["bar_dist"]),
            multi_bar=bool(sti_opts["multi_bar"]),
            t_onset=t_onset,
            n_t=int(sbar_gt.n_t),
            ms_sti=sti_opts.get("ms_sti"),
            delta_ms=float(sti_opts["delta_ms"]),
            i_baseline=i_baseline_from_i_sti(i_sti),
            i_sti=float(i_sti[contrast]),
        )
    )
    sti_opts["n_t"] = int(sbar_gt.n_t)
    sti_opts["spec_tokens"] = list(sbar_gt.spec_tokens)
    if cost_radius is not None:
        sti_opts["cost_radius"] = int(cost_radius)
    sti_opts["gt_cells"] = list(sbar_gt.active_gts)
    pack = SbarPack(
        task="sbar",
        contrast=contrast,
        i_sti=pack_i_sti,
        gts=sbar_gt.gts,
        cost_scales=sbar_gt.cost_scales,
        entry_bs=sbar_gt.entry_bs,
        entry_nodes=sbar_gt.entry_nodes,
        cost_t0s=sbar_gt.cost_t0s,
        cost_ts=build_cost_ts(sti_opts, cost_ms=opts.get("cost_ms")),
        cost_radius=cost_radius,
        t_onset=t_onset,
        i_sti_pulse=i_sti_pulse,
        sti_bs=sti_bs,
        sti_nodes=sti_nodes,
        a_sti_mid_idxs=a_sti_mid_idxs,
        entry_part_keys=sbar_gt.entry_part_keys,
        cost_part_plot_specs=_sbar_cost_part_plot_specs(
            sbar_gt.entry_part_keys,
            sbar_gt.entry_nodes,
            sbar_gt.cost_scales,
            connectome,
            contrast,
        ),
    )
    return pack, sti_opts, (
        f"static-bar {contrast} (B={sbar_gt.n_b} stis, "
        f"{int(sbar_gt.entry_bs.shape[0])} cost nodes, "
        f"{cost_hex_label(cost_radius, sbar_gt.n_cost_hex)})"
    )
