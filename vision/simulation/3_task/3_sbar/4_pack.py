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
from task.mbar.gt import GT_CELLS
from task.mbar.sti_geo import sbar_line_hex_mask
from task.sbar.gt import load_gt
from task.sbar.sti_geo import sti_hexes
from task.sbar.sti_spec import (
    build_sbar_signals,
    gruntman_sbar_specs,
)
from task.spread.pack import CostPartPlotSpec, cost_hex_label, cost_sti_hexes
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
    cost_radius: Optional[int] = None
    entry_part_keys: Tuple[str, ...] = ()
    cost_part_plot_specs: Optional[Dict[str, CostPartPlotSpec]] = None


def part_key(contrast: str, cell: str, mid) -> str:
    return f"sbar_{contrast}_{cell}_mid{(f'{int(mid):+d}' if float(mid).is_integer() else f'{float(mid):+.1f}')}"


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
    multi_bar: bool = True,
    cost_radius: Optional[int] = None,
    i_baseline: float,
    i_sti: float,
    contrasts: Sequence[str],
    gt_cells: Optional[Sequence[str]] = None,
) -> SbarGt:
    specs = gruntman_sbar_specs(
        contrasts=tuple(contrasts),
        bar_directions=bar_directions,
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
    bar_span = 1 + int(bar_dist)
    t0 = int(sti.t_onset)
    n_t_cost = int(t_from_ms(float(ms_response), delta_ms=float(delta_ms)) + 1)

    entry_bs, entry_nodes, entry_gts, entry_cost_scales, entry_cost_t0s = (
        [], [], [], [], [],
    )
    entry_part_keys: List[str] = []
    hex_idx = {
        (int(hex.u), int(hex.v)): hex_idx
        for hex_idx, hex in enumerate(hexes)
    }

    for b, spec in enumerate(sti.specs):
        hex_mask = sbar_line_hex_mask(
            hexes, spec.direction, int(bar_dist), multi_bar=bool(multi_bar),
        )
        if spec.direction in ("right", "left"):
            mids = sorted(
                float(hexes[hex_idx].x)
                for hex_idx in range(len(hexes))
                if hex_mask[hex_idx] > 0.0
            )
        else:
            mids = sorted(
                float(hexes[hex_idx].y)
                for hex_idx in range(len(hexes))
                if hex_mask[hex_idx] > 0.0
            )
        mids_set = set(mids)
        for hex in cost_hexes:
            if hex_mask[hex_idx[(int(hex.u), int(hex.v))]] <= 0.0:
                continue
            mid = float(hex.x if spec.direction in ("right", "left") else hex.y)
            for cell in active:
                nodes = connectome.nodes_at_uv(hex.u, hex.v, cell, cells=cells)
                if len(nodes) == 0:
                    continue
                gt = None
                for mid0 in mids:
                    if not (0.0 < mid - mid0 < float(bar_span)):
                        continue
                    if (mid0 + bar_span) not in mids_set:
                        continue
                    b_off = mid - mid0 - bar_span
                    if (
                        f"{cell[:2]}_{'PC' if (cell.startswith('T4') and spec.contrast == 'bright') or (cell.startswith('T5') and spec.contrast == 'dark') else 'NC'}_pos{(f'{int(mid):+d}' if float(mid).is_integer() else f'{float(mid):+.1f}')}_w1" in gts
                        and f"{cell[:2]}_{'PC' if (cell.startswith('T4') and spec.contrast == 'bright') or (cell.startswith('T5') and spec.contrast == 'dark') else 'NC'}_pos{(f'{int(b_off):+d}' if float(b_off).is_integer() else f'{float(b_off):+.1f}')}_w1" in gts
                    ):
                        gt = (
                            gts[f"{cell[:2]}_{'PC' if (cell.startswith('T4') and spec.contrast == 'bright') or (cell.startswith('T5') and spec.contrast == 'dark') else 'NC'}_pos{(f'{int(mid):+d}' if float(mid).is_integer() else f'{float(mid):+.1f}')}_w1"][t0:t0 + n_t_cost]
                            + gts[f"{cell[:2]}_{'PC' if (cell.startswith('T4') and spec.contrast == 'bright') or (cell.startswith('T5') and spec.contrast == 'dark') else 'NC'}_pos{(f'{int(b_off):+d}' if float(b_off).is_integer() else f'{float(b_off):+.1f}')}_w1"][t0:t0 + n_t_cost]
                        )
                        break
                if gt is None:
                    if f"{cell[:2]}_{'PC' if (cell.startswith('T4') and spec.contrast == 'bright') or (cell.startswith('T5') and spec.contrast == 'dark') else 'NC'}_pos{(f'{int(mid):+d}' if float(mid).is_integer() else f'{float(mid):+.1f}')}_w1" not in gts:
                        continue
                    gt = gts[f"{cell[:2]}_{'PC' if (cell.startswith('T4') and spec.contrast == 'bright') or (cell.startswith('T5') and spec.contrast == 'dark') else 'NC'}_pos{(f'{int(mid):+d}' if float(mid).is_integer() else f'{float(mid):+.1f}')}_w1"][t0:t0 + n_t_cost]
                for node in nodes:
                    entry_bs.append(b)
                    entry_nodes.append(int(node))
                    entry_gts.append(gt)
                    entry_cost_scales.append(1.0)
                    entry_cost_t0s.append(t0)
                    entry_part_keys.append(part_key(spec.contrast, cell, mid))

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
        i_sti=torch.as_tensor(sti.i_sti, dtype=torch.float64),
        gts=gts,
        power=power,
        cost_scales=cost_scales,
        entry_bs=entry_bs,
        entry_nodes=entry_nodes,
        cost_t0s=cost_t0s,
        n_b=sti.n_b,
        n_t=int(sti.n_t),
        n_cost_hex=len(cost_hexes),
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
    gt_cells=None,
):
    sti_opts = {
        "ms_pre": ms_pre,
        "ms_response": ms_response,
        "ms_post": ms_post,
        "delta_ms": delta_ms,
        "delta_ms_pre": delta_ms_pre,
        "bar_dist": int(bar_dist),
        "bar_directions": list(bar_directions),
        "multi_bar": bool(multi_bar),
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
    del gt_amp, opts
    if not sti_opts:
        raise ValueError("sbar requires sti opts (from resolve_train_opts / CLI)")
    sti_opts = dict(sti_opts)
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
        multi_bar=bool(sti_opts["multi_bar"]),
    )
    sti_opts["n_t"] = int(sbar_gt.n_t)
    sti_opts["spec_tokens"] = list(sbar_gt.spec_tokens)
    if cost_radius is not None:
        sti_opts["cost_radius"] = int(cost_radius)
    sti_opts["gt_cells"] = list(sbar_gt.active_gts)
    pack = SbarPack(
        task="sbar",
        contrast=contrast,
        i_sti=sbar_gt.i_sti,
        gts=sbar_gt.gts,
        cost_scales=sbar_gt.cost_scales,
        entry_bs=sbar_gt.entry_bs,
        entry_nodes=sbar_gt.entry_nodes,
        cost_t0s=sbar_gt.cost_t0s,
        cost_radius=cost_radius,
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
