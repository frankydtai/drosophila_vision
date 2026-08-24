# -*- coding: utf-8 -*-
"""Spread pack: uniform ``i_sti`` over all sti nodes; ir-only gt cost."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import build_hex
from import_bootstrap import parse_comma_list
from network import path  # noqa: F401 -- FAFBv783 on sys.path
from network.construction import (
    cost_radius_mask,
    gt_cells_from_opts,
    node_cells,
)
from neuron.borst import t_from_ms
from task.spread.gt import (
    GT_CELLS,
    RF_SIGN,
    expand_gt_cells,
    gt_sign,
    load_ir,
    spread_gt_active,
)
from task.spread.sti_spec import i_baseline_from_i_sti, sti_mask


@dataclass(frozen=True)
class CostPartPlotSpec:
    part_key: str
    cell: Optional[str]
    series: tuple
    label: str


@dataclass(frozen=True)
class SpreadPack:
    """One train pack: task×contrast drive + entries + gts."""

    task: str
    contrast: str
    i_sti: torch.Tensor  # (B, T, N)
    gts: torch.Tensor  # (n_cost, T')
    entry_bs: torch.Tensor  # (n_cost,)
    entry_nodes: torch.Tensor  # (n_cost,)
    cost_ts: Optional[torch.Tensor] = None
    t_onset: Optional[int] = None
    entry_part_keys: Tuple[str, ...] = ()
    cost_part_plot_specs: Optional[Dict[str, CostPartPlotSpec]] = None


def part_key(contrast: str, cell: str) -> str:
    return f"spread_{contrast}_{cell}"


def _spread_cost_part_plot_specs(
    entry_part_keys: Sequence[str],
    entry_nodes,
    connectome,
    contrast: str,
) -> Dict[str, CostPartPlotSpec]:
    specs: Dict[str, CostPartPlotSpec] = {}
    entry_nodes_np = np.asarray(entry_nodes, dtype=np.int64)
    node_cells_np = connectome.node_cells[entry_nodes_np]
    if torch.is_tensor(node_cells_np):
        node_cells_np = node_cells_np.detach().cpu().numpy()
    cells = connectome.cells
    for entry, part_key in enumerate(entry_part_keys):
        if part_key in specs:
            continue
        cell = str(cells[int(node_cells_np[entry])])
        specs[part_key] = CostPartPlotSpec(
            part_key, cell, ("spread", contrast), contrast,
        )
    return specs


@dataclass
class SpreadGt:
    i_sti: torch.Tensor
    gts: torch.Tensor
    entry_bs: torch.Tensor
    entry_nodes: torch.Tensor
    n_cost_hex: int = 0
    entry_part_keys: Tuple[str, ...] = ()


def build_spread_gt(
    connectome,
    *,
    n_t: int,
    t_onset: int,
    i_baseline: float,
    i_sti: float,
    contrast: str,
    gt_amp: float,
    delta_ms: float,
    ms_sti: Optional[float] = None,
    ms_response: Optional[float] = None,
    gt_cells: Optional[Sequence[str]] = None,
    filter: str = "none",
    spread_gt_mode: str,
) -> SpreadGt:
    if ms_response is None:
        raise ValueError("build_spread_gt requires ms_response")
    n_t_gt = int(t_onset) + t_from_ms(float(ms_response), delta_ms=float(delta_ms)) + 1
    if n_t_gt > int(n_t):
        raise ValueError(
            f"spread gt n_t={n_t_gt} exceeds forward n_t={n_t} "
            f"(ms_response={ms_response:g}, t_onset={t_onset})"
        )
    ir = load_ir(
        t_onset=t_onset,
        n_t=n_t_gt,
        ms_sti=ms_sti,
        delta_ms=delta_ms,
        filter=filter,
    )
    cell_idx = {cell: i for i, cell in enumerate(GT_CELLS)}
    gt_cells = [
        cell
        for cell in (GT_CELLS if gt_cells is None else expand_gt_cells(gt_cells))
        if cell in connectome.cells
    ]
    if not gt_cells:
        raise ValueError(f"spread has no gt cells (requested subset of {list(GT_CELLS)!r})")
    cells = node_cells(connectome)
    hexes = cost_sti_hexes(connectome)
    i_sti_pulse = (float(i_sti) - i_baseline) * sti_mask(
        t_onset, n_t, ms_sti, delta_ms=delta_ms,
    )
    sti_nodes = np.asarray(connectome.sti_nodes, dtype=np.int64)
    i_sti = np.zeros((1, n_t, connectome.n_node), dtype=np.float64)
    if len(sti_nodes):
        i_sti[:, :, sti_nodes] = float(i_baseline) + i_sti_pulse[:, None]
    entry_nodes = []
    gts = []
    entry_part_keys: List[str] = []
    for cell in gt_cells:
        if not spread_gt_active(spread_gt_mode, contrast, int(RF_SIGN[cell])):
            continue
        gt = ir[cell_idx[cell]][slice(t_onset, n_t_gt)] * gt_amp * gt_sign(contrast, RF_SIGN[cell])
        for hex in hexes:
            for node in connectome.nodes_at_uv(hex.u, hex.v, cell, cells=cells):
                entry_nodes.append(int(node))
                gts.append(gt)
                entry_part_keys.append(part_key(contrast, cell))
    if not entry_nodes:
        raise ValueError("no spread cost nodes (check gt cells)")
    gts = np.asarray(gts, dtype=np.float64)
    entry_nodes = np.asarray(entry_nodes, dtype=np.int64)
    return SpreadGt(
        i_sti=i_sti,
        gts=gts,
        entry_bs=np.zeros(len(entry_nodes), dtype=np.int64),
        entry_nodes=entry_nodes,
        n_cost_hex=len(hexes),
        entry_part_keys=tuple(entry_part_keys),
    )


def post_onset_n_t(opts) -> tuple[int, float]:
    ms_response = float(opts["ms_response"])
    if opts.get("ms_sti") is not None:
        ms_response = max(ms_response, float(opts["ms_sti"]))
    delta_ms = float(opts["delta_ms"])
    return int(t_from_ms(ms_response, delta_ms=delta_ms)) + 1, delta_ms


def cost_mss(cost_ms, *, post, delta_ms) -> list:
    """``cost_ms`` scalar → interval grid; sequence → explicit post-onset ``mss``."""
    if isinstance(cost_ms, bool) or cost_ms is None:
        raise ValueError("cost_ms must be an interval or a list of ms")
    if isinstance(cost_ms, dict):
        raise ValueError("cost_ms must be an interval or a list of ms, not a radius map")
    if isinstance(cost_ms, (int, float)):
        interval = float(cost_ms)
        if interval <= 0:
            raise ValueError("cost_ms interval must be > 0")
        if post <= 0:
            raise ValueError("post-onset window must be > 0 for cost_ms")
        step = max(1, int(round(interval / delta_ms)))
        return [t * delta_ms for t in range(0, post, step)]
    if isinstance(cost_ms, str):
        tokens = parse_comma_list(cost_ms)
        if len(tokens) == 1:
            return cost_mss(float(tokens[0]), post=post, delta_ms=delta_ms)
        cost_ms = tokens
    mss = [float(ms) for ms in cost_ms]
    if not mss:
        raise ValueError("cost_ms list must have at least one ms")
    return mss


def build_cost_ts(opts, *, cost_ms):
    if cost_ms is None:
        return None
    post, delta_ms = post_onset_n_t(opts)
    ts = set()
    for ms in cost_mss(cost_ms, post=post, delta_ms=delta_ms):
        t = int(round(float(ms) / delta_ms))
        if t < 0 or t >= post:
            raise ValueError(
                f"cost time {ms} ms post-onset t out of range [0,{post})"
            )
        ts.add(t)
    return np.asarray(sorted(ts), dtype=np.int64)


def cost_hex_label(cost_radius, n_cost_hex) -> str:
    radius_label = "all hexes" if cost_radius is None else f"radius={int(cost_radius)}"
    if isinstance(n_cost_hex, dict):
        hex_labels = ", ".join(
            f"b{int(b)}={int(n_hex)}"
            for b, n_hex in sorted(n_cost_hex.items())
        )
        return f"cost hexes per b [{hex_labels}], {radius_label}"
    return f"{int(n_cost_hex)} cost hexes, {radius_label}"


@dataclass(frozen=True)
class CostStiHex:
    """One sti hex used for cost selection and at_xy filtering."""

    u: int
    v: int
    x: float
    y: float


def cost_sti_hexes(connectome, cost_radius=None) -> List[CostStiHex]:
    """Sti hexes used for cost (optional central hex disc)."""
    hexes_by_uv: Dict[Tuple[int, int], CostStiHex] = {}
    for node in connectome.sti_nodes:
        u = int(connectome.us[node])
        v = int(connectome.vs[node])
        if (u, v) in hexes_by_uv or not cost_radius_mask(u, v, cost_radius):
            continue
        x, y = build_hex.xy_from_uv(u, v)
        hexes_by_uv[(u, v)] = CostStiHex(u=u, v=v, x=float(x), y=float(y))
    return [hexes_by_uv[(u, v)] for u, v in sorted(hexes_by_uv)]


def resolve_spread_sti_opts(opts, **_):
    ms_sti = opts["ms_sti"]
    gt_cells = opts.get("gt_cells")
    sti_opts = {
        "ms_pre": opts["ms_pre"],
        "ms_response": opts["ms_response"],
        "ms_post": opts["ms_post"],
        "delta_ms": opts["delta_ms"],
        "delta_ms_pre": opts["delta_ms_pre"],
    }
    if ms_sti is not None:
        sti_opts["ms_sti"] = ms_sti
    if gt_cells is not None:
        sti_opts["gt_cells"] = list(gt_cells)
    ms_response = sti_opts.get("ms_response")
    if ms_sti is not None and ms_response is not None:
        sti_opts["ms_response"] = max(float(ms_response), float(ms_sti))
    return sti_opts


def build_spread_pack(
    connectome,
    *,
    contrast: str,
    gt_amp: float,
    i_sti: Dict[str, float],
    sti_opts: Optional[dict],
    opts: dict,
) -> Tuple[SpreadPack, dict, str]:
    from config import TRAIN_OPTIMIZATION

    if not sti_opts:
        raise ValueError("spread requires sti opts (from resolve_train_opts / CLI)")
    sti_opts = dict(sti_opts)
    spread_gt_mode = str(opts["spread_gt_mode"])
    cost_ms = opts.get("cost_ms", TRAIN_OPTIMIZATION["cost_ms"])
    filter = str(opts.get("filter", "none"))
    ms_sti = sti_opts.get("ms_sti")
    ms_response = sti_opts.get("ms_response")
    if ms_sti is not None and ms_response is not None:
        sti_opts["ms_response"] = max(float(ms_response), float(ms_sti))
    for key in ("ms_pre", "ms_response", "delta_ms", "delta_ms_pre"):
        if sti_opts.get(key) is None:
            raise ValueError(f"spread sti opts require {key}")
    ms_pre = float(sti_opts["ms_pre"])
    ms_response = float(sti_opts["ms_response"])
    delta_ms = float(sti_opts["delta_ms"])
    delta_ms_pre = float(sti_opts["delta_ms_pre"])
    ms_post = float(sti_opts.get("ms_post", 0.0))
    ms_sti = sti_opts.get("ms_sti")
    t_onset = int(t_from_ms(ms_pre, delta_ms=delta_ms_pre))
    n_t = int(
        t_onset
        + t_from_ms(ms_response, delta_ms=delta_ms)
        + t_from_ms(ms_post, delta_ms=delta_ms)
        + 1
    )
    spread_gt = build_spread_gt(
        connectome,
        n_t=n_t,
        t_onset=t_onset,
        i_baseline=i_baseline_from_i_sti(i_sti),
        i_sti=float(i_sti[contrast]),
        contrast=contrast,
        gt_amp=gt_amp,
        delta_ms=delta_ms,
        ms_sti=ms_sti,
        ms_response=ms_response,
        gt_cells=gt_cells_from_opts(sti_opts),
        filter=filter,
        spread_gt_mode=spread_gt_mode,
    )
    pack = SpreadPack(
        task="spread",
        contrast=contrast,
        i_sti=spread_gt.i_sti,
        gts=spread_gt.gts,
        entry_bs=spread_gt.entry_bs,
        entry_nodes=spread_gt.entry_nodes,
        cost_ts=build_cost_ts(
            sti_opts,
            cost_ms=cost_ms,
        ),
        t_onset=int(t_onset),
        entry_part_keys=spread_gt.entry_part_keys,
        cost_part_plot_specs=_spread_cost_part_plot_specs(
            spread_gt.entry_part_keys,
            spread_gt.entry_nodes,
            connectome,
            contrast,
        ),
    )
    return (
        pack,
        dict(sti_opts),
        f"spread {contrast} ({int(spread_gt.gts.shape[0])} cost nodes, "
        f"{int(spread_gt.n_cost_hex)} cost hexes)",
    )
