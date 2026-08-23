# -*- coding: utf-8 -*-
"""Spot pack: bind GT numbers to the network for cost.

Cost-radius scales, cost hexes, sti ``i_sti``, and :class:`SpotGt` packing.
GT traces come from :mod:`task.spot.gt`. :class:`SpotPack` is spot-specific.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from network import path  # noqa: F401 -- FAFBv783 on sys.path
import build_hex
from network.construction import (
    cost_radius_mask,
    active_gt_cells,
    node_cells,
    gt_cells_from_opts,
    standardize_cost_radius,
)
from neuron.borst import t_from_ms
from task.spread.gt import GT_CELLS, RF_SIGN, gt_sign, spread_gt_active
from task.spread.pack import (
    CostPartPlotSpec,
    build_cost_ts,
    cost_hex_label,
)
from task.spread.sti_spec import i_baseline_from_i_sti, sti_mask
from task.spot.gt import (
    _spot_readout_a_radius,
    load_rf_ir,
)
from task.spot.sti_geo import (
    SpotB,
    resolve_spot,
    spot_sti_bs,
)
from task.spot.sti_spec import build_spot_a_sti_radius_drive


def part_key(contrast: str, cell: str, radius) -> str:
    return f"spot_{contrast}_{cell}_r{int(radius)}"


def _spot_cost_part_plot_specs(
    entry_part_keys: Sequence[str],
    entry_radii: torch.Tensor,
    entry_nodes: torch.Tensor,
    cost_scales: torch.Tensor,
    connectome,
    contrast: str,
) -> Dict[str, CostPartPlotSpec]:
    specs: Dict[str, CostPartPlotSpec] = {}
    radii = entry_radii.detach().cpu().numpy()
    node_cells = connectome.node_cells[entry_nodes].detach().cpu().numpy()
    cells = connectome.cells
    for entry, part_key in enumerate(entry_part_keys):
        if float(cost_scales[entry]) <= 0.0:
            continue
        if part_key in specs:
            continue
        cell = str(cells[int(node_cells[entry])])
        radius = int(radii[entry])
        specs[part_key] = CostPartPlotSpec(
            part_key,
            cell,
            ("spot_radius", radius),
            f"R{radius} ({contrast})",
        )
    return specs


# -- Cost-radius scales ------------------------------------------------------


def resolve_spot_cost_radii(
    cost_radius_scales: Dict[int, float],
    spot_cost_radii: Tuple[int, ...],
) -> Tuple[int, ...]:
    return tuple(
        radius for radius in spot_cost_radii
        if cost_radius_scales.get(radius, 0.0) != 0.0
    )


def build_a_sti_radius_mask(
    cost_radius_scales: Dict[int, float],
    a_sti_radii: Tuple[int, ...],
) -> Tuple[float, ...]:
    """Per ``a_sti_radii`` radius: ``1`` if cost-radius scale ≠ 0 else ``0``.

    Forward multiplies ``a_sti_radius`` by this mask (indi or fixed).
    """
    return tuple(
        0.0 if cost_radius_scales.get(radius, 0.0) == 0.0 else 1.0
        for radius in a_sti_radii
    )


def spot_a_sti_radii() -> tuple[int, ...]:
    from config import SPOT_PACK
    return tuple(int(radius) for radius in SPOT_PACK["a_sti_radii"])


# -- Cost hexes / network readout --------------------------------------------


def spot_cost_hexes(
    spot_bs: Sequence[SpotB],
    cost_radii,
    cost_radius,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Cost readouts: ``(b, mu, mv, radius, su, sv)`` per sti radius."""
    cost_radii = sorted({int(radius) for radius in cost_radii})
    for radius in cost_radii:
        if radius < 0:
            raise ValueError(f"spot cost radius must be >= 0, got {radius!r}")
    cost_hexes: List[Tuple[int, int, int, int, int, int]] = []
    for b, spot_b in enumerate(spot_bs):
        for su, sv in spot_b.sti_uv:
            for radius in cost_radii:
                for du, dv in build_hex.shell_hexes(radius):
                    mu, mv = su + int(du), sv + int(dv)
                    if not cost_radius_mask(mu, mv, cost_radius):
                        continue
                    cost_hexes.append((
                        b, int(mu), int(mv), int(radius), int(su), int(sv),
                    ))
    return cost_hexes


def spot_n_cost_hex(cost_hexes):
    if not cost_hexes:
        return 0
    n_by_b: Dict[int, int] = {}
    for b, _mu, _mv, _radius, _su, _sv in cost_hexes:
        n_by_b[b] = n_by_b.get(b, 0) + 1
    if len(ns := set(n_by_b.values())) == 1:
        return next(iter(ns))
    return {b: n_by_b[b] for b in sorted(n_by_b)}


def build_spot_cost_readout(connectome, spot_bs, cost_radii, cost_radius):
    network_node_u = np.asarray(connectome.us)
    network_node_v = np.asarray(connectome.vs)
    type_all = connectome.node_cells.detach().cpu().numpy()
    bs, nodes, radius, type_idx, sti_u, sti_v = [], [], [], [], [], []
    for b, mu, mv, cell_radius, su, sv in spot_cost_hexes(
        spot_bs, cost_radii, cost_radius,
    ):
        on_hex = (network_node_u == mu) & (network_node_v == mv)
        for candidate_node in np.where(on_hex)[0]:
            bs.append(b)
            nodes.append(int(candidate_node))
            radius.append(cell_radius)
            type_idx.append(int(type_all[candidate_node]))
            sti_u.append(int(su))
            sti_v.append(int(sv))
    return (
        np.asarray(bs, dtype=np.int64),
        np.asarray(nodes, dtype=np.int64),
        np.asarray(radius, dtype=np.int64),
        np.asarray(type_idx, dtype=np.int64),
        np.asarray(sti_u, dtype=np.int64),
        np.asarray(sti_v, dtype=np.int64),
    )


def build_spot_center_readout(connectome, spot_bs, cost_radii, cost_radius):
    """Cost-node readout plus ``center_entry_mask`` for sti-on hex (radius 0)."""
    bs, nodes, radius, type_idx, sti_u, sti_v = build_spot_cost_readout(
        connectome, spot_bs, cost_radii, cost_radius,
    )
    sti_u = np.asarray(sti_u, dtype=np.int64)
    sti_v = np.asarray(sti_v, dtype=np.int64)
    network_node_u = np.asarray(connectome.us)
    network_node_v = np.asarray(connectome.vs)
    du = np.asarray(network_node_u[nodes] - sti_u, dtype=np.int64)
    dv = np.asarray(network_node_v[nodes] - sti_v, dtype=np.int64)
    center_entry_mask = (du == 0) & (dv == 0)
    return (
        np.asarray(bs, dtype=np.int64),
        np.asarray(nodes, dtype=np.int64),
        np.asarray(radius, dtype=np.int64),
        np.asarray(type_idx, dtype=np.int64),
        sti_u,
        sti_v,
        du,
        dv,
        center_entry_mask,
    )


@dataclass
class SpotGt:
    i_sti: torch.Tensor          # (B, T, N)
    gts: torch.Tensor            # (n_cost, T')
    power: torch.Tensor           # scalar
    cost_scales: torch.Tensor     # (n_cost,)
    entry_radii: torch.Tensor  # (n_cost,) long hex-lattice radius per entry
    entry_bs: torch.Tensor   # (n_cost,) long
    entry_nodes: torch.Tensor    # (n_cost,) long
    entry_sti_us: torch.Tensor  # (n_cost,) long
    entry_sti_vs: torch.Tensor  # (n_cost,) long
    n_b: int
    n_cost_hex: int
    n_center: int
    n_shift: int
    entry_part_keys: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SpotPack:
    """One spot train pack: task×contrast drive + entries + gts."""

    task: str
    contrast: str
    i_sti: torch.Tensor
    gts: torch.Tensor
    cost_scales: torch.Tensor
    entry_bs: torch.Tensor
    entry_nodes: torch.Tensor
    cost_radius: Optional[int] = None
    entry_radii: Optional[torch.Tensor] = None
    cost_sti_us: Optional[torch.Tensor] = None
    cost_sti_vs: Optional[torch.Tensor] = None
    cost_ts: Optional[torch.Tensor] = None
    t_onset: Optional[int] = None
    i_sti_pulse: Optional[torch.Tensor] = None
    sti_bs: Optional[torch.Tensor] = None
    sti_nodes: Optional[torch.Tensor] = None
    a_sti_radius_idxs: Optional[torch.Tensor] = None
    a_sti_radius_mask: Optional[torch.Tensor] = None
    entry_part_keys: Tuple[str, ...] = ()
    cost_part_plot_specs: Optional[Dict[str, CostPartPlotSpec]] = None

    def forward_i_sti(self) -> torch.Tensor:
        i_sti = self.i_sti
        if i_sti.dim() == 3 and int(i_sti.shape[0]) == 1:
            return i_sti.squeeze(0)
        return i_sti


def build_spot_gt(
    connectome,
    *,
    spot_radius: float,
    multi_spot: bool,
    fully_inside: bool,
    shift_radius: int,
    n_t: int,
    t_onset: int,
    i_baseline: float,
    i_sti: float,
    contrast: str,
    gt_amp: float,
    delta_ms: float,
    cost_radius_scales: Dict[int, float],
    spot_cost_radii: Tuple[int, ...],
    cost_radius: Optional[int] = None,
    ms_sti: Optional[float] = None,
    ms_response: Optional[float] = None,
    gt_cells: Optional[Sequence[str]] = None,
    filter: str = "none",
    spread_gt_mode: str,
) -> SpotGt:
    i_baseline = float(i_baseline)
    if ms_response is None:
        raise ValueError("build_spot_gt requires ms_response")
    n_t_gt = int(t_onset) + t_from_ms(float(ms_response), delta_ms=float(delta_ms)) + 1
    if n_t_gt > int(n_t):
        raise ValueError(
            f"spot gt n_t={n_t_gt} exceeds forward n_t={n_t} "
            f"(ms_response={ms_response:g}, t_onset={t_onset})"
        )
    rf, ir = load_rf_ir(
        t_onset=t_onset, n_t=n_t_gt, ms_sti=ms_sti, delta_ms=delta_ms,
        filter=filter,
    )
    gt_type_idx = dict(zip(
        [str(cell) for cell in GT_CELLS], range(len(GT_CELLS)),
    ))
    if gt_cells is not None:
        bad = [str(t) for t in gt_cells if str(t) not in gt_type_idx]
        if bad:
            raise ValueError(
                f"unknown spot gt cell(s) {bad!r} "
                f"(expected subset of {list(GT_CELLS)})",
            )

    spot = resolve_spot(
        connectome,
        spot_radius=spot_radius,
        shift_radius=shift_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
    )
    cells = node_cells(connectome)
    active = active_gt_cells(gt_cells, GT_CELLS, connectome.cells, context="spot")

    spot_bs = spot_sti_bs(spot)
    n_b = len(spot_bs)

    drive = i_baseline + (float(i_sti) - i_baseline) * sti_mask(
        t_onset, n_t, ms_sti, delta_ms=delta_ms,
    )
    sti_nodes = np.asarray(connectome.sti_nodes, dtype=np.int64)
    i_sti = np.zeros((n_b, n_t, connectome.n_node), dtype=np.float64)
    if len(sti_nodes):
        i_sti[:, :, sti_nodes] = float(i_baseline)
    for b, spot_b in enumerate(spot_bs):
        for su, sv in spot_b.sti_uv:
            nodes = connectome.sti_nodes_at_uv(su, sv)
            if len(nodes):
                i_sti[b, :, nodes] = drive[:, None]

    resp = slice(t_onset, n_t_gt)  # cost window: response only (no ms_post)

    cost_hexes = spot_cost_hexes(
        spot_bs,
        resolve_spot_cost_radii(cost_radius_scales, spot_cost_radii),
        cost_radius,
    )

    cost_bs, cost_node, entry_radii_vals, cost_readout, cost_scales_vals = [], [], [], [], []
    cost_sti_us, cost_sti_vs = [], []
    entry_part_keys: List[str] = []
    trace_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for b, mu, mv, radius, su, sv in cost_hexes:
        scale = cost_radius_scales.get(radius, 0.0)
        if scale == 0.0:
            continue
        for gt_cell in active:
            gt_idx = gt_type_idx[gt_cell]
            nodes = connectome.nodes_at_uv(mu, mv, gt_cell, cells=cells)
            if len(nodes) == 0:
                continue
            rf_sign = int(RF_SIGN[gt_cell])
            if not spread_gt_active(spread_gt_mode, contrast, rf_sign):
                continue
            cache_digest = (int(radius), gt_idx)
            if cache_digest not in trace_cache:
                trace_cache[cache_digest] = (
                    _spot_readout_a_radius(rf[gt_idx], int(radius), spot_radius)
                    * ir[gt_idx][resp]
                    * gt_amp
                    * gt_sign(contrast, rf_sign)
                )
            trace = trace_cache[cache_digest]
            for node in nodes:
                cost_bs.append(b)
                cost_node.append(int(node))
                entry_radii_vals.append(int(radius))
                cost_readout.append(trace)
                cost_scales_vals.append(scale)
                cost_sti_us.append(int(su))
                cost_sti_vs.append(int(sv))
                entry_part_keys.append(part_key(contrast, gt_cell, radius))

    if not cost_bs:
        raise ValueError("no spot cost nodes (check cost_radius and gt cells)")

    gts = np.asarray(cost_readout, dtype=np.float64)
    cost_scales = np.asarray(cost_scales_vals, dtype=np.float64)
    entry_radii = np.asarray(entry_radii_vals, dtype=np.int64)
    entry_bs = np.asarray(cost_bs, dtype=np.int64)
    entry_nodes = np.asarray(cost_node, dtype=np.int64)
    entry_sti_us = np.asarray(cost_sti_us, dtype=np.int64)
    entry_sti_vs = np.asarray(cost_sti_vs, dtype=np.int64)

    power = float(np.sum(cost_scales[:, None] * gts ** 2))
    if power == 0.0:
        power = 1.0

    return SpotGt(
        i_sti=i_sti,
        gts=gts,
        power=power,
        cost_scales=cost_scales,
        entry_radii=entry_radii,
        entry_bs=entry_bs,
        entry_nodes=entry_nodes,
        entry_sti_us=entry_sti_us,
        entry_sti_vs=entry_sti_vs,
        n_b=n_b,
        n_cost_hex=spot_n_cost_hex(cost_hexes),
        n_center=len(spot.centers),
        n_shift=len(spot.shifts),
        entry_part_keys=tuple(entry_part_keys),
    )


def build_spot_sti_opts(
    *,
    ms_pre,
    ms_response,
    delta_ms,
    delta_ms_pre,
    shift_radius: int,
    spot_radius,
    multi_spot: bool,
    fully_inside: bool,
    ms_sti=None,
    ms_post=0.0,
    gt_cells=None,
):
    """Spot sti opts: timing / geometry only (currents live on session ``i_sti``)."""
    opts = {
        "ms_pre": ms_pre,
        "ms_response": ms_response,
        "ms_post": ms_post,
        "delta_ms": delta_ms,
        "delta_ms_pre": delta_ms_pre,
        "shift_radius": shift_radius,
        "spot_radius": spot_radius,
        "multi_spot": multi_spot,
        "fully_inside": fully_inside,
    }
    if ms_sti is not None:
        opts["ms_sti"] = ms_sti
    if gt_cells is not None:
        opts["gt_cells"] = list(gt_cells)
    ms_sti = opts.get("ms_sti")
    ms_response = opts.get("ms_response")
    if ms_sti is not None and ms_response is not None:
        opts["ms_response"] = max(float(ms_response), float(ms_sti))
    return opts


def resolve_spot_sti_opts(opts, **_):
    return build_spot_sti_opts(
        ms_pre=opts["ms_pre"],
        ms_response=opts["ms_response"],
        ms_post=opts["ms_post"],
        delta_ms=opts["delta_ms"],
        delta_ms_pre=opts["delta_ms_pre"],
        shift_radius=opts["shift_radius"],
        spot_radius=opts["spot_radius"],
        multi_spot=opts["multi_spot"],
        fully_inside=opts["fully_inside"],
        ms_sti=opts.get("ms_sti"),
        gt_cells=opts.get("gt_cells"),
    )


def build_spot_pack(
    connectome,
    *,
    contrast: str,
    gt_amp: float,
    i_sti: Dict[str, float],
    sti_opts: Optional[dict],
    opts: dict,
) -> Tuple[SpotPack, dict, str]:
    from config import SPOT_PACK, TRAIN_OPTIMIZATION

    if not sti_opts:
        raise ValueError("spot requires sti opts (from resolve_train_opts / CLI)")
    sti_opts = dict(sti_opts)
    filter = str(opts.get("filter", "none"))
    spread_gt_mode = str(opts["spread_gt_mode"])
    cost_ms = opts.get("cost_ms", TRAIN_OPTIMIZATION["cost_ms"])
    cost_radius_scales = SPOT_PACK["spot_cost_radius_scale"]
    spot_cost_radii = SPOT_PACK["spot_cost_radii"]
    a_sti_radii = spot_a_sti_radii()
    ms_sti = sti_opts.get("ms_sti")
    ms_response = sti_opts.get("ms_response")
    if ms_sti is not None and ms_response is not None:
        sti_opts["ms_response"] = max(float(ms_response), float(ms_sti))
    for key in ("ms_pre", "ms_response", "delta_ms", "delta_ms_pre"):
        if sti_opts.get(key) is None:
            raise ValueError(f"spot sti opts require {key}")
    ms_pre = float(sti_opts["ms_pre"])
    ms_response = float(sti_opts["ms_response"])
    delta_ms = float(sti_opts["delta_ms"])
    delta_ms_pre = float(sti_opts["delta_ms_pre"])
    ms_post = float(sti_opts.get("ms_post", 0.0))
    ms_sti = sti_opts.get("ms_sti")
    cost_radius = standardize_cost_radius(sti_opts.get("cost_radius"))
    shift_radius = sti_opts["shift_radius"]
    spot_radius = sti_opts["spot_radius"]
    multi_spot = sti_opts["multi_spot"]
    fully_inside = sti_opts["fully_inside"]
    t_onset = int(t_from_ms(ms_pre, delta_ms=delta_ms_pre))
    n_t = int(
        t_onset
        + t_from_ms(ms_response, delta_ms=delta_ms)
        + t_from_ms(ms_post, delta_ms=delta_ms)
        + 1
    )
    i_baseline = i_baseline_from_i_sti(i_sti)
    spot_gt = build_spot_gt(
        connectome,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        shift_radius=shift_radius,
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
        spot_cost_radii=spot_cost_radii,
        gt_cells=gt_cells_from_opts(sti_opts),
        filter=str(filter),
        spread_gt_mode=str(spread_gt_mode),
    )
    i_sti, i_sti_pulse, sti_bs, sti_nodes, a_sti_radius_idxs = build_spot_a_sti_radius_drive(
        connectome,
        spot_sti_bs(resolve_spot(connectome, sti_opts=sti_opts)),
        a_sti_radii=a_sti_radii,
        t_onset=int(t_onset),
        n_t=int(n_t),
        ms_sti=ms_sti,
        delta_ms=delta_ms,
        i_baseline=i_baseline,
        i_sti=float(i_sti[contrast]),
    )
    pack = SpotPack(
        task="spot",
        contrast=contrast,
        i_sti=i_sti,
        gts=spot_gt.gts,
        cost_scales=spot_gt.cost_scales,
        entry_bs=spot_gt.entry_bs,
        entry_nodes=spot_gt.entry_nodes,
        cost_sti_us=spot_gt.entry_sti_us,
        cost_sti_vs=spot_gt.entry_sti_vs,
        cost_radius=cost_radius,
        entry_radii=spot_gt.entry_radii,
        cost_ts=None if int(spot_gt.entry_radii.shape[0]) == 0 else build_cost_ts(
            sti_opts, cost_ms=cost_ms,
        ),
        t_onset=int(t_onset),
        i_sti_pulse=i_sti_pulse,
        sti_bs=sti_bs,
        sti_nodes=sti_nodes,
        a_sti_radius_idxs=a_sti_radius_idxs,
        a_sti_radius_mask=build_a_sti_radius_mask(cost_radius_scales, a_sti_radii),
        entry_part_keys=spot_gt.entry_part_keys,
        cost_part_plot_specs=_spot_cost_part_plot_specs(
            spot_gt.entry_part_keys,
            spot_gt.entry_radii,
            spot_gt.entry_nodes,
            spot_gt.cost_scales,
            connectome,
            contrast,
        ),
    )
    return pack, dict(sti_opts), (
        f"spot {contrast} (B={spot_gt.n_b} stis [{spot_gt.n_center} centers simultaneous "
        f"x {spot_gt.n_shift} shifts], {int(spot_gt.gts.shape[0])} cost nodes, "
        f"{cost_hex_label(cost_radius, spot_gt.n_cost_hex)})"
    )
