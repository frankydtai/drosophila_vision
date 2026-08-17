# -*- coding: utf-8 -*-
"""Spot pack: bind GT numbers to the network for cost.

Cost-radius scales, cost hexes, sti ``i_sti``, and :class:`SpotGt` packing.
GT traces come from :mod:`task.spot.gt`. Sparse cost time points and
``Pack`` wrapping lives in the ``train`` layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from network import path  # noqa: F401 -- FAFBv783 on sys.path
import build_hex
from import_bootstrap import parse_comma_list
from network.construction import (
    hex2gt,
    cost_radius_mask,
    active_gt_cells,
    node_cells,
)
from neuron.borst import t_from_ms
from task.spread.gt import GT_CELLS, RF_SIGN, contrast_sign, spread_gt_active
from task.spread.sti_spec import standardize_sti_timing, sti_mask
from task.spot.gt import (
    _spot_readout_a_radius,
    load_rf_ir,
    spot_gt_active,
)
from task.spot.sti_geo import (
    SpotB,
    resolve_spot,
    spot_sti_bs,
)

# Spot contrasts (independent of task; bright | dark).
SPOT_CONTRASTS = frozenset({"bright", "dark"})


# -- Cost-radius scales ------------------------------------------------------


def standardize_spot_cost_radius(radius) -> int:
    """Parse a spot cost / drive radius to an int hex-lattice radius."""
    if isinstance(radius, bool):
        raise ValueError(f"invalid spot cost radius {radius!r}")
    if isinstance(radius, int):
        return int(radius)
    if isinstance(radius, float):
        if not float(radius).is_integer():
            raise ValueError(f"spot cost radius must be an int, got {radius!r}")
        return int(radius)
    token = str(radius).strip()
    try:
        return int(token, 10)
    except ValueError as exc:
        raise ValueError(f"spot cost radius must be an int, got {radius!r}") from exc


def parse_spot_cost_radius_scale_value(token: str) -> float:
    token = str(token).strip()
    if "/" in token:
        num, den = token.split("/", 1)
        return float(num.strip()) / float(den.strip())
    return float(token)


def expand_spot_cost_radius_scale(
    spot_cost_radius_scale: Optional[dict] = None,
    *,
    sti_opts: Optional[dict] = None,
) -> Optional[Dict[int, float]]:
    if sti_opts is not None:
        spot_cost_radius_scale = (sti_opts or {}).get("spot_cost_radius_scale")
    if not spot_cost_radius_scale:
        return None
    return {
        standardize_spot_cost_radius(radius): parse_spot_cost_radius_scale_value(scale)
        for radius, scale in spot_cost_radius_scale.items()
    }


def expand_cost_ms(
    *,
    cost_ms: Optional[dict] = None,
) -> Dict[int, Tuple[float, ...]]:
    """Radius → explicit post-onset ms; empty when unset."""
    if not cost_ms:
        return {}
    out: Dict[int, Tuple[float, ...]] = {}
    for radius, mss in cost_ms.items():
        radius = standardize_spot_cost_radius(radius)
        mss = parse_comma_list(mss) if isinstance(mss, str) else list(mss)
        if not mss:
            raise ValueError(f"cost_ms[{radius!r}] must list at least one ms")
        out[radius] = tuple(float(ms) for ms in mss)
    return out


def parse_cost_ms_tokens(
    tokens: Optional[Sequence[str]],
) -> Optional[Dict[int, Tuple[float, ...]]]:
    """Parse ``--cost-ms``: ``none``/``off`` → ``{}``; else ``R=MS,...``. Omit → ``None``."""
    if tokens is None:
        return None
    if len(tokens) == 1 and str(tokens[0]).strip().lower() in ("none", "off"):
        return {}
    out: Dict[int, Tuple[float, ...]] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"expected R=MS,... or none|off, got {token!r}")
        radius, val = token.split("=", 1)
        vals = parse_comma_list(val)
        if not vals:
            raise ValueError(f"--cost-ms {radius}=... must list at least one ms")
        out[standardize_spot_cost_radius(radius)] = tuple(float(x) for x in vals)
    return out


def resolve_spot_cost_radius_scale(
    tokens: Optional[Sequence[str]],
    *,
    cost_radius_scales: Dict[int, float],
    spot_cost_radii: Tuple[int, ...],
) -> Optional[Dict[int, float]]:
    """Parse ``--spot-cost-radius-scale`` space-separated ``R`` / ``R=S`` tokens."""
    if not tokens:
        return None
    bare: list[int] = []
    explicit: Dict[int, float] = {}
    for token in tokens:
        if "=" in token:
            radius, val = token.split("=", 1)
            explicit[standardize_spot_cost_radius(radius)] = (
                parse_spot_cost_radius_scale_value(val)
            )
        else:
            bare.append(standardize_spot_cost_radius(token))
    if bare:
        scales = {int(radius): 0.0 for radius in spot_cost_radii}
        for radius in bare:
            scales[radius] = 1.0
    else:
        scales = dict(cost_radius_scales)
    scales.update(explicit)
    return scales


def resolve_spot_cost_radii(
    spot_cost_radius_scale: Optional[Dict[int, float]] = None,
    *,
    cost_radius_scales: Dict[int, float],
    spot_cost_radii: Tuple[int, ...],
    sti_opts: Optional[dict] = None,
) -> Tuple[int, ...]:
    if sti_opts is not None:
        spot_cost_radius_scale = expand_spot_cost_radius_scale(sti_opts=sti_opts)
    scales = (
        dict(cost_radius_scales)
        if spot_cost_radius_scale is None
        else spot_cost_radius_scale
    )
    return tuple(
        int(radius) for radius in spot_cost_radii
        if float(scales.get(int(radius), 0.0)) != 0.0
    )


def build_a_sti_radius_mask(
    spot_cost_radius_scale: Optional[Dict[int, float]] = None,
    *,
    cost_radius_scales: Dict[int, float],
    a_sti_radii: Tuple[int, ...],
) -> Tuple[float, ...]:
    """Per ``a_sti_radii`` radius: ``1`` if cost-radius scale ≠ 0 else ``0``.

    Forward multiplies ``a_sti_radius`` by this mask (indi or fixed).
    """
    scales = (
        dict(cost_radius_scales)
        if spot_cost_radius_scale is None
        else spot_cost_radius_scale
    )
    return tuple(
        0.0 if float(scales.get(int(radius), 0.0)) == 0.0 else 1.0
        for radius in a_sti_radii
    )


def spot_cost_node_scale(
    radius: int,
    spot_cost_radius_scale: Optional[Dict[int, float]],
    *,
    cost_radius_scales: Dict[int, float],
) -> float:
    scales = (
        dict(cost_radius_scales)
        if spot_cost_radius_scale is None
        else spot_cost_radius_scale
    )
    return float(scales.get(int(radius), 0.0))


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
    vals = set(n_by_b.values())
    if len(vals) == 1:
        return next(iter(vals))
    return {b: n_by_b[b] for b in sorted(n_by_b)}


def _as_np(arr) -> np.ndarray:
    return arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)


def build_spot_cost_readout(connectome, spot_bs, cost_radii, cost_radius):
    network_node_u = _as_np(connectome.us)
    network_node_v = _as_np(connectome.vs)
    type_all = _as_np(connectome.node_cells)
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
    network_node_u = _as_np(connectome.us)
    network_node_v = _as_np(connectome.vs)
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
    device: Optional[str] = None,
    cost_radius: Optional[int] = None,
    spot_cost_radius_scale: Optional[Dict[int, float]] = None,
    sim_dtype: torch.dtype,
    ms_sti: Optional[float] = None,
    ms_response: Optional[float] = None,
    gt_cells: Optional[Sequence[str]] = None,
    filter: str = "none",
    spread_gt_mode: str = "all",
) -> SpotGt:
    if contrast not in SPOT_CONTRASTS:
        raise ValueError(f"contrast must be 'bright' or 'dark', got {contrast!r}")
    i_baseline = float(i_baseline)
    i_sti = float(i_sti)
    device = device or connectome.device
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
    node_cell = node_cells(connectome)
    active = active_gt_cells(gt_cells, GT_CELLS, connectome.cells, context="spot")

    spot_bs = spot_sti_bs(spot)
    n_b = len(spot_bs)

    # Single sti_mask source (step or finite spot) shared with the ir gt.
    mask = sti_mask(t_onset, n_t, ms_sti, delta_ms=delta_ms)
    drive = torch.as_tensor(
        i_baseline + (i_sti - i_baseline) * mask, dtype=sim_dtype, device=device,
    )
    # All sti hexes hold i_baseline; sti_uv hexes then get the step/spot drive.
    sti_nodes = torch.as_tensor(connectome.sti_nodes, dtype=torch.long, device=device)
    i_sti = torch.zeros((n_b, n_t, connectome.n_node), dtype=sim_dtype, device=device)
    if len(sti_nodes):
        i_sti[:, :, sti_nodes] = float(i_baseline)
    for b, spot_b in enumerate(spot_bs):
        for su, sv in spot_b.sti_uv:
            nodes = connectome.sti_nodes_at_uv(su, sv)
            if len(nodes):
                idx = torch.as_tensor(nodes, dtype=torch.long, device=device)
                i_sti[b, :, idx] = drive[:, None]

    resp = slice(t_onset, n_t_gt)  # cost window: response only (no ms_post)

    cost_radii = resolve_spot_cost_radii(
        spot_cost_radius_scale,
        cost_radius_scales=cost_radius_scales,
        spot_cost_radii=spot_cost_radii,
    )
    cost_hexes = spot_cost_hexes(spot_bs, cost_radii, cost_radius)

    cost_bs, cost_node, entry_radii_vals, cost_readout, cost_scales_vals = [], [], [], [], []
    cost_sti_us, cost_sti_vs = [], []
    trace_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for b, mu, mv, radius, su, sv in cost_hexes:
        scale = spot_cost_node_scale(
            radius, spot_cost_radius_scale, cost_radius_scales=cost_radius_scales,
        )
        if scale == 0.0:
            continue
        for gt_cell in active:
            gt_idx = gt_type_idx[gt_cell]
            nodes = hex2gt(connectome, mu, mv, gt_cell, node_cell)
            if len(nodes) == 0:
                continue
            rf_sign = int(RF_SIGN[gt_cell])
            if not spot_gt_active(spread_gt_mode, contrast, rf_sign):
                continue
            cache_digest = (int(radius), gt_idx)
            if cache_digest not in trace_cache:
                a_radius = _spot_readout_a_radius(rf[gt_idx], int(radius), spot_radius)
                trace = (
                    a_radius
                    * ir[gt_idx][resp]
                    * gt_amp
                    * float(contrast_sign(contrast))
                )
                trace_cache[cache_digest] = trace
            trace = trace_cache[cache_digest]
            for node in nodes:
                cost_bs.append(b)
                cost_node.append(int(node))
                entry_radii_vals.append(int(radius))
                cost_readout.append(trace)
                cost_scales_vals.append(scale)
                cost_sti_us.append(int(su))
                cost_sti_vs.append(int(sv))

    if not cost_bs:
        raise ValueError("no spot cost nodes (check cost_radius and gt cells)")

    gts = torch.tensor(np.asarray(cost_readout), dtype=sim_dtype, device=device)  # (n_cost,T')
    cost_scales = torch.tensor(np.asarray(cost_scales_vals), dtype=sim_dtype, device=device)
    entry_radii = torch.tensor(
        np.asarray(entry_radii_vals, dtype=np.int64), dtype=torch.long, device=device,
    )
    entry_bs = torch.tensor(np.asarray(cost_bs), dtype=torch.long, device=device)
    entry_nodes = torch.tensor(np.asarray(cost_node), dtype=torch.long, device=device)
    entry_sti_us = torch.tensor(np.asarray(cost_sti_us), dtype=torch.long, device=device)
    entry_sti_vs = torch.tensor(np.asarray(cost_sti_vs), dtype=torch.long, device=device)

    power = torch.sum(cost_scales[:, None] * gts ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)

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
    return standardize_sti_timing(opts)
