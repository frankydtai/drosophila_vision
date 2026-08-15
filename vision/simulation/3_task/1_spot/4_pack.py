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
from import_bootstrap import parse_comma_list
from network.construction import (
    hex2gt,
    hex_in_cost_radius,
    active_gt_cells,
    node_cells,
)
from neuron.param import t_from_ms
from task.spot.gt import (
    GT_CELLS,
    RF_SIGN,
    _spot_readout_a_radius,
    load_rf_ir,
    contrast_sign,
    spot_gt_active,
)
from task.spot.sti_geo import (
    SpotB,
    members_by_radius,
    resolve_spot,
    spot_radius_half_steps,
    spot_sti_bs,
)
from task.spot.sti_spec import (
    standardize_sti_timing,
    sti_waveform,
)

# Spot contrasts (distinct from the task tokens in train.config).
SPOT_CONTRASTS = frozenset({"bright", "dark"})
_SPOT_I_BASELINE = "i_baseline_spot"
_SPOT_I_PEAK = {"bright": "i_bright_spot", "dark": "i_dark_spot"}


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
    text = str(radius).strip()
    try:
        return int(text, 10)
    except ValueError as exc:
        raise ValueError(f"spot cost radius must be an int, got {radius!r}") from exc


def parse_spot_cost_radius_scale_value(text: str) -> float:
    tok = str(text).strip()
    if "/" in tok:
        num, den = tok.split("/", 1)
        return float(num.strip()) / float(den.strip())
    return float(tok)


def expand_spot_cost_radius_scale_dict(
    kv: Optional[dict] = None,
    *,
    sti_opts: Optional[dict] = None,
) -> Optional[Dict[int, float]]:
    if sti_opts is not None:
        kv = (sti_opts or {}).get("spot_cost_radius_scale")
    if not kv:
        return None
    return {
        standardize_spot_cost_radius(k): parse_spot_cost_radius_scale_value(v)
        for k, v in kv.items()
    }


def expand_cost_ms_dict(
    *,
    cost_ms: Optional[dict] = None,
) -> Dict[int, Tuple[float, ...]]:
    """Radius → explicit post-onset ms; empty when unset."""
    kv = cost_ms
    if not kv:
        return {}
    out: Dict[int, Tuple[float, ...]] = {}
    for k, v in kv.items():
        r = standardize_spot_cost_radius(k)
        vals = parse_comma_list(v) if isinstance(v, str) else list(v)
        if not vals:
            raise ValueError(f"cost_ms[{k!r}] must list at least one ms")
        out[r] = tuple(float(x) for x in vals)
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
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"expected R=MS,... or none|off, got {tok!r}")
        radius, val = tok.split("=", 1)
        vals = parse_comma_list(val)
        if not vals:
            raise ValueError(f"--cost-ms {radius}=... must list at least one ms")
        out[standardize_spot_cost_radius(radius)] = tuple(float(x) for x in vals)
    return out


def resolve_spot_cost_radius_scale_defaults(
    spot_radius: float,
    *,
    scales: Dict[int, float],
    scales_radius1: Dict[int, float],
) -> Dict[int, float]:
    """Cost-radius scales for ``spot_radius`` (radius-1 folds r=2 into r=1)."""
    # spot_radius == 1 → half_steps == 2
    if spot_radius_half_steps(spot_radius) == 2:
        return dict(scales_radius1)
    return dict(scales)


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
    for tok in tokens:
        if "=" in tok:
            radius, val = tok.split("=", 1)
            explicit[standardize_spot_cost_radius(radius)] = (
                parse_spot_cost_radius_scale_value(val)
            )
        else:
            bare.append(standardize_spot_cost_radius(tok))
    if bare:
        scales = {int(r): 0.0 for r in spot_cost_radii}
        for r in bare:
            scales[r] = 1.0
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
        spot_cost_radius_scale = expand_spot_cost_radius_scale_dict(sti_opts=sti_opts)
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
        0.0 if float(scales.get(int(r), 0.0)) == 0.0 else 1.0
        for r in a_sti_radii
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
    by_radius = members_by_radius(cost_radii)
    cost_hexes: List[Tuple[int, int, int, int, int, int]] = []
    for b, spot_b in enumerate(spot_bs):
        for su, sv in spot_b.sti_uv:
            for radius, members in by_radius.items():
                for du, dv in members:
                    mu, mv = su + du, sv + dv
                    if not hex_in_cost_radius(mu, mv, cost_radius):
                        continue
                    cost_hexes.append((
                        b, int(mu), int(mv), int(radius), int(su), int(sv),
                    ))
    return cost_hexes


def spot_n_cost_hexes(cost_hexes):
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
    bs, node_indices, radius, type_idx, sti_u, sti_v = [], [], [], [], [], []
    for b, mu, mv, cell_radius, su, sv in spot_cost_hexes(
        spot_bs, cost_radii, cost_radius,
    ):
        on_hex = (network_node_u == mu) & (network_node_v == mv)
        for candidate_node_idx in np.where(on_hex)[0]:
            bs.append(b)
            node_indices.append(int(candidate_node_idx))
            radius.append(cell_radius)
            type_idx.append(int(type_all[candidate_node_idx]))
            sti_u.append(int(su))
            sti_v.append(int(sv))
    return (
        np.asarray(bs, dtype=np.int64),
        np.asarray(node_indices, dtype=np.int64),
        np.asarray(radius, dtype=np.int64),
        np.asarray(type_idx, dtype=np.int64),
        np.asarray(sti_u, dtype=np.int64),
        np.asarray(sti_v, dtype=np.int64),
    )


def build_spot_center_readout(connectome, spot_bs, cost_radii, cost_radius):
    """Cost-node readout plus ``center_entry_mask`` for sti-on hex (radius 0)."""
    bs, node_idx, radius, type_idx, sti_u, sti_v = build_spot_cost_readout(
        connectome, spot_bs, cost_radii, cost_radius,
    )
    sti_u = np.asarray(sti_u, dtype=np.int64)
    sti_v = np.asarray(sti_v, dtype=np.int64)
    network_node_u = _as_np(connectome.us)
    network_node_v = _as_np(connectome.vs)
    du = np.asarray(network_node_u[node_idx] - sti_u, dtype=np.int64)
    dv = np.asarray(network_node_v[node_idx] - sti_v, dtype=np.int64)
    center_entry_mask = (du == 0) & (dv == 0)
    return (
        np.asarray(bs, dtype=np.int64),
        np.asarray(node_idx, dtype=np.int64),
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
    info: dict


def build_spot_gt(
    connectome,
    *,
    spot_radius: float,
    multi_spot: bool,
    fully_inside: bool,
    shift_radius: int,
    n_t: int,
    t_onset: int,
    i_baseline_spot: float,
    i_bright_spot: float,
    i_dark_spot: float,
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
    spot_gt_mode: str = "all",
) -> SpotGt:
    if contrast not in SPOT_CONTRASTS:
        raise ValueError(f"contrast must be 'bright' or 'dark', got {contrast!r}")
    i_baseline = float(i_baseline_spot)
    i_spot = float(i_bright_spot if contrast == "bright" else i_dark_spot)
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
    gt_type_idx = {str(gt_cell): i for i, gt_cell in enumerate(GT_CELLS)}
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

    # Single sti waveform source (step or finite spot) shared with the ir gt.
    u = sti_waveform(t_onset, n_t, ms_sti, delta_ms=delta_ms)
    drive = torch.as_tensor(
        i_baseline + (i_spot - i_baseline) * u, dtype=sim_dtype, device=device,
    )
    # All sti hexes hold i_baseline; sti_uv hexes then get the step/spot drive.
    sti_idx = torch.as_tensor(connectome.sti_node_indices, dtype=torch.long, device=device)
    i_sti = torch.zeros((n_b, n_t, connectome.n_nodes), dtype=sim_dtype, device=device)
    if len(sti_idx):
        i_sti[:, :, sti_idx] = float(i_baseline)
    for b, spot_b in enumerate(spot_bs):
        for su, sv in spot_b.sti_uv:
            nodes = connectome.sti_nodes_at(su, sv)
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
            if not spot_gt_active(spot_gt_mode, contrast, rf_sign):
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
            for node_idx in nodes:
                cost_bs.append(b)
                cost_node.append(int(node_idx))
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

    info = {
        "n_b": n_b,
        "n_cost": gts.shape[0],
        "n_cost_hexes": spot_n_cost_hexes(cost_hexes),
        "n_centers": len(spot.centers),
        "n_shifts": len(spot.shifts),
        "cost_radius": cost_radius,
        "spot_radius": float(spot_radius),
        "multi_spot": bool(multi_spot),
        "fully_inside": bool(fully_inside),
        "spot_cost_radius_scale": spot_cost_radius_scale,
        "spot_cost_radii": list(cost_radii),
        "active_gts": active,
        "i_baseline_spot": float(i_baseline),
        "i_bright_spot": float(i_bright_spot),
        "i_dark_spot": float(i_dark_spot),
        "contrast": str(contrast),
        "ms_sti": None if ms_sti is None else float(ms_sti),
        "ms_response": float(ms_response),
        "t_onset": int(t_onset),
        "n_t": int(n_t),
        "n_t_gt": int(n_t_gt),
        "filter": str(filter),
        "spot_gt_mode": str(spot_gt_mode),
    }
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
        info=info,
    )


def build_spot_sti_opts(
    contrast: str,
    *,
    i_baseline_spot,
    i_spot,
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
    """Sti step/spot sti opts for ``spot_{contrast}``."""
    if contrast not in SPOT_CONTRASTS:
        raise ValueError(f"spot contrast must be 'bright' or 'dark', got {contrast!r}")
    i_peak = _SPOT_I_PEAK[contrast]
    opts = {
        _SPOT_I_BASELINE: i_baseline_spot,
        i_peak: i_spot,
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
