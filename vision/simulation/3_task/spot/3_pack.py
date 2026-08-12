# -*- coding: utf-8 -*-
"""Spot paradigm pack: bind GT numbers to the network for cost.

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
    normalize_gt_cells,
    node_cell_names,
)
from neuron.param import t_from_ms
from task.spot.gt import (
    GT_CELLS,
    RF_SIGN,
    _spot_readout_a_radius,
    load_RecF_ImpR,
    contrast_sign,
    spot_gt_active,
)
from task.spot.input import (
    SpotBatch,
    members_by_euclid_radius,
    normalize_spot_timing,
    spot_radius_folds_r2_into_r1,
    spot_input_waveform,
    spot_sti_batches,
    spot_from_opts,
)

# Spot paradigm contrasts (distinct from the task NAMES in train.config).
SPOT_CONTRASTS = frozenset({"bright", "dark"})
_SPOT_BASELINE_KEY = "i_baseline_spot"
_SPOT_I_KEY = {"bright": "i_bright_spot", "dark": "i_dark_spot"}


# -- Cost-radius scales ------------------------------------------------------


def normalize_spot_cost_radius_key(key, *, aliases: Dict[str, float]) -> float:
    if isinstance(key, (int, float)):
        return round(float(key), 6)
    text = str(key).strip().lower()
    if text in aliases:
        return round(float(aliases[text]), 6)
    return round(float(text), 6)


def parse_spot_cost_radius_scale_value(text: str) -> float:
    tok = str(text).strip()
    if "/" in tok:
        num, den = tok.split("/", 1)
        return float(num.strip()) / float(den.strip())
    return float(tok)


def expand_spot_cost_r_s_dict(
    kv: Optional[dict] = None,
    *,
    sti_opts: Optional[dict] = None,
    aliases: Dict[str, float],
) -> Optional[Dict[float, float]]:
    if sti_opts is not None:
        kv = (sti_opts or {}).get("spot_cost_radius_scale")
    if not kv:
        return None
    return {
        normalize_spot_cost_radius_key(k, aliases=aliases): parse_spot_cost_radius_scale_value(v)
        for k, v in kv.items()
    }


def expand_cost_ms_dict(
    *,
    sti_opts: Optional[dict] = None,
    aliases: Dict[str, float],
) -> Dict[float, Tuple[float, ...]]:
    """Radius → explicit post-onset ms; empty when unset."""
    kv = (sti_opts or {}).get("cost_ms") if sti_opts is not None else None
    if not kv:
        return {}
    out: Dict[float, Tuple[float, ...]] = {}
    for k, v in kv.items():
        r = normalize_spot_cost_radius_key(k, aliases=aliases)
        vals = parse_comma_list(v) if isinstance(v, str) else list(v)
        if not vals:
            raise ValueError(f"cost_ms[{k!r}] must list at least one ms")
        out[r] = tuple(float(x) for x in vals)
    return out


def parse_cost_ms_tokens(
    tokens: Optional[Sequence[str]],
    *,
    aliases: Dict[str, float],
) -> Optional[Dict[float, Tuple[float, ...]]]:
    """Parse ``--cost-ms``: ``none``/``off`` → ``{}``; else ``R=MS,...``. Omit → ``None``."""
    if tokens is None:
        return None
    if len(tokens) == 1 and str(tokens[0]).strip().lower() in ("none", "off"):
        return {}
    out: Dict[float, Tuple[float, ...]] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"expected R=MS,... or none|off, got {tok!r}")
        key, val = tok.split("=", 1)
        vals = parse_comma_list(val)
        if not vals:
            raise ValueError(f"--cost-ms {key}=... must list at least one ms")
        out[normalize_spot_cost_radius_key(key, aliases=aliases)] = tuple(
            float(x) for x in vals
        )
    return out


def default_spot_cost_radius_scale(
    spot_radius: float,
    *,
    scales: Dict[float, float],
    scales_radius1: Dict[float, float],
) -> Dict[float, float]:
    """Cost-radius scales for ``spot_radius`` (radius-1 folds r=2 into r=1)."""
    if spot_radius_folds_r2_into_r1(spot_radius):
        return dict(scales_radius1)
    return dict(scales)


def parse_spot_cost_r_s_tokens(
    tokens: Optional[Sequence[str]],
    *,
    default_scales: Dict[float, float],
    spot_cost_radii: Tuple[float, ...],
    aliases: Dict[str, float],
) -> Optional[Dict[float, float]]:
    """Parse ``--spot-cost-r-s`` space-separated ``R`` / ``R=S`` tokens."""
    if not tokens:
        return None
    bare: list[float] = []
    explicit: Dict[float, float] = {}
    for tok in tokens:
        if "=" in tok:
            key, val = tok.split("=", 1)
            explicit[normalize_spot_cost_radius_key(key, aliases=aliases)] = (
                parse_spot_cost_radius_scale_value(val)
            )
        else:
            bare.append(normalize_spot_cost_radius_key(tok, aliases=aliases))
    if bare:
        scales = {round(float(r), 6): 0.0 for r in spot_cost_radii}
        for r in bare:
            scales[r] = 1.0
    else:
        scales = dict(default_scales)
    scales.update(explicit)
    return scales


def resolve_spot_cost_radii(
    spot_cost_radius_scale: Optional[Dict[float, float]] = None,
    *,
    default_scales: Dict[float, float],
    spot_cost_radii: Tuple[float, ...],
    sti_opts: Optional[dict] = None,
    aliases: Optional[Dict[str, float]] = None,
) -> Tuple[float, ...]:
    if sti_opts is not None:
        if aliases is None:
            raise ValueError("resolve_spot_cost_radii with sti_opts requires aliases")
        spot_cost_radius_scale = expand_spot_cost_r_s_dict(
            sti_opts=sti_opts, aliases=aliases,
        )
    scales = (
        dict(default_scales)
        if spot_cost_radius_scale is None
        else spot_cost_radius_scale
    )
    return tuple(
        radius for radius in spot_cost_radii
        if float(scales.get(round(radius, 6), 0.0)) != 0.0
    )


def build_a_sti_radius_mask(
    spot_cost_radius_scale: Optional[Dict[float, float]] = None,
    *,
    default_scales: Dict[float, float],
    a_sti_radii: Tuple[float, ...],
) -> Tuple[float, ...]:
    """Per ``a_sti_radii`` slot: ``1`` if cost-radius scale ≠ 0 else ``0``.

    Forward multiplies ``a_sti_radius`` by this mask (indi or fixed).
    """
    scales = (
        dict(default_scales)
        if spot_cost_radius_scale is None
        else spot_cost_radius_scale
    )
    return tuple(
        0.0 if float(scales.get(round(float(r), 6), 0.0)) == 0.0 else 1.0
        for r in a_sti_radii
    )


def spot_cost_node_scale(
    radius: float,
    spot_cost_radius_scale: Optional[Dict[float, float]],
    *,
    default_scales: Dict[float, float],
) -> float:
    scales = (
        dict(default_scales)
        if spot_cost_radius_scale is None
        else spot_cost_radius_scale
    )
    return float(scales.get(round(radius, 6), 0.0))


# -- Cost hexes / network readout --------------------------------------------


def spot_cost_hexes(
    batches: Sequence[SpotBatch],
    cost_radii,
    cost_radius,
) -> List[Tuple[int, int, int, float, int, int]]:
    """Cost readouts: ``(batch, mu, mv, radius_key, su, sv)`` per sti radius."""
    by_radius = members_by_euclid_radius(cost_radii)
    cost_hexes: List[Tuple[int, int, int, float, int, int]] = []
    for b, batch in enumerate(batches):
        for su, sv in batch.sti_uv:
            for radius_key, members in by_radius.items():
                for du, dv in members:
                    mu, mv = su + du, sv + dv
                    if not hex_in_cost_radius(mu, mv, cost_radius):
                        continue
                    cost_hexes.append((
                        b, int(mu), int(mv), float(radius_key), int(su), int(sv),
                    ))
    return cost_hexes


def spot_n_cost_hexes(cost_hexes):
    if not cost_hexes:
        return 0
    n_by_batch: Dict[int, int] = {}
    for b, _mu, _mv, _radius, _su, _sv in cost_hexes:
        n_by_batch[b] = n_by_batch.get(b, 0) + 1
    vals = set(n_by_batch.values())
    if len(vals) == 1:
        return next(iter(vals))
    return {b: n_by_batch[b] for b in sorted(n_by_batch)}


def _as_np(arr) -> np.ndarray:
    return arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)


def build_spot_cost_readout(connectome, batches, cost_radii, cost_radius):
    network_node_u = _as_np(connectome.us)
    network_node_v = _as_np(connectome.vs)
    type_all = _as_np(connectome.node_cells)
    batch_idx, node_indices, radius, type_idx, sti_u, sti_v = [], [], [], [], [], []
    for b, mu, mv, cell_radius, su, sv in spot_cost_hexes(
        batches, cost_radii, cost_radius,
    ):
        on_hex = (network_node_u == mu) & (network_node_v == mv)
        for candidate_node_idx in np.where(on_hex)[0]:
            batch_idx.append(b)
            node_indices.append(int(candidate_node_idx))
            radius.append(cell_radius)
            type_idx.append(int(type_all[candidate_node_idx]))
            sti_u.append(int(su))
            sti_v.append(int(sv))
    return (
        np.asarray(batch_idx, dtype=np.int64),
        np.asarray(node_indices, dtype=np.int64),
        np.asarray(radius, dtype=np.float64),
        np.asarray(type_idx, dtype=np.int64),
        np.asarray(sti_u, dtype=np.int64),
        np.asarray(sti_v, dtype=np.int64),
    )


def build_spot_center_readout(connectome, batches, cost_radii, cost_radius):
    """Cost-node readout plus ``center_entry_mask`` for sti-on hex (radius 0)."""
    batch_idx, node_idx, radius, type_idx, sti_u, sti_v = build_spot_cost_readout(
        connectome, batches, cost_radii, cost_radius,
    )
    sti_u = np.asarray(sti_u, dtype=np.int64)
    sti_v = np.asarray(sti_v, dtype=np.int64)
    network_node_u = _as_np(connectome.us)
    network_node_v = _as_np(connectome.vs)
    du = np.asarray(network_node_u[node_idx] - sti_u, dtype=np.int64)
    dv = np.asarray(network_node_v[node_idx] - sti_v, dtype=np.int64)
    center_entry_mask = (du == 0) & (dv == 0)
    return (
        np.asarray(batch_idx, dtype=np.int64),
        np.asarray(node_idx, dtype=np.int64),
        np.asarray(radius, dtype=np.float64),
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
    entry_radii: torch.Tensor  # (n_cost,) Euclidean per entry
    entry_batches: torch.Tensor   # (n_cost,) long
    entry_nodes: torch.Tensor    # (n_cost,) long
    entry_sti_us: torch.Tensor  # (n_cost,) long
    entry_sti_vs: torch.Tensor  # (n_cost,) long
    n_batch: int
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
    default_cost_scales: Dict[float, float],
    spot_cost_radii: Tuple[float, ...],
    device: Optional[str] = None,
    cost_radius: Optional[int] = None,
    spot_cost_radius_scale: Optional[Dict[float, float]] = None,
    sim_dtype: torch.dtype,
    ms_spot: Optional[float] = None,
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
    recf_gt, recf_peak_scale, impr_gt = load_RecF_ImpR(
        t_onset=t_onset, n_t=n_t_gt, ms_spot=ms_spot, delta_ms=delta_ms,
        filter=filter,
    )
    gt_type_idx = {str(rt): i for i, rt in enumerate(GT_CELLS)}
    if gt_cells is not None:
        bad = [str(t) for t in gt_cells if str(t) not in gt_type_idx]
        if bad:
            raise ValueError(
                f"unknown spot gt cell(s) {bad!r} "
                f"(expected subset of {list(GT_CELLS)})",
            )

    spot = spot_from_opts(
        connectome,
        spot_radius=spot_radius,
        shift_radius=shift_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
    )
    names = node_cell_names(connectome)
    active = active_gt_cells(gt_cells, GT_CELLS, connectome.cells, context="spot")

    batches = spot_sti_batches(spot)
    n_batch = len(batches)

    # Single sti waveform source (step or finite spot) shared with the ImpR gt.
    u = spot_input_waveform(t_onset, n_t, ms_spot, delta_ms=delta_ms)
    drive = torch.as_tensor(
        i_baseline + (i_spot - i_baseline) * u, dtype=sim_dtype, device=device,
    )
    # All sti hexes hold i_baseline; sti_uv hexes then get the step/spot drive.
    sti_idx = torch.as_tensor(connectome.sti_node_indices, dtype=torch.long, device=device)
    i_sti = torch.zeros((n_batch, n_t, connectome.n_nodes), dtype=sim_dtype, device=device)
    if len(sti_idx):
        i_sti[:, :, sti_idx] = float(i_baseline)
    for b, batch in enumerate(batches):
        for su, sv in batch.sti_uv:
            nodes = connectome.sti_nodes_at(su, sv)
            if len(nodes):
                idx = torch.as_tensor(nodes, dtype=torch.long, device=device)
                i_sti[b, :, idx] = drive[:, None]

    resp = slice(t_onset, n_t_gt)  # cost window: response only (no ms_post)

    cost_radii = resolve_spot_cost_radii(
        spot_cost_radius_scale,
        default_scales=default_cost_scales,
        spot_cost_radii=spot_cost_radii,
    )
    cost_hexes = spot_cost_hexes(batches, cost_radii, cost_radius)

    cost_batch, cost_node, entry_radii_vals, cost_readout, cost_scales_vals = [], [], [], [], []
    cost_sti_us, cost_sti_vs = [], []
    trace_cache: Dict[Tuple[float, int], np.ndarray] = {}
    for b, mu, mv, radius, su, sv in cost_hexes:
        w = spot_cost_node_scale(
            radius, spot_cost_radius_scale, default_scales=default_cost_scales,
        )
        if w == 0.0:
            continue
        for rt in active:
            gt_idx = gt_type_idx[rt]
            if str(filter) == "ca" and float(recf_peak_scale[gt_idx]) == 0.0:
                continue
            nodes = hex2gt(connectome, mu, mv, rt, names)
            if len(nodes) == 0:
                continue
            rf_sign = int(RF_SIGN[gt_idx])
            if not spot_gt_active(spot_gt_mode, contrast, rf_sign):
                continue
            cache_key = (round(float(radius), 6), gt_idx)
            if cache_key not in trace_cache:
                a_radius = _spot_readout_a_radius(recf_gt[gt_idx], radius, spot_radius)
                trace = (
                    a_radius
                    * impr_gt[gt_idx][resp]
                    * gt_amp
                    * float(contrast_sign(contrast))
                )
                trace_cache[cache_key] = trace
            trace = trace_cache[cache_key]
            for node_idx in nodes:
                cost_batch.append(b)
                cost_node.append(int(node_idx))
                entry_radii_vals.append(radius)
                cost_readout.append(trace)
                cost_scales_vals.append(w)
                cost_sti_us.append(int(su))
                cost_sti_vs.append(int(sv))

    if not cost_batch:
        raise ValueError("no spot cost nodes (check cost_radius and gt cells)")

    gts = torch.tensor(np.asarray(cost_readout), dtype=sim_dtype, device=device)  # (n_cost,T')
    cost_scales = torch.tensor(np.asarray(cost_scales_vals), dtype=sim_dtype, device=device)
    entry_radii = torch.tensor(
        np.asarray(entry_radii_vals), dtype=sim_dtype, device=device,
    )
    entry_batches = torch.tensor(np.asarray(cost_batch), dtype=torch.long, device=device)
    entry_nodes = torch.tensor(np.asarray(cost_node), dtype=torch.long, device=device)
    entry_sti_us = torch.tensor(np.asarray(cost_sti_us), dtype=torch.long, device=device)
    entry_sti_vs = torch.tensor(np.asarray(cost_sti_vs), dtype=torch.long, device=device)

    power = torch.sum(cost_scales[:, None] * gts ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)

    info = {
        "n_batch": n_batch,
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
        "ms_spot": None if ms_spot is None else float(ms_spot),
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
        entry_batches=entry_batches,
        entry_nodes=entry_nodes,
        entry_sti_us=entry_sti_us,
        entry_sti_vs=entry_sti_vs,
        n_batch=n_batch,
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
    ms_spot=None,
    ms_post=0.0,
    cost_interval_ms=None,
    cost_ms=None,
    gt_cells=None,
):
    """Sti step/spot sti opts for ``spot_{contrast}``."""
    if contrast not in SPOT_CONTRASTS:
        raise ValueError(f"spot contrast must be 'bright' or 'dark', got {contrast!r}")
    peak_key = _SPOT_I_KEY[contrast]
    opts = {
        _SPOT_BASELINE_KEY: i_baseline_spot,
        peak_key: i_spot,
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
    if ms_spot is not None:
        opts["ms_spot"] = ms_spot
    if cost_interval_ms is not None:
        # Keep raw (may be a ``{v,ca}`` branch dict). Branch selection and
        # numeric casting must happen after ``open_session``'s resolve step.
        opts["cost_interval_ms"] = cost_interval_ms
    if cost_ms is not None:
        # Keep raw to support per-branch ms values inside cost_ms.
        opts["cost_ms"] = cost_ms
    rs = normalize_gt_cells(gt_cells)
    if rs is not None:
        opts["gt_cells"] = rs
    return normalize_spot_timing(opts)
