# -*- coding: utf-8 -*-
"""Moving-bar sti spec: conditions, timing, speed, contrast, and hex currents."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from path import MBAR_CACHE_DIRNAME
from neuron.borst import t_from_ms
from task.mbar.sti_geo import (
    BAR_RADIUS,
    GRUNTMAN_directions,
    GRUNTMAN_WS_DEG,
    Hex,
    StiHex,
    bar_bound0_bar_bound1s,
    _fill,
    bar_bounds,
    view_bounds,
    i_sti_nodes_from_hexes,
    sti_hexes,
)

logger = logging.getLogger(__name__)

# Gruntman Fig. 1 Ci fast condition: 40 ms / 2.25 deg per LED step.
GRUNTMAN_SPEED_DEG_OVER_S = 56.25

# Moving-bar per-hex cost window relative to first-sti alignment.
COST_WINDOW_MS = 900.0
COST_ALIGNED_FIRST_STI_MS = 300.0
COST_WINDOW_BEFORE_MS = COST_ALIGNED_FIRST_STI_MS
COST_WINDOW_AFTER_MS = COST_WINDOW_MS - COST_ALIGNED_FIRST_STI_MS

# Post-sweep tail: baseline after bar exit through ``t_first_sti + after`` plus pad.
T_TAIL_PAD_MS = 50.0
MBAR_TAIL_MS = COST_WINDOW_AFTER_MS + T_TAIL_PAD_MS

PD_ND_LABELS = ("PD", "ND")
PD_IDX, ND_IDX = 0, 1


@dataclass(frozen=True)
class MbarSpec:
    direction: str
    contrast: str
    w_deg: float
    speed_deg_over_s: float = GRUNTMAN_SPEED_DEG_OVER_S

    @property
    def token(self) -> str:
        w_token = "w1" if self.w_deg <= 3.0 else "w4"
        return f"{self.direction}_{self.contrast}_{w_token}"


def gruntman_mbar_specs(
    *,
    contrasts: Sequence[str],
    directions: Sequence[str] = GRUNTMAN_directions,
    ws_deg: Sequence[float] = GRUNTMAN_WS_DEG,
    speed_deg_over_s: float = GRUNTMAN_SPEED_DEG_OVER_S,
) -> List[MbarSpec]:
    """The Gruntman-style whole-view moving-bar conditions for ``contrasts``."""
    return [
        MbarSpec(
            direction=direction,
            contrast=contrast,
            w_deg=w_deg,
            speed_deg_over_s=speed_deg_over_s,
        )
        for direction in directions
        for contrast in contrasts
        for w_deg in ws_deg
    ]


def t_from_bar_bound(
    spec: MbarSpec,
    bar_bound: float,
    bar_bound_end: float,
    t_onset: int,
    delta_ms: float,
    n_t: Optional[int] = None,
) -> int:
    shift_deg = float(spec.speed_deg_over_s) * (float(delta_ms) / 1000.0)
    if spec.direction in ("left", "down"):
        shift_deg = -shift_deg
    elif spec.direction not in ("right", "up"):
        raise ValueError(f"unknown direction {spec.direction!r}")
    if abs(shift_deg) < 1e-15:
        return t_onset
    k = int(round((bar_bound_end - bar_bound) / shift_deg))
    t = t_onset + max(0, k)
    if n_t is not None:
        t = min(t, n_t - 1)
    return t


def mbar_sweep_end_t(
    specs: Sequence[MbarSpec],
    view_deg: Tuple[float, float, float, float],
    bar_radius: int,
    *,
    multi_bar: bool = True,
    t_onset: int = None,
    delta_ms: float,
) -> int:
    """Exclusive t index where all bars finish their local sweep (no tail)."""
    if not specs:
        return t_onset + 1
    t_exit = t_onset
    for spec in specs:
        w_deg = float(spec.w_deg)
        for bar_bound0, bar_bound1 in bar_bound0_bar_bound1s(
            spec, view_deg, bar_radius, multi_bar=multi_bar,
        ):
            if spec.direction in ("right", "up"):
                bar_bound, bar_bound_end = float(bar_bound0) - w_deg, float(bar_bound1)
            elif spec.direction in ("left", "down"):
                bar_bound, bar_bound_end = float(bar_bound1) + w_deg, float(bar_bound0)
            else:
                raise ValueError(f"unknown direction {spec.direction!r}")
            t_exit = max(
                t_exit,
                t_from_bar_bound(spec, bar_bound, bar_bound_end, t_onset, delta_ms),
            )
    return t_exit + 1


def mbar_n_t(
    specs: Sequence[MbarSpec],
    view_deg: Tuple[float, float, float, float],
    bar_radius: int,
    *,
    multi_bar: bool = True,
    t_onset: int = None,
    delta_ms: float,
    t_tail_ms: float = MBAR_TAIL_MS,
) -> int:
    """Simulation length: baseline + multi-bar sweep + post-sweep tail."""
    t_tail = t_from_ms(t_tail_ms, delta_ms=delta_ms)
    return mbar_sweep_end_t(
        specs, view_deg, bar_radius, multi_bar=multi_bar, t_onset=t_onset, delta_ms=delta_ms,
    ) + t_tail


def mbar_transit_times(
    spec: MbarSpec,
    view_deg: Tuple[float, float, float, float],
    bar_radius: int,
    *,
    multi_bar: bool = True,
    t_onset: int = None,
    n_t: Optional[int] = None,
    delta_ms: float,
) -> Tuple[int, int, int]:
    """Return ``(entry, mid, exit)`` t idxs for the first simultaneous bar."""
    bar_bound0, bar_bound1 = bar_bound0_bar_bound1s(
        spec, view_deg, bar_radius, multi_bar=multi_bar,
    )[0]
    w_deg = float(spec.w_deg)
    shift_deg = float(spec.speed_deg_over_s) * (float(delta_ms) / 1000.0)
    if spec.direction in ("left", "down"):
        shift_deg = -shift_deg
    elif spec.direction not in ("right", "up"):
        raise ValueError(f"unknown direction {spec.direction!r}")
    if spec.direction in ("right", "up"):
        bar_bound, bar_bound_end = float(bar_bound0) - w_deg, float(bar_bound1)
    else:
        bar_bound, bar_bound_end = float(bar_bound1) + w_deg, float(bar_bound0)
    bar_bound_entry = bar_bound + shift_deg
    mid = float(bar_bound0) + 0.5 * (
        float(bar_bound1) - float(bar_bound0) - math.copysign(1.0, shift_deg) * w_deg
    )
    bar_bound_exit = bar_bound_end - shift_deg
    return (
        t_from_bar_bound(spec, bar_bound, bar_bound_entry, t_onset, delta_ms, n_t),
        t_from_bar_bound(spec, bar_bound, mid, t_onset, delta_ms, n_t),
        t_from_bar_bound(spec, bar_bound, bar_bound_exit, t_onset, delta_ms, n_t),
    )


def hex_first_sti_t(
    i_sti_hex: np.ndarray,
    i_baseline: float,
    *,
    atol: float = 1e-12,
) -> int:
    """First t where a hex i_sti differs from baseline (``t_first_sti``)."""
    curr = np.asarray(i_sti_hex, dtype=np.float64).reshape(-1)
    mask = ~np.isclose(curr, float(i_baseline), atol=atol, rtol=0.0)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("hex has no non-baseline sti sample")
    return int(idx[0])


def build_i_sti_hex(
    hexes: Sequence[Hex],
    specs: Sequence[MbarSpec],
    n_t: int,
    bar_radius: int,
    *,
    multi_bar: bool = True,
    t_onset: int = None,
    delta_ms: float,
    i_baseline: float,
    i_sti: float,
) -> np.ndarray:
    """Multi-b hex currents ``(B, T, n_hex)``.

    Each b row superposes simultaneous bars for one ``MbarSpec``.
    Specs that share direction / w / speed reuse one ``fills`` stack;
    only the contrast peak current ``i_sti`` differs when callers rebuild per contrast.
    """
    n_b = len(specs)
    n_hex = len(hexes)
    if n_hex == 0 or n_b == 0:
        return np.zeros((n_b, n_t, n_hex), dtype=np.float64)

    i_sti_hex = np.zeros((n_b, n_t, n_hex), dtype=np.float64)
    for b in range(n_b):
        i_sti_hex[b, :t_onset] = i_baseline

    view_deg = view_bounds(hexes)

    by_geometry: dict[Tuple[str, float, float], List[int]] = {}
    for b, spec in enumerate(specs):
        geometry = (spec.direction, float(spec.w_deg), float(spec.speed_deg_over_s))
        by_geometry.setdefault(geometry, []).append(b)

    for bs in by_geometry.values():
        spec = specs[bs[0]]
        shift_deg = float(spec.speed_deg_over_s) * (float(delta_ms) / 1000.0)
        if spec.direction in ("left", "down"):
            shift_deg = -shift_deg
        elif spec.direction not in ("right", "up"):
            raise ValueError(f"unknown direction {spec.direction!r}")
        w_deg = float(spec.w_deg)
        n_post = n_t - t_onset
        fills = np.zeros((n_post, n_hex), dtype=np.float64)
        for bar_bound0, bar_bound1 in bar_bound0_bar_bound1s(
            spec, view_deg, bar_radius, multi_bar=multi_bar,
        ):
            if spec.direction in ("right", "up"):
                bar_bound = float(bar_bound0) - w_deg
            else:
                bar_bound = float(bar_bound1) + w_deg
            for t in range(n_post):
                visible_bar = bar_bounds(spec, bar_bound, view_deg, bar_bound0, bar_bound1)
                if visible_bar is not None:
                    x_deg0, y_deg0, x_deg1, y_deg1 = visible_bar
                    for hex_idx in range(n_hex):
                        hex = hexes[hex_idx]
                        fills[t, hex_idx] += _fill(
                            hex.vertex_x_degs, hex.vertex_y_degs,
                            x_deg0, y_deg0, x_deg1, y_deg1,
                        )
                bar_bound += shift_deg
        fills = np.clip(fills, 0.0, 1.0)
        for b in bs:
            i_sti_hex[b, t_onset:] = (
                i_baseline + fills * (float(i_sti) - i_baseline)
            )
    return i_sti_hex


@dataclass
class MbarSti:
    i_sti: torch.Tensor
    i_sti_hex: np.ndarray
    specs: List[MbarSpec]
    n_b: int
    n_t: int
    t_onset: int
    view_deg: Tuple[float, float, float, float]
    i_baseline: float
    sweep_t: int
    sweep_s: float
    bar_radius: int
    multi_bar: bool


from task.spread.sti_spec import i_baseline_from_i_sti
@dataclass
class MbarT0Grids:
    t0_bn: np.ndarray
    before_t: Dict[str, int]
    after_t: Dict[str, int]


def mbar_spec_horizon(t_first_stis: Sequence[int], n_t: int) -> Tuple[int, int, int]:
    """Return ``(fb, before_t, after_t)`` for one spec over all hexes."""
    fb = int(min(t_first_stis))
    before = fb
    after = int(n_t) - int(max(t_first_stis))
    return fb, before, after


def mbar_network_t0_bn(connectome, filt_hexes: Sequence[StiHex], n_b: int, t0_map: dict) -> np.ndarray:
    """Expand per-hex ``t0`` values to a full node grid ``(B, N_nodes)``."""
    node_us = np.asarray(connectome.us, dtype=np.int64)
    node_vs = np.asarray(connectome.vs, dtype=np.int64)
    t0_bn = np.full((n_b, connectome.n_node), -1, dtype=np.int64)
    for b in range(n_b):
        for hex in filt_hexes:
            t0 = t0_map.get((b, int(hex.u), int(hex.v)))
            if t0 is None:
                continue
            on_hex = (node_us == int(hex.u)) & (node_vs == int(hex.v))
            t0_bn[b, on_hex] = t0
    return t0_bn


def build_mbar_t0_grids(
    i_sti_hex: np.ndarray,
    specs: Sequence[MbarSpec],
    n_t: int,
    i_baseline: float,
    *,
    hex_idxs: Sequence[int],
    filt_hex_idxs: Sequence[int],
    connectome,
    filt_network_hexes: Sequence[StiHex],
) -> MbarT0Grids:
    """Plot/train-aligned ``t0`` grid and per-spec full horizons."""
    before_t: Dict[str, int] = {}
    after_t: Dict[str, int] = {}
    n_b = len(specs)
    i_baseline = float(i_baseline)

    t0_map: dict = {}
    for b, spec in enumerate(specs):
        t_first_all = [
            hex_first_sti_t(i_sti_hex[b, :, hex_idx], i_baseline=i_baseline)
            for hex_idx in hex_idxs
        ]
        fb, before, after = mbar_spec_horizon(t_first_all, n_t)
        before_t[spec.token] = before
        after_t[spec.token] = after
        t_first_filt = [
            hex_first_sti_t(i_sti_hex[b, :, hex_idx], i_baseline=i_baseline)
            for hex_idx in filt_hex_idxs
        ]
        for hex, t_first in zip(filt_network_hexes, t_first_filt):
            t0_map[(b, int(hex.u), int(hex.v))] = t_first - fb
    t0_bn = mbar_network_t0_bn(connectome, filt_network_hexes, n_b, t0_map)
    return MbarT0Grids(t0_bn=t0_bn, before_t=before_t, after_t=after_t)


def _mbar_cache_digest(
    network_json: Path,
    specs: Sequence[MbarSpec],
    hex_uv: Sequence[Tuple[int, int]],
    n_t: int,
    t_onset: int,
    delta_ms: float,
    i_baseline: float,
    bar_radius: int,
    multi_bar: bool = True,
    i_sti: Optional[float] = None,
) -> str:
    stat = network_json.stat()
    payload = {
        "network": str(network_json.resolve()),
        "network_mtime_ns": stat.st_mtime_ns,
        "network_size": stat.st_size,
        "hex_uv": list(hex_uv),
        "bar_radius": int(bar_radius),
        "multi_bar": bool(multi_bar),
        "specs": [
            {
                "direction": spec.direction,
                "contrast": spec.contrast,
                "w_deg": spec.w_deg,
                "speed_deg_over_s": spec.speed_deg_over_s,
            }
            for spec in specs
        ],
        "n_t": n_t,
        "t_onset": t_onset,
        "delta_ms": delta_ms,
        "i_baseline": i_baseline,
        "i_sti": None if i_sti is None else float(i_sti),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def _mbar_cache_path(
    network_json: Path,
    specs: Sequence[MbarSpec],
    hex_uv: Sequence[Tuple[int, int]],
    n_t: int,
    t_onset: int,
    delta_ms: float,
    i_baseline: float,
    bar_radius: int,
    multi_bar: bool = True,
    i_sti: Optional[float] = None,
) -> Path:
    digest = _mbar_cache_digest(
        network_json, specs, hex_uv, n_t, t_onset, delta_ms,
        i_baseline, bar_radius, multi_bar, i_sti,
    )
    return Path(network_json).resolve().parent / MBAR_CACHE_DIRNAME / f"{digest}.npz"


def _load_mbar_hex_cache(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data["i_sti_hex"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Ignoring corrupt moving-bar cache %s: %s", path, exc)
        return None


def _save_mbar_hex_cache(path: Path, i_sti_hex: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, i_sti_hex=i_sti_hex)
    logger.info("Cached moving-bar i_sti_hex to %s", path)


def build_mbar_signals(
    connectome,
    specs: Sequence[MbarSpec],
    n_t: Optional[int] = None,
    t_onset: int = None,
    *,
    delta_ms: float,
    bar_radius: int = BAR_RADIUS,
    multi_bar: bool = True,
    i_baseline: float,
    i_sti: float,
    device: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    network_json: Optional[Path] = None,
    sim_dtype: torch.dtype,
) -> MbarSti:
    """Build sti current for moving-bar stis.

    Returns ``i_sti`` with shape ``(B, T, N_nodes)`` where ``B = len(specs)``.
    Peak current ``i_sti`` is for this build (one contrast at a time).
    """
    device = device or connectome.device
    bar_radius = int(bar_radius)
    multi_bar = bool(multi_bar)
    specs = list(specs)
    i_sti = float(i_sti)
    i_baseline = float(i_baseline)
    sti = sti_hexes(connectome)
    view_deg = view_bounds(sti)
    if n_t is None:
        n_t = mbar_n_t(
            specs, view_deg, bar_radius, multi_bar=multi_bar, t_onset=t_onset, delta_ms=delta_ms,
        )
    n_b = len(specs)
    n_node = connectome.n_node
    sweep_end = mbar_sweep_end_t(
        specs, view_deg, bar_radius, multi_bar=multi_bar, t_onset=t_onset, delta_ms=delta_ms,
    )
    sweep_t = sweep_end - t_onset

    cache_path: Optional[Path] = None
    source_json = Path(network_json) if network_json is not None else getattr(connectome, "source_json", None)
    hex_uv = [(hex.u, hex.v) for hex in sti]
    if source_json is not None:
        cache_path = _mbar_cache_path(
            source_json, specs, hex_uv, n_t, t_onset, delta_ms,
            i_baseline, bar_radius, multi_bar, i_sti,
        )

    i_sti_hex: Optional[np.ndarray] = None
    if cache_path is not None and use_cache and not refresh_cache:
        i_sti_hex = _load_mbar_hex_cache(cache_path)
        if i_sti_hex is not None:
            logger.info("Loaded moving-bar i_sti_hex from cache %s", cache_path)

    if i_sti_hex is None:
        i_sti_hex = build_i_sti_hex(
            sti, specs, n_t=n_t, bar_radius=bar_radius, multi_bar=multi_bar,
            t_onset=t_onset, delta_ms=delta_ms,
            i_baseline=i_baseline, i_sti=i_sti,
        )
        if cache_path is not None and use_cache:
            _save_mbar_hex_cache(cache_path, i_sti_hex)

    i_sti_np = i_sti_nodes_from_hexes(i_sti_hex, sti, n_node)

    return MbarSti(
        i_sti=torch.as_tensor(i_sti_np, dtype=sim_dtype, device=device),
        i_sti_hex=i_sti_hex,
        specs=specs,
        n_b=n_b,
        n_t=n_t,
        t_onset=t_onset,
        view_deg=view_deg,
        i_baseline=i_baseline,
        sweep_t=sweep_t,
        sweep_s=sweep_t * delta_ms / 1000.0,
        bar_radius=bar_radius,
        multi_bar=multi_bar,
    )
