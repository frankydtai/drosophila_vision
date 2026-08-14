# -*- coding: utf-8 -*-
"""Moving-bar sti spec: conditions, timing, speed, contrast, and hex currents."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from path import moving_bar_cache_dir
from neuron.param import t_from_ms
from task.moving_bar.sti_geo import (
    DEFAULT_BAR_RADIUS,
    GRUNTMAN_DIRECTIONS,
    GRUNTMAN_WIDTHS_DEG,
    Hex,
    StiHex,
    bar_rect_lane_clipped,
    coverage_batch,
    view_bounds,
    i_sti_nodes_from_hex,
    lane_sweep_trail_range,
    motion_lanes,
    sti_hexes,
)

logger = logging.getLogger(__name__)

# Gruntman Fig. 1 Ci fast condition: 40 ms / 2.25 deg per LED step.
GRUNTMAN_SPEED_DEG_S = 56.0
GRUNTMAN_CONTRASTS = ("bright", "dark")

# Moving-bar per-hex cost window relative to first-sti alignment.
COST_WINDOW_MS = 900.0
COST_ALIGNED_FIRST_STI_MS = 300.0
COST_WINDOW_BEFORE_MS = COST_ALIGNED_FIRST_STI_MS
COST_WINDOW_AFTER_MS = COST_WINDOW_MS - COST_ALIGNED_FIRST_STI_MS

# Post-sweep tail: baseline after bar exit through ``t_first_sti + after`` plus pad.
T_TAIL_PAD_MS = 50.0
MOVING_BAR_TAIL_MS = COST_WINDOW_AFTER_MS + T_TAIL_PAD_MS

PD_IDX, ND_IDX = 0, 1


def cost_window_before_t(delta_ms: float) -> int:
    """``t_from_ms(COST_WINDOW_BEFORE_MS)``."""
    return t_from_ms(COST_WINDOW_BEFORE_MS, delta_ms=delta_ms)


def cost_window_after_t(delta_ms: float) -> int:
    """``t_from_ms(COST_WINDOW_AFTER_MS)``."""
    return t_from_ms(COST_WINDOW_AFTER_MS, delta_ms=delta_ms)


@dataclass(frozen=True)
class MovingBarSpec:
    direction: str
    contrast: str
    width_deg: float
    speed_deg_s: float = GRUNTMAN_SPEED_DEG_S

    @property
    def name(self) -> str:
        wtag = "w1" if self.width_deg <= 3.0 else "w4"
        return f"{self.direction}_{self.contrast}_{wtag}"


def gruntman_moving_bar_specs(
    directions: Sequence[str] = GRUNTMAN_DIRECTIONS,
    contrasts: Sequence[str] = GRUNTMAN_CONTRASTS,
    widths_deg: Sequence[float] = GRUNTMAN_WIDTHS_DEG,
    speed_deg_s: float = GRUNTMAN_SPEED_DEG_S,
) -> List[MovingBarSpec]:
    """The 16 Gruntman-style whole-view moving-bar conditions."""
    return [
        MovingBarSpec(direction=d, contrast=c, width_deg=w, speed_deg_s=speed_deg_s)
        for d in directions
        for c in contrasts
        for w in widths_deg
    ]


def _trail_shift_deg(spec: MovingBarSpec, dt_s: float) -> float:
    """Signed trail advance (deg) in one sample: ``±speed_deg_s * dt_s``."""
    s = float(spec.speed_deg_s) * dt_s
    if spec.direction in ("right", "up"):
        return s
    if spec.direction in ("left", "down"):
        return -s
    raise ValueError(f"unknown direction {spec.direction!r}")


def _coverage_time_series(
    hex_stack: np.ndarray,
    spec: MovingBarSpec,
    view_deg: Tuple[float, float, float, float],
    n_t: int,
    t_onset: int,
    delta_ms: float,
    bar_radius: int,
    *,
    multi_bar: bool = True,
) -> np.ndarray:
    """Coverage from simultaneous per-lane bars (connectome field)."""
    dt_s = delta_ms / 1000.0
    trail_shift_deg = _trail_shift_deg(spec, dt_s)
    lane_origins = motion_lanes(spec, view_deg, bar_radius, multi_bar=multi_bar)
    n_hexes = hex_stack.shape[0]
    n_post = n_t - t_onset
    out = np.zeros((n_post, n_hexes), dtype=np.float64)
    for lane_origin, lane_pitch in lane_origins:
        trail_start, trail_exit = lane_sweep_trail_range(spec, lane_origin, lane_pitch)
        trail = float(trail_start)
        for i in range(n_post):
            rect = bar_rect_lane_clipped(
                spec, trail, lane_origin, lane_pitch, view_deg,
            )
            if rect is not None:
                bx0, by0, bx1, by1 = rect
                out[i] += coverage_batch(hex_stack, bx0, by0, bx1, by1)
            trail += trail_shift_deg
    return np.clip(out, 0.0, 1.0)


def t_from_trail(
    spec: MovingBarSpec,
    trail_start: float,
    trail_end: float,
    t_onset: int,
    delta_ms: float,
    n_t: Optional[int] = None,
) -> int:
    trail_shift_deg = _trail_shift_deg(spec, delta_ms / 1000.0)
    if abs(trail_shift_deg) < 1e-15:
        return t_onset
    k = int(round((trail_end - trail_start) / trail_shift_deg))
    t = t_onset + max(0, k)
    if n_t is not None:
        t = min(t, n_t - 1)
    return t


def moving_bar_sweep_end_t(
    specs: Sequence[MovingBarSpec],
    view_deg: Tuple[float, float, float, float],
    bar_radius: int,
    *,
    multi_bar: bool = True,
    t_onset: int = None,
    delta_ms: float,
) -> int:
    """Exclusive t index where all lanes finish their local sweep (no tail)."""
    if not specs:
        return t_onset + 1
    t_exit = t_onset
    for spec in specs:
        for lane_origin, lane_pitch in motion_lanes(spec, view_deg, bar_radius, multi_bar=multi_bar):
            trail_start, trail_exit = lane_sweep_trail_range(spec, lane_origin, lane_pitch)
            t_exit = max(
                t_exit,
                t_from_trail(spec, trail_start, trail_exit, t_onset, delta_ms),
            )
    return t_exit + 1


def moving_bar_n_t(
    specs: Sequence[MovingBarSpec],
    view_deg: Tuple[float, float, float, float],
    bar_radius: int,
    *,
    multi_bar: bool = True,
    t_onset: int = None,
    delta_ms: float,
    t_tail_ms: float = MOVING_BAR_TAIL_MS,
) -> int:
    """Simulation length: baseline + multi-bar sweep + post-sweep tail."""
    t_tail = t_from_ms(t_tail_ms, delta_ms=delta_ms)
    return moving_bar_sweep_end_t(
        specs, view_deg, bar_radius, multi_bar=multi_bar, t_onset=t_onset, delta_ms=delta_ms,
    ) + t_tail


def moving_bar_transit_times(
    spec: MovingBarSpec,
    view_deg: Tuple[float, float, float, float],
    bar_radius: int,
    *,
    multi_bar: bool = True,
    t_onset: int = None,
    n_t: Optional[int] = None,
    delta_ms: float,
) -> Tuple[int, int, int]:
    """Return ``(entry, mid, exit)`` t indices for the first multi-bar lane."""
    lane_origin, lane_pitch = motion_lanes(spec, view_deg, bar_radius, multi_bar=multi_bar)[0]
    trail_start, trail_exit = lane_sweep_trail_range(spec, lane_origin, lane_pitch)
    w = float(spec.width_deg)
    trail_shift_deg = _trail_shift_deg(spec, delta_ms / 1000.0)
    origin = float(lane_origin)
    trail_entry = float(trail_start) + trail_shift_deg
    trail_mid = origin + 0.5 * (
        float(lane_pitch) - math.copysign(1.0, trail_shift_deg) * w
    )
    trail_exit_vis = float(trail_exit) - trail_shift_deg
    return (
        t_from_trail(spec, trail_start, trail_entry, t_onset, delta_ms, n_t),
        t_from_trail(spec, trail_start, trail_mid, t_onset, delta_ms, n_t),
        t_from_trail(spec, trail_start, trail_exit_vis, t_onset, delta_ms, n_t),
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


def bar_lane_rects_at_t(
    spec: MovingBarSpec,
    view_deg: Tuple[float, float, float, float],
    bar_radius: int,
    t: int,
    *,
    multi_bar: bool = True,
    t_onset: int = None,
    delta_ms: float,
) -> List[Tuple[float, float, float, float]]:
    """All lane bar rectangles at simulation time ``t`` (empty outside local sweep)."""
    trail_shift_deg = _trail_shift_deg(spec, delta_ms / 1000.0)
    rects: List[Tuple[float, float, float, float]] = []
    for lane_origin, lane_pitch in motion_lanes(spec, view_deg, bar_radius, multi_bar=multi_bar):
        trail_start, _ = lane_sweep_trail_range(spec, lane_origin, lane_pitch)
        trail = float(trail_start) + (t - t_onset) * trail_shift_deg
        rect = bar_rect_lane_clipped(
            spec, trail, lane_origin, lane_pitch, view_deg,
        )
        if rect is not None:
            rects.append(rect)
    return rects


def _current_from_coverage(
    coverage: np.ndarray,
    contrast: str,
    i_baseline: float,
    *,
    i_bright_moving_bar: float,
    i_dark_moving_bar: float,
) -> np.ndarray:
    if contrast == "bright":
        peak = float(i_bright_moving_bar)
    elif contrast == "dark":
        peak = float(i_dark_moving_bar)
    else:
        raise ValueError(f"unknown contrast {contrast!r}")
    return i_baseline + coverage * (peak - i_baseline)


def build_batched_i_sti_hex(
    hexes: Sequence[Hex],
    specs: Sequence[MovingBarSpec],
    n_t: int,
    bar_radius: int,
    *,
    multi_bar: bool = True,
    t_onset: int = None,
    delta_ms: float,
    i_baseline: float,
    i_bright_moving_bar: Optional[float] = None,
    i_dark_moving_bar: Optional[float] = None,
) -> np.ndarray:
    """Batched multi-bar hex currents ``(B, T, n_hexes)``.

    Each batch row superposes simultaneous lane bars for one ``MovingBarSpec``.
    Specs that share direction / width / speed reuse one coverage time series;
    only the bright/dark contrast scaling differs.
    """
    n_batch = len(specs)
    n_hexes = len(hexes)
    if n_hexes == 0 or n_batch == 0:
        return np.zeros((n_batch, n_t, n_hexes), dtype=np.float64)

    out = np.zeros((n_batch, n_t, n_hexes), dtype=np.float64)
    for b in range(n_batch):
        out[b, :t_onset] = i_baseline

    view_deg = view_bounds(hexes)
    hex_stack = np.stack([c.hex_xy for c in hexes], axis=0)

    by_geometry: dict[Tuple[str, float, float], List[int]] = {}
    for b, spec in enumerate(specs):
        key = (spec.direction, float(spec.width_deg), float(spec.speed_deg_s))
        by_geometry.setdefault(key, []).append(b)

    for batch_idxs in by_geometry.values():
        cov_ts = _coverage_time_series(
            hex_stack, specs[batch_idxs[0]], view_deg,
            n_t=n_t, t_onset=t_onset, delta_ms=delta_ms,
            bar_radius=bar_radius,
            multi_bar=multi_bar,
        )
        for b in batch_idxs:
            out[b, t_onset:] = _current_from_coverage(
                cov_ts, specs[b].contrast, i_baseline=i_baseline,
                i_bright_moving_bar=i_bright_moving_bar, i_dark_moving_bar=i_dark_moving_bar,
            )
    return out


@dataclass
class MovingBarSti:
    i_sti: torch.Tensor
    i_sti_hex: np.ndarray
    specs: List[MovingBarSpec]
    info: dict = field(default_factory=dict)


def resolve_i_baseline(value: float) -> float:
    """Cast sti baseline current (pA)."""
    return float(value)


def moving_bar_i_baseline_from_opts(train_opts) -> float:
    """``i_baseline_moving_bar`` from moving-bar sti opts on a train session."""
    opts = train_opts or {}
    for key in ("moving_bar_bright_sti_opts", "moving_bar_dark_sti_opts"):
        sub = opts.get(key) or {}
        if "i_baseline_moving_bar" in sub:
            return resolve_i_baseline(float(sub["i_baseline_moving_bar"]))
    raise ValueError(
        "moving-bar sti opts require i_baseline_moving_bar "
        "(inject via default_params.NETWORK_CONSTRUCTION['i_baseline'] / CLI)"
    )


@dataclass
class MovingBarT0Grids:
    t0_bn: np.ndarray
    before_t: Dict[str, int]
    after_t: Dict[str, int]


def moving_bar_spec_horizon(t_first_stis: Sequence[int], n_t: int) -> Tuple[int, int, int]:
    """Return ``(fb, before_t, after_t)`` for one spec over all hexes."""
    fb = int(min(t_first_stis))
    before = fb
    after = int(n_t) - int(max(t_first_stis))
    return fb, before, after


def moving_bar_network_t0_bn(connectome, filt_hexes: Sequence[StiHex], n_batch: int, t0_map: dict) -> np.ndarray:
    """Expand per-hex ``t0`` values to a full node grid ``(B, N_nodes)``."""
    node_u_np = np.asarray(connectome.us, dtype=np.int64)
    node_v_np = np.asarray(connectome.vs, dtype=np.int64)
    t0_bn = np.full((n_batch, connectome.n_nodes), -1, dtype=np.int64)
    for bi in range(n_batch):
        for c in filt_hexes:
            t0 = t0_map.get((bi, int(c.u), int(c.v)))
            if t0 is None:
                continue
            on_hex = (node_u_np == int(c.u)) & (node_v_np == int(c.v))
            t0_bn[bi, on_hex] = t0
    return t0_bn


def build_moving_bar_t0_grids(
    i_sti_hex: np.ndarray,
    specs: Sequence[MovingBarSpec],
    n_t: int,
    i_baseline: float,
    *,
    hex_idxs: Sequence[int],
    filt_hex_idxs: Sequence[int],
    connectome,
    filt_network_hexes: Sequence[StiHex],
) -> MovingBarT0Grids:
    """Plot/train-aligned ``t0`` grid and per-spec full horizons."""
    before_t: Dict[str, int] = {}
    after_t: Dict[str, int] = {}
    n_batch = len(specs)
    i_baseline = resolve_i_baseline(i_baseline)

    t0_map: dict = {}
    for bi, spec in enumerate(specs):
        t_first_all = [
            hex_first_sti_t(i_sti_hex[bi, :, hex_idx], i_baseline=i_baseline)
            for hex_idx in hex_idxs
        ]
        fb, before, after = moving_bar_spec_horizon(t_first_all, n_t)
        before_t[spec.name] = before
        after_t[spec.name] = after
        t_first_filt = [
            hex_first_sti_t(i_sti_hex[bi, :, hex_idx], i_baseline=i_baseline)
            for hex_idx in filt_hex_idxs
        ]
        for c, tc in zip(filt_network_hexes, t_first_filt):
            t0_map[(bi, int(c.u), int(c.v))] = tc - fb
    t0_bn = moving_bar_network_t0_bn(connectome, filt_network_hexes, n_batch, t0_map)
    return MovingBarT0Grids(t0_bn=t0_bn, before_t=before_t, after_t=after_t)


def _moving_bar_cache_key(
    network_json: Path,
    specs: Sequence[MovingBarSpec],
    hex_uv: Sequence[Tuple[int, int]],
    n_t: int,
    t_onset: int,
    delta_ms: float,
    i_baseline: float,
    bar_radius: int,
    multi_bar: bool = True,
    i_bright_moving_bar: Optional[float] = None,
    i_dark_moving_bar: Optional[float] = None,
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
                "direction": s.direction,
                "contrast": s.contrast,
                "width_deg": s.width_deg,
                "speed_deg_s": s.speed_deg_s,
            }
            for s in specs
        ],
        "n_t": n_t,
        "t_onset": t_onset,
        "delta_ms": delta_ms,
        "i_baseline_moving_bar": i_baseline,
    }
    if i_bright_moving_bar is not None:
        payload["i_bright_moving_bar"] = i_bright_moving_bar
    if i_dark_moving_bar is not None:
        payload["i_dark_moving_bar"] = i_dark_moving_bar
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def _moving_bar_cache_path(
    network_json: Path,
    specs: Sequence[MovingBarSpec],
    hex_uv: Sequence[Tuple[int, int]],
    n_t: int,
    t_onset: int,
    delta_ms: float,
    i_baseline: float,
    bar_radius: int,
    multi_bar: bool = True,
    i_bright_moving_bar: Optional[float] = None,
    i_dark_moving_bar: Optional[float] = None,
) -> Path:
    key = _moving_bar_cache_key(
        network_json, specs, hex_uv, n_t, t_onset, delta_ms,
        i_baseline, bar_radius, multi_bar, i_bright_moving_bar, i_dark_moving_bar,
    )
    return moving_bar_cache_dir(network_json) / f"{key}.npz"


def _load_moving_bar_hex_cache(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data["i_sti_hex"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Ignoring corrupt moving-bar cache %s: %s", path, exc)
        return None


def _save_moving_bar_hex_cache(path: Path, i_sti_hex: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, i_sti_hex=i_sti_hex)
    logger.info("Cached moving-bar i_sti_hex to %s", path)


def build_moving_bar_signals(
    connectome,
    specs: Optional[Sequence[MovingBarSpec]] = None,
    n_t: Optional[int] = None,
    t_onset: int = None,
    *,
    delta_ms: float,
    bar_radius: int = DEFAULT_BAR_RADIUS,
    multi_bar: bool = True,
    i_baseline: float,
    i_bright_moving_bar: Optional[float] = None,
    i_dark_moving_bar: Optional[float] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    network_json: Optional[Path] = None,
    sim_dtype: torch.dtype,
) -> MovingBarSti:
    """Build batched sti current for moving-bar stis.

    Returns ``i_sti`` with shape ``(B, T, N_nodes)`` where ``B = len(specs)``.
    """
    device = device or connectome.device
    bar_radius = int(bar_radius)
    multi_bar = bool(multi_bar)
    specs = list(specs if specs is not None else gruntman_moving_bar_specs())
    contrasts = {s.contrast for s in specs}
    i_bright = None
    i_dark = None
    if "bright" in contrasts:
        if i_bright_moving_bar is None:
            raise ValueError("i_bright_moving_bar required for bright contrast")
        i_bright = float(i_bright_moving_bar)
    if "dark" in contrasts:
        if i_dark_moving_bar is None:
            raise ValueError("i_dark_moving_bar required for dark contrast")
        i_dark = float(i_dark_moving_bar)
    sti = sti_hexes(connectome)
    view_deg = view_bounds(sti)
    if n_t is None:
        n_t = moving_bar_n_t(
            specs, view_deg, bar_radius, multi_bar=multi_bar, t_onset=t_onset, delta_ms=delta_ms,
        )
    n_batch = len(specs)
    n_nodes = connectome.n_nodes
    sweep_end = moving_bar_sweep_end_t(
        specs, view_deg, bar_radius, multi_bar=multi_bar, t_onset=t_onset, delta_ms=delta_ms,
    )
    sweep_t = sweep_end - t_onset
    tail_t = n_t - sweep_end

    cache_path: Optional[Path] = None
    source_json = Path(network_json) if network_json is not None else getattr(connectome, "source_json", None)
    hex_uv = [(c.u, c.v) for c in sti]
    if source_json is not None:
        cache_path = _moving_bar_cache_path(
            source_json, specs, hex_uv, n_t, t_onset, delta_ms,
            i_baseline, bar_radius, multi_bar, i_bright, i_dark,
        )

    i_sti_hex: Optional[np.ndarray] = None
    if cache_path is not None and use_cache and not refresh_cache:
        i_sti_hex = _load_moving_bar_hex_cache(cache_path)
        if i_sti_hex is not None:
            logger.info("Loaded moving-bar i_sti_hex from cache %s", cache_path)

    if i_sti_hex is None:
        i_sti_hex = build_batched_i_sti_hex(
            sti, specs, n_t=n_t, bar_radius=bar_radius, multi_bar=multi_bar,
            t_onset=t_onset, delta_ms=delta_ms,
            i_baseline=i_baseline, i_bright_moving_bar=i_bright, i_dark_moving_bar=i_dark,
        )
        if cache_path is not None and use_cache:
            _save_moving_bar_hex_cache(cache_path, i_sti_hex)

    i_sti_np = i_sti_nodes_from_hex(i_sti_hex, sti, n_nodes)

    info = {
        "n_batch": n_batch,
        "n_sti_hexes": len(sti),
        "bar_radius": bar_radius,
        "multi_bar": multi_bar,
        "view_deg": view_deg,
        "n_t": n_t,
        "t_onset": t_onset,
        "sweep_end": sweep_end,
        "sweep_t": sweep_t,
        "sweep_time_s": sweep_t * delta_ms / 1000.0,
        "tail_t": tail_t,
        "tail_time_s": tail_t * delta_ms / 1000.0,
        "i_baseline_moving_bar": i_baseline,
        "speed_deg_s": specs[0].speed_deg_s if specs else GRUNTMAN_SPEED_DEG_S,
        "spec_names": [s.name for s in specs],
    }
    if i_bright is not None:
        info["i_bright_moving_bar"] = i_bright
    if i_dark is not None:
        info["i_dark_moving_bar"] = i_dark
    return MovingBarSti(
        i_sti=torch.as_tensor(i_sti_np, dtype=sim_dtype, device=device),
        i_sti_hex=i_sti_hex,
        specs=specs,
        info=info,
    )
