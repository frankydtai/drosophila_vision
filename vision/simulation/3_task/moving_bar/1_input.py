# -*- coding: utf-8 -*-
"""Gruntman-style moving-bar stimulus: geometry, timing, and hex currents.

Pure visual-field math on hexes (degrees, coverage, pA). No connectome
or node indexing — :mod:`network.moving_bar_readout` maps these currents onto a network.
"""
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

from network import path  # noqa: F401 -- FAFBv783 on sys.path

from build_hex import DEG, HEX_PATCH_RADIUS, hex_vertices, xy_from_uv, xy_deg_from_uv
from path import moving_bar_cache_dir
from neuron.param import t_from_ms
from network.construction import hex_in_cost_radius

logger = logging.getLogger(__name__)

# Gruntman Fig. 1 Ci fast condition: 40 ms / 2.25 deg per LED step.
GRUNTMAN_SPEED_DEG_S = 56.0
GRUNTMAN_WIDTHS_DEG = (2.25, 9.0)
GRUNTMAN_DIRECTIONS = ("right", "left", "up", "down")
GRUNTMAN_CONTRASTS = ("bright", "dark")

# Per-lane spacing width in hex nodes (``spacing_deg = bar_radius * DEG``).
DEFAULT_BAR_RADIUS = 2

# Moving-bar per-hex cost window relative to first-stimulus alignment.
COST_WINDOW_MS = 900.0
COST_ALIGNED_FIRST_STI_MS = 300.0
COST_WINDOW_BEFORE_MS = COST_ALIGNED_FIRST_STI_MS
COST_WINDOW_AFTER_MS = COST_WINDOW_MS - COST_ALIGNED_FIRST_STI_MS

# Post-sweep tail: baseline after bar exit through ``t_first_sti + after`` plus pad.
T_TAIL_PAD_MS = 50.0
MOVING_BAR_TAIL_MS = COST_WINDOW_AFTER_MS + T_TAIL_PAD_MS


def cost_window_before_t(delta_ms: float) -> int:
    """``t_from_ms(COST_WINDOW_BEFORE_MS)``."""
    return t_from_ms(COST_WINDOW_BEFORE_MS, delta_ms=delta_ms)


def cost_window_after_t(delta_ms: float) -> int:
    """``t_from_ms(COST_WINDOW_AFTER_MS)``."""
    return t_from_ms(COST_WINDOW_AFTER_MS, delta_ms=delta_ms)


_HEX_AREA = 1.5 * math.sqrt(3.0) * float(HEX_PATCH_RADIUS) ** 2


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


@dataclass
class Hex:
    """One FAFB sti hex: axial coords, hex-step ``(x,y)``, degree ``(x_deg,y_deg)``."""

    u: int
    v: int
    x: float
    y: float
    x_deg: float
    y_deg: float
    hex_xy: np.ndarray


def hex_from_uv(u: int, v: int) -> Hex:
    """Build one FAFB sti hex from axial ``(u, v)``."""
    x, y = xy_from_uv(u, v)
    x, y = float(x), float(y)
    x_deg, y_deg = xy_deg_from_uv(u, v)
    x_deg, y_deg = float(x_deg), float(y_deg)
    return Hex(
        u=int(u),
        v=int(v),
        x=x,
        y=y,
        x_deg=x_deg,
        y_deg=y_deg,
        hex_xy=hex_vertices(x_deg, y_deg),
    )


def gruntman_moving_bar_specs(
    directions: Sequence[str] = GRUNTMAN_DIRECTIONS,
    contrasts: Sequence[str] = GRUNTMAN_CONTRASTS,
    widths_deg: Sequence[float] = GRUNTMAN_WIDTHS_DEG,
    speed_deg_s: float = GRUNTMAN_SPEED_DEG_S,
) -> List[MovingBarSpec]:
    """The 16 Gruntman-style whole-field moving-bar conditions."""
    return [
        MovingBarSpec(direction=d, contrast=c, width_deg=w, speed_deg_s=speed_deg_s)
        for d in directions
        for c in contrasts
        for w in widths_deg
    ]


def _cross2(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _seg_intersect(
    px0: float, py0: float, px1: float, py1: float,
    qx0: float, qy0: float, qx1: float, qy1: float,
) -> Tuple[float, float]:
    rx, ry = px1 - px0, py1 - py0
    sx, sy = qx1 - qx0, qy1 - qy0
    denom = _cross2(rx, ry, sx, sy)
    if abs(denom) < 1e-12:
        return px1, py1
    t = _cross2(qx0 - px0, qy0 - py0, sx, sy) / denom
    return px0 + t * rx, py0 + t * ry


def _clip_halfplane(
    px: np.ndarray,
    py: np.ndarray,
    n: int,
    bound: float,
    *,
    axis: int,
    keep_ge: bool,
    outx: np.ndarray,
    outy: np.ndarray,
) -> int:
    """Sutherland–Hodgman half-plane clip; ``axis`` 0=x / 1=y."""
    m = 0
    prev_x, prev_y = float(px[n - 1]), float(py[n - 1])
    prev_c = prev_x if axis == 0 else prev_y
    prev_in = prev_c >= bound if keep_ge else prev_c <= bound
    if axis == 0:
        qx0, qy0, qx1, qy1 = bound, -1e6, bound, 1e6
    else:
        qx0, qy0, qx1, qy1 = -1e6, bound, 1e6, bound
    for i in range(n):
        cur_x, cur_y = float(px[i]), float(py[i])
        cur_c = cur_x if axis == 0 else cur_y
        cur_in = cur_c >= bound if keep_ge else cur_c <= bound
        if cur_in:
            if not prev_in:
                ix, iy = _seg_intersect(prev_x, prev_y, cur_x, cur_y, qx0, qy0, qx1, qy1)
                outx[m], outy[m] = ix, iy
                m += 1
            outx[m], outy[m] = cur_x, cur_y
            m += 1
        elif prev_in:
            ix, iy = _seg_intersect(prev_x, prev_y, cur_x, cur_y, qx0, qy0, qx1, qy1)
            outx[m], outy[m] = ix, iy
            m += 1
        prev_x, prev_y, prev_in = cur_x, cur_y, cur_in
    return m


def _poly_area_xy(px: np.ndarray, py: np.ndarray, n: int) -> float:
    if n < 3:
        return 0.0
    x = px[:n]
    y = py[:n]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_rect_area(
    hex_xy: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    hex_area: float = _HEX_AREA,
) -> float:
    """Axis-aligned rect ∩ hex area / hex_area, using fixed buffers (hot path)."""
    w1x = np.empty(12, dtype=np.float64)
    w1y = np.empty(12, dtype=np.float64)
    w1x[:6] = hex_xy[:, 0]
    w1y[:6] = hex_xy[:, 1]
    w2x = np.empty(12, dtype=np.float64)
    w2y = np.empty(12, dtype=np.float64)
    px, py, ox, oy = w1x, w1y, w2x, w2y
    n = 6

    for axis, bound, keep_ge in (
        (0, xmin, True),
        (0, xmax, False),
        (1, ymin, True),
        (1, ymax, False),
    ):
        n = _clip_halfplane(px, py, n, bound, axis=axis, keep_ge=keep_ge, outx=ox, outy=oy)
        if n == 0:
            return 0.0
        px, py, ox, oy = ox, oy, px, py

    return min(1.0, _poly_area_xy(px, py, n) / hex_area)


def _coverage_batch(
    hex_stack: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> np.ndarray:
    """Coverage fraction for every hex against one bar rectangle."""
    n_hexes = hex_stack.shape[0]
    out = np.empty(n_hexes, dtype=np.float64)
    for j in range(n_hexes):
        out[j] = _clip_rect_area(hex_stack[j], xmin, ymin, xmax, ymax)
    return out


def bar_lane_pitch_deg(
    spec: MovingBarSpec,
    bar_radius: int,
    *,
    field_deg: Optional[Tuple[float, float, float, float]] = None,
    multi_bar: bool = True,
) -> float:
    """Lane pitch in degrees.

    Default (``multi_bar=True``): integer width+spacing hex counts.
    ``multi_bar=False`` + ``field_deg``: one lane spans the full sti-field degree on the motion axis.
    """
    if int(bar_radius) < 0:
        raise ValueError(f"bar_radius must be >= 0, got {bar_radius!r}")
    if not bool(multi_bar):
        if field_deg is None:
            raise ValueError("field_deg required when multi_bar=False")
        x0, y0, x1, y1 = field_deg
        if spec.direction in ("right", "left"):
            return float(x1) - float(x0)
        if spec.direction in ("up", "down"):
            return float(y1) - float(y0)
        raise ValueError(f"unknown direction {spec.direction!r}")
    width_cols = int(math.ceil(float(spec.width_deg) / float(DEG)))
    return float(width_cols + int(bar_radius)) * float(DEG)


def _motion_field_lo_hi(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Motion-axis field bounds ``(lo, hi)`` in degrees."""
    x0, y0, x1, y1 = field_deg
    if spec.direction in ("right", "left"):
        return float(x0), float(x1)
    if spec.direction in ("up", "down"):
        return float(y0), float(y1)
    raise ValueError(f"unknown direction {spec.direction!r}")


def _motion_entry_exit(
    spec: MovingBarSpec,
    field_lo: float,
    field_hi: float,
) -> Tuple[float, float]:
    """``(entry_s, exit_s)`` on the motion axis (bar enters at ``entry_s``)."""
    if spec.direction in ("right", "up"):
        return field_lo, field_hi
    if spec.direction in ("left", "down"):
        return field_hi, field_lo
    raise ValueError(f"unknown direction {spec.direction!r}")


def _motion_lanes(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    bar_radius: int,
    *,
    multi_bar: bool = True,
) -> List[Tuple[float, float]]:
    """``(lane_origin, lane_pitch)`` tiled from the motion entry edge through field exit."""
    field_lo, field_hi = _motion_field_lo_hi(spec, field_deg)
    if not bool(multi_bar):
        return [(field_lo, field_hi - field_lo)]
    pitch = bar_lane_pitch_deg(spec, bar_radius, field_deg=field_deg, multi_bar=True)
    entry_s, exit_s = _motion_entry_exit(spec, field_lo, field_hi)
    eps = 1e-9
    lanes: List[Tuple[float, float]] = []
    if float(entry_s) < float(exit_s):
        p = float(entry_s)
        exit_v = float(exit_s)
        while p < exit_v - eps:
            lane_pitch = min(pitch, exit_v - p)
            lanes.append((p, lane_pitch))
            p += pitch
    else:
        p = float(entry_s) - pitch
        exit_v = float(exit_s)
        while p + pitch > exit_v + eps:
            if p < exit_v - eps:
                lanes.append((exit_v, (p + pitch) - exit_v))
                break
            lanes.append((p, pitch))
            p -= pitch
    if not lanes:
        origin = min(float(entry_s), float(exit_s))
        lanes.append((origin, min(pitch, abs(float(exit_s) - float(entry_s)))))
    return lanes


def _lane_sweep_trail_range(
    spec: MovingBarSpec,
    lane_origin: float,
    lane_pitch: float,
) -> Tuple[float, float]:
    """``(trail_start, trail_exit)`` for one lane; bar clips to ``[origin, origin+pitch]``."""
    w = float(spec.width_deg)
    pitch = float(lane_pitch)
    origin = float(lane_origin)
    if spec.direction in ("right", "up"):
        return origin - w, origin + pitch
    if spec.direction in ("left", "down"):
        return origin + pitch + w, origin
    raise ValueError(f"unknown direction {spec.direction!r}")


def _bar_rect_lane_clipped(
    spec: MovingBarSpec,
    trail: float,
    lane_origin: float,
    lane_pitch: float,
    field_deg: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float]]:
    """Bar rectangle clipped to one lane; ``None`` when zero visible width."""
    x0, y0, x1, y1 = field_deg
    w = float(spec.width_deg)
    origin = float(lane_origin)
    lane_end = origin + float(lane_pitch)
    d = spec.direction
    if d in ("right", "up"):
        vis_lo = max(float(trail), origin)
        vis_hi = min(float(trail) + w, lane_end)
    elif d in ("left", "down"):
        vis_lo = max(float(trail) - w, origin)
        vis_hi = min(float(trail), lane_end)
    else:
        raise ValueError(f"unknown direction {d!r}")
    if vis_lo >= vis_hi - 1e-12:
        return None
    if d in ("right", "left"):
        return vis_lo, y0, vis_hi, y1
    return x0, vis_lo, x1, vis_hi



def _coverage_time_series(
    hex_stack: np.ndarray,
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
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
    lane_origins = _motion_lanes(spec, field_deg, bar_radius, multi_bar=multi_bar)
    n_hexes = hex_stack.shape[0]
    n_post = n_t - t_onset
    out = np.zeros((n_post, n_hexes), dtype=np.float64)
    for lane_origin, lane_pitch in lane_origins:
        trail_start, trail_exit = _lane_sweep_trail_range(spec, lane_origin, lane_pitch)
        trail = float(trail_start)
        for i in range(n_post):
            rect = _bar_rect_lane_clipped(
                spec, trail, lane_origin, lane_pitch, field_deg,
            )
            if rect is not None:
                bx0, by0, bx1, by1 = rect
                out[i] += _coverage_batch(hex_stack, bx0, by0, bx1, by1)
            trail += trail_shift_deg
    return np.clip(out, 0.0, 1.0)


def field_bounds(hexes: Sequence[Hex]) -> Tuple[float, float, float, float]:
    """Sti-field degree bounds from hex vertices (not centers)."""
    if not hexes:
        return 0.0, 0.0, 0.0, 0.0
    xmins = [float(c.hex_xy[:, 0].min()) for c in hexes]
    ymins = [float(c.hex_xy[:, 1].min()) for c in hexes]
    xmaxs = [float(c.hex_xy[:, 0].max()) for c in hexes]
    ymaxs = [float(c.hex_xy[:, 1].max()) for c in hexes]
    return min(xmins), min(ymins), max(xmaxs), max(ymaxs)


def _trail_shift_deg(spec: MovingBarSpec, dt_s: float) -> float:
    """Signed trail advance (deg) in one sample: ``±speed_deg_s * dt_s``."""
    s = float(spec.speed_deg_s) * dt_s
    if spec.direction in ("right", "up"):
        return s
    if spec.direction in ("left", "down"):
        return -s
    raise ValueError(f"unknown direction {spec.direction!r}")


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
    field_deg: Tuple[float, float, float, float],
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
        for lane_origin, lane_pitch in _motion_lanes(spec, field_deg, bar_radius, multi_bar=multi_bar):
            trail_start, trail_exit = _lane_sweep_trail_range(spec, lane_origin, lane_pitch)
            t_exit = max(
                t_exit,
                t_from_trail(spec, trail_start, trail_exit, t_onset, delta_ms),
            )
    return t_exit + 1



def moving_bar_n_t(
    specs: Sequence[MovingBarSpec],
    field_deg: Tuple[float, float, float, float],
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
        specs, field_deg, bar_radius, multi_bar=multi_bar, t_onset=t_onset, delta_ms=delta_ms,
    ) + t_tail



def moving_bar_transit_times(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    bar_radius: int,
    *,
    multi_bar: bool = True,
    t_onset: int = None,
    n_t: Optional[int] = None,
    delta_ms: float,
) -> Tuple[int, int, int]:
    """Return ``(entry, center, exit)`` t indices for the first multi-bar lane."""
    lane_origin, lane_pitch = _motion_lanes(spec, field_deg, bar_radius, multi_bar=multi_bar)[0]
    trail_start, trail_exit = _lane_sweep_trail_range(spec, lane_origin, lane_pitch)
    w = float(spec.width_deg)
    trail_shift_deg = _trail_shift_deg(spec, delta_ms / 1000.0)
    origin = float(lane_origin)
    # Signed ``trail_shift_deg`` encodes sweep direction; same trail ends for all directions.
    trail_entry = float(trail_start) + trail_shift_deg
    trail_center = origin + 0.5 * (
        float(lane_pitch) - math.copysign(1.0, trail_shift_deg) * w
    )
    trail_exit_vis = float(trail_exit) - trail_shift_deg
    return (
        t_from_trail(spec, trail_start, trail_entry, t_onset, delta_ms, n_t),
        t_from_trail(spec, trail_start, trail_center, t_onset, delta_ms, n_t),
        t_from_trail(spec, trail_start, trail_exit_vis, t_onset, delta_ms, n_t),
    )



def hex_first_stim_t(
    i_sti_hex: np.ndarray,
    i_baseline: float,
    *,
    atol: float = 1e-12,
) -> int:
    """First t where a hex i_sti differs from baseline (``t_first_sti``)."""
    curr = np.asarray(i_sti_hex, dtype=np.float64).reshape(-1)
    active = ~np.isclose(curr, float(i_baseline), atol=atol, rtol=0.0)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        raise ValueError("hex has no non-baseline stimulus sample")
    return int(idx[0])


def bar_lane_rects_at_t(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
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
    for lane_origin, lane_pitch in _motion_lanes(spec, field_deg, bar_radius, multi_bar=multi_bar):
        trail_start, _ = _lane_sweep_trail_range(spec, lane_origin, lane_pitch)
        trail = float(trail_start) + (t - t_onset) * trail_shift_deg
        rect = _bar_rect_lane_clipped(
            spec, trail, lane_origin, lane_pitch, field_deg,
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

    field_deg = field_bounds(hexes)
    hex_stack = np.stack([c.hex_xy for c in hexes], axis=0)

    by_geometry: dict[Tuple[str, float, float], List[int]] = {}
    for b, spec in enumerate(specs):
        key = (spec.direction, float(spec.width_deg), float(spec.speed_deg_s))
        by_geometry.setdefault(key, []).append(b)

    for batch_idxs in by_geometry.values():
        cov_ts = _coverage_time_series(
            hex_stack, specs[batch_idxs[0]], field_deg,
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


# -- Connectome mapping: hex currents -> node ``i_sti`` (was moving_bar_readout) --

PD_IDX, ND_IDX = 0, 1


@dataclass
class StiHex(Hex):
    """One sti hex on a connectome, with node indices for writing onto ``i_sti``."""

    node_idx: np.ndarray


@dataclass
class MovingBarStimulus:
    i_sti: torch.Tensor
    i_sti_hex: np.ndarray
    specs: List[MovingBarSpec]
    info: dict = field(default_factory=dict)


def sti_hexes(connectome) -> List[StiHex]:
    """Sti hexes with sti nodes (one per axial ``(u, v)``)."""
    by_uv: Dict[Tuple[int, int], StiHex] = {}
    for node_idx in connectome.sti_node_idx:
        u, v = int(connectome.u[node_idx]), int(connectome.v[node_idx])
        key = (u, v)
        if key in by_uv:
            continue
        nodes = connectome.sti_nodes_at(u, v)
        if len(nodes) == 0:
            continue
        base = hex_from_uv(key[0], key[1])
        by_uv[key] = StiHex(
            u=base.u, v=base.v, x=base.x, y=base.y,
            x_deg=base.x_deg, y_deg=base.y_deg, hex_xy=base.hex_xy,
            node_idx=np.asarray(nodes, dtype=np.int64),
        )
    return [by_uv[k] for k in sorted(by_uv)]


def moving_bar_cost_hexes(connectome, cost_radius=None) -> List[StiHex]:
    """Sti hexes used for moving-bar cost (optional central hex disc)."""
    hexes = sti_hexes(connectome)
    if cost_radius is None:
        return hexes
    return [c for c in hexes if hex_in_cost_radius(c.u, c.v, cost_radius)]


def _as_int64_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return np.asarray(x.detach().cpu().numpy(), dtype=np.int64)
    return np.asarray(x, dtype=np.int64)


def network_uv_np(connectome) -> Tuple[np.ndarray, np.ndarray]:
    """connectome axial ``(u, v)`` per node as int64 numpy arrays."""
    return _as_int64_np(connectome.u), _as_int64_np(connectome.v)


def _coord_matches(val, axis_filter, tol=1e-6) -> bool:
    if axis_filter is None:
        return True
    if isinstance(axis_filter, (list, tuple)):
        return any(np.isclose(val, float(v), atol=tol) for v in axis_filter)
    return np.isclose(val, float(axis_filter), atol=tol)


def filter_sti_hexes(hexes, *, at_x=None, at_y=None, tol=1e-6):
    """Keep network sti hexes whose hex-step ``(x, y)`` matches ``at_x`` / ``at_y``."""
    if at_x is None and at_y is None:
        return list(hexes)
    out = []
    for hex in hexes:
        if not _coord_matches(hex.x, at_x, tol=tol):
            continue
        if not _coord_matches(hex.y, at_y, tol=tol):
            continue
        out.append(hex)
    return out


def resolve_i_baseline(value: float) -> float:
    """Cast sti baseline current (pA)."""
    return float(value)


def moving_bar_i_baseline_from_opts(train_opts) -> float:
    """``i_baseline_moving_bar`` from moving-bar stimulus opts on a train session."""
    opts = train_opts or {}
    for key in ("moving_bar_bright_stimulus_opts", "moving_bar_dark_stimulus_opts"):
        sub = opts.get(key) or {}
        if "i_baseline_moving_bar" in sub:
            return resolve_i_baseline(float(sub["i_baseline_moving_bar"]))
    raise ValueError(
        "moving-bar stimulus opts require i_baseline_moving_bar "
        "(inject via param_defaults.I_BASELINE / CLI)"
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
    node_u_np = np.asarray(connectome.u, dtype=np.int64)
    node_v_np = np.asarray(connectome.v, dtype=np.int64)
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
    all_hex_idxs: Sequence[int],
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
            hex_first_stim_t(i_sti_hex[bi, :, hex_idx], i_baseline=i_baseline)
            for hex_idx in all_hex_idxs
        ]
        fb, before, after = moving_bar_spec_horizon(t_first_all, n_t)
        before_t[spec.name] = before
        after_t[spec.name] = after
        t_first_filt = [
            hex_first_stim_t(i_sti_hex[bi, :, hex_idx], i_baseline=i_baseline)
            for hex_idx in filt_hex_idxs
        ]
        for c, tc in zip(filt_network_hexes, t_first_filt):
            t0_map[(bi, int(c.u), int(c.v))] = tc - fb
    t0_bn = moving_bar_network_t0_bn(connectome, filt_network_hexes, n_batch, t0_map)
    return MovingBarT0Grids(t0_bn=t0_bn, before_t=before_t, after_t=after_t)


def _hex_node_map(hexes: Sequence[StiHex]) -> Tuple[np.ndarray, np.ndarray]:
    hex_idx: List[int] = []
    node_idx: List[int] = []
    for j, hex in enumerate(hexes):
        for u in np.asarray(hex.node_idx).ravel():
            hex_idx.append(j)
            node_idx.append(int(u))
    return (
        np.asarray(hex_idx, dtype=np.int64),
        np.asarray(node_idx, dtype=np.int64),
    )


def i_sti_nodes_from_hex(i_sti_hex, hexes, n_nodes):
    """Map ``(B, T, n_hexes)`` i_sti_hex to ``(B, T, n_nodes)`` by hex→node index."""
    n_batch, n_t, _ = i_sti_hex.shape
    out = np.zeros((n_batch, n_t, n_nodes), dtype=np.float64)
    hex_idx, node_idx = _hex_node_map(hexes)
    if len(hex_idx):
        out[:, :, node_idx] = i_sti_hex[:, :, hex_idx]
    return out


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
) -> MovingBarStimulus:
    """Build batched sti current for moving-bar stimuli.

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
    field_deg = field_bounds(sti)
    if n_t is None:
        n_t = moving_bar_n_t(
            specs, field_deg, bar_radius, multi_bar=multi_bar, t_onset=t_onset, delta_ms=delta_ms,
        )
    n_batch = len(specs)
    n_nodes = connectome.n_nodes
    sweep_end = moving_bar_sweep_end_t(
        specs, field_deg, bar_radius, multi_bar=multi_bar, t_onset=t_onset, delta_ms=delta_ms,
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
        "field_deg": field_deg,
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
    return MovingBarStimulus(
        i_sti=torch.as_tensor(i_sti_np, dtype=sim_dtype, device=device),
        i_sti_hex=i_sti_hex,
        specs=specs,
        info=info,
    )

