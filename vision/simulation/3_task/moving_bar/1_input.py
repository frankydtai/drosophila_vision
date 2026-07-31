# -*- coding: utf-8 -*-
"""Gruntman-style moving-bar stimulus: geometry, timing, and column currents.

Pure visual-field math on hex columns (degrees, coverage, pA). No connectome
or unit indexing — :mod:`network.moving_bar_target` maps these currents onto a network.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from network import path  # noqa: F401 -- FAFBv783 on sys.path

from build_hex import DEG, HEX_PATCH_RADIUS, hex_vertices, uv_to_xy, uv_to_xy_deg
from path import moving_bar_cache_dir
from neuron.params import ms_to_t
from network.layout import column_in_cost_extent

logger = logging.getLogger(__name__)

# Gruntman Fig. 1 Ci fast condition: 40 ms / 2.25 deg per LED step.
GRUNTMAN_SPEED_DEG_S = 56.0
GRUNTMAN_WIDTHS_DEG = (2.25, 9.0)
GRUNTMAN_DIRECTIONS = ("right", "left", "up", "down")
GRUNTMAN_CONTRASTS = ("bright", "dark")

# Per-lane spacing width in hex-column units (``spacing_deg = bar_extent * DEG``).
DEFAULT_BAR_EXTENT = 2

# Moving-bar per-column cost window relative to first-stimulus alignment.
COST_WINDOW_MS = 900.0
COST_ALIGNED_FIRST_STI_MS = 300.0
COST_WINDOW_BEFORE_MS = COST_ALIGNED_FIRST_STI_MS
COST_WINDOW_AFTER_MS = COST_WINDOW_MS - COST_ALIGNED_FIRST_STI_MS

# Post-sweep tail: baseline after bar exit through ``t_first_sti + after`` plus pad.
T_TAIL_PAD_MS = 50.0
MOVING_BAR_TAIL_MS = COST_WINDOW_AFTER_MS + T_TAIL_PAD_MS

def cost_window_before_t(delta_ms: float) -> int:
    """``ms_to_t(COST_WINDOW_BEFORE_MS)``."""
    return ms_to_t(COST_WINDOW_BEFORE_MS, delta_ms=delta_ms)


def cost_window_after_t(delta_ms: float) -> int:
    """``ms_to_t(COST_WINDOW_AFTER_MS)``."""
    return ms_to_t(COST_WINDOW_AFTER_MS, delta_ms=delta_ms)


def cost_window_t(delta_ms: float) -> int:
    """``ms_to_t(COST_WINDOW_MS) + 1`` (inclusive window length)."""
    return ms_to_t(COST_WINDOW_MS, delta_ms=delta_ms) + 1


def t_tail(delta_ms: float) -> int:
    """``ms_to_t(MOVING_BAR_TAIL_MS)``."""
    return ms_to_t(MOVING_BAR_TAIL_MS, delta_ms=delta_ms)

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
class HexColumn:
    """One FAFB sti column: axial coords, hex-step ``(x,y)``, degree ``(x_deg,y_deg)``."""

    u: int
    v: int
    x: float
    y: float
    x_deg: float
    y_deg: float
    hex_xy: np.ndarray


def hex_column_from_uv(u: int, v: int) -> HexColumn:
    """Build one FAFB sti column from axial ``(u, v)``."""
    x, y = uv_to_xy(u, v)
    x, y = float(x), float(y)
    x_deg, y_deg = uv_to_xy_deg(u, v)
    x_deg, y_deg = float(x_deg), float(y_deg)
    return HexColumn(
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


def _clip_halfplane_xmin(
    px: np.ndarray, py: np.ndarray, n: int, xmin: float,
    outx: np.ndarray, outy: np.ndarray,
) -> int:
    m = 0
    prev_x, prev_y = float(px[n - 1]), float(py[n - 1])
    prev_in = prev_x >= xmin
    for i in range(n):
        cur_x, cur_y = float(px[i]), float(py[i])
        cur_in = cur_x >= xmin
        if cur_in:
            if not prev_in:
                ix, iy = _seg_intersect(prev_x, prev_y, cur_x, cur_y, xmin, -1e6, xmin, 1e6)
                outx[m], outy[m] = ix, iy
                m += 1
            outx[m], outy[m] = cur_x, cur_y
            m += 1
        elif prev_in:
            ix, iy = _seg_intersect(prev_x, prev_y, cur_x, cur_y, xmin, -1e6, xmin, 1e6)
            outx[m], outy[m] = ix, iy
            m += 1
        prev_x, prev_y, prev_in = cur_x, cur_y, cur_in
    return m


def _clip_halfplane_xmax(
    px: np.ndarray, py: np.ndarray, n: int, xmax: float,
    outx: np.ndarray, outy: np.ndarray,
) -> int:
    m = 0
    prev_x, prev_y = float(px[n - 1]), float(py[n - 1])
    prev_in = prev_x <= xmax
    for i in range(n):
        cur_x, cur_y = float(px[i]), float(py[i])
        cur_in = cur_x <= xmax
        if cur_in:
            if not prev_in:
                ix, iy = _seg_intersect(prev_x, prev_y, cur_x, cur_y, xmax, -1e6, xmax, 1e6)
                outx[m], outy[m] = ix, iy
                m += 1
            outx[m], outy[m] = cur_x, cur_y
            m += 1
        elif prev_in:
            ix, iy = _seg_intersect(prev_x, prev_y, cur_x, cur_y, xmax, -1e6, xmax, 1e6)
            outx[m], outy[m] = ix, iy
            m += 1
        prev_x, prev_y, prev_in = cur_x, cur_y, cur_in
    return m


def _clip_halfplane_ymin(
    px: np.ndarray, py: np.ndarray, n: int, ymin: float,
    outx: np.ndarray, outy: np.ndarray,
) -> int:
    m = 0
    prev_x, prev_y = float(px[n - 1]), float(py[n - 1])
    prev_in = prev_y >= ymin
    for i in range(n):
        cur_x, cur_y = float(px[i]), float(py[i])
        cur_in = cur_y >= ymin
        if cur_in:
            if not prev_in:
                ix, iy = _seg_intersect(prev_x, prev_y, cur_x, cur_y, -1e6, ymin, 1e6, ymin)
                outx[m], outy[m] = ix, iy
                m += 1
            outx[m], outy[m] = cur_x, cur_y
            m += 1
        elif prev_in:
            ix, iy = _seg_intersect(prev_x, prev_y, cur_x, cur_y, -1e6, ymin, 1e6, ymin)
            outx[m], outy[m] = ix, iy
            m += 1
        prev_x, prev_y, prev_in = cur_x, cur_y, cur_in
    return m


def _clip_halfplane_ymax(
    px: np.ndarray, py: np.ndarray, n: int, ymax: float,
    outx: np.ndarray, outy: np.ndarray,
) -> int:
    m = 0
    prev_x, prev_y = float(px[n - 1]), float(py[n - 1])
    prev_in = prev_y <= ymax
    for i in range(n):
        cur_x, cur_y = float(px[i]), float(py[i])
        cur_in = cur_y <= ymax
        if cur_in:
            if not prev_in:
                ix, iy = _seg_intersect(prev_x, prev_y, cur_x, cur_y, -1e6, ymax, 1e6, ymax)
                outx[m], outy[m] = ix, iy
                m += 1
            outx[m], outy[m] = cur_x, cur_y
            m += 1
        elif prev_in:
            ix, iy = _seg_intersect(prev_x, prev_y, cur_x, cur_y, -1e6, ymax, 1e6, ymax)
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

    n = _clip_halfplane_xmin(px, py, n, xmin, ox, oy)
    if n == 0:
        return 0.0
    px, py, ox, oy = ox, oy, px, py

    n = _clip_halfplane_xmax(px, py, n, xmax, ox, oy)
    if n == 0:
        return 0.0
    px, py, ox, oy = ox, oy, px, py

    n = _clip_halfplane_ymin(px, py, n, ymin, ox, oy)
    if n == 0:
        return 0.0
    px, py, ox, oy = ox, oy, px, py

    n = _clip_halfplane_ymax(px, py, n, ymax, ox, oy)
    if n == 0:
        return 0.0

    return min(1.0, _poly_area_xy(ox, oy, n) / hex_area)


def _coverage_batch(
    hex_stack: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> np.ndarray:
    """Coverage fraction for every column hex against one bar rectangle."""
    n_cols = hex_stack.shape[0]
    out = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        out[j] = _clip_rect_area(hex_stack[j], xmin, ymin, xmax, ymax)
    return out


def _geometry_key(spec: MovingBarSpec) -> Tuple[str, float, float]:
    return (spec.direction, float(spec.width_deg), float(spec.speed_deg_s))


def bar_spacing_deg(bar_extent: int) -> float:
    """Lane spacing width in degrees (``bar_extent`` hex-column units × ``DEG``)."""
    if int(bar_extent) < 0:
        raise ValueError(f"bar_extent must be >= 0, got {bar_extent!r}")
    return float(bar_extent) * float(DEG)


def bar_width_cols(spec: MovingBarSpec) -> int:
    """Bar width in whole columns, quantized by ``ceil(width_deg / DEG)``."""
    return int(math.ceil(float(spec.width_deg) / float(DEG)))


def bar_lane_pitch_deg(
    spec: MovingBarSpec,
    bar_extent: int,
    *,
    field_deg: Optional[Tuple[float, float, float, float]] = None,
    multi_bar: bool = True,
) -> float:
    """Lane pitch in degrees.

    Default (``multi_bar=True``): integer width+spacing column counts.
    ``multi_bar=False`` + ``field_deg``: one lane spans the full sti-field extent on the motion axis.
    """
    if int(bar_extent) < 0:
        raise ValueError(f"bar_extent must be >= 0, got {bar_extent!r}")
    if not bool(multi_bar):
        if field_deg is None:
            raise ValueError("field_deg required when multi_bar=False")
        x0, y0, x1, y1 = field_deg
        if spec.direction in ("right", "left"):
            return float(x1) - float(x0)
        if spec.direction in ("up", "down"):
            return float(y1) - float(y0)
        raise ValueError(f"unknown direction {spec.direction!r}")
    pitch_cols = bar_width_cols(spec) + int(bar_extent)
    return float(pitch_cols) * float(DEG)


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
    bar_extent: int,
    *,
    multi_bar: bool = True,
) -> List[Tuple[float, float]]:
    """``(lane_origin, lane_pitch)`` tiled from the motion entry edge through field exit."""
    field_lo, field_hi = _motion_field_lo_hi(spec, field_deg)
    if not bool(multi_bar):
        return [(field_lo, field_hi - field_lo)]
    pitch = bar_lane_pitch_deg(spec, bar_extent, field_deg=field_deg, multi_bar=True)
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
    d = spec.direction
    if d == "right":
        return origin - w, origin + pitch
    if d == "left":
        return origin + pitch + w, origin
    if d == "up":
        return origin - w, origin + pitch
    if d == "down":
        return origin + pitch + w, origin
    raise ValueError(f"unknown direction {d!r}")


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
    if d == "right":
        vis_lo = max(float(trail), origin)
        vis_hi = min(float(trail) + w, lane_end)
        if vis_lo >= vis_hi - 1e-12:
            return None
        return vis_lo, y0, vis_hi, y1
    if d == "left":
        vis_lo = max(float(trail) - w, origin)
        vis_hi = min(float(trail), lane_end)
        if vis_lo >= vis_hi - 1e-12:
            return None
        return vis_lo, y0, vis_hi, y1
    if d == "up":
        vis_lo = max(float(trail), origin)
        vis_hi = min(float(trail) + w, lane_end)
        if vis_lo >= vis_hi - 1e-12:
            return None
        return x0, vis_lo, x1, vis_hi
    if d == "down":
        vis_lo = max(float(trail) - w, origin)
        vis_hi = min(float(trail), lane_end)
        if vis_lo >= vis_hi - 1e-12:
            return None
        return x0, vis_lo, x1, vis_hi
    raise ValueError(f"unknown direction {d!r}")



def _coverage_time_series(
    hex_stack: np.ndarray,
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    n_t: int,
    t_on: int,
    delta_ms: float,
    bar_extent: int,
    *,
    multi_bar: bool = True,
) -> np.ndarray:
    """Superposed coverage from simultaneous per-lane bars (network connectome field)."""
    dt_s = delta_ms / 1000.0
    trail_shift_deg = _trail_shift_deg(spec, dt_s)
    lane_origins = _motion_lanes(spec, field_deg, bar_extent, multi_bar=multi_bar)
    n_cols = hex_stack.shape[0]
    n_post = n_t - t_on
    out = np.zeros((n_post, n_cols), dtype=np.float64)
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


def coverage_hex_bar(
    hex_xy: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    hex_area: float = _HEX_AREA,
) -> float:
    """Fraction of a column hex covered by an axis-aligned bar rectangle."""
    return _clip_rect_area(hex_xy, xmin, ymin, xmax, ymax, hex_area=hex_area)


def field_bounds(columns: Sequence[HexColumn]) -> Tuple[float, float, float, float]:
    """Sti-field extent in degrees from column hex vertices (not centers)."""
    if not columns:
        return 0.0, 0.0, 0.0, 0.0
    xmins = [float(c.hex_xy[:, 0].min()) for c in columns]
    ymins = [float(c.hex_xy[:, 1].min()) for c in columns]
    xmaxs = [float(c.hex_xy[:, 0].max()) for c in columns]
    ymaxs = [float(c.hex_xy[:, 1].max()) for c in columns]
    return min(xmins), min(ymins), max(xmaxs), max(ymaxs)


def _bar_rect(
    spec: MovingBarSpec,
    trail_pos: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> Tuple[float, float, float, float]:
    w = float(spec.width_deg)
    d = spec.direction
    if d == "right":
        return trail_pos, y0, trail_pos + w, y1
    if d == "left":
        return trail_pos - w, y0, trail_pos, y1
    if d == "up":
        return x0, trail_pos, x1, trail_pos + w
    if d == "down":
        return x0, trail_pos - w, x1, trail_pos
    raise ValueError(f"unknown direction {d!r}")


def _trail_start(spec: MovingBarSpec, x0: float, y0: float, x1: float, y1: float) -> float:
    w = float(spec.width_deg)
    if spec.direction == "right":
        return x0 - w
    if spec.direction == "left":
        return x1 + w
    if spec.direction == "up":
        return y0 - w
    if spec.direction == "down":
        return y1 + w
    raise ValueError(f"unknown direction {spec.direction!r}")


def _trail_exit(spec: MovingBarSpec, x0: float, y0: float, x1: float, y1: float) -> float:
    w = float(spec.width_deg)
    if spec.direction == "right":
        return x1 - w
    if spec.direction == "left":
        return x0 + w
    if spec.direction == "up":
        return y1 - w
    if spec.direction == "down":
        return y0 + w
    raise ValueError(f"unknown direction {spec.direction!r}")


def _trail_center_target(spec: MovingBarSpec, x0: float, y0: float, x1: float, y1: float) -> float:
    w = float(spec.width_deg)
    if spec.direction == "right":
        return 0.5 * (x0 + x1) - 0.5 * w
    if spec.direction == "left":
        return 0.5 * (x0 + x1) + 0.5 * w
    if spec.direction == "up":
        return 0.5 * (y0 + y1) - 0.5 * w
    if spec.direction == "down":
        return 0.5 * (y0 + y1) + 0.5 * w
    raise ValueError(f"unknown direction {spec.direction!r}")


def _trail_shift_deg(spec: MovingBarSpec, dt_s: float) -> float:
    """Signed trail advance (deg) in one sample: ``±speed_deg_s * dt_s``."""
    s = float(spec.speed_deg_s) * dt_s
    if spec.direction == "right":
        return s
    if spec.direction == "left":
        return -s
    if spec.direction == "up":
        return s
    if spec.direction == "down":
        return -s
    raise ValueError(f"unknown direction {spec.direction!r}")


def _trail_to_t(
    spec: MovingBarSpec,
    trail_start: float,
    trail_target: float,
    t_on: int,
    delta_ms: float,
    n_t: Optional[int] = None,
) -> int:
    trail_shift_deg = _trail_shift_deg(spec, delta_ms / 1000.0)
    if abs(trail_shift_deg) < 1e-15:
        return t_on
    k = int(round((trail_target - trail_start) / trail_shift_deg))
    t = t_on + max(0, k)
    if n_t is not None:
        t = min(t, n_t - 1)
    return t


def moving_bar_sweep_end_t(
    specs: Sequence[MovingBarSpec],
    field_deg: Tuple[float, float, float, float],
    bar_extent: int,
    *,
    multi_bar: bool = True,
    t_on: int = None,
    delta_ms: float,
) -> int:
    """Exclusive t index where all lanes finish their local sweep (no tail)."""
    if not specs:
        return t_on + 1
    t_exit = t_on
    for spec in specs:
        for lane_origin, lane_pitch in _motion_lanes(spec, field_deg, bar_extent, multi_bar=multi_bar):
            trail_start, trail_exit = _lane_sweep_trail_range(spec, lane_origin, lane_pitch)
            t_exit = max(
                t_exit,
                _trail_to_t(spec, trail_start, trail_exit, t_on, delta_ms),
            )
    return t_exit + 1



def moving_bar_n_t(
    specs: Sequence[MovingBarSpec],
    field_deg: Tuple[float, float, float, float],
    bar_extent: int,
    *,
    multi_bar: bool = True,
    t_on: int = None,
    delta_ms: float,
    t_tail_ms: float = MOVING_BAR_TAIL_MS,
) -> int:
    """Simulation length: baseline + multi-bar sweep + post-sweep tail."""
    t_tail = ms_to_t(t_tail_ms, delta_ms=delta_ms)
    return moving_bar_sweep_end_t(
        specs, field_deg, bar_extent, multi_bar=multi_bar, t_on=t_on, delta_ms=delta_ms,
    ) + t_tail



def moving_bar_transit_times(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    bar_extent: int,
    *,
    multi_bar: bool = True,
    t_on: int = None,
    n_t: Optional[int] = None,
    delta_ms: float,
) -> Tuple[int, int, int]:
    """Return ``(entry, center, exit)`` t indices for the first multi-bar lane."""
    lane_origin, lane_pitch = _motion_lanes(spec, field_deg, bar_extent, multi_bar=multi_bar)[0]
    trail_start, trail_exit = _lane_sweep_trail_range(spec, lane_origin, lane_pitch)
    w = float(spec.width_deg)
    trail_shift_deg = _trail_shift_deg(spec, delta_ms / 1000.0)
    origin = float(lane_origin)
    # Signed ``trail_shift_deg`` encodes sweep direction; same trail targets for all directions.
    trail_entry = float(trail_start) + trail_shift_deg
    trail_center = origin + 0.5 * (
        float(lane_pitch) - math.copysign(1.0, trail_shift_deg) * w
    )
    trail_exit_vis = float(trail_exit) - trail_shift_deg
    return (
        _trail_to_t(spec, trail_start, trail_entry, t_on, delta_ms, n_t),
        _trail_to_t(spec, trail_start, trail_center, t_on, delta_ms, n_t),
        _trail_to_t(spec, trail_start, trail_exit_vis, t_on, delta_ms, n_t),
    )



def column_first_stim_t(
    column_current: np.ndarray,
    i_baseline: float,
    *,
    atol: float = 1e-12,
) -> int:
    """First t where a column current differs from baseline (``t_first_sti``)."""
    curr = np.asarray(column_current, dtype=np.float64).reshape(-1)
    active = ~np.isclose(curr, float(i_baseline), atol=atol, rtol=0.0)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        raise ValueError("column has no non-baseline stimulus sample")
    return int(idx[0])


def bar_lane_rects_at_t(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    bar_extent: int,
    t: int,
    *,
    multi_bar: bool = True,
    t_on: int = None,
    delta_ms: float,
) -> List[Tuple[float, float, float, float]]:
    """All lane bar rectangles at simulation time ``t`` (empty outside local sweep)."""
    trail_shift_deg = _trail_shift_deg(spec, delta_ms / 1000.0)
    rects: List[Tuple[float, float, float, float]] = []
    for lane_origin, lane_pitch in _motion_lanes(spec, field_deg, bar_extent, multi_bar=multi_bar):
        trail_start, _trail_exit = _lane_sweep_trail_range(spec, lane_origin, lane_pitch)
        trail = float(trail_start) + (t - t_on) * trail_shift_deg
        rect = _bar_rect_lane_clipped(
            spec, trail, lane_origin, lane_pitch, field_deg,
        )
        if rect is not None:
            rects.append(rect)
    return rects


def bar_trail_at_t(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    t: int,
    t_on: int = None,
    *,
    delta_ms: float) -> float:
    x0, y0, x1, y1 = field_deg
    trail = _trail_start(spec, x0, y0, x1, y1)
    trail_shift_deg = _trail_shift_deg(spec, delta_ms / 1000.0)
    return trail + (t - t_on) * trail_shift_deg


def bar_rect_at_t(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    t: int,
    t_on: int = None,
    *,
    delta_ms: float) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = field_deg
    trail = bar_trail_at_t(spec, field_deg, t, t_on=t_on, delta_ms=delta_ms)
    return _bar_rect(spec, trail, x0, y0, x1, y1)


def _current_from_coverage(
    coverage: np.ndarray,
    contrast: str,
    i_baseline: float,
    *,
    i_bright_bar: float,
    i_dark_bar: float,
) -> np.ndarray:
    if contrast == "bright":
        peak = float(i_bright_bar)
    elif contrast == "dark":
        peak = float(i_dark_bar)
    else:
        raise ValueError(f"unknown contrast {contrast!r}")
    return i_baseline + coverage * (peak - i_baseline)


def build_column_current(
    columns: Sequence[HexColumn],
    spec: MovingBarSpec,
    n_t: int,
    bar_extent: int,
    *,
    multi_bar: bool = True,
    t_on: int = None,
    delta_ms: float,
    i_baseline: float,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> np.ndarray:
    """Column-level current ``(T, n_cols)`` for one multi-bar condition."""
    n_cols = len(columns)
    out = np.full((n_t, n_cols), i_baseline, dtype=np.float64)
    if n_cols == 0:
        return out

    field_deg = field_bounds(columns)
    hex_stack = np.stack([c.hex_xy for c in columns], axis=0)
    cov_ts = _coverage_time_series(
        hex_stack, spec, field_deg, n_t=n_t, t_on=t_on, delta_ms=delta_ms,
        bar_extent=bar_extent,
        multi_bar=multi_bar,
    )
    out[t_on:] = _current_from_coverage(
        cov_ts, spec.contrast, i_baseline=i_baseline,
        i_bright_bar=i_bright_bar, i_dark_bar=i_dark_bar,
    )
    return out


def build_batched_column_current(
    columns: Sequence[HexColumn],
    specs: Sequence[MovingBarSpec],
    n_t: int,
    bar_extent: int,
    *,
    multi_bar: bool = True,
    t_on: int = None,
    delta_ms: float,
    i_baseline: float,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> np.ndarray:
    """Batched multi-bar column currents ``(B, T, n_cols)``.

    Each batch row superposes simultaneous lane bars for one ``MovingBarSpec``.
    Specs that share direction / width / speed reuse one coverage time series;
    only the bright/dark contrast scaling differs.
    """
    n_batch = len(specs)
    n_cols = len(columns)
    if n_cols == 0 or n_batch == 0:
        return np.zeros((n_batch, n_t, n_cols), dtype=np.float64)

    out = np.zeros((n_batch, n_t, n_cols), dtype=np.float64)
    for b in range(n_batch):
        out[b, :t_on] = i_baseline

    field_deg = field_bounds(columns)
    hex_stack = np.stack([c.hex_xy for c in columns], axis=0)

    groups: dict[Tuple[str, float, float], List[int]] = {}
    for b, spec in enumerate(specs):
        groups.setdefault(_geometry_key(spec), []).append(b)

    for batch_idxs in groups.values():
        cov_ts = _coverage_time_series(
            hex_stack, specs[batch_idxs[0]], field_deg,
            n_t=n_t, t_on=t_on, delta_ms=delta_ms,
            bar_extent=bar_extent,
            multi_bar=multi_bar,
        )
        for b in batch_idxs:
            out[b, t_on:] = _current_from_coverage(
                cov_ts, specs[b].contrast, i_baseline=i_baseline,
                i_bright_bar=i_bright_bar, i_dark_bar=i_dark_bar,
            )
    return out


# -- Connectome mapping: column currents -> unit ``signal`` (was moving_bar_target) --

PD_IDX, ND_IDX = 0, 1
_TRACE_CACHE: Dict[str, np.ndarray] = {}


@dataclass
class StiColumn(HexColumn):
    """One sti column on a connectome, with unit indices for scattering."""

    unit_idx: np.ndarray


def _sti_column_from_uv(u: int, v: int, unit_idx: np.ndarray) -> StiColumn:
    base = hex_column_from_uv(u, v)
    return StiColumn(
        u=base.u, v=base.v, x=base.x, y=base.y,
        x_deg=base.x_deg, y_deg=base.y_deg, hex_xy=base.hex_xy,
        unit_idx=unit_idx,
    )


@dataclass
class MovingBarStimulus:
    signal: torch.Tensor
    column_current: np.ndarray
    specs: List[MovingBarSpec]
    info: dict = field(default_factory=dict)


def sti_columns(C) -> List[StiColumn]:
    """Sti columns with photoreceptor units (one per axial ``(u, v)``)."""
    cols: Dict[Tuple[int, int], StiColumn] = {}
    u_in = C.u[C.is_input]
    v_in = C.v[C.is_input]
    for u, v in zip(u_in.tolist(), v_in.tolist()):
        key = (int(u), int(v))
        if key in cols:
            continue
        units = C.input_units_at(key[0], key[1])
        if len(units) == 0:
            continue
        cols[key] = _sti_column_from_uv(key[0], key[1], np.asarray(units, dtype=np.int64))
    return [cols[k] for k in sorted(cols)]


def moving_bar_cost_columns(C, cost_extent=None) -> List[StiColumn]:
    """Sti columns used for moving-bar cost (optional central hex disc)."""
    cols = sti_columns(C)
    if cost_extent is None:
        return cols
    return [c for c in cols if column_in_cost_extent(c.u, c.v, cost_extent)]


def _as_int64_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return np.asarray(x.detach().cpu().numpy(), dtype=np.int64)
    return np.asarray(x, dtype=np.int64)


def network_uv_np(C) -> Tuple[np.ndarray, np.ndarray]:
    """connectome axial ``(u, v)`` per unit as int64 numpy arrays."""
    return _as_int64_np(C.u), _as_int64_np(C.v)


def _coord_matches(val, axis_filter, tol=1e-6) -> bool:
    if axis_filter is None:
        return True
    if isinstance(axis_filter, (list, tuple)):
        return any(np.isclose(val, float(v), atol=tol) for v in axis_filter)
    return np.isclose(val, float(axis_filter), atol=tol)


def filter_sti_columns(cols, *, at_x=None, at_y=None, tol=1e-6):
    """Keep network sti columns whose hex-step ``(x, y)`` matches ``at_x`` / ``at_y``."""
    if at_x is None and at_y is None:
        return list(cols)
    out = []
    for col in cols:
        if not _coord_matches(col.x, at_x, tol=tol):
            continue
        if not _coord_matches(col.y, at_y, tol=tol):
            continue
        out.append(col)
    return out


def resolve_i_baseline(value: float) -> float:
    """Cast photoreceptor baseline current (pA)."""
    return float(value)


def moving_bar_i_baseline_from_opts(train_opts) -> float:
    """``i_baseline`` from moving-bar stimulus opts on a train session."""
    opts = train_opts or {}
    for key in ("moving_bar_bright_stimulus_opts", "moving_bar_dark_stimulus_opts"):
        sub = opts.get(key) or {}
        if "i_baseline" in sub:
            return resolve_i_baseline(float(sub["i_baseline"]))
    raise ValueError(
        "moving-bar stimulus opts require i_baseline "
        "(inject via training.defaults.I_BASELINE / CLI)"
    )


def moving_bar_window_t_rel(t0, t_on: int, win: int):
    """Window index into post-``t_on`` trace columns (numpy)."""
    t0 = np.asarray(t0, dtype=np.int64)
    win_ix = np.arange(int(win), dtype=np.int64)
    t_rel = t0[..., None] - int(t_on) + win_ix
    return t_rel, t_rel < 0


def moving_bar_window_t_rel_torch(t0, t_on: int, win: int, *, device=None):
    """Torch counterpart of :func:`moving_bar_window_t_rel`."""
    if not torch.is_tensor(t0):
        t0 = torch.as_tensor(t0, dtype=torch.long, device=device)
    else:
        t0 = t0.to(dtype=torch.long)
    if device is None:
        device = t0.device
    win_ix = torch.arange(int(win), dtype=torch.long, device=device)
    t_rel = t0[..., None] - int(t_on) + win_ix
    return t_rel, t_rel < 0


@dataclass
class MovingBarT0Grids:
    t0_bn: np.ndarray
    before_t: Dict[str, int]
    after_t: Dict[str, int]


def moving_bar_spec_horizon(t_first_stis: Sequence[int], n_t: int) -> Tuple[int, int, int]:
    """Return ``(fb, before_t, after_t)`` for one spec over all columns."""
    fb = int(min(t_first_stis))
    before = fb
    after = int(n_t) - int(max(t_first_stis))
    return fb, before, after


def moving_bar_network_t0_bn(C, filt_cols: Sequence[StiColumn], n_batch: int, t0_map: dict) -> np.ndarray:
    """Expand per-column ``t0`` values to a full unit grid ``(B, N_units)``."""
    u_np = np.asarray(C.u, dtype=np.int64)
    v_np = np.asarray(C.v, dtype=np.int64)
    t0_bn = np.full((n_batch, C.n_units), -1, dtype=np.int64)
    for bi in range(n_batch):
        for c in filt_cols:
            t0 = t0_map.get((bi, int(c.u), int(c.v)))
            if t0 is None:
                continue
            on_col = (u_np == int(c.u)) & (v_np == int(c.v))
            t0_bn[bi, on_col] = t0
    return t0_bn


def build_moving_bar_t0_grids(
    column_current: np.ndarray,
    specs: Sequence[MovingBarSpec],
    n_t: int,
    i_baseline: float,
    *,
    all_col_idxs: Sequence[int],
    filt_col_idxs: Sequence[int],
    network_C,
    filt_network_cols: Sequence[StiColumn],
) -> MovingBarT0Grids:
    """Plot/training-aligned ``t0`` grid and per-spec full horizons."""
    before_t: Dict[str, int] = {}
    after_t: Dict[str, int] = {}
    n_batch = len(specs)
    i_baseline = resolve_i_baseline(i_baseline)

    t0_map: dict = {}
    for bi, spec in enumerate(specs):
        t_first_all = [
            column_first_stim_t(column_current[bi, :, col_idx], i_baseline=i_baseline)
            for col_idx in all_col_idxs
        ]
        fb, before, after = moving_bar_spec_horizon(t_first_all, n_t)
        before_t[spec.name] = before
        after_t[spec.name] = after
        t_first_filt = [
            column_first_stim_t(column_current[bi, :, col_idx], i_baseline=i_baseline)
            for col_idx in filt_col_idxs
        ]
        for c, tc in zip(filt_network_cols, t_first_filt):
            t0_map[(bi, int(c.u), int(c.v))] = tc - fb
    t0_bn = moving_bar_network_t0_bn(network_C, filt_network_cols, n_batch, t0_map)
    return MovingBarT0Grids(t0_bn=t0_bn, before_t=before_t, after_t=after_t)


def _column_unit_map(columns: Sequence[StiColumn]) -> Tuple[np.ndarray, np.ndarray]:
    col_idx: List[int] = []
    unit_idx: List[int] = []
    for j, col in enumerate(columns):
        for u in np.asarray(col.unit_idx).ravel():
            col_idx.append(j)
            unit_idx.append(int(u))
    return (
        np.asarray(col_idx, dtype=np.int64),
        np.asarray(unit_idx, dtype=np.int64),
    )


def scatter_column_current(column_current, columns, n_units):
    """Broadcast column current ``(T, n_cols)`` to unit current ``(T, n_units)``."""
    n_t = column_current.shape[0]
    out = np.zeros((n_t, n_units), dtype=np.float64)
    col_idx, unit_idx = _column_unit_map(columns)
    if len(col_idx):
        out[:, unit_idx] = column_current[:, col_idx]
    return out


def scatter_column_current_batched(column_current, columns, n_units):
    """Broadcast ``(B, T, n_cols)`` column current to ``(B, T, n_units)``."""
    n_batch, n_t, _ = column_current.shape
    out = np.zeros((n_batch, n_t, n_units), dtype=np.float64)
    col_idx, unit_idx = _column_unit_map(columns)
    if len(col_idx):
        out[:, :, unit_idx] = column_current[:, :, col_idx]
    return out


def _column_uv(columns: Sequence[StiColumn]) -> List[Tuple[int, int]]:
    return [(c.u, c.v) for c in columns]


def _spec_contrast_set(specs: Sequence[MovingBarSpec]) -> frozenset:
    return frozenset(s.contrast for s in specs)


def _moving_bar_cache_key(
    network_json: Path,
    specs: Sequence[MovingBarSpec],
    column_uv: Sequence[Tuple[int, int]],
    n_t: int,
    t_on: int,
    delta_ms: float,
    i_baseline: float,
    bar_extent: int,
    multi_bar: bool = True,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> str:
    stat = network_json.stat()
    payload = {
        "network": str(network_json.resolve()),
        "network_mtime_ns": stat.st_mtime_ns,
        "network_size": stat.st_size,
        "column_uv": list(column_uv),
        "bar_extent": int(bar_extent),
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
        "t_on": t_on,
        "delta_ms": delta_ms,
        "i_baseline": i_baseline,
    }
    if i_bright_bar is not None:
        payload["i_bright_bar"] = i_bright_bar
    if i_dark_bar is not None:
        payload["i_dark_bar"] = i_dark_bar
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def _moving_bar_cache_path(
    network_json: Path,
    specs: Sequence[MovingBarSpec],
    column_uv: Sequence[Tuple[int, int]],
    n_t: int,
    t_on: int,
    delta_ms: float,
    i_baseline: float,
    bar_extent: int,
    multi_bar: bool = True,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> Path:
    key = _moving_bar_cache_key(
        network_json, specs, column_uv, n_t, t_on, delta_ms,
        i_baseline, bar_extent, multi_bar, i_bright_bar, i_dark_bar,
    )
    return moving_bar_cache_dir(network_json) / f"{key}.npz"


def _load_moving_bar_column_cache(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data["column_current"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Ignoring corrupt moving-bar cache %s: %s", path, exc)
        return None


def _save_moving_bar_column_cache(path: Path, column_current: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, column_current=column_current)
    logger.info("Cached moving-bar column current to %s", path)


def build_moving_bar_signals(
    C,
    specs: Optional[Sequence[MovingBarSpec]] = None,
    n_t: Optional[int] = None,
    t_on: int = None,
    *,
    delta_ms: float,
    bar_extent: int = DEFAULT_BAR_EXTENT,
    multi_bar: bool = True,
    i_baseline: float,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    network_json: Optional[Path] = None,
    sim_dtype: torch.dtype,
) -> MovingBarStimulus:
    """Build batched photoreceptor current for moving-bar stimuli (network connectome).

    Returns ``signal`` with shape ``(B, T, N_units)`` where ``B = len(specs)``.
    """
    device = device or C.device
    bar_extent = int(bar_extent)
    multi_bar = bool(multi_bar)
    specs = list(specs if specs is not None else gruntman_moving_bar_specs())
    contrasts = _spec_contrast_set(specs)
    i_bright = None
    i_dark = None
    if "bright" in contrasts:
        if i_bright_bar is None:
            raise ValueError("i_bright_bar required for bright contrast")
        i_bright = float(i_bright_bar)
    if "dark" in contrasts:
        if i_dark_bar is None:
            raise ValueError("i_dark_bar required for dark contrast")
        i_dark = float(i_dark_bar)
    sti_cols = sti_columns(C)
    field_deg = field_bounds(sti_cols)
    if n_t is None:
        n_t = moving_bar_n_t(
            specs, field_deg, bar_extent, multi_bar=multi_bar, t_on=t_on, delta_ms=delta_ms,
        )
    n_batch = len(specs)
    n_units = C.n_units
    sweep_end = moving_bar_sweep_end_t(
        specs, field_deg, bar_extent, multi_bar=multi_bar, t_on=t_on, delta_ms=delta_ms,
    )
    sweep_t = sweep_end - t_on
    tail_t = n_t - sweep_end

    cache_path: Optional[Path] = None
    source_json = Path(network_json) if network_json is not None else getattr(C, "source_json", None)
    column_uv = _column_uv(sti_cols)
    if source_json is not None:
        cache_path = _moving_bar_cache_path(
            source_json, specs, column_uv, n_t, t_on, delta_ms,
            i_baseline, bar_extent, multi_bar, i_bright, i_dark,
        )

    col_curr: Optional[np.ndarray] = None
    if cache_path is not None and use_cache and not refresh_cache:
        col_curr = _load_moving_bar_column_cache(cache_path)
        if col_curr is not None:
            logger.info("Loaded moving-bar column current from cache %s", cache_path)

    if col_curr is None:
        col_curr = build_batched_column_current(
            sti_cols, specs, n_t=n_t, bar_extent=bar_extent, multi_bar=multi_bar,
            t_on=t_on, delta_ms=delta_ms,
            i_baseline=i_baseline, i_bright_bar=i_bright, i_dark_bar=i_dark,
        )
        if cache_path is not None and use_cache:
            _save_moving_bar_column_cache(cache_path, col_curr)

    signal_np = scatter_column_current_batched(col_curr, sti_cols, n_units)

    info = {
        "n_batch": n_batch,
        "n_sti_columns": len(sti_cols),
        "bar_extent": bar_extent,
        "multi_bar": multi_bar,
        "field_deg": field_deg,
        "n_t": n_t,
        "t_on": t_on,
        "sweep_end": sweep_end,
        "sweep_t": sweep_t,
        "sweep_time_s": sweep_t * delta_ms / 1000.0,
        "tail_t": tail_t,
        "tail_time_s": tail_t * delta_ms / 1000.0,
        "i_baseline": i_baseline,
        "speed_deg_s": specs[0].speed_deg_s if specs else GRUNTMAN_SPEED_DEG_S,
        "spec_names": [s.name for s in specs],
    }
    if i_bright is not None:
        info["i_bright_bar"] = i_bright
    if i_dark is not None:
        info["i_dark_bar"] = i_dark
    return MovingBarStimulus(
        signal=torch.as_tensor(signal_np, dtype=sim_dtype, device=device),
        column_current=col_curr,
        specs=specs,
        info=info,
    )

