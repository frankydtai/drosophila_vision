# -*- coding: utf-8 -*-
"""Gruntman-style moving-bar stimulus: geometry, timing, and column currents.

Pure visual-field math on hex columns (degrees, coverage, pA). No connectome
or unit indexing — :mod:`network.moving_bar_target` maps these currents onto a network.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

import network_bootstrap  # noqa: F401

from column_mapper import DEG, HEX_PATCH_RADIUS, hex_vertices, uv_to_xy, uv_to_xy_deg
from Medulla_Library import I_BASELINE, I_BRIGHT, I_DARK
from training_config import DELTAT_MS, MOVING_BAR_TAIL_MS, T_ON, ms_to_steps

# Gruntman Fig. 1 Ci fast condition: 40 ms / 2.25 deg per LED step.
GRUNTMAN_SPEED_DEG_S = 56.0
GRUNTMAN_WIDTHS_DEG = (2.25, 9.0)
GRUNTMAN_DIRECTIONS = ("right", "left", "up", "down")
GRUNTMAN_CONTRASTS = ("bright", "dark")

# Per-lane spacing width in hex-column units (``spacing_deg = bar_extent * DEG``).
DEFAULT_BAR_EXTENT = 2

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


def _coverage_time_series_whole_field(
    hex_stack: np.ndarray,
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    maxtime: int,
    t_on: int,
    deltat_ms: float,
) -> np.ndarray:
    """Coverage ``(maxtime - t_on, n_cols)`` for one whole-field bar (Borst only)."""
    x0, y0, x1, y1 = field_deg
    dt_s = deltat_ms / 1000.0
    step = _trail_step(spec, dt_s)
    trail = _trail_start(spec, x0, y0, x1, y1)
    n_cols = hex_stack.shape[0]
    n_steps = maxtime - t_on
    out = np.empty((n_steps, n_cols), dtype=np.float64)
    for i in range(n_steps):
        bx0, by0, bx1, by1 = _bar_rect(spec, trail, x0, y0, x1, y1)
        out[i] = _coverage_batch(hex_stack, bx0, by0, bx1, by1)
        trail += step
    return out


def _coverage_time_series(
    hex_stack: np.ndarray,
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    maxtime: int,
    t_on: int,
    deltat_ms: float,
    bar_extent: int,
    *,
    multi_bar: bool = True,
) -> np.ndarray:
    """Superposed coverage from simultaneous per-lane bars (network connectome field)."""
    dt_s = deltat_ms / 1000.0
    step = _trail_step(spec, dt_s)
    lane_origins = _motion_lanes(spec, field_deg, bar_extent, multi_bar=multi_bar)
    n_cols = hex_stack.shape[0]
    n_steps = maxtime - t_on
    out = np.zeros((n_steps, n_cols), dtype=np.float64)
    for lane_origin, lane_pitch in lane_origins:
        trail_start, trail_exit = _lane_sweep_trail_range(spec, lane_origin, lane_pitch)
        trail = float(trail_start)
        for i in range(n_steps):
            rect = _bar_rect_lane_clipped(
                spec, trail, lane_origin, lane_pitch, field_deg,
            )
            if rect is not None:
                bx0, by0, bx1, by1 = rect
                out[i] += _coverage_batch(hex_stack, bx0, by0, bx1, by1)
            trail += step
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


def _trail_step(spec: MovingBarSpec, dt_s: float) -> float:
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


def _trail_to_step(
    spec: MovingBarSpec,
    trail_start: float,
    trail_target: float,
    t_on: int,
    deltat_ms: float,
    maxtime: Optional[int] = None,
) -> int:
    step = _trail_step(spec, deltat_ms / 1000.0)
    if abs(step) < 1e-15:
        return t_on
    k = int(round((trail_target - trail_start) / step))
    t = t_on + max(0, k)
    if maxtime is not None:
        t = min(t, maxtime - 1)
    return t


def moving_bar_sweep_end_step(
    specs: Sequence[MovingBarSpec],
    field_deg: Tuple[float, float, float, float],
    bar_extent: int,
    *,
    multi_bar: bool = True,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
) -> int:
    """Exclusive step index where all lanes finish their local sweep (no tail)."""
    if not specs:
        return t_on + 1
    t_exit = t_on
    for spec in specs:
        for lane_origin, lane_pitch in _motion_lanes(spec, field_deg, bar_extent, multi_bar=multi_bar):
            trail_start, trail_exit = _lane_sweep_trail_range(spec, lane_origin, lane_pitch)
            t_exit = max(
                t_exit,
                _trail_to_step(spec, trail_start, trail_exit, t_on, deltat_ms),
            )
    return t_exit + 1


def whole_field_moving_bar_sweep_end_step(
    specs: Sequence[MovingBarSpec],
    field_deg: Tuple[float, float, float, float],
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
) -> int:
    """Exclusive sweep end for one whole-field bar (Borst horizontal demo only)."""
    x0, y0, x1, y1 = field_deg
    if not specs:
        return t_on + 1
    t_exit = t_on
    for spec in specs:
        trail_start = _trail_start(spec, x0, y0, x1, y1)
        trail_exit = _trail_exit(spec, x0, y0, x1, y1)
        t_exit = max(
            t_exit,
            _trail_to_step(spec, trail_start, trail_exit, t_on, deltat_ms),
        )
    return t_exit + 1


def moving_bar_maxtime(
    specs: Sequence[MovingBarSpec],
    field_deg: Tuple[float, float, float, float],
    bar_extent: int,
    *,
    multi_bar: bool = True,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
    t_tail_ms: float = MOVING_BAR_TAIL_MS,
) -> int:
    """Simulation length: ``T_ON_MS`` baseline + multi-bar sweep + post-sweep tail."""
    t_tail = ms_to_steps(t_tail_ms, deltat_ms=deltat_ms)
    return moving_bar_sweep_end_step(
        specs, field_deg, bar_extent, multi_bar=multi_bar, t_on=t_on, deltat_ms=deltat_ms,
    ) + t_tail


def whole_field_moving_bar_maxtime(
    specs: Sequence[MovingBarSpec],
    field_deg: Tuple[float, float, float, float],
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
    t_tail_ms: float = MOVING_BAR_TAIL_MS,
) -> int:
    """Simulation length for Borst whole-field moving-bar (no multi-bar lanes)."""
    t_tail = ms_to_steps(t_tail_ms, deltat_ms=deltat_ms)
    return whole_field_moving_bar_sweep_end_step(
        specs, field_deg, t_on=t_on, deltat_ms=deltat_ms,
    ) + t_tail


def moving_bar_transit_times(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    bar_extent: int,
    *,
    multi_bar: bool = True,
    t_on: int = T_ON,
    maxtime: Optional[int] = None,
    deltat_ms: float = DELTAT_MS,
) -> Tuple[int, int, int]:
    """Return ``(entry, center, exit)`` step indices for the first multi-bar lane."""
    lane_origin, lane_pitch = _motion_lanes(spec, field_deg, bar_extent, multi_bar=multi_bar)[0]
    trail_start, trail_exit = _lane_sweep_trail_range(spec, lane_origin, lane_pitch)
    w = float(spec.width_deg)
    step = _trail_step(spec, deltat_ms / 1000.0)
    origin = float(lane_origin)
    # Signed ``step`` encodes sweep direction; same trail targets for all directions.
    trail_entry = float(trail_start) + step
    trail_center = origin + 0.5 * (float(lane_pitch) - math.copysign(1.0, step) * w)
    trail_exit_vis = float(trail_exit) - step
    return (
        _trail_to_step(spec, trail_start, trail_entry, t_on, deltat_ms, maxtime),
        _trail_to_step(spec, trail_start, trail_center, t_on, deltat_ms, maxtime),
        _trail_to_step(spec, trail_start, trail_exit_vis, t_on, deltat_ms, maxtime),
    )


def whole_field_moving_bar_transit_times(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    t_on: int = T_ON,
    maxtime: Optional[int] = None,
    deltat_ms: float = DELTAT_MS,
) -> Tuple[int, int, int]:
    """Return ``(entry, center, exit)`` step indices for one whole-field bar (Borst)."""
    x0, y0, x1, y1 = field_deg
    trail_start = _trail_start(spec, x0, y0, x1, y1)
    return (
        _trail_to_step(spec, trail_start, trail_start, t_on, deltat_ms, maxtime),
        _trail_to_step(spec, trail_start, _trail_center_target(spec, x0, y0, x1, y1), t_on, deltat_ms, maxtime),
        _trail_to_step(spec, trail_start, _trail_exit(spec, x0, y0, x1, y1), t_on, deltat_ms, maxtime),
    )


def column_first_stim_step(
    column_current: np.ndarray,
    i_baseline: float = I_BASELINE,
    *,
    atol: float = 1e-12,
) -> int:
    """First step where a column current differs from baseline (``t_first_sti``)."""
    curr = np.asarray(column_current, dtype=np.float64).reshape(-1)
    active = ~np.isclose(curr, float(i_baseline), atol=atol, rtol=0.0)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        raise ValueError("column has no non-baseline stimulus step")
    return int(idx[0])


def bar_lane_rects_at_step(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    bar_extent: int,
    t: int,
    *,
    multi_bar: bool = True,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
) -> List[Tuple[float, float, float, float]]:
    """All lane bar rectangles at simulation step ``t`` (empty outside local sweep)."""
    step = _trail_step(spec, deltat_ms / 1000.0)
    rects: List[Tuple[float, float, float, float]] = []
    for lane_origin, lane_pitch in _motion_lanes(spec, field_deg, bar_extent, multi_bar=multi_bar):
        trail_start, _trail_exit = _lane_sweep_trail_range(spec, lane_origin, lane_pitch)
        trail = float(trail_start) + (t - t_on) * step
        rect = _bar_rect_lane_clipped(
            spec, trail, lane_origin, lane_pitch, field_deg,
        )
        if rect is not None:
            rects.append(rect)
    return rects


def bar_trail_at_step(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    t: int,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
) -> float:
    x0, y0, x1, y1 = field_deg
    trail = _trail_start(spec, x0, y0, x1, y1)
    step = _trail_step(spec, deltat_ms / 1000.0)
    return trail + (t - t_on) * step


def bar_rect_at_step(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    t: int,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = field_deg
    trail = bar_trail_at_step(spec, field_deg, t, t_on=t_on, deltat_ms=deltat_ms)
    return _bar_rect(spec, trail, x0, y0, x1, y1)


def _current_from_coverage(
    coverage: np.ndarray,
    contrast: str,
    i_baseline: float = I_BASELINE,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> np.ndarray:
    if contrast == "bright":
        peak = I_BRIGHT if i_bright_bar is None else i_bright_bar
    elif contrast == "dark":
        peak = I_DARK if i_dark_bar is None else i_dark_bar
    else:
        raise ValueError(f"unknown contrast {contrast!r}")
    return i_baseline + coverage * (peak - i_baseline)


def build_column_current(
    columns: Sequence[HexColumn],
    spec: MovingBarSpec,
    maxtime: int,
    bar_extent: int,
    *,
    multi_bar: bool = True,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
    i_baseline: float = I_BASELINE,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> np.ndarray:
    """Column-level current ``(T, n_cols)`` for one multi-bar condition."""
    n_cols = len(columns)
    out = np.full((maxtime, n_cols), i_baseline, dtype=np.float64)
    if n_cols == 0:
        return out

    field_deg = field_bounds(columns)
    hex_stack = np.stack([c.hex_xy for c in columns], axis=0)
    cov_ts = _coverage_time_series(
        hex_stack, spec, field_deg, maxtime=maxtime, t_on=t_on, deltat_ms=deltat_ms,
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
    maxtime: int,
    bar_extent: int,
    *,
    multi_bar: bool = True,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
    i_baseline: float = I_BASELINE,
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
        return np.zeros((n_batch, maxtime, n_cols), dtype=np.float64)

    out = np.zeros((n_batch, maxtime, n_cols), dtype=np.float64)
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
            maxtime=maxtime, t_on=t_on, deltat_ms=deltat_ms,
            bar_extent=bar_extent,
            multi_bar=multi_bar,
        )
        for b in batch_idxs:
            out[b, t_on:] = _current_from_coverage(
                cov_ts, specs[b].contrast, i_baseline=i_baseline,
                i_bright_bar=i_bright_bar, i_dark_bar=i_dark_bar,
            )
    return out


def borst_whole_field_batched_column_current(
    columns: Sequence[HexColumn],
    specs: Sequence[MovingBarSpec],
    maxtime: int,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
    i_baseline: float = I_BASELINE,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> np.ndarray:
    """Borst-only whole-field batched column currents ``(B, T, n_cols)``."""
    n_batch = len(specs)
    n_cols = len(columns)
    if n_cols == 0 or n_batch == 0:
        return np.zeros((n_batch, maxtime, n_cols), dtype=np.float64)

    out = np.zeros((n_batch, maxtime, n_cols), dtype=np.float64)
    for b in range(n_batch):
        out[b, :t_on] = i_baseline

    field_deg = field_bounds(columns)
    hex_stack = np.stack([c.hex_xy for c in columns], axis=0)

    groups: dict[Tuple[str, float, float], List[int]] = {}
    for b, spec in enumerate(specs):
        groups.setdefault(_geometry_key(spec), []).append(b)

    for batch_idxs in groups.values():
        cov_ts = _coverage_time_series_whole_field(
            hex_stack, specs[batch_idxs[0]], field_deg,
            maxtime=maxtime, t_on=t_on, deltat_ms=deltat_ms,
        )
        for b in batch_idxs:
            out[b, t_on:] = _current_from_coverage(
                cov_ts, specs[b].contrast, i_baseline=i_baseline,
                i_bright_bar=i_bright_bar, i_dark_bar=i_dark_bar,
            )
    return out
