# -*- coding: utf-8 -*-
"""Gruntman-style moving-bar stimulus: geometry, timing, and column currents.

Pure visual-field math on hex columns (degrees, coverage, pA). No connectome
or unit indexing — :mod:`network.stimulus` maps these currents onto a network.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

import network_bootstrap  # noqa: F401

from column_mapper import HEX_PATCH_RADIUS, hex_vertices
from Medulla_Library import I_BASELINE, I_BRIGHT, I_DARK, T_ON, T_TAIL

# Gruntman Fig. 1 Ci fast condition: 40 ms / 2.25 deg per LED step.
GRUNTMAN_SPEED_DEG_S = 56.0
GRUNTMAN_WIDTHS_DEG = (2.25, 9.0)
GRUNTMAN_DIRECTIONS = ("right", "left", "up", "down")
GRUNTMAN_CONTRASTS = ("bright", "dark")

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
    """One sti column in degree coordinates (no connectome unit indices)."""

    u: int
    v: int
    x: float
    y: float
    hex_xy: np.ndarray


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


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


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


def _coverage_time_series(
    hex_stack: np.ndarray,
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    maxtime: int,
    t_on: int,
    deltat_ms: float,
) -> np.ndarray:
    """Coverage ``(maxtime - t_on, n_cols)`` for one bar geometry."""
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


def _segment_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r = p2 - p1
    s = q2 - q1
    denom = _cross(r, s)
    if abs(denom) < 1e-12:
        return p2
    t = _cross(q1 - p1, s) / denom
    return p1 + t * r


def _clip_polygon_to_halfplane(
    poly: np.ndarray,
    edge_p1: np.ndarray,
    edge_p2: np.ndarray,
    inside_fn,
) -> np.ndarray:
    if len(poly) == 0:
        return poly
    out: List[np.ndarray] = []
    prev = poly[-1]
    prev_in = inside_fn(prev)
    for cur in poly:
        cur_in = inside_fn(cur)
        if cur_in:
            if not prev_in:
                out.append(_segment_intersection(prev, cur, edge_p1, edge_p2))
            out.append(cur)
        elif prev_in:
            out.append(_segment_intersection(prev, cur, edge_p1, edge_p2))
        prev, prev_in = cur, cur_in
    return np.asarray(out, dtype=np.float64) if out else np.empty((0, 2), dtype=np.float64)


def _clip_polygon_to_rect(poly: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float) -> np.ndarray:
    big = 1e6
    clips = (
        (np.array([xmin, -big]), np.array([xmin, big]), lambda p: p[0] >= xmin),
        (np.array([xmax, -big]), np.array([xmax, big]), lambda p: p[0] <= xmax),
        (np.array([-big, ymin]), np.array([big, ymin]), lambda p: p[1] >= ymin),
        (np.array([-big, ymax]), np.array([big, ymax]), lambda p: p[1] <= ymax),
    )
    for edge_p1, edge_p2, inside in clips:
        poly = _clip_polygon_to_halfplane(poly, edge_p1, edge_p2, inside)
        if len(poly) == 0:
            return poly
    return poly


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


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
    t_on: int = T_ON,
    deltat_ms: float = 10.0,
) -> int:
    """Exclusive step index where the bar finishes sweeping the field (no tail)."""
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
    t_on: int = T_ON,
    deltat_ms: float = 10.0,
    t_tail: int = T_TAIL,
) -> int:
    """Simulation length: ``t_on`` (0.5 s) + sweep + ``t_tail`` (0.5 s post-stimulus).

    Returns the exclusive upper time index (``range(maxtime)``). Stimulus current
    is baseline before ``t_on`` and after the sweep; the tail holds baseline while
    the network settles for per-column ``t_center ± 0.45 s`` training windows.
    """
    return moving_bar_sweep_end_step(specs, field_deg, t_on=t_on, deltat_ms=deltat_ms) + t_tail


def moving_bar_transit_times(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    t_on: int = T_ON,
    maxtime: Optional[int] = None,
    deltat_ms: float = 10.0,
) -> Tuple[int, int, int]:
    """Return ``(entry, center, exit)`` simulation step indices for one bar."""
    x0, y0, x1, y1 = field_deg
    trail_start = _trail_start(spec, x0, y0, x1, y1)
    return (
        _trail_to_step(spec, trail_start, trail_start, t_on, deltat_ms, maxtime),
        _trail_to_step(spec, trail_start, _trail_center_target(spec, x0, y0, x1, y1), t_on, deltat_ms, maxtime),
        _trail_to_step(spec, trail_start, _trail_exit(spec, x0, y0, x1, y1), t_on, deltat_ms, maxtime),
    )


def _trail_target_for_column_center(
    col_x: float,
    col_y: float,
    spec: MovingBarSpec,
) -> float:
    """Trail position when the bar centre sits at ``col_x`` / ``col_y`` on the motion axis."""
    w = float(spec.width_deg)
    d = spec.direction
    if d == "right":
        return float(col_x) - 0.5 * w
    if d == "left":
        return float(col_x) + 0.5 * w
    if d == "up":
        return float(col_y) - 0.5 * w
    if d == "down":
        return float(col_y) + 0.5 * w
    raise ValueError(f"unknown direction {d!r}")


def column_bar_center_step(
    col_x: float,
    col_y: float,
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    t_on: int = T_ON,
    deltat_ms: float = 10.0,
    maxtime: Optional[int] = None,
) -> int:
    """Simulation step when the bar centre crosses a column on the motion axis.

    Horizontal motion (``right`` / ``left``) uses ``col_x`` only; vertical motion
    (``up`` / ``down``) uses ``col_y`` only. The bar spans the full field extent
    perpendicular to motion (see ``_bar_rect``), so all columns sharing the same
    motion-axis coordinate share the same ``t_center``.
    """
    x0, y0, x1, y1 = field_deg
    trail_start = _trail_start(spec, x0, y0, x1, y1)
    trail_target = _trail_target_for_column_center(col_x, col_y, spec)
    return _trail_to_step(spec, trail_start, trail_target, t_on, deltat_ms, maxtime)


def bar_trail_at_step(
    spec: MovingBarSpec,
    field_deg: Tuple[float, float, float, float],
    t: int,
    t_on: int = T_ON,
    deltat_ms: float = 10.0,
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
    deltat_ms: float = 10.0,
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
    t_on: int = T_ON,
    deltat_ms: float = 10.0,
    i_baseline: float = I_BASELINE,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> np.ndarray:
    """Column-level current ``(T, n_cols)`` for one moving-bar condition."""
    n_cols = len(columns)
    out = np.full((maxtime, n_cols), i_baseline, dtype=np.float64)
    if n_cols == 0:
        return out

    field_deg = field_bounds(columns)
    hex_stack = np.stack([c.hex_xy for c in columns], axis=0)
    cov_ts = _coverage_time_series(
        hex_stack, spec, field_deg, maxtime=maxtime, t_on=t_on, deltat_ms=deltat_ms,
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
    t_on: int = T_ON,
    deltat_ms: float = 10.0,
    i_baseline: float = I_BASELINE,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> np.ndarray:
    """Batched column currents ``(B, T, n_cols)``.

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
        )
        for b in batch_idxs:
            out[b, t_on:] = _current_from_coverage(
                cov_ts, specs[b].contrast, i_baseline=i_baseline,
                i_bright_bar=i_bright_bar, i_dark_bar=i_dark_bar,
            )
    return out
