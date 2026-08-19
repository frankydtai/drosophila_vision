"""Moving-bar geometry: moving/multi-bar lane logic + shared sbar exports."""
from __future__ import annotations

import math
from typing import List, Optional, Protocol, Tuple

from build_hex import DEG
from task.sbar.sti_geo import (
    Hex,
    StiHex,
    bar_rect_lane_clipped,
    clip_rect_areas,
    filter_sti_hexes,
    hex_from_uv,
    i_sti_nodes_from_hexes,
    mbar_cost_hexes,
    node_us_vs,
    sti_hexes,
    view_bounds,
)

GRUNTMAN_WS_DEG = (2.25, 9.0)
GRUNTMAN_DIRECTIONS = ("right", "left", "up", "down")

# Per-lane spacing w in hex nodes (``spacing_deg = bar_radius * DEG``).
BAR_RADIUS = 2


class _BarGeo(Protocol):
    direction: str
    w_deg: float


def bar_lane_pitch_deg(
    bar: _BarGeo,
    bar_radius: int,
    *,
    view_deg: Optional[Tuple[float, float, float, float]] = None,
    multi_bar: bool = True,
) -> float:
    """Lane pitch in degrees.

    Default (``multi_bar=True``): integer w+spacing hex counts.
    ``multi_bar=False`` + ``view_deg``: one lane spans the full sti-view degree on the motion axis.
    """
    if int(bar_radius) < 0:
        raise ValueError(f"bar_radius must be >= 0, got {bar_radius!r}")
    if not bool(multi_bar):
        if view_deg is None:
            raise ValueError("view_deg required when multi_bar=False")
        x0, y0, x1, y1 = view_deg
        if bar.direction in ("right", "left"):
            return float(x1) - float(x0)
        if bar.direction in ("up", "down"):
            return float(y1) - float(y0)
        raise ValueError(f"unknown direction {bar.direction!r}")
    w_cols = int(math.ceil(float(bar.w_deg) / float(DEG)))
    return float(w_cols + int(bar_radius)) * float(DEG)


def _motion_view_lo_hi(
    bar: _BarGeo,
    view_deg: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Motion-axis view bounds ``(lo, hi)`` in degrees."""
    x0, y0, x1, y1 = view_deg
    if bar.direction in ("right", "left"):
        return float(x0), float(x1)
    if bar.direction in ("up", "down"):
        return float(y0), float(y1)
    raise ValueError(f"unknown direction {bar.direction!r}")


def _motion_entry_exit(
    bar: _BarGeo,
    view_lo: float,
    view_hi: float,
) -> Tuple[float, float]:
    """``(entry_deg, exit_deg)`` on the motion axis (bar enters at ``entry_deg``)."""
    if bar.direction in ("right", "up"):
        return view_lo, view_hi
    if bar.direction in ("left", "down"):
        return view_hi, view_lo
    raise ValueError(f"unknown direction {bar.direction!r}")


def motion_lanes(
    bar: _BarGeo,
    view_deg: Tuple[float, float, float, float],
    bar_radius: int,
    *,
    multi_bar: bool = True,
) -> List[Tuple[float, float]]:
    """``(lane_origin, lane_pitch)`` tiled from the motion entry edge through field exit."""
    view_lo, view_hi = _motion_view_lo_hi(bar, view_deg)
    if not bool(multi_bar):
        return [(view_lo, view_hi - view_lo)]
    pitch = bar_lane_pitch_deg(bar, bar_radius, view_deg=view_deg, multi_bar=True)
    entry_deg, exit_deg = _motion_entry_exit(bar, view_lo, view_hi)
    eps = 1e-9
    lanes: List[Tuple[float, float]] = []
    if float(entry_deg) < float(exit_deg):
        p = float(entry_deg)
        exit_v = float(exit_deg)
        while p < exit_v - eps:
            lane_pitch = min(pitch, exit_v - p)
            lanes.append((p, lane_pitch))
            p += pitch
    else:
        p = float(entry_deg) - pitch
        exit_v = float(exit_deg)
        while p + pitch > exit_v + eps:
            if p < exit_v - eps:
                lanes.append((exit_v, (p + pitch) - exit_v))
                break
            lanes.append((p, pitch))
            p -= pitch
    if not lanes:
        origin = min(float(entry_deg), float(exit_deg))
        lanes.append((origin, min(pitch, abs(float(exit_deg) - float(entry_deg)))))
    return lanes


def lane_sweep_trail_range(
    bar: _BarGeo,
    lane_origin: float,
    lane_pitch: float,
) -> Tuple[float, float]:
    """``(trail_start, trail_exit)`` for one lane; bar clips to ``[origin, origin+pitch]``."""
    w_deg = float(bar.w_deg)
    pitch = float(lane_pitch)
    origin = float(lane_origin)
    if bar.direction in ("right", "up"):
        return origin - w_deg, origin + pitch
    if bar.direction in ("left", "down"):
        return origin + pitch + w_deg, origin
    raise ValueError(f"unknown direction {bar.direction!r}")


