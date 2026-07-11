"""Tests for per-column moving-bar centre timing."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from training_config import COST_HALF_WINDOW_STEPS, COST_WINDOW_STEPS
from visual_stimulus.moving_bar_stimulus import (
    HexColumn,
    MovingBarSpec,
    column_bar_center_step,
    field_bounds,
    hex_vertices,
    moving_bar_maxtime,
    moving_bar_sweep_end_step,
    moving_bar_transit_times,
)


def _demo_columns() -> list[HexColumn]:
    cols: list[HexColumn] = []
    for u in range(-4, 5):
        for v in range(-4, 5):
            x = float(u) * 2.25
            y = float(v) * (3.0 ** 0.5) * 1.5
            cols.append(HexColumn(u=u, v=v, x=x, y=y, hex_xy=hex_vertices(x, y)))
    return cols


def test_field_center_matches_transit_times():
    cols = _demo_columns()
    field_deg = field_bounds(cols)
    x0, y0, x1, y1 = field_deg
    spec = MovingBarSpec("right", "bright", 2.25)
    maxtime = moving_bar_maxtime([spec], field_deg)
    _, t_center_field, _ = moving_bar_transit_times(spec, field_deg, maxtime=maxtime)
    col_x = 0.5 * (x0 + x1)
    col_y = 0.5 * (y0 + y1)
    t_col = column_bar_center_step(col_x, col_y, spec, field_deg, maxtime=maxtime)
    assert t_col == t_center_field


def test_monotonic_along_motion_axis():
    cols = _demo_columns()
    field_deg = field_bounds(cols)
    spec = MovingBarSpec("right", "bright", 2.25)
    maxtime = moving_bar_maxtime([spec], field_deg)
    ordered = sorted(cols, key=lambda c: c.x)
    times = [column_bar_center_step(c.x, c.y, spec, field_deg, maxtime=maxtime) for c in ordered]
    assert times == sorted(times)
    assert len(set(times)) > 1


def test_perpendicular_columns_share_t_center():
    cols = _demo_columns()
    field_deg = field_bounds(cols)
    spec = MovingBarSpec("right", "bright", 2.25)
    maxtime = moving_bar_maxtime([spec], field_deg)
    by_x: dict[float, list[int]] = {}
    for c in cols:
        t = column_bar_center_step(c.x, c.y, spec, field_deg, maxtime=maxtime)
        by_x.setdefault(round(c.x, 6), []).append(t)
    for ts in by_x.values():
        assert len(set(ts)) == 1


def test_vertical_motion_uses_col_y():
    cols = _demo_columns()
    field_deg = field_bounds(cols)
    spec = MovingBarSpec("up", "bright", 2.25)
    maxtime = moving_bar_maxtime([spec], field_deg)
    c0, c1 = sorted(cols, key=lambda c: c.y)[:2]
    t0 = column_bar_center_step(c0.x, c0.y, spec, field_deg, maxtime=maxtime)
    t1 = column_bar_center_step(c1.x, c1.y, spec, field_deg, maxtime=maxtime)
    assert t1 >= t0
    # Same y, different x -> same t_center for vertical bar.
    same_y = [c for c in cols if abs(c.y - c0.y) < 1e-9]
    ts = {column_bar_center_step(c.x, c.y, spec, field_deg, maxtime=maxtime) for c in same_y}
    assert len(ts) == 1


def test_cost_window_fits_in_maxtime():
    cols = _demo_columns()
    field_deg = field_bounds(cols)
    spec = MovingBarSpec("right", "bright", 2.25)
    maxtime = moving_bar_maxtime([spec], field_deg)
    assert maxtime == moving_bar_sweep_end_step([spec], field_deg) + 50
    for c in cols:
        t_center = column_bar_center_step(c.x, c.y, spec, field_deg, maxtime=maxtime)
        t0 = t_center - COST_HALF_WINDOW_STEPS
        t1 = t_center + COST_HALF_WINDOW_STEPS
        assert t0 >= 0, (c.u, c.v, t_center, t0)
        assert t1 < maxtime, (c.u, c.v, t_center, t1, maxtime)
        assert t1 - t0 == COST_WINDOW_STEPS


if __name__ == "__main__":
    test_field_center_matches_transit_times()
    test_monotonic_along_motion_axis()
    test_perpendicular_columns_share_t_center()
    test_vertical_motion_uses_col_y()
    test_cost_window_fits_in_maxtime()
    print("ok")
