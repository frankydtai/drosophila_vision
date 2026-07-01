#!/usr/bin/env python3
"""Regression: fast moving-bar coverage matches the legacy clipper."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import connectome_path  # noqa: F401
from visual_stimulus.moving_bar_stimulus import (
    HexColumn,
    MovingBarSpec,
    _coverage_hex_bar_legacy,
    build_batched_column_current,
    coverage_hex_bar,
    hex_vertices,
    gruntman_moving_bar_specs,
)
import numpy as np


def _demo_columns() -> list[HexColumn]:
    cols: list[HexColumn] = []
    for u in range(-4, 5):
        for v in range(-4, 5):
            x = float(u) * 2.25
            y = float(v) * (3.0 ** 0.5) * 1.5
            cols.append(HexColumn(u=u, v=v, x=x, y=y, hex_xy=hex_vertices(x, y)))
    return cols


def test_coverage_matches_legacy():
    cols = _demo_columns()
    hex_stack = np.stack([c.hex_xy for c in cols], axis=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        xmin, ymin = rng.uniform(-20, 0, size=2)
        xmax, ymax = xmin + rng.uniform(1, 15), ymin + rng.uniform(1, 15)
        j = int(rng.integers(0, len(cols)))
        fast = coverage_hex_bar(hex_stack[j], xmin, ymin, xmax, ymax)
        legacy = _coverage_hex_bar_legacy(hex_stack[j], xmin, ymin, xmax, ymax)
        assert abs(fast - legacy) < 1e-12, (fast, legacy, xmin, ymin, xmax, ymax)


def test_batched_shapes_and_baseline():
    cols = _demo_columns()
    specs = gruntman_moving_bar_specs()
    maxtime = 120
    out = build_batched_column_current(cols, specs, maxtime=maxtime, t_on=50)
    assert out.shape == (len(specs), maxtime, len(cols))
    assert np.all(out[:, :50, :] == 20.0)
    assert np.any(out[:, 50:, :] != 20.0)


def test_geometry_grouping_contrast_only():
    cols = _demo_columns()
    spec_b = MovingBarSpec("right", "bright", 2.25)
    spec_d = MovingBarSpec("right", "dark", 2.25)
    maxtime = 100
    one = build_batched_column_current(cols, [spec_b], maxtime=maxtime)
    two = build_batched_column_current(cols, [spec_b, spec_d], maxtime=maxtime)
    np.testing.assert_allclose(one[0], two[0])
    assert not np.allclose(two[0], two[1])


if __name__ == "__main__":
    test_coverage_matches_legacy()
    test_batched_shapes_and_baseline()
    test_geometry_grouping_contrast_only()
    print("ok")
