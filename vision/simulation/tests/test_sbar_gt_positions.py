from __future__ import annotations

import os
import sys


SIMULATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMULATION_ROOT not in sys.path:
    sys.path.insert(0, SIMULATION_ROOT)

import import_bootstrap  # noqa: E402,F401
from task.sbar.gt import load_gt_stats, position_label  # noqa: E402


def test_t4_t5_raw_position_indices_are_divided_by_two_without_interpolation():
    gts, gt_stds = load_gt_stats(
        t_onset=0,
        ms_response=160,
        ms_sti=160,
        delta_ms=1.0,
    )
    expected_mids = [position / 2 for position in range(-4, 5)]

    for cell in ("T4", "T5"):
        for pathway in ("PC", "NC"):
            expected_keys = {
                f"{cell}_{pathway}_pos{position_label(mid)}_w1"
                for mid in expected_mids
            }
            prefix = f"{cell}_{pathway}_"
            assert {key for key in gts if key.startswith(prefix)} == expected_keys
            assert {
                key for key in gt_stds if key.startswith(prefix)
            } == expected_keys
