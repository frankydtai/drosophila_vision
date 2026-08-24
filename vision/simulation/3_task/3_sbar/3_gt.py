# -*- coding: utf-8 -*-
"""Static-bar GT numbers: Gruntman Fig.2 digitized width-1 traces.

Data source: ``figure_digitization/gruntman21/2ax2bc_digitized.csv``
(Gruntman 2021 Fig.2a/2b/2c digitized flash-response traces).

GT layout: each entry is one T4/T5 cell's response to a width-1 static bar
placed at one spatial position on the motion axis.

CRITICAL NUMBERS — do not guess, do not round, do not say "approximately":
- The CSV ``position`` field has EXACTLY 9 distinct values: -2, -1.5, -1, -0.5, 0,
  0.5, 1, 1.5, 2 (degrees, measured from the RF centre along the bar motion axis).
  These are NOT just -2..+2 integers — 0.5-step positions exist and are required.
- The CSV ``target_width_led`` == 1 filter selects ONLY width-1 traces (9 positions).
  The same CSV also has ``target_width_led`` == 2 (width-2 traces, 2.25° wide, pos -2..+2
  in 0.5° steps); those are for a different sbar variant and must be excluded here.
- The ``trace_id`` field encodes: {cell_prefix}_{PC|NC}_{posSIGN}_w1
  e.g. T4_PC_pos-2_w1, T5_NC_pos+0_w1, T4_PC_pos+1.5_w1.
  PC = ON-pathway (T4 bright / T5 dark); NC = OFF-pathway (T4 dark / T5 bright).
- The ``position`` column (column 6, 0-indexed) and the ``pos{N}_w1`` trace_id
  suffix MUST agree numerically — both carry the same float position value.
- Each position trace has ~90-100 time samples from -355ms to ~+400ms (pre- and
  post-stimulus window); ``t_axis`` is ``(arange(n_t) - t_onset) * delta_ms`` so
  CSV ``time_ms=0`` lands on sample ``t_onset`` (same onset paste as spread Arenz).
- There are 8 cells × 9 positions = 72 possible width-1 traces; the actual
  available traces depend on which PC/NC pathway each cell uses for each contrast.

COMMON MISTAKES this module has been fixed from:
- Claiming positions are integers only (wrong: 0.5-step exists).
- Claiming there are "10 positions" or "tens of positions" (wrong: exactly 9).
- Claiming each cell has a different set of positions (wrong: all cells share
  the same 9-position grid).
- Confusing the CSV ``position`` column (float degrees) with the hex-coordinate
  ``mid`` used in ``build_sbar_gt`` — the CSV position is the biological
  stimulus position; ``mid`` is the hex-axis coordinate of the cost node.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from neuron.borst import t_from_ms


GT_CELLS: Tuple[str, ...] = (
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
)

GT_CELL_ALIASES: dict = {
    "T4": tuple(cell for cell in GT_CELLS if cell.startswith("T4")),
    "T5": tuple(cell for cell in GT_CELLS if cell.startswith("T5")),
}


def expand_gt_cells(cells: Sequence[str]) -> Tuple[str, ...]:
    """Expand static-bar cell tokens via ``GT_CELL_ALIASES`` (e.g. T4, T5)."""
    if not cells:
        raise ValueError("gt_cells must not be empty")
    gt_cells: list = []
    seen: set = set()
    for token in cells:
        token = str(token).strip()
        if token in GT_CELL_ALIASES:
            pool = GT_CELL_ALIASES[token]
        elif token in GT_CELLS:
            pool = (token,)
        else:
            raise ValueError(
                f"unknown gt cell {token!r} (expected {', '.join((*GT_CELL_ALIASES, *GT_CELLS))})"
            )
        for cell in pool:
            if cell not in seen:
                seen.add(cell)
                gt_cells.append(cell)
    return tuple(gt_cells)


def load_gt(
    *,
    t_onset,
    ms_response,
    ms_sti,
    delta_ms: float,
    ms_post=0.0,
):
    """Load width-1 digitized traces keyed by CSV ``trace_id``.

    The ``target_width_led == 1`` filter (line 130) is MANDATORY.
    The same CSV also has ``target_width_led`` == 2 (width-2 traces, 2.25° wide,
    same 9 positions); those are for a different sbar variant and must be excluded here.

    The ``position`` field in the CSV has exactly 9 distinct values:
    -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2 degrees.
    The ``trace_id`` encodes this as ``pos{-2}_w1``, ``pos{-1.5}_w1``, …,
    ``pos{+2}_w1``. Both the column-6 float and the trace_id suffix agree.

    Return: dict keyed by trace_id string e.g. ``"T4_PC_pos-2_w1"``, value is
    a (n_t,) np.float64 trace aligned to the simulation time axis.
    """
    ms_response = float(ms_response)
    if ms_sti is not None:
        ms_response = max(ms_response, float(ms_sti))
    t_onset = int(t_onset)
    # Like spread load_ir(ca): CSV time_ms=0 at sample t_onset (not sim t=0).
    n_t = int(
        t_onset
        + t_from_ms(ms_response, delta_ms=float(delta_ms))
        + t_from_ms(float(ms_post), delta_ms=float(delta_ms))
        + 1
    )
    t_axis = (np.arange(n_t, dtype=np.float64) - t_onset) * float(delta_ms)
    gts = {}
    # CRITICAL: target_width_led == 1 selects only width-1 traces (9 positions, 2.25° wide).
    # The same CSV also has target_width_led == 2 (width-2 traces, 4.5° wide).
    # Do NOT remove this filter — loading width-2 traces will silently produce
    # wrong GT for all cost entries.
    for _, grouped in pd.read_csv(
        Path(__file__).resolve().parents[4]
        / "figure_digitization"
        / "gruntman21"
        / "2ax2bc_digitized.csv"
    ).query("target_width_led == 1").groupby("trace_id"):
        gt = np.interp(
            t_axis,
            grouped["time_ms"].to_numpy(dtype=np.float64),
            grouped["vm_mv"].to_numpy(dtype=np.float64),
            left=float(grouped["vm_mv"].iloc[0]),
            right=float(grouped["vm_mv"].iloc[-1]),
        )
        gts[str(grouped["trace_id"].iloc[0])] = gt
    return gts
