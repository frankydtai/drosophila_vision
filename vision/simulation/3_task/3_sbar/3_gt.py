# -*- coding: utf-8 -*-
"""Static-bar GT numbers from the original Gruntman recordings.

Data sources:
- T4/T5: ``t4_t5_w1_160ms_mean_std.csv`` generated directly from
  ``singleBarStT4.mat`` / ``singleBarStT5.mat``.  Each source trace is the
  repeat-average of one biological cell; mean and sample SD are then taken
  across biological cells.
- Mi1/Mi4: ``gruntman21_mi1_position_traces.csv`` /
  ``gruntman21_mi4_position_traces.csv`` (three-source spatial decomposition
  contributions at the same nine flash positions; PC/NC from the T4 pathway).

GT layout: each entry is one cell's response to a width-1 static bar
placed at one spatial position on the motion axis.

CRITICAL NUMBERS:
- The raw-data CSV has exactly nine measured aligned positions: integers -4..+4.
- It contains only width 1 and duration 160 ms.
- Half-step positions required by the vertical hex axis are spatially linearly
  interpolated between the two adjacent measured positions at load time.
- T4/T5 ``trace_id`` encodes: {cell_prefix}_{PC|NC}_{posSIGN}_w1
  e.g. T4_PC_pos-2_w1, T5_NC_pos+0_w1, T4_PC_pos+1.5_w1.
  PC = ON-pathway (T4 bright / T5 dark); NC = OFF-pathway (T4 dark / T5 bright).
- Mi1/Mi4 keys use the full cell name: Mi1_PC_pos+0_w1, Mi4_NC_pos-1.5_w1.
  Bright → PC, dark → NC (same as T4; traces come from the T4 PC/NC fit).
- ``vm_std_mv`` is sample SD across biological-cell repeat averages (ddof=1),
  never SD across pooled repeat recordings.

COMMON MISTAKES this module has been fixed from:
- Confusing the nine measured integer positions with the derived half-step
  positions used only for vertical-axis geometry.
- Claiming there are "10 positions" or "tens of positions" (wrong: exactly 9).
- Claiming each cell has a different set of positions (wrong: all cells share
  the same 9-position grid).
- Confusing the CSV ``position`` column (float degrees) with the hex-coordinate
  ``mid`` used in ``build_sbar_gt`` — the CSV position is the biological
  stimulus position; ``mid`` is the hex-axis coordinate of the cost node.
- Using ``cell[:2]`` for Mi1/Mi4 keys (wrong: both become ``Mi_…`` and collide).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from neuron.borst import t_from_ms


GT_CELLS: Tuple[str, ...] = (
    "Mi1", "Mi4",
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
)

GT_CELL_ALIASES: dict = {
    "T4": tuple(cell for cell in GT_CELLS if cell.startswith("T4")),
    "T5": tuple(cell for cell in GT_CELLS if cell.startswith("T5")),
}

_GRUNTMAN21 = (
    Path(__file__).resolve().parents[4]
    / "figure_digitization"
    / "gruntman21"
)
_T4_T5_CSV = _GRUNTMAN21 / "t4_t5_w1_160ms_mean_std.csv"
_SOURCE_POSITION_CSVS = {
    "Mi1": _GRUNTMAN21 / "gruntman21_mi1_position_traces.csv",
    "Mi4": _GRUNTMAN21 / "gruntman21_mi4_position_traces.csv",
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


def position_label(position: float) -> str:
    """Format a flash/mid position for a GT trace key."""
    return (
        f"{int(position):+d}"
        if float(position).is_integer()
        else f"{float(position):+.1f}"
    )


def gt_trace_key(cell: str, contrast: str, position: float) -> str:
    """Build GT dict key for one (cell, bright/dark contrast, position)."""
    if cell in ("Mi1", "Mi4"):
        prefix = cell
        pathway = "PC" if contrast == "bright" else "NC"
    elif cell.startswith("T4") or cell.startswith("T5"):
        prefix = cell[:2]
        pathway = (
            "PC" if (cell.startswith("T4") and contrast == "bright")
            or (cell.startswith("T5") and contrast == "dark") else "NC"
        )
    else:
        raise ValueError(f"unknown sbar gt cell: {cell!r}")
    return f"{prefix}_{pathway}_pos{position_label(position)}_w1"


def _interp_trace(t_axis: np.ndarray, time_ms, vm_mv) -> np.ndarray:
    time_ms = np.asarray(time_ms, dtype=np.float64)
    vm_mv = np.asarray(vm_mv, dtype=np.float64)
    return np.interp(
        t_axis,
        time_ms,
        vm_mv,
        left=float(vm_mv[0]),
        right=float(vm_mv[-1]),
    )


def load_gt_stats(
    *,
    t_onset,
    ms_response,
    ms_sti,
    delta_ms: float,
    ms_post=0.0,
):
    """Return ``(means, stds)`` on the simulation time axis.

    ``stds`` contains T4/T5 only because the Mi1/Mi4 decomposition has no
    biological-cell variation data.  Measured integer T4/T5 positions are
    supplemented with half-step spatial interpolation for the vertical hex
    axis; measured keys themselves are never replaced.
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
    gt_stds = {}
    frame = pd.read_csv(_T4_T5_CSV)
    if set(frame["width_led"].unique()) != {1} or set(
        frame["duration_ms"].unique()
    ) != {160}:
        raise ValueError(f"unexpected width/duration in {_T4_T5_CSV}")
    for _, grouped in frame.groupby("trace_id", sort=False):
        key = str(grouped["trace_id"].iloc[0])
        gts[key] = _interp_trace(
            t_axis,
            grouped["time_ms"],
            grouped["vm_mean_mv"],
        )
        gt_stds[key] = _interp_trace(
            t_axis,
            grouped["time_ms"],
            grouped["vm_std_mv"],
        )

    # Hex y coordinates include half steps, while the raw aligned-position grid
    # is integer.  Interpolate the two statistics spatially without modifying
    # the directly measured integer-position traces.
    for cell in ("T4", "T5"):
        for pathway in ("PC", "NC"):
            for position2 in range(-7, 8, 2):
                position = position2 / 2.0
                lo = float(np.floor(position))
                hi = float(np.ceil(position))
                lo_key = f"{cell}_{pathway}_pos{position_label(lo)}_w1"
                hi_key = f"{cell}_{pathway}_pos{position_label(hi)}_w1"
                key = f"{cell}_{pathway}_pos{position_label(position)}_w1"
                gts[key] = 0.5 * (gts[lo_key] + gts[hi_key])
                gt_stds[key] = 0.5 * (gt_stds[lo_key] + gt_stds[hi_key])
    for cell, path in _SOURCE_POSITION_CSVS.items():
        # Mi4 CSV contribution is opposite the cell Vm used as sbar GT.
        gt_sign = -1.0 if cell == "Mi4" else 1.0
        for (pathway, position), grouped in pd.read_csv(path).groupby(
            ["contrast", "position"],
            sort=False,
        ):
            key = f"{cell}_{pathway}_pos{position_label(float(position))}_w1"
            gts[key] = gt_sign * _interp_trace(
                t_axis,
                grouped["time_ms"],
                grouped["contribution_vm_mv"],
            )
    return gts, gt_stds


def load_gt(**kwargs):
    """Backward-compatible mean-only GT loader."""
    return load_gt_stats(**kwargs)[0]
