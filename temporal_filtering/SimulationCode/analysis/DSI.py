# -*- coding: utf-8 -*-
"""Direction selectivity index (DSI) for moving-bar plot traces."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

_AXIS_PAIRS = {
    "right": ("right", "left"),
    "left": ("right", "left"),
    "up": ("up", "down"),
    "down": ("up", "down"),
}


def parse_moving_bar_spec(sname: str) -> Tuple[str, str, str]:
    direction, contrast, wtag = str(sname).split("_", 2)
    return direction, contrast, wtag


def trace_peak(trace: np.ndarray) -> float:
    return float(np.max(np.asarray(trace, dtype=np.float64)))


def _peak_pair(
    trace_map: Mapping[tuple, np.ndarray],
    cell_name: str,
    contrast: str,
    wtag: str,
    pos_dir: str,
    neg_dir: str,
) -> Optional[Tuple[float, float]]:
    pos_key = (cell_name, f"{pos_dir}_{contrast}_{wtag}")
    neg_key = (cell_name, f"{neg_dir}_{contrast}_{wtag}")
    if pos_key not in trace_map or neg_key not in trace_map:
        return None
    return trace_peak(trace_map[pos_key]), trace_peak(trace_map[neg_key])


def moving_bar_dsi_for_spec(
    trace_map: Mapping[tuple, np.ndarray],
    cell_name: str,
    spec_name: str,
) -> Optional[float]:
    direction, contrast, wtag = parse_moving_bar_spec(spec_name)
    if direction not in _AXIS_PAIRS:
        return None
    pos_dir, neg_dir = _AXIS_PAIRS[direction]
    peaks = _peak_pair(trace_map, cell_name, contrast, wtag, pos_dir, neg_dir)
    if peaks is None:
        return None
    peak_pos, peak_neg = peaks
    denom = peak_pos + peak_neg
    if denom <= 0.0:
        return None
    if direction in ("right", "up"):
        return (peak_pos - peak_neg) / denom
    return (peak_neg - peak_pos) / denom


def moving_bar_dsi_lookup(
    trace_map: Mapping[tuple, np.ndarray],
    cell_names: Sequence[str],
    spec_names: Sequence[str],
) -> dict[tuple[str, str], Optional[float]]:
    out: dict[tuple[str, str], Optional[float]] = {}
    for cell in cell_names:
        for spec in spec_names:
            key = (cell, spec)
            if key not in out:
                out[key] = moving_bar_dsi_for_spec(trace_map, cell, spec)
    return out


def moving_bar_cell_title(
    label: str,
    n: Optional[int] = None,
    model_dsi: Optional[float] = None,
    data_dsi: Optional[float] = None,
    *,
    has_data: bool = False,
) -> str:
    if n is None:
        lines = [str(label)]
    else:
        lines = [f"{label} (n={int(n)})"]
    if model_dsi is not None:
        lines.append(f"DSI={model_dsi:.3f}")
    if has_data and data_dsi is not None:
        lines.append(f"data DSI={data_dsi:.3f}")
    return "\n".join(lines)
