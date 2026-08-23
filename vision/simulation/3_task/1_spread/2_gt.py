# -*- coding: utf-8 -*-
"""Spread GT numbers: ir(t) → gt (no rf, no network binding)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from task.spread.sti_spec import sti_mask

GT_CELLS: Tuple[str, ...] = (
    "L1", "L2", "L3", "L4", "L5",
    "Mi1", "Tm3", "Mi4", "Mi9",
    "T4a", "T4b", "T4c", "T4d",
    "Tm1", "Tm2", "Tm4", "Tm9",
)

RF_SIGN = {
    "L1": -1, "L2": -1, "L3": -1, "L4": -1, "L5": 1,
    "Mi1": 1, "Tm3": 1, "Mi4": 1, "Mi9": -1,
    "T4a": 1, "T4b": 1, "T4c": 1, "T4d": 1,
    "Tm1": -1, "Tm2": -1, "Tm4": -1, "Tm9": -1,
}

_ARENZ_DIR = Path(__file__).resolve().parents[4] / "figure_digitization" / "arenz"
ARENZ_L_DIGITIZED_CSV = _ARENZ_DIR / "L_digitized.csv"
ARENZ_4_DIGITIZED_CSV = _ARENZ_DIR / "4_digitized.csv"

IR_hp_ms = {
    "L1": 391.0, "L2": 288.0, "L3": 0.0, "L4": 381.0, "L5": 127.0,
    "Mi1": 318.0, "Tm3": 260.0, "Mi4": 0.0, "Mi9": 0.0,
    "T4a": 0.0, "T4b": 0.0, "T4c": 0.0, "T4d": 0.0,
    "Tm1": 296.0, "Tm2": 153.0, "Tm4": 249.0, "Tm9": 0.0,
}
IR_lp_ms = {
    "L1": 38.0, "L2": 58.0, "L3": 54.0, "L4": 23.0, "L5": 42.0,
    "Mi1": 54.0, "Tm3": 27.0, "Mi4": 38.0, "Mi9": 77.0,
    "T4a": 82.4, "T4b": 82.4, "T4c": 82.4, "T4d": 82.4,
    "Tm1": 44.0, "Tm2": 14.0, "Tm4": 24.0, "Tm9": 107.0,
}
IR_delay_ms = {
    "L1": 5.0, "L2": 5.0, "L3": 5.0, "L4": 10.0, "L5": 10.0,
    "Mi1": 15.0, "Tm3": 15.0, "Mi4": 15.0, "Mi9": 15.0,
    "T4a": 25.0, "T4b": 25.0, "T4c": 25.0, "T4d": 25.0,
    "Tm1": 15.0, "Tm2": 15.0, "Tm4": 15.0, "Tm9": 15.0,
}


def expand_gt_cells(cells: Sequence[str]) -> Tuple[str, ...]:
    if not cells:
        raise ValueError("gt_cells must not be empty")
    keys = tuple(str(cell).strip() for cell in cells)
    for key in keys:
        if key not in GT_CELLS:
            raise ValueError(f"unknown gt cell {key!r} (expected {', '.join(GT_CELLS)})")
    return tuple(dict.fromkeys(keys))


def _lowpass(mask, tau_ms, *, delta_ms: float):
    mask = mask.transpose(np.roll(np.arange(mask.ndim), 1))
    tau_ms = float(tau_ms)
    if tau_ms < delta_ms:
        return mask.transpose(np.roll(np.arange(mask.ndim), -1))
    result = np.empty_like(mask)
    result[0] = mask[0]
    for t in range(mask.shape[0] - 1):
        result[t + 1] = result[t] + (delta_ms / tau_ms) * (mask[t] - result[t])
    return result.transpose(np.roll(np.arange(result.ndim), -1))


def _bandpass(mask, hp_tau_ms, lp_tau_ms, *, delta_ms: float):
    result = _lowpass(mask, lp_tau_ms, delta_ms=delta_ms)
    if hp_tau_ms != 0:
        result = result - _lowpass(result, hp_tau_ms, delta_ms=delta_ms)
    return result


def normalize_ir(ir):
    ir = ir - ir[0]
    ir_max = np.nanmax(ir)
    ir_min = np.nanmin(ir)
    if ir_max == ir_min:
        return ir * 0.0
    return ir / max(abs(ir_max), abs(ir_min))


def _shift_right(ir, t_delay: int):
    ir = np.asarray(ir)
    t_delay = int(t_delay)
    if t_delay <= 0:
        return ir
    delayed_ir = np.zeros_like(ir)
    delayed_ir[t_delay:] = ir[:-t_delay]
    return delayed_ir


def build_ir_lti(cell: str, mask: np.ndarray, *, delta_ms: float) -> np.ndarray:
    return _shift_right(
        normalize_ir(
            _bandpass(mask, IR_hp_ms[cell], IR_lp_ms[cell], delta_ms=delta_ms)
        ),
        int(round(float(IR_delay_ms[cell]) / float(delta_ms))),
    )


def t_delay_from_ir(*, delta_ms: float, filter: str = "none") -> np.ndarray:
    if str(filter) == "ca":
        return np.zeros(len(GT_CELLS), dtype=np.int64)
    return np.asarray(
        [round(float(IR_delay_ms[cell]) / float(delta_ms)) for cell in GT_CELLS],
        dtype=np.int64,
    )


def _arenz_traces_from_csv(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    by_cell: Dict[str, list] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cell = str(row["cell"]).strip()
            by_cell.setdefault(cell, []).append(
                (float(row["time_s"]), float(row["amplitude"]))
            )
    return {
        cell: tuple(
            np.asarray(column, dtype=np.float64)
            for column in zip(*sorted(pairs, key=lambda pair: pair[0]))
        )
        for cell, pairs in by_cell.items()
    }


def load_ir_arenz(*, n_t, delta_ms: float) -> np.ndarray:
    n_t = int(n_t)
    delta_ms = float(delta_ms)
    if delta_ms <= 0:
        raise ValueError(f"delta_ms must be > 0, got {delta_ms}")
    ir = np.zeros((len(GT_CELLS), n_t), dtype=np.float64)
    if n_t <= 0:
        return ir
    traces = {}
    traces.update(_arenz_traces_from_csv(ARENZ_L_DIGITIZED_CSV))
    traces.update(_arenz_traces_from_csv(ARENZ_4_DIGITIZED_CSV))
    for cell_idx, cell in enumerate(GT_CELLS):
        if cell not in traces:
            continue
        t_csv, amp = traces[cell]
        ir[cell_idx] = np.interp(
            np.arange(n_t, dtype=np.float64) * (delta_ms / 1000.0),
            t_csv,
            amp,
            left=amp[0],
            right=amp[-1],
        )
    return ir


def contrast_sign(contrast: str) -> int:
    if contrast == "bright":
        return 1
    if contrast == "dark":
        return -1
    raise ValueError(f"contrast must be 'bright' or 'dark', got {contrast!r}")


def gt_sign(contrast: str, rf_sign: int) -> float:
    return float(int(rf_sign) * contrast_sign(contrast))


def spread_gt_active(spread_gt_mode: str, contrast: str, rf_sign: int) -> bool:
    return (str(spread_gt_mode) == "all") or (gt_sign(contrast, rf_sign) > 0)


def load_ir(*, t_onset=None, n_t=None, ms_sti=None, delta_ms: float, filter="none"):
    if t_onset is None or n_t is None:
        raise ValueError("load_ir requires t_onset and n_t")
    t_onset = int(t_onset)
    n_t = int(n_t)
    delta_ms = float(delta_ms)
    if delta_ms <= 0:
        raise ValueError(f"delta_ms must be > 0, got {delta_ms}")
    if str(filter) == "ca":
        ir = np.zeros((len(GT_CELLS), n_t))
        if t_onset < n_t:
            ir[:, t_onset:] = load_ir_arenz(n_t=n_t - t_onset, delta_ms=delta_ms)
        return ir
    mask = sti_mask(t_onset, n_t, ms_sti, delta_ms=delta_ms)
    mask = mask / np.max(mask)
    ir = np.zeros((len(GT_CELLS), n_t))
    for cell_idx, cell in enumerate(GT_CELLS):
        ir[cell_idx] = build_ir_lti(cell, mask, delta_ms=delta_ms)
    return ir
