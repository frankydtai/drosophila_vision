# -*- coding: utf-8 -*-
"""Spot GT numbers: rf × ir → gt (ir from :mod:`task.spread.gt`)."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from task.spread.gt import (
    GT_CELLS,
    RF_SIGN,
    gt_sign,
    load_ir,
    spread_gt_active,
    expand_gt_cells,
    t_delay_from_ir,
)

RF_CENTER_RADIUS = 0
RF_N_RADII = 5
RF_RADIUS_DEG = 5.0

RF_SCALE = {
    "L1": [1.000, 0.137, -0.009, -0.008, -0.006],
    "L2": [1.000, 0.231, -0.009, -0.008, -0.005],
    "L3": [1.000, -0.215, -0.178, -0.038, -0.004],
    "L4": [1.000, 0.299, -0.038, -0.038, -0.024],
    "L5": [1.000, 0.212, -0.030, -0.024, -0.014],
    "Mi1": [1.000, 0.126, -0.018, -0.012, -0.007],
    "Tm3": [1.000, 0.618, 0.146, 0.013, 0.000],
    "Mi4": [1.000, -0.057, -0.111, -0.029, -0.004],
    "Mi9": [1.000, 0.081, -0.054, -0.030, -0.013],
    "T4a": [1.000, 0.714, 0.460, 0.000, 0.000],
    "T4b": [1.000, 0.714, 0.460, 0.000, 0.000],
    "T4c": [1.000, 0.714, 0.460, 0.000, 0.000],
    "T4d": [1.000, 0.714, 0.460, 0.000, 0.000],
    "Tm1": [1.000, 0.301, -0.031, -0.028, -0.015],
    "Tm2": [1.000, 0.308, -0.025, -0.027, -0.016],
    "Tm4": [1.000, 0.523, 0.026, -0.059, -0.044],
    "Tm9": [1.000, 0.195, -0.042, -0.025, -0.011],
}


def build_rf(cell: str) -> np.ndarray:
    return np.asarray(RF_SCALE[cell], dtype=np.float64).copy()


def spot_gt_active(spread_gt_mode: str, contrast: str, rf_sign: int) -> bool:
    return spread_gt_active(spread_gt_mode, contrast, rf_sign)


def load_rf_ir(*, t_onset=None, n_t=None, ms_sti=None, delta_ms: float, filter="none"):
    if t_onset is None or n_t is None:
        raise ValueError("load_rf_ir requires t_onset and n_t")
    t_onset = int(t_onset)
    n_t = int(n_t)
    delta_ms = float(delta_ms)
    if delta_ms <= 0:
        raise ValueError(f"delta_ms must be > 0, got {delta_ms}")
    rf = np.zeros((len(GT_CELLS), RF_N_RADII))
    gt_cell_idx = dict(zip(GT_CELLS, range(len(GT_CELLS))))
    for cell in GT_CELLS:
        rf[gt_cell_idx[cell]] = build_rf(cell)
    ir = load_ir(
        t_onset=t_onset, n_t=n_t, ms_sti=ms_sti, delta_ms=delta_ms, filter=str(filter),
    )
    return rf, ir


def _gt_from_rf_ir(rf: np.ndarray, ir: np.ndarray) -> np.ndarray:
    gt = np.zeros((rf.shape[0], RF_N_RADII, ir.shape[1]))
    gt_cell_idx = dict(zip(GT_CELLS, range(len(GT_CELLS))))
    for cell in GT_CELLS:
        for radius in range(RF_N_RADII):
            gt[gt_cell_idx[cell], radius] = (
                rf[gt_cell_idx[cell], radius] * ir[gt_cell_idx[cell]]
            )
    return gt


def _spot_gt(
    contrast: str,
    *,
    t_onset=None,
    n_t=None,
    ms_sti=None,
    delta_ms: float,
    filter="none",
    spread_gt_mode: str = "all",
) -> np.ndarray:
    gts = _gt_from_rf_ir(*load_rf_ir(
        t_onset=t_onset, n_t=n_t, ms_sti=ms_sti, delta_ms=delta_ms, filter=filter,
    ))
    gt_cell_idx = dict(zip(GT_CELLS, range(len(GT_CELLS))))
    for cell in GT_CELLS:
        rf_sign = int(RF_SIGN[cell])
        if not spot_gt_active(spread_gt_mode, contrast, rf_sign):
            gts[gt_cell_idx[cell]] = 0.0
        else:
            gts[gt_cell_idx[cell]] *= gt_sign(contrast, rf_sign)
    return gts


def load_gt(
    *,
    t_onset=None,
    n_t=None,
    ms_sti=None,
    delta_ms: float,
    filter="none",
    spread_gt_mode: str = "all",
):
    return _spot_gt(
        "bright",
        t_onset=t_onset, n_t=n_t, ms_sti=ms_sti, delta_ms=delta_ms,
        filter=filter, spread_gt_mode=spread_gt_mode,
    )


def load_gt_dark(
    *,
    t_onset=None,
    n_t=None,
    ms_sti=None,
    delta_ms: float,
    filter="none",
    spread_gt_mode: str = "all",
):
    return _spot_gt(
        "dark",
        t_onset=t_onset, n_t=n_t, ms_sti=ms_sti, delta_ms=delta_ms,
        filter=filter, spread_gt_mode=spread_gt_mode,
    )


def _spot_readout_a_radius(
    rf: np.ndarray,
    radius: int,
    spot_radius: float,
) -> float:
    radius = int(radius)
    if radius < 0 or radius >= RF_N_RADII:
        raise ValueError(f"spot rf radius out of range: {radius!r}")
    if float(spot_radius) == 1:
        if radius == 1:
            return float(rf[1]) + float(rf[2])
        if radius == 2:
            return 0.0
    return float(rf[radius])
