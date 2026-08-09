# -*- coding: utf-8 -*-
"""Spot paradigm GT numbers: RecF x ImpR traces (no network binding).

Synthesizes the Medulla_Library RecF/ImpR library (bandpass/lowpass ImpR
shaping -- gt-only, not ``neuron.filter_ca``). PR drive is
:func:`task.spot.input.spot_input_waveform`, shared with network ``i_sti``.

Network mapping, cost hexes, and :class:`task.spot.readout.SpotGt` packing
live in :mod:`task.spot.readout`.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from task.spot.input import (
    spot_extent_folds_r2_into_r1,
    spot_input_waveform,
)

# ImpR / RecF gt row order (13 gt cells).
GT_CELLS: Tuple[str, ...] = (
    "L1", "L2", "L3", "L4", "L5", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9",
)


def expand_gt_cells(names: Sequence[str]) -> Tuple[str, ...]:
    """Validate ``--gt`` spot cell tokens against ``GT_CELLS`` (final keep-set)."""
    if not names:
        raise ValueError("gt_cells must not be empty")
    out: list = []
    seen: set = set()
    for raw in names:
        key = str(raw).strip()
        if key not in GT_CELLS:
            valid = ", ".join(GT_CELLS)
            raise ValueError(f"unknown gt cell {key!r} (expected {valid})")
        if key not in seen:
            seen.add(key)
            out.append(key)
    return tuple(out)


# RecF sample grid: center at sample 22; one integer radius step = 5 samples.
# Profile cube axis is Euclidean radius (0 .. RF_N_RADII-1), not a mirrored bin.
_RF_CENTER_SAMPLE = 22
_RF_SAMPLES_PER_COL = 5
_RF_NSAMPLES = 45
RF_CENTER_RADIUS = 0
RF_N_RADII = 5
RF_RADIUS_DEG = _RF_SAMPLES_PER_COL  # degrees per integer radius on RF plots
# Gt-only ImpR shaping helpers (not the unused Ca filter). Inlined from the
# old blindschleiche_py3 module so spot/gt owns this path alone.


def _gauss1d(fwhm, rfsize):
    myrange = rfsize / 2
    sigma = fwhm / (2.0 * np.sqrt(2 * np.log(2)))
    x = np.arange(-myrange, (myrange + 1), 1) * 1.0
    z = np.exp(-x ** 2 / (2 * (sigma ** 2)))
    return z / np.sum(z)


def _lowpass(x, tau_ms, *, delta_ms: float):
    """Euler low-pass; ``tau_ms`` is the time constant in milliseconds."""
    x = x.transpose(np.roll(np.arange(x.ndim), 1))
    n = x.shape[0]
    result = np.zeros_like(x)
    tau_ms = float(tau_ms)
    dt = float(delta_ms)
    if dt <= 0:
        raise ValueError(f"delta_ms must be > 0, got {dt}")
    if tau_ms < dt:
        result = x
    else:
        alpha = dt / tau_ms
        result[0] = x[0]
        for i in range(0, n - 1):
            result[i + 1] = alpha * (x[i] - result[i]) + result[i]
    return result.transpose(np.roll(np.arange(result.ndim), -1))


def _bandpass(x, hp_tau_ms, lp_tau_ms, *, delta_ms: float):
    result = _lowpass(x, lp_tau_ms, delta_ms=delta_ms)
    if hp_tau_ms != 0:
        result = result - _lowpass(result, hp_tau_ms, delta_ms=delta_ms)
    return result


def normalize_gt(x):
    x = x - x[0]
    mymax = np.nanmax(x)
    mymin = np.nanmin(x)
    if mymax == mymin:
        return x * 0.0
    return x / max(abs(mymax), abs(mymin))


def _shift_right(y, k: int):
    """Delay ``y`` by ``k`` samples (leading zeros; trailing samples dropped)."""
    y = np.asarray(y)
    k = int(k)
    if k <= 0:
        return y
    out = np.zeros_like(y)
    out[k:] = y[:-k]
    return out


# ImpR onset delay (samples / t-index); same for all gt cells.
_IMPR_SHIFT = 5


def read_RecF_ImpR(*, t_onset=None, n_t=None, ms_spot=None, delta_ms: float):
    """Return ``(RecF_gt, ImpR_gt)`` for the 13 gt cells.

    Shapes: ``RecF_gt`` ``(13, 45)``; ``ImpR_gt`` ``(13, n_t)``. The
    drive is :func:`task.spot.input.spot_input_waveform` (step or finite spot).
    ImpR filter taus are in ms (scaled by ``delta_ms``); delay is in samples.
    """
    if t_onset is None or n_t is None:
        raise ValueError("read_RecF_ImpR requires t_onset and n_t")
    t_onset = int(t_onset)
    n_t = int(n_t)
    delta_ms = float(delta_ms)
    if delta_ms <= 0:
        raise ValueError(f"delta_ms must be > 0, got {delta_ms}")
    n_cells = len(GT_CELLS)

    RF_center_width = np.array([6, 7, 6, 8, 7, 6, 12, 6, 6, 8, 8, 11, 7])
    RF_surrnd_width = np.array([41, 29, 15, 33, 31, 29, 7, 16, 24, 27, 31, 35, 24])
    RF_surrnd_weight = np.array(
        [0.012, 0.013, 0.19, 0.046, 0.035, 0.022, 0.000, 0.132, 0.063, 0.040, 0.035, 0.054, 0.046]
    ) * 5.0
    RF_sign = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

    RecF_gt = np.zeros((n_cells, 45))
    for i in range(n_cells):
        center = _gauss1d(RF_center_width[i], 44)
        surrnd = _gauss1d(RF_surrnd_width[i], 44)
        RecF_gt[i] = normalize_gt(
            (center - RF_surrnd_weight[i] * surrnd) * RF_sign[i]
        )

    # ImpR HP / LP time constants (ms).
    IR_hp_ms = np.array(
        [391.0, 288.0, 0.0, 381.0, 127.0, 318.0, 260.0, 0.0, 0.0, 296.0, 153.0, 249.0, 0.0]
    )
    IR_lp_ms = np.array(
        [38.0, 58.0, 54.0, 23.0, 42.0, 54.0, 27.0, 38.0, 77.0, 44.0, 14.0, 24.0, 107.0]
    )

    u = spot_input_waveform(t_onset, n_t, ms_spot, delta_ms=delta_ms)
    u = u / np.max(u)

    ImpR_gt = np.zeros((n_cells, n_t))
    for i in range(n_cells):
        ImpR_gt[i] = _shift_right(
            normalize_gt(_bandpass(u, IR_hp_ms[i], IR_lp_ms[i], delta_ms=delta_ms)),
            _IMPR_SHIFT,
        )

    return RecF_gt, ImpR_gt


def read_RecF_gt(*, t_onset=None, n_t=None, ms_spot=None, delta_ms: float):
    """Spatial x temporal spot cube ``(n_cells, RF_N_RADII, n_t)``; axis = radius."""
    RecF_gt, ImpR_gt = read_RecF_ImpR(
        t_onset=t_onset, n_t=n_t, ms_spot=ms_spot, delta_ms=delta_ms,
    )
    mt = ImpR_gt.shape[1]
    n_cells = len(GT_CELLS)
    gt = np.zeros((n_cells, RF_N_RADII, mt))
    for i in range(n_cells):
        for radius in range(RF_N_RADII):
            sample = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * radius
            gt[i, radius] = RecF_gt[i, sample] * ImpR_gt[i]
    return gt


def read_RecF_gt_dark(*, t_onset=None, n_t=None, ms_spot=None, delta_ms: float):
    """Dark spot spatial x temporal cube: negated bright ``read_RecF_gt()``."""
    return -read_RecF_gt(
        t_onset=t_onset, n_t=n_t, ms_spot=ms_spot, delta_ms=delta_ms,
    )


def _recf_at(recf_row: np.ndarray, radius: float) -> float:
    idx = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * radius
    idx = min(max(idx, 0.0), _RF_NSAMPLES - 1)
    return float(np.interp(idx, np.arange(_RF_NSAMPLES), recf_row))


def _spot_readout_amp(recf_row: np.ndarray, radius: float, spot_extent: float) -> float:
    """RecF amplitude at Euclidean ``radius`` (extent-1 folds r=2 into r=1)."""
    r = round(float(radius), 6)
    if spot_extent_folds_r2_into_r1(spot_extent):
        if r == 1.0:
            return _recf_at(recf_row, 1.0) + _recf_at(recf_row, 2.0)
        if r == 2.0:
            return 0.0
    return _recf_at(recf_row, r)
