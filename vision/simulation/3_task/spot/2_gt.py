# -*- coding: utf-8 -*-
"""Spot paradigm GT numbers: RecF x ImpR traces (no network binding).

Synthesizes the Medulla_Library RecF/ImpR library (bandpass/lowpass ImpR
shaping -- gt-only, not ``neuron.filter_ca``). With ``filter==\"ca\"``, ImpR
is replaced by Arenz digitized CSV traces (``t=0`` at stimulus onset).
PR drive is :func:`task.spot.input.spot_input_waveform`, shared with network
``i_sti``.

Network mapping, cost hexes, and :class:`task.spot.readout.SpotGt` packing
live in :mod:`task.spot.readout`.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from task.spot.input import (
    spot_radius_folds_r2_into_r1,
    spot_input_waveform,
)

# ImpR / RecF gt row order (13 gt cells).
GT_CELLS: Tuple[str, ...] = (
    "L1", "L2", "L3", "L4", "L5", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9",
)

# Repo root: vision/simulation/3_task/spot/2_gt.py → parents[4].
_ARENZ_DIR = Path(__file__).resolve().parents[4] / "figure_digitization" / "arenz"
ARENZ_L_DIGITIZED_CSV = _ARENZ_DIR / "L_digitized.csv"
ARENZ_4_DIGITIZED_CSV = _ARENZ_DIR / "4_digitized.csv"


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


def _load_arenz_csv_traces(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """``cell → (time_s, gt)`` from Arenz digitized CSV (``t=0`` = onset)."""
    by_cell: Dict[str, list] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cell = str(row["cell"]).strip()
            by_cell.setdefault(cell, []).append(
                (float(row["time_s"]), float(row["amplitude"]))
            )
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cell, pairs in by_cell.items():
        pairs.sort(key=lambda p: p[0])
        t = np.asarray([p[0] for p in pairs], dtype=np.float64)
        gt = np.asarray([p[1] for p in pairs], dtype=np.float64)
        out[cell] = (t, gt)
    return out


def load_arenz_digitized_impr(*, t_onset, n_t, delta_ms: float) -> np.ndarray:
    """Arenz digitized temporal gt ``(13, n_t)``; CSV ``t=0`` at ``t_onset``.

    Pre-onset samples are 0. Post-onset times use ``(t - t_onset) * delta_ms``.
    """
    t_onset = int(t_onset)
    n_t = int(n_t)
    delta_ms = float(delta_ms)
    if delta_ms <= 0:
        raise ValueError(f"delta_ms must be > 0, got {delta_ms}")
    traces = {}
    traces.update(_load_arenz_csv_traces(ARENZ_L_DIGITIZED_CSV))
    traces.update(_load_arenz_csv_traces(ARENZ_4_DIGITIZED_CSV))
    missing = [c for c in GT_CELLS if c not in traces]
    if missing:
        raise ValueError(f"Arenz digitized CSV missing cells {missing}")
    out = np.zeros((len(GT_CELLS), n_t), dtype=np.float64)
    if t_onset >= n_t:
        return out
    t_rel_s = np.arange(n_t - t_onset, dtype=np.float64) * (delta_ms / 1000.0)
    for i, cell in enumerate(GT_CELLS):
        t_csv, gt = traces[cell]
        out[i, t_onset:] = np.interp(t_rel_s, t_csv, gt, left=gt[0], right=gt[-1])
    return out


def load_RecF_ImpR(*, t_onset=None, n_t=None, ms_spot=None, delta_ms: float, filter="none"):
    """Return ``(RecF_gt, ImpR_gt)`` for the 13 gt cells.

    Shapes: ``RecF_gt`` ``(13, 45)``; ``ImpR_gt`` ``(13, n_t)``. The
    drive is :func:`task.spot.input.spot_input_waveform` (step or finite spot).
    ImpR filter taus are in ms (scaled by ``delta_ms``); delay is in samples.
    With ``filter==\"ca\"``, ImpR is Arenz digitized (``t=0`` at onset);
    RecF keeps DoG radius signs but omits cell ``RF_sign`` (Arenz has polarity).
    """
    if t_onset is None or n_t is None:
        raise ValueError("load_RecF_ImpR requires t_onset and n_t")
    t_onset = int(t_onset)
    n_t = int(n_t)
    delta_ms = float(delta_ms)
    if delta_ms <= 0:
        raise ValueError(f"delta_ms must be > 0, got {delta_ms}")
    n_cells = len(GT_CELLS)
    use_ca = str(filter) == "ca"

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
        dog = center - RF_surrnd_weight[i] * surrnd
        # ca: polarity from Arenz ImpR; keep DoG signed across radii (no RF_sign).
        RecF_gt[i] = normalize_gt(dog if use_ca else dog * RF_sign[i])

    if use_ca:
        return RecF_gt, load_arenz_digitized_impr(
            t_onset=t_onset, n_t=n_t, delta_ms=delta_ms,
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


def load_RecF_gt(*, t_onset=None, n_t=None, ms_spot=None, delta_ms: float, filter="none"):
    """Spatial x temporal spot cube ``(n_cells, RF_N_RADII, n_t)``; axis = radius."""
    RecF_gt, ImpR_gt = load_RecF_ImpR(
        t_onset=t_onset, n_t=n_t, ms_spot=ms_spot, delta_ms=delta_ms, filter=filter,
    )
    mt = ImpR_gt.shape[1]
    n_cells = len(GT_CELLS)
    gt = np.zeros((n_cells, RF_N_RADII, mt))
    for i in range(n_cells):
        for radius in range(RF_N_RADII):
            sample = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * radius
            gt[i, radius] = RecF_gt[i, sample] * ImpR_gt[i]
    return gt


def load_RecF_gt_dark(*, t_onset=None, n_t=None, ms_spot=None, delta_ms: float, filter="none"):
    """Dark spot spatial x temporal cube: negated bright ``load_RecF_gt()``."""
    return -load_RecF_gt(
        t_onset=t_onset, n_t=n_t, ms_spot=ms_spot, delta_ms=delta_ms, filter=filter,
    )


def _recf_at(recf_row: np.ndarray, radius: float) -> float:
    idx = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * radius
    idx = min(max(idx, 0.0), _RF_NSAMPLES - 1)
    return float(np.interp(idx, np.arange(_RF_NSAMPLES), recf_row))


def _spot_readout_a_radius(recf_row: np.ndarray, radius: float, spot_radius: float) -> float:
    """RecF ``a_radius`` at Euclidean ``radius`` (radius-1 folds r=2 into r=1)."""
    r = round(float(radius), 6)
    if spot_radius_folds_r2_into_r1(spot_radius):
        if r == 1.0:
            return _recf_at(recf_row, 1.0) + _recf_at(recf_row, 2.0)
        if r == 2.0:
            return 0.0
    return _recf_at(recf_row, r)
