# -*- coding: utf-8 -*-
"""Spot paradigm GT numbers: RecF x ImpR traces (no network binding).

All GT literals and helpers in this module are **owned here** — nothing is
imported at runtime from legacy SimulationCode (Medulla_Library,
blindschleiche_py3). Per-cell DoG RecF widths and ImpR HP/LP taus are
hardcoded arrays indexed by :data:`GT_CELLS`; the ``filter=\"ca\"`` path also
reads external CSVs under ``figure_digitization/arenz/``.

``filter=\"none\"``: peak-normalized RecF ``(n_cell, RF_N_RADII)`` × bandpass/LP
ImpR from :func:`spot_input_waveform`. DoG RecF rows are multiplied by
``RF_SIGN``; Gruntman 2018 Fig. 2B direct RecF rows (T4a–T4d) use gain ratios
at ``r=0..2`` and shared center ``tau_ms`` as LP (``IR_hp_ms=0``).

``filter=\"ca\"``: DoG RecF without ``RF_SIGN``; ImpR from Arenz digitized CSV
(``t=0`` at sti onset) when a row exists. Rows without Arenz ImpR stay zero.

RecF holds peak-normalized samples at Euclidean radii ``0 .. RF_N_RADII-1``
(``RecF(r) = raw(r) / max(|raw(0..RF_N_RADII-1)|)``; no ``x -= x[0]``).
Fractional cost radii (e.g. ``sqrt(3)``) evaluate the same DoG at ``x = radius *
RF_RADIUS_DEG`` with the integer-radius peak scale.

Cost GT membership is gated by ``spot_gt_mode`` (``all`` | ``positive``) via
:func:`spot_gt_active`; waveform × :func:`contrast_sign` only (dark = −1).
Sti drive is :func:`task.spot.input.spot_input_waveform`, shared with network
``i_sti``.

Network mapping, cost hexes, and :class:`task.spot.pack.SpotGt` packing
live in :mod:`task.spot.pack`.
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

GT_CELLS: Tuple[str, ...] = (
    "L1", "L2", "L3", "L4", "L5", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9",
    "T4a", "T4b", "T4c", "T4d",
)

# Per-cell ON (+1) / OFF (−1); same order as ``GT_CELLS``.
RF_SIGN = np.array(
    [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
    dtype=np.int8,
)

# Gruntman 2018 Fig. 2B location-shared LP fit (gruntman18_fit_lp_shared_location.csv).
_GRUNTMAN_CENTER_GAIN_MV = 25.731779521277698
_GRUNTMAN_RECF_R1 = (
    (22.237813729937685 + 14.5102594676567) / 2.0 / _GRUNTMAN_CENTER_GAIN_MV
)
_GRUNTMAN_RECF_R2 = (
    (17.38170393943792 + 6.269266224825205) / 2.0 / _GRUNTMAN_CENTER_GAIN_MV
)
_GRUNTMAN_RECF_ROW = np.array(
    [1.0, _GRUNTMAN_RECF_R1, _GRUNTMAN_RECF_R2, 0.0, 0.0], dtype=np.float64,
)
_N_RECF_DOG = 13

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


# RecF axis: Euclidean hex radius r → visual-field deg = r * RF_RADIUS_DEG.
RF_CENTER_RADIUS = 0
RF_N_RADII = 5
RF_RADIUS_DEG = 5.0

RF_CENTER_WIDTH = np.array([6, 7, 6, 8, 7, 6, 12, 6, 6, 8, 8, 11, 7])
RF_SURRND_WIDTH = np.array([41, 29, 15, 33, 31, 29, 7, 16, 24, 27, 31, 35, 24])
RF_SURRND_SCALE = np.array(
    [0.012, 0.013, 0.19, 0.046, 0.035, 0.022, 0.000, 0.132, 0.063, 0.040, 0.035, 0.054, 0.046]
) * 5.0

IR_hp_ms = np.array(
    [391.0, 288.0, 0.0, 381.0, 127.0, 318.0, 260.0, 0.0, 0.0, 296.0, 153.0, 249.0, 0.0]
    + [0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
IR_lp_ms = np.array(
    [38.0, 58.0, 54.0, 23.0, 42.0, 54.0, 27.0, 38.0, 77.0, 44.0, 14.0, 24.0, 107.0]
    + [82.40496136373703, 82.40496136373703, 82.40496136373703, 82.40496136373703],
    dtype=np.float64,
)


def _gauss1d_at(fwhm: float, x_deg: float) -> float:
    """Sum-normalized 1D Gaussian at offset ``x_deg`` from center (legacy grid)."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    xs = np.arange(-22.0, 23.0, 1.0)
    total = float(np.sum(np.exp(-xs ** 2 / (2.0 * sigma ** 2))))
    if total == 0.0:
        return 0.0
    return float(np.exp(-float(x_deg) ** 2 / (2.0 * sigma ** 2)) / total)


def _raw_signed_dog_at(cell_idx: int, radius: float, *, use_ca: bool) -> float:
    x_deg = float(radius) * RF_RADIUS_DEG
    center = _gauss1d_at(float(RF_CENTER_WIDTH[cell_idx]), x_deg)
    surrnd = _gauss1d_at(float(RF_SURRND_WIDTH[cell_idx]), x_deg)
    raw = center - float(RF_SURRND_SCALE[cell_idx]) * surrnd
    if not use_ca:
        raw *= float(RF_SIGN[cell_idx])
    return raw


def _build_recf_row(cell_idx: int, *, use_ca: bool) -> tuple[np.ndarray, float]:
    if cell_idx >= _N_RECF_DOG:
        if use_ca:
            return np.zeros(RF_N_RADII, dtype=np.float64), 0.0
        return _GRUNTMAN_RECF_ROW.copy(), 1.0
    raw_r = np.array(
        [_raw_signed_dog_at(cell_idx, r, use_ca=use_ca) for r in range(RF_N_RADII)],
        dtype=np.float64,
    )
    peak = max(abs(float(np.max(raw_r))), abs(float(np.min(raw_r))))
    if peak == 0.0:
        return raw_r * 0.0, 0.0
    return raw_r / peak, peak


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
    """Arenz digitized temporal gt ``(len(GT_CELLS), n_t)``; CSV ``t=0`` at ``t_onset``.

    Rows are filled only for :data:`GT_CELLS` names present in the Arenz CSVs;
    other rows stay zero. Pre-onset samples are 0. Post-onset times use
    ``(t - t_onset) * delta_ms``.
    """
    t_onset = int(t_onset)
    n_t = int(n_t)
    delta_ms = float(delta_ms)
    if delta_ms <= 0:
        raise ValueError(f"delta_ms must be > 0, got {delta_ms}")
    traces = {}
    traces.update(_load_arenz_csv_traces(ARENZ_L_DIGITIZED_CSV))
    traces.update(_load_arenz_csv_traces(ARENZ_4_DIGITIZED_CSV))
    missing = [GT_CELLS[i] for i in range(_N_RECF_DOG) if GT_CELLS[i] not in traces]
    if missing:
        raise ValueError(f"Arenz digitized CSV missing cells {missing}")
    out = np.zeros((len(GT_CELLS), n_t), dtype=np.float64)
    if t_onset >= n_t:
        return out
    t_rel_s = np.arange(n_t - t_onset, dtype=np.float64) * (delta_ms / 1000.0)
    for i, cell in enumerate(GT_CELLS):
        if cell not in traces:
            continue
        t_csv, gt = traces[cell]
        out[i, t_onset:] = np.interp(t_rel_s, t_csv, gt, left=gt[0], right=gt[-1])
    return out


def contrast_sign(contrast: str) -> int:
    """``+1`` bright / ``−1`` dark (sole GT waveform ±1 for contrast)."""
    if contrast == "bright":
        return 1
    if contrast == "dark":
        return -1
    raise ValueError(f"contrast must be 'bright' or 'dark', got {contrast!r}")


def spot_gt_active(spot_gt_mode: str, contrast: str, rf_sign: int) -> bool:
    """Cost GT membership for ``spot_gt_mode`` × ``contrast`` × ``rf_sign``.

    ``all``: always active. ``positive``: ``rf_sign * contrast_sign(contrast) > 0``.
    """
    return (str(spot_gt_mode) == "all") or (
        int(rf_sign) * contrast_sign(contrast) > 0
    )


def load_RecF_ImpR(*, t_onset=None, n_t=None, ms_spot=None, delta_ms: float, filter="none"):
    """Return ``(RecF_gt, RecF_peak_scale, ImpR_gt)`` for ``GT_CELLS``.

    Shapes: ``RecF_gt`` ``(n_cell, RF_N_RADII)``; ``RecF_peak_scale`` ``(n_cell,)``;
    ``ImpR_gt`` ``(n_cell, n_t)``.

    ``filter=\"none\"``: peak-normalized RecF; ImpR =
    ``normalize_gt(bandpass(u, IR_hp_ms, IR_lp_ms))`` with ``_IMPR_SHIFT``.

    ``filter=\"ca\"``: DoG RecF without ``RF_SIGN``; Arenz digitized ImpR when present.
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

    RecF_gt = np.zeros((n_cells, RF_N_RADII))
    RecF_peak_scale = np.zeros(n_cells)
    for i in range(n_cells):
        RecF_gt[i], RecF_peak_scale[i] = _build_recf_row(i, use_ca=use_ca)

    if use_ca:
        return RecF_gt, RecF_peak_scale, load_arenz_digitized_impr(
            t_onset=t_onset, n_t=n_t, delta_ms=delta_ms,
        )

    u = spot_input_waveform(t_onset, n_t, ms_spot, delta_ms=delta_ms)
    u = u / np.max(u)

    ImpR_gt = np.zeros((n_cells, n_t))
    for i in range(n_cells):
        ImpR_gt[i] = _shift_right(
            normalize_gt(_bandpass(u, IR_hp_ms[i], IR_lp_ms[i], delta_ms=delta_ms)),
            _IMPR_SHIFT,
        )

    return RecF_gt, RecF_peak_scale, ImpR_gt


def _recf_impr_rt(RecF_gt: np.ndarray, ImpR_gt: np.ndarray) -> np.ndarray:
    """``(n_cells, RF_N_RADII, n_t)`` = RecF(r) × ImpR (no membership gate)."""
    mt = ImpR_gt.shape[1]
    n_cells = RecF_gt.shape[0]
    gt = np.zeros((n_cells, RF_N_RADII, mt))
    for i in range(n_cells):
        for radius in range(RF_N_RADII):
            gt[i, radius] = RecF_gt[i, radius] * ImpR_gt[i]
    return gt


def _recf_gt_rt(
    contrast: str,
    *,
    t_onset=None,
    n_t=None,
    ms_spot=None,
    delta_ms: float,
    filter="none",
    spot_gt_mode: str = "all",
) -> np.ndarray:
    """Spatial×temporal rt; inactive rows (see :func:`spot_gt_active`) are zero."""
    RecF_gt, _RecF_peak_scale, ImpR_gt = load_RecF_ImpR(
        t_onset=t_onset, n_t=n_t, ms_spot=ms_spot, delta_ms=delta_ms, filter=filter,
    )
    gts = _recf_impr_rt(RecF_gt, ImpR_gt)
    for i in range(gts.shape[0]):
        rf_sign = int(RF_SIGN[i])
        if not spot_gt_active(spot_gt_mode, contrast, rf_sign):
            gts[i] = 0.0
        else:
            gts[i] *= float(contrast_sign(contrast))
    return gts


def load_RecF_gt(
    *,
    t_onset=None,
    n_t=None,
    ms_spot=None,
    delta_ms: float,
    filter="none",
    spot_gt_mode: str = "all",
):
    """Bright spatial x temporal rt ``(n_cells, RF_N_RADII, n_t)``."""
    return _recf_gt_rt(
        "bright",
        t_onset=t_onset, n_t=n_t, ms_spot=ms_spot, delta_ms=delta_ms,
        filter=filter, spot_gt_mode=spot_gt_mode,
    )


def load_RecF_gt_dark(
    *,
    t_onset=None,
    n_t=None,
    ms_spot=None,
    delta_ms: float,
    filter="none",
    spot_gt_mode: str = "all",
):
    """Dark spatial x temporal rt ``(n_cells, RF_N_RADII, n_t)``."""
    return _recf_gt_rt(
        "dark",
        t_onset=t_onset, n_t=n_t, ms_spot=ms_spot, delta_ms=delta_ms,
        filter=filter, spot_gt_mode=spot_gt_mode,
    )


def _recf_at_from_row(recf_row: np.ndarray, radius: float) -> float:
    x = float(radius)
    if x <= 0.0:
        return float(recf_row[0])
    if x >= RF_N_RADII - 1:
        return float(recf_row[RF_N_RADII - 1])
    i0 = int(np.floor(x))
    i1 = min(i0 + 1, RF_N_RADII - 1)
    frac = x - i0
    return float((1.0 - frac) * recf_row[i0] + frac * recf_row[i1])


def _spot_readout_a_radius(
    recf_row: np.ndarray,
    radius: float,
    spot_radius: float,
) -> float:
    """RecF ``a_radius`` at Euclidean ``radius`` (radius-1 folds r=2 into r=1)."""
    r = round(float(radius), 6)
    if spot_radius_folds_r2_into_r1(spot_radius):
        if r == 1.0:
            return _recf_at_from_row(recf_row, 1.0) + _recf_at_from_row(recf_row, 2.0)
        if r == 2.0:
            return 0.0
    return _recf_at_from_row(recf_row, radius)
