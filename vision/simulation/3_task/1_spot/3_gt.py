# -*- coding: utf-8 -*-
"""Spot paradigm GT numbers: rf × ir → gt (no network binding).

All GT literals and helpers in this module are **owned here** — nothing is
imported at runtime from legacy SimulationCode (Medulla_Library,
blindschleiche_py3). Per-cell rf samples and ir HP/LP/delay are hardcoded
dicts keyed by cell name (:data:`GT_CELLS`); the ``filter=\"ca\"`` path also
reads external CSVs under ``figure_digitization/arenz/``.

``filter=\"none\"``: :func:`build_rf` × :func:`build_ir_lti` (bandpass/LP on
:func:`sti_waveform`). T4a–T4d share Gruntman 2018 Fig. 2B spatial
samples in :data:`RF_SCALE` and LP ``IR_lp_ms`` (``IR_hp_ms=0``).

``filter=\"ca\"``: same :data:`RF_SCALE` (unsigned; no ``RF_SIGN`` on rf);
ir from :func:`load_ir_arenz` when a CSV row exists.

:data:`RF_SCALE` holds peak-normalized unsigned samples at Euclidean radii
``0 .. RF_N_RADII-1`` (center = +1). Assembled gt is ``rf(r) × ir(t)`` per
cell, shape ``(RF_N_RADII, n_t)``. Fractional cost radii interpolate
:data:`RF_SCALE`.

Cost GT membership is gated by ``spot_gt_mode`` (``all`` | ``positive``) via
:func:`spot_gt_active` (still uses :data:`RF_SIGN`); waveform ×
:func:`contrast_sign` only (dark = −1). Sti drive is
:func:`task.spot.sti_spec.sti_waveform`, shared with network ``i_sti``.

Network mapping, cost hexes, and :class:`task.spot.pack.SpotGt` packing
live in :mod:`task.spot.pack`.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from task.spot.sti_geo import spot_radius_folds_r2_into_r1
from task.spot.sti_spec import sti_waveform

GT_CELLS: Tuple[str, ...] = (
    "L1", "L2", "L3", "L4", "L5",
    "Mi1", "Tm3", "Mi4", "Mi9",
    "T4a", "T4b", "T4c", "T4d",
    "Tm1", "Tm2", "Tm4", "Tm9",
)

# Per-cell ON (+1) / OFF (−1); membership only (not multiplied into rf).
RF_SIGN = {
    "L1": -1, "L2": -1, "L3": -1, "L4": -1, "L5": 1,
    "Mi1": 1, "Tm3": 1, "Mi4": 1, "Mi9": -1,
    "T4a": 1, "T4b": 1, "T4c": 1, "T4d": 1,
    "Tm1": -1, "Tm2": -1, "Tm4": -1, "Tm9": -1,
}

# # Gruntman 2018 Fig. 2B location-shared LP fit (gruntman18_fit_lp_shared_location.csv).
# _GRUNTMAN_CENTER_GAIN_MV = 25.731779521277698
# _GRUNTMAN_RF_R1 = (
#     (22.237813729937685 + 14.5102594676567) / 2.0 / _GRUNTMAN_CENTER_GAIN_MV
# )
# _GRUNTMAN_RF_R2 = (
#     (17.38170393943792 + 6.269266224825205) / 2.0 / _GRUNTMAN_CENTER_GAIN_MV
# )
# RF_GRUNTMAN_ROW = np.array(
#     [1.0, _GRUNTMAN_RF_R1, _GRUNTMAN_RF_R2, 0.0, 0.0], dtype=np.float64,
# )

# Repo root: vision/simulation/3_task/1_spot/3_gt.py → parents[4].
_ARENZ_DIR = Path(__file__).resolve().parents[4] / "figure_digitization" / "arenz"
ARENZ_L_DIGITIZED_CSV = _ARENZ_DIR / "L_digitized.csv"
ARENZ_4_DIGITIZED_CSV = _ARENZ_DIR / "4_digitized.csv"


def expand_gt_cells(cells: Sequence[str]) -> Tuple[str, ...]:
    """Validate ``--gt`` spot cell tokens against ``GT_CELLS`` (final keep-set)."""
    if not cells:
        raise ValueError("gt_cells must not be empty")
    out: list = []
    seen: set = set()
    for raw in cells:
        key = str(raw).strip()
        if key not in GT_CELLS:
            valid = ", ".join(GT_CELLS)
            raise ValueError(f"unknown gt cell {key!r} (expected {valid})")
        if key not in seen:
            seen.add(key)
            out.append(key)
    return tuple(out)


# rf axis: Euclidean hex radius r → visual-field deg = r * RF_RADIUS_DEG.
RF_CENTER_RADIUS = 0
RF_N_RADII = 5
RF_RADIUS_DEG = 5.0

# # Legacy DoG params (baked into RF_SCALE below).
# RF_CENTER_WIDTH = {
#     "L1": 6, "L2": 7, "L3": 6, "L4": 8, "L5": 7,
#     "Mi1": 6, "Tm3": 12, "Mi4": 6, "Mi9": 6,
#     "Tm1": 8, "Tm2": 8, "Tm4": 11, "Tm9": 7,
# }
# RF_SURRND_WIDTH = {
#     "L1": 41, "L2": 29, "L3": 15, "L4": 33, "L5": 31,
#     "Mi1": 29, "Tm3": 7, "Mi4": 16, "Mi9": 24,
#     "Tm1": 27, "Tm2": 31, "Tm4": 35, "Tm9": 24,
# }
# RF_SURRND_SCALE = {
#     "L1": 0.06, "L2": 0.065, "L3": 0.95, "L4": 0.23,
#     "L5": 0.175, "Mi1": 0.11, "Tm3": 0.0, "Mi4": 0.66,
#     "Mi9": 0.315, "Tm1": 0.2, "Tm2": 0.175, "Tm4": 0.27,
#     "Tm9": 0.23,
# }

# Peak-normalized unsigned rf(r) at r=0..4 (center=+1); no RF_SIGN.
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


# def _gauss1d_at(fwhm: float, x_deg: float) -> float:
#     """Sum-normalized 1D Gaussian at offset ``x_deg`` from center (legacy grid)."""
#     sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
#     xs = np.arange(-22.0, 23.0, 1.0)
#     total = float(np.sum(np.exp(-xs ** 2 / (2.0 * sigma ** 2))))
#     if total == 0.0:
#         return 0.0
#     return float(np.exp(-float(x_deg) ** 2 / (2.0 * sigma ** 2)) / total)
#
#
# def _rf_dog_at(cell: str, radius: float, *, signed: bool) -> float:
#     """Single-radius DoG sample (before peak normalize)."""
#     x_deg = float(radius) * RF_RADIUS_DEG
#     center = _gauss1d_at(float(RF_CENTER_WIDTH[cell]), x_deg)
#     surrnd = _gauss1d_at(float(RF_SURRND_WIDTH[cell]), x_deg)
#     raw = center - float(RF_SURRND_SCALE[cell]) * surrnd
#     if signed:
#         raw *= float(RF_SIGN[cell])
#     return raw
#
#
# def build_rf_dog(cell: str, *, signed: bool) -> tuple[np.ndarray, float]:
#     """Peak-normalized DoG rf ``(RF_N_RADII,)`` and pre-norm peak scale."""
#     raw_r = np.array(
#         [_rf_dog_at(cell, r, signed=signed) for r in range(RF_N_RADII)],
#         dtype=np.float64,
#     )
#     peak = max(abs(float(np.max(raw_r))), abs(float(np.min(raw_r))))
#     if peak == 0.0:
#         return raw_r * 0.0, 0.0
#     return raw_r / peak, peak
#
#
# def build_rf_gruntman() -> tuple[np.ndarray, float]:
#     """Gruntman 2018 Fig. 2B rf ``(RF_N_RADII,)``; peak scale 1."""
#     return RF_GRUNTMAN_ROW.copy(), 1.0
#
#
# def build_rf_zero() -> tuple[np.ndarray, float]:
#     """Zero rf; peak scale 0."""
#     return np.zeros(RF_N_RADII, dtype=np.float64), 0.0
#
#
# def build_rf_row(cell: str, *, filter: str) -> tuple[np.ndarray, float]:
#     """Dispatch rf source by cell × ``filter``."""
#     if cell not in RF_CENTER_WIDTH:
#         return build_rf_zero() if str(filter) == "ca" else build_rf_gruntman()
#     return build_rf_dog(cell, signed=(str(filter) != "ca"))


def build_rf(cell: str) -> np.ndarray:
    """``RF_SCALE[cell]`` as ``(RF_N_RADII,)``."""
    return np.asarray(RF_SCALE[cell], dtype=np.float64).copy()


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
        result[0] = x[0]
        for i in range(0, n - 1):
            result[i + 1] = result[i] + (dt / tau_ms) * (x[i] - result[i])
    return result.transpose(np.roll(np.arange(result.ndim), -1))


def _bandpass(x, hp_tau_ms, lp_tau_ms, *, delta_ms: float):
    result = _lowpass(x, lp_tau_ms, delta_ms=delta_ms)
    if hp_tau_ms != 0:
        result = result - _lowpass(result, hp_tau_ms, delta_ms=delta_ms)
    return result


def normalize_ir(x):
    """Baseline-subtract and peak-normalize an ir trace."""
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


def build_ir_lti(cell: str, u: np.ndarray, *, delta_ms: float) -> np.ndarray:
    """LTI ir: bandpass/LP on sti drive ``u``, normalize, then ``IR_delay_ms``.

    ``IR_hp_ms[cell] == 0`` → LP-only; else HP+LP bandpass.
    Delay is ``round(IR_delay_ms[cell] / delta_ms)`` samples.
    """
    t_delay = int(round(float(IR_delay_ms[cell]) / float(delta_ms)))
    return _shift_right(
        normalize_ir(
            _bandpass(u, IR_hp_ms[cell], IR_lp_ms[cell], delta_ms=delta_ms)
        ),
        t_delay,
    )


def t_delay_from_ir(
    *,
    delta_ms: float,
    filter: str = "none",
) -> np.ndarray:
    """Per-``GT_CELLS`` onset delay in samples for plot peak windows.

    ``filter=\"ca\"`` → all zeros (Arenz ir has no LTI delay). Otherwise
    ``round(IR_delay_ms[cell] / delta_ms)``.
    """
    n = len(GT_CELLS)
    if str(filter) == "ca":
        return np.zeros(n, dtype=np.int64)
    return np.asarray(
        [round(float(IR_delay_ms[c]) / float(delta_ms)) for c in GT_CELLS],
        dtype=np.int64,
    )


def _arenz_traces_from_csv(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """``cell → (time_s, amplitude)`` from Arenz digitized CSV (``t=0`` = onset)."""
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
        amp = np.asarray([p[1] for p in pairs], dtype=np.float64)
        out[cell] = (t, amp)
    return out


def load_ir_arenz(*, t_onset, n_t, delta_ms: float) -> np.ndarray:
    """Arenz digitized ir ``(len(GT_CELLS), n_t)``; CSV ``t=0`` at ``t_onset``.

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
    traces.update(_arenz_traces_from_csv(ARENZ_L_DIGITIZED_CSV))
    traces.update(_arenz_traces_from_csv(ARENZ_4_DIGITIZED_CSV))
    out = np.zeros((len(GT_CELLS), n_t), dtype=np.float64)
    if t_onset >= n_t:
        return out
    t_rel_s = np.arange(n_t - t_onset, dtype=np.float64) * (delta_ms / 1000.0)
    for i, cell in enumerate(GT_CELLS):
        if cell not in traces:
            continue
        t_csv, amp = traces[cell]
        out[i, t_onset:] = np.interp(t_rel_s, t_csv, amp, left=amp[0], right=amp[-1])
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


def load_rf_ir(*, t_onset=None, n_t=None, ms_sti=None, delta_ms: float, filter="none"):
    """Return ``(rf, ir)`` for ``GT_CELLS``.

    Shapes: ``rf`` ``(n_cell, RF_N_RADII)``; ``ir`` ``(n_cell, n_t)``.

    ``filter=\"none\"``: :func:`build_rf` × :func:`build_ir_lti`.
    ``filter=\"ca\"``: :func:`build_rf` × :func:`load_ir_arenz`.
    """
    if t_onset is None or n_t is None:
        raise ValueError("load_rf_ir requires t_onset and n_t")
    t_onset = int(t_onset)
    n_t = int(n_t)
    delta_ms = float(delta_ms)
    if delta_ms <= 0:
        raise ValueError(f"delta_ms must be > 0, got {delta_ms}")
    n_cells = len(GT_CELLS)
    filter = str(filter)

    rf = np.zeros((n_cells, RF_N_RADII))
    for i, cell in enumerate(GT_CELLS):
        rf[i] = build_rf(cell)

    if filter == "ca":
        return rf, load_ir_arenz(
            t_onset=t_onset, n_t=n_t, delta_ms=delta_ms,
        )

    u = sti_waveform(t_onset, n_t, ms_sti, delta_ms=delta_ms)
    u = u / np.max(u)
    ir = np.zeros((n_cells, n_t))
    for i, cell in enumerate(GT_CELLS):
        ir[i] = build_ir_lti(cell, u, delta_ms=delta_ms)
    return rf, ir


def _gt_from_rf_ir(rf: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """``(n_cells, RF_N_RADII, n_t)`` = rf(r) × ir(t) (no membership gate)."""
    n_t = ir.shape[1]
    n_cells = rf.shape[0]
    gt = np.zeros((n_cells, RF_N_RADII, n_t))
    for i in range(n_cells):
        for radius in range(RF_N_RADII):
            gt[i, radius] = rf[i, radius] * ir[i]
    return gt


def _spot_gt(
    contrast: str,
    *,
    t_onset=None,
    n_t=None,
    ms_sti=None,
    delta_ms: float,
    filter="none",
    spot_gt_mode: str = "all",
) -> np.ndarray:
    """Assembled gt ``(n_cells, RF_N_RADII, n_t)``; inactive rows are zero."""
    rf, ir = load_rf_ir(
        t_onset=t_onset, n_t=n_t, ms_sti=ms_sti, delta_ms=delta_ms, filter=filter,
    )
    gts = _gt_from_rf_ir(rf, ir)
    for i, cell in enumerate(GT_CELLS):
        rf_sign = int(RF_SIGN[cell])
        if not spot_gt_active(spot_gt_mode, contrast, rf_sign):
            gts[i] = 0.0
        else:
            gts[i] *= float(contrast_sign(contrast))
    return gts


def load_gt(
    *,
    t_onset=None,
    n_t=None,
    ms_sti=None,
    delta_ms: float,
    filter="none",
    spot_gt_mode: str = "all",
):
    """Bright gt ``(n_cells, RF_N_RADII, n_t)``."""
    return _spot_gt(
        "bright",
        t_onset=t_onset, n_t=n_t, ms_sti=ms_sti, delta_ms=delta_ms,
        filter=filter, spot_gt_mode=spot_gt_mode,
    )


def load_gt_dark(
    *,
    t_onset=None,
    n_t=None,
    ms_sti=None,
    delta_ms: float,
    filter="none",
    spot_gt_mode: str = "all",
):
    """Dark gt ``(n_cells, RF_N_RADII, n_t)``."""
    return _spot_gt(
        "dark",
        t_onset=t_onset, n_t=n_t, ms_sti=ms_sti, delta_ms=delta_ms,
        filter=filter, spot_gt_mode=spot_gt_mode,
    )


def _rf_at(rf: np.ndarray, radius: float) -> float:
    x = float(radius)
    if x <= 0.0:
        return float(rf[0])
    if x >= RF_N_RADII - 1:
        return float(rf[RF_N_RADII - 1])
    i0 = int(np.floor(x))
    i1 = min(i0 + 1, RF_N_RADII - 1)
    frac = x - i0
    return float((1.0 - frac) * rf[i0] + frac * rf[i1])


def _spot_readout_a_radius(
    rf: np.ndarray,
    radius: float,
    spot_radius: float,
) -> float:
    """rf ``a_radius`` at Euclidean ``radius`` (radius-1 folds r=2 into r=1)."""
    r = round(float(radius), 6)
    if spot_radius_folds_r2_into_r1(spot_radius):
        if r == 1.0:
            return _rf_at(rf, 1.0) + _rf_at(rf, 2.0)
        if r == 2.0:
            return 0.0
    return _rf_at(rf, radius)
