"""Compare old vs new spot RecF spatial normalize formulas.

Old (45-sample profile, current ``task.spot.gt``):

    center = Gauss1D(FWHM_center, 44)
    surrnd = Gauss1D(FWHM_surrnd, 44)
    raw(x) = (center(x) - scale * surrnd(x)) * RF_SIGN     x = -22..+22 deg
    RecF_old = normalize_gt(raw)                           45 samples
    RecF_old(r) = RecF_old[22 + 5*r]                       integer cost radius r
    RecF_old(sqrt3) = interp(22 + 5*sqrt3, RecF_old)       fractional radius

New (5 radii, peak normalize — no subtract):

    raw_r = raw(5*r) for r = 0..4
    scale = max(|raw_0|, ..., |raw_4|)
    RecF_new(r) = raw_r / scale
    RecF_new(sqrt3) = raw(5*sqrt3) / scale               same scale as r=0..4

No x -= x[0]; center stays ±1 when it is the 5-point peak.

Usage (from ``simulation/``):

    ../.venv/bin/python test/plot_recf_normalize_compare.py
    ../.venv/bin/python test/plot_recf_normalize_compare.py --show
"""
from __future__ import annotations

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from figure.panel import save_figure
from task.spot.gt import (
    GT_CELLS,
    RF_N_RADII,
    RF_RADIUS_DEG,
    RF_SIGN,
    _RF_CENTER_SAMPLE,
    _RF_NSAMPLES,
    _RF_SAMPLES_PER_COL,
    _gauss1d,
    normalize_gt,
)

DEFAULT_SAVE = os.path.join(HERE, "recf_normalize_compare.png")

RF_CENTER_W = np.array([6, 7, 6, 8, 7, 6, 12, 6, 6, 8, 8, 11, 7])
RF_SURRND_W = np.array([41, 29, 15, 33, 31, 29, 7, 16, 24, 27, 31, 35, 24])
RF_SURRND_SCALE = np.array(
    [0.012, 0.013, 0.19, 0.046, 0.035, 0.022, 0.000, 0.132, 0.063, 0.040, 0.035, 0.054, 0.046]
) * 5.0


def raw_signed_dog_row(cell_idx: int) -> np.ndarray:
    center = _gauss1d(RF_CENTER_W[cell_idx], 44)
    surrnd = _gauss1d(RF_SURRND_W[cell_idx], 44)
    return (center - RF_SURRND_SCALE[cell_idx] * surrnd) * float(RF_SIGN[cell_idx])


def raw_at_radius(raw_row: np.ndarray, radius: float) -> float:
    sample = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * radius
    sample = min(max(sample, 0.0), _RF_NSAMPLES - 1)
    return float(np.interp(sample, np.arange(_RF_NSAMPLES), raw_row))


def recf_new_row(cell_idx: int) -> tuple[np.ndarray, float]:
    """Peak-normalized RecF at r=0..4 and shared scale for sqrt3."""
    raw = raw_signed_dog_row(cell_idx)
    five = np.array([raw_at_radius(raw, r) for r in range(RF_N_RADII)])
    scale = max(abs(float(np.nanmax(five))), abs(float(np.nanmin(five))))
    if scale == 0.0:
        return five * 0.0, 0.0
    return five / scale, scale


def recf_old_at_radius(recf_row: np.ndarray, radius: float) -> float:
    sample = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * radius
    sample = min(max(sample, 0.0), _RF_NSAMPLES - 1)
    return float(np.interp(sample, np.arange(_RF_NSAMPLES), recf_row))


def plot_compare(*, save_path: str, show: bool) -> None:
    radius_x = np.arange(RF_N_RADII, dtype=float) * RF_RADIUS_DEG
    sqrt3 = math.sqrt(3.0)
    sqrt3_x = sqrt3 * RF_RADIUS_DEG

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Spot RecF: old 45-pt normalize_gt vs new 5-r peak normalize (no subtract)",
        fontsize=14,
        y=0.98,
    )

    ax_formula = fig.add_axes([0.05, 0.72, 0.90, 0.22])
    ax_formula.axis("off")
    ax_formula.text(
        0.0,
        1.0,
        (
            "Shared raw DoG (per cell):\n"
            "  raw(x) = (Gauss1D(FWHM_center,44)(x) - scale*Gauss1D(FWHM_surrnd,44)(x)) * RF_SIGN\n"
            "  normalize_gt(y) = (y - y[0]) / max(|y|)\n\n"
            "OLD (current):  RecF_old = normalize_gt(raw) on x = -22..+22 (45 samples)\n"
            "                RecF_old(r) = RecF_old[22 + 5*r];  sqrt3 via interp at 22+5*sqrt3\n\n"
            "NEW (5 radii):  raw_r = raw(5*r) for r=0..4;  scale = max(|raw_r|)\n"
            "                RecF_new(r) = raw_r / scale   (no subtract)\n"
            "                RecF_new(sqrt3) = raw(5*sqrt3) / scale"
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    axes = fig.subplots(4, 4).flat
    for cell_idx, (ax, cell) in enumerate(zip(axes, GT_CELLS)):
        raw = raw_signed_dog_row(cell_idx)
        old_row = normalize_gt(raw)
        new_row, scale = recf_new_row(cell_idx)

        deg = np.arange(_RF_NSAMPLES) - _RF_CENTER_SAMPLE
        pos = deg >= 0
        ax.plot(deg[pos], old_row[pos], color="#2474b5", lw=1.5, label="old 45-pt")

        old_mark = np.array([recf_old_at_radius(old_row, r) for r in range(RF_N_RADII)])
        ax.plot(radius_x, old_mark, "o", color="#2474b5", ms=5, label="old @ r=0..4")

        ax.plot(radius_x, new_row, "s", color="#c0392b", ms=5, label="new peak 5-r")

        old_sqrt3 = recf_old_at_radius(old_row, sqrt3)
        new_sqrt3 = raw_at_radius(raw, sqrt3) / scale if scale > 0.0 else 0.0
        ax.axvline(sqrt3_x, color="0.75", lw=0.8, ls=":")
        ax.plot([sqrt3_x], [old_sqrt3], "^", color="#2474b5", ms=6)
        ax.plot([sqrt3_x], [new_sqrt3], "v", color="#c0392b", ms=6)
        ax.annotate(
            f"sqrt3 old={old_sqrt3:.3f} new={new_sqrt3:.3f}",
            (sqrt3_x, old_sqrt3),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="0.25",
        )

        max_diff = float(np.max(np.abs(new_row - old_mark)))
        ax.set_title(f"{cell}  max|r=0..4 diff|={max_diff:.3f}", fontsize=9)
        ax.set_xlim(-1, 21)
        ax.axhline(0.0, color="0.85", lw=0.6)
        ax.axvline(0.0, color="0.85", lw=0.6)
        if cell_idx % 4 == 0:
            ax.set_ylabel("RecF")
        if cell_idx >= 12:
            ax.set_xlabel("deg from center")
        if cell_idx == 3:
            ax.legend(fontsize=7, loc="upper right")

    for ax in axes[len(GT_CELLS):]:
        ax.axis("off")

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.68, hspace=0.45, wspace=0.28)
    save_figure(fig, save_path)
    print(f"wrote {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", default=DEFAULT_SAVE)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_compare(save_path=args.save, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
