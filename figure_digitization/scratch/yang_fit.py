"""Fit GCaMP forward LP on Yang extrap V vs digitized Ca.

Two outputs from one run:
  yang_fit.png          g·LP_τ(rect(V+bias))
  yang_fit_no_bias.png  g·LP_τ(V) — no bias, no rect

Run:  ../.venv/bin/python yang_fit.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yang_extrap import (
    COLORS,
    DT,
    ORDER_B,
    ORDER_E,
    TAIL450_MS,
    YANG_CSV,
    baseline0,
    build_final_v,
    resample,
)
from yang_lp import lowpass, lp_rect_bias

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "yang_fit.png"
OUT_NO_BIAS_PNG = HERE / "yang_fit_no_bias.png"

TAUS_MS = np.unique(
    np.concatenate(
        [
            np.arange(50, 501, 25),
            np.arange(500, 2001, 20),
            np.arange(2000, 5001, 50),
        ]
    )
)
BIAS_GRID = np.arange(-5.0, 5.01, 0.25)


def r2_g(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(pred) & np.isfinite(target)
    p, t = pred[m], target[m]
    if len(p) < 5 or np.allclose(p, 0):
        return -np.inf, np.nan
    g = float(np.dot(p, t) / np.dot(p, p))
    resid = t - g * p
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot <= 0:
        return -np.inf, g
    return 1.0 - float(np.sum(resid**2)) / ss_tot, g


def load_pair(
    df: pd.DataFrame, cell: str, stimulus: str, t_grid: np.ndarray, tail_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict]:
    v_sub = df[(df.cell == cell) & (df.modality == "voltage") & (df.stimulus == stimulus)]
    c_sub = df[(df.cell == cell) & (df.modality == "calcium") & (df.stimulus == stimulus)]
    vm = baseline0(resample(v_sub.time_s.values, v_sub.dff_percent.values, t_grid))
    v_ext, meta = build_final_v(vm, t_grid, tail_s)
    i0 = int(np.flatnonzero(np.isfinite(v_ext))[0])
    v_fill = np.nan_to_num(v_ext, nan=0.0)
    ca = baseline0(resample(c_sub.time_s.values, c_sub.dff_percent.values, t_grid))
    ca = np.where(np.isfinite(ca), ca, np.nan)
    return vm, v_ext, v_fill, ca, i0, meta


def lp_v(v_fill: np.ndarray, tau_s: float, i0: int) -> np.ndarray:
    lp0 = lowpass(v_fill, tau_s, DT)
    return lp0 - lp0[i0]


def fit_rect(
    v_ext: np.ndarray, v_fill: np.ndarray, ca: np.ndarray, i0: int
) -> dict:
    mask = np.isfinite(v_ext) & np.isfinite(ca)
    best = None
    for tau_ms in TAUS_MS:
        tau_s = tau_ms / 1000.0
        for bias in BIAS_GRID:
            p = lp_rect_bias(v_fill, float(bias), tau_s, DT, i0)
            r2, g = r2_g(p[mask], ca[mask])
            if best is None or r2 > best["R2"]:
                best = dict(tau_ms=int(tau_ms), bias=float(bias), R2=float(r2), g=float(g))
    return best


def fit_lp(v_ext: np.ndarray, v_fill: np.ndarray, ca: np.ndarray, i0: int) -> dict:
    mask = np.isfinite(v_ext) & np.isfinite(ca)
    best = None
    for tau_ms in TAUS_MS:
        p = lp_v(v_fill, tau_ms / 1000.0, i0)
        r2, g = r2_g(p[mask], ca[mask])
        if best is None or r2 > best["R2"]:
            best = dict(tau_ms=int(tau_ms), R2=float(r2), g=float(g))
    return best


def plot_panels(
    fits: dict[tuple[str, str], dict],
    df: pd.DataFrame,
    t_grid: np.ndarray,
    tail_s: float,
    path: Path,
    *,
    rect: bool,
    title: str,
    pred_label: str,
) -> None:
    fig, axes = plt.subplots(6, 2, figsize=(9, 14), sharex=True)
    for r, cell in enumerate(ORDER_B + ORDER_E):
        for c, stimulus in enumerate(["light", "dark"]):
            ax = axes[r, c]
            row = fits[(cell, stimulus)]
            _, v_ext, v_fill, ca, i0, meta = load_pair(df, cell, stimulus, t_grid, tail_s)
            tau_s = row["tau_ms"] / 1000.0
            if rect:
                p = lp_rect_bias(v_fill, row["bias"], tau_s, DT, i0)
            else:
                p = lp_v(v_fill, tau_s, i0)
            pred = row["g"] * p
            ext_m = np.isfinite(v_ext)
            color = COLORS[cell]
            ax.plot(t_grid, ca, color=color, lw=1.6, label="Ca digitize")
            ax.plot(
                t_grid[ext_m], v_ext[ext_m], color="0.55", lw=1.2, ls="--", label="V extrap"
            )
            ax.plot(t_grid, pred, color="0.1", lw=1.6, ls="--", label=pred_label)
            ax.axvline(meta["t_peak"], color="0.35", lw=0.5, ls="-.")
            ax.axvline(meta["t_end"], color="0.45", lw=0.5, ls="--")
            ax.axhline(0, color="0.75", lw=0.5)
            if r == 0:
                ax.set_title(stimulus, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{cell}\nΔF/F %", fontsize=9)
            if r == 5:
                ax.set_xlabel("time (s)", fontsize=9)
            ann = f"τ={row['tau_ms']} ms  R²={row['R2']:.3f}  g={row['g']:.3g}"
            if rect:
                ann = f"τ={row['tau_ms']} ms  bias={row['bias']:+.2g}  R²={row['R2']:.3f}  g={row['g']:.3g}"
            ax.text(0.02, 0.98, ann, transform=ax.transAxes, va="top", fontsize=7)
            if r == 0 and c == 1:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    df = pd.read_csv(YANG_CSV)
    tail_s = TAIL450_MS / 1000.0
    t_max = float(df[df.modality == "calcium"].time_s.max()) + DT
    t_grid = np.arange(0.0, t_max + 1e-9, DT)
    fits_rect: dict[tuple[str, str], dict] = {}
    fits_lp: dict[tuple[str, str], dict] = {}
    for cell in ORDER_B + ORDER_E:
        for stimulus in ["light", "dark"]:
            _, v_ext, v_fill, ca, i0, _ = load_pair(df, cell, stimulus, t_grid, tail_s)
            key = (cell, stimulus)
            fits_rect[key] = fit_rect(v_ext, v_fill, ca, i0)
            fits_lp[key] = fit_lp(v_ext, v_fill, ca, i0)
    plot_panels(
        fits_rect,
        df,
        t_grid,
        tail_s,
        OUT_PNG,
        rect=True,
        title="Yang S5: g·LP_τ(rect(V_extrap+bias)) vs Ca — fit full span",
        pred_label="g·LP rect(V+bias)",
    )
    plot_panels(
        fits_lp,
        df,
        t_grid,
        tail_s,
        OUT_NO_BIAS_PNG,
        rect=False,
        title="Yang S5: g·LP_τ(V_extrap) vs Ca — fit full span (no bias, no rect)",
        pred_label="g·LP(V)",
    )
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_NO_BIAS_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
