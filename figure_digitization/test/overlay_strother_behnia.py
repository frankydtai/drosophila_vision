"""Fit GCaMP 1st-order LP tau (Arenz-style) and overlay Strother vs Behnia.

Forward model (fit tau):  Ca ≈ g · LP_τ(Vm_Behnia)
Inverse (plot):           freq-domain deconv with high-freq cutoff, scale to mV.

Per cell×polarity, pick Behnia step (0.1→0.5 or 0.5→0.9) with max R².

Outputs
-------
overlay_forward.png   — g·LP(Vm) vs Strother Ca  (validates τ)
overlay_strother_behnia.png — regularized deconv Strother vs Behnia Vm
overlay_strother_behnia_fit.csv

Run:  ../.venv/bin/python overlay_strother_behnia.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ST_CSV = ROOT / "strother" / "2b_digitized.csv"
BH_CSV = ROOT / "behnia" / "ex2_digitized.csv"
OUT_DECONV = HERE / "overlay_strother_behnia.png"
OUT_FORWARD = HERE / "overlay_forward.png"
OUT_CSV = HERE / "overlay_strother_behnia_fit.csv"

DT = 0.025
T_MAX = 4.0
F_CUT_HZ = 1.5  # discard digitization jitter above this (Arenz-like)
SMOOTH_S = 0.40  # savgol window before / after deconv
TAUS_MS = np.unique(
    np.concatenate(
        [
            np.arange(50, 501, 25),
            np.arange(500, 2001, 20),
            np.arange(2000, 5001, 50),
        ]
    )
)
STEPS = [(0.1, 0.5), (0.5, 0.9)]
BRANCH = {"ON": "positive", "OFF": "negative"}
CELLS = ["Mi1", "Tm3"]
POLARITIES = ["ON", "OFF"]
COLORS = {"Mi1": "#c0392b", "Tm3": "#d4a017"}


def resample(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    order = np.argsort(t)
    t, y = t[order], y[order]
    _, idx = np.unique(t, return_index=True)
    t, y = t[idx], y[idx]
    return np.interp(t_grid, t, y, left=np.nan, right=np.nan)


def lowpass(x: np.ndarray, tau_s: float, dt: float) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    a = dt / tau_s
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


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


def baseline0(y: np.ndarray) -> np.ndarray:
    i0 = 0 if np.isfinite(y[0]) else int(np.flatnonzero(np.isfinite(y))[0])
    return y - y[i0]


def smooth(y: np.ndarray, dt: float, win_s: float = SMOOTH_S) -> np.ndarray:
    n = len(y)
    win = max(5, int(round(win_s / dt)) | 1)
    win = min(win, n if n % 2 == 1 else n - 1)
    return savgol_filter(y, window_length=win, polyorder=2, mode="interp")


def deconv_lp(ca: np.ndarray, tau_s: float, dt: float, f_cut: float = F_CUT_HZ) -> np.ndarray:
    """Freq-domain inverse of 1st-order LP with high-freq cutoff."""
    ca_s = smooth(ca, dt)
    n = len(ca_s)
    freqs = np.fft.rfftfreq(n, d=dt)
    # H_lp(f) = 1 / (1 + j 2π f τ)  →  divide Ca by H = multiply by (1 + j 2π f τ)
    H_inv = 1.0 + 1j * 2.0 * np.pi * freqs * tau_s
    Ca_f = np.fft.rfft(ca_s)
    V_f = Ca_f * H_inv
    V_f[freqs > f_cut] = 0.0
    return smooth(np.fft.irfft(V_f, n=n), dt)


def load_pair(
    st: pd.DataFrame,
    bh: pd.DataFrame,
    cell: str,
    pol: str,
    step: tuple[float, float],
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    f, to = step
    st_sub = st[(st.cell == cell) & (st.polarity == pol)]
    ca = baseline0(resample(st_sub.time_s.values, st_sub.dff.values, t_grid))
    bh_sub = bh[
        (bh.cell == cell)
        & np.isclose(bh.intensity_from, f)
        & np.isclose(bh.intensity_to, to)
        & (bh.branch == BRANCH[pol])
    ]
    vm = baseline0(resample(bh_sub.time_s.values, bh_sub.vm_mv.values, t_grid))
    return ca, vm


def fit_all(st: pd.DataFrame, bh: pd.DataFrame, t_grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for cell in CELLS:
        for pol in POLARITIES:
            best = None
            for step in STEPS:
                ca, vm = load_pair(st, bh, cell, pol, step, t_grid)
                for tau_ms in TAUS_MS:
                    y = lowpass(np.nan_to_num(vm, nan=0.0), tau_ms / 1000.0, DT)
                    y = np.where(np.isfinite(vm), y, np.nan)
                    r2, g = r2_g(y, ca)
                    rec = dict(
                        cell=cell,
                        polarity=pol,
                        step=f"{step[0]}->{step[1]}",
                        branch=BRANCH[pol],
                        tau_ms=int(tau_ms),
                        R2=r2,
                        g_forward=g,
                    )
                    if best is None or r2 > best["R2"]:
                        best = rec
            rows.append(best)
    return pd.DataFrame(rows)


def _panel_title(row: pd.Series, r2_ov: float, g: float, extra: str = "") -> str:
    return (
        f"{row.cell} {row.polarity}  step {row.step}  τ={row.tau_ms} ms\n"
        f"forward R²={row.R2:.3f}  overlay R²={r2_ov:.3f}  g={g:.3g}{extra}"
    )


def plot_forward(fits: pd.DataFrame, st: pd.DataFrame, bh: pd.DataFrame, path: Path) -> None:
    t_grid = np.arange(0.0, T_MAX + 1e-9, DT)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for ax, (_, row) in zip(axes.ravel(), fits.iterrows()):
        f, to = map(float, row.step.split("->"))
        ca, vm = load_pair(st, bh, row.cell, row.polarity, (f, to), t_grid)
        y = lowpass(np.nan_to_num(vm, nan=0.0), row.tau_ms / 1000.0, DT)
        y = np.where(np.isfinite(vm), y, np.nan)
        r2, g = r2_g(y, ca)
        pred = g * y
        color = COLORS[row.cell]
        ax.plot(t_grid, ca, color=color, lw=1.6, label="Strother ΔF/F")
        ax.plot(t_grid, pred, color="0.25", lw=1.4, ls="--", label=r"$g\cdot$LP$_\tau$(Behnia)")
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axvline(0, color="0.75", lw=0.6)
        ax.set_title(_panel_title(row, r2, g), fontsize=9)
        ax.set_ylabel("ΔF/F")
        ax.legend(fontsize=8, loc="best")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Forward check: LP(Behnia Vm) vs Strother Ca", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_deconv(fits: pd.DataFrame, st: pd.DataFrame, bh: pd.DataFrame, path: Path) -> None:
    t_grid = np.arange(0.0, T_MAX + 1e-9, DT)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for ax, (_, row) in zip(axes.ravel(), fits.iterrows()):
        f, to = map(float, row.step.split("->"))
        ca, vm = load_pair(st, bh, row.cell, row.polarity, (f, to), t_grid)
        v_hat = deconv_lp(np.nan_to_num(ca, nan=0.0), row.tau_ms / 1000.0, DT)
        v_hat = np.where(np.isfinite(ca), v_hat, np.nan)
        r2_ov, g_inv = r2_g(v_hat, vm)
        v_scaled = g_inv * v_hat
        color = COLORS[row.cell]
        ax.plot(t_grid, vm, color=color, lw=1.6, label="Behnia Vm")
        ax.plot(
            t_grid,
            v_scaled,
            color="0.25",
            lw=1.4,
            ls="--",
            label=rf"Strother deconv (f$<$ {F_CUT_HZ:g} Hz)",
        )
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axvline(0, color="0.75", lw=0.6)
        ax.set_title(_panel_title(row, r2_ov, g_inv), fontsize=9)
        ax.set_ylabel("Vm (mV)")
        ax.legend(fontsize=8, loc="best")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(
        "Regularized GCaMP deconv (Strother) vs Behnia Vm",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    st = pd.read_csv(ST_CSV)
    bh = pd.read_csv(BH_CSV)
    t_grid = np.arange(0.0, T_MAX + 1e-9, DT)
    fits = fit_all(st, bh, t_grid)
    fits.to_csv(OUT_CSV, index=False)
    plot_forward(fits, st, bh, OUT_FORWARD)
    plot_deconv(fits, st, bh, OUT_DECONV)
    print(fits.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_FORWARD}")
    print(f"wrote {OUT_DECONV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
