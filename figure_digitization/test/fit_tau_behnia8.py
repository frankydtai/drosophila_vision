"""Fit GCaMP deconv tau: Behnia Vm ≈ g · deconv_τ(Strother Ca).

Fit window: t ∈ [0, 1] s only (full traces still plotted).
Strother Ca is Butterworth-cleaned, then Arenz inverse-LP:
  V̂ = Ca + τ · dCa/dt
Eight panels = Mi1/Tm3 × ON/OFF × two Behnia amplitude steps.

Run:  ../.venv/bin/python fit_tau_behnia8.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ST_CSV = ROOT / "strother" / "2b_digitized.csv"
BH_CSV = ROOT / "behnia" / "ex2_digitized.csv"
OUT_CSV = HERE / "fit_tau_behnia8.csv"
OUT_PNG = HERE / "fit_tau_behnia8.png"
OUT_PNG_350 = HERE / "fit_tau_behnia8_tau350.png"
TAU_ARENZ_MS = 350

DT = 0.025
T_MAX = 4.0
T_FIT = 1.0  # fit tau only on t ∈ [0, T_FIT]
FC_HZ = 0.5
SAVGOL_S = 0.50
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


def clean_ca(raw: np.ndarray) -> np.ndarray:
    b, a = butter(2, FC_HZ, btype="low", fs=1.0 / DT)
    return filtfilt(b, a, raw)


def deconv_gcamp(ca: np.ndarray, tau_s: float) -> np.ndarray:
    """V̂ = Ca + τ dCa/dt (savgol value + derivative on cleaned Ca)."""
    n = len(ca)
    win = max(5, int(round(SAVGOL_S / DT)) | 1)
    win = min(win, n if n % 2 == 1 else n - 1)
    ca_s = savgol_filter(ca, win, 2, mode="interp")
    dca = savgol_filter(ca, win, 2, deriv=1, delta=DT, mode="interp")
    return ca_s + tau_s * dca


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


def load_strother_ca(st: pd.DataFrame, cell: str, pol: str, t_grid: np.ndarray) -> np.ndarray:
    sub = st[(st.cell == cell) & (st.polarity == pol)]
    raw = resample(sub.time_s.values, sub.dff.values, t_grid)
    raw = np.nan_to_num(raw, nan=0.0)
    raw = raw - raw[0]
    ca = clean_ca(raw)
    return ca - ca[0]


def load_behnia_vm(
    bh: pd.DataFrame, cell: str, pol: str, step: tuple[float, float], t_grid: np.ndarray
) -> np.ndarray:
    f, to = step
    sub = bh[
        (bh.cell == cell)
        & np.isclose(bh.intensity_from, f)
        & np.isclose(bh.intensity_to, to)
        & (bh.branch == BRANCH[pol])
    ]
    vm = resample(sub.time_s.values, sub.vm_mv.values, t_grid)
    vm = np.nan_to_num(vm, nan=0.0)
    return vm - vm[0]


def fit_one(ca: np.ndarray, vm: np.ndarray, t_grid: np.ndarray) -> dict:
    """Deconv full traces; score R²/g only on t ≤ T_FIT."""
    mask = t_grid <= T_FIT + 1e-12
    best = None
    for tau_ms in TAUS_MS:
        v_hat = deconv_gcamp(ca, tau_ms / 1000.0)
        v_hat = v_hat - v_hat[0]
        r2, g = r2_g(v_hat[mask], vm[mask])
        if best is None or r2 > best["R2"]:
            best = dict(tau_ms=int(tau_ms), R2=float(r2), g=float(g))
    return best


def fit_fixed_tau(
    ca: np.ndarray, vm: np.ndarray, t_grid: np.ndarray, tau_ms: int
) -> dict:
    mask = t_grid <= T_FIT + 1e-12
    v_hat = deconv_gcamp(ca, tau_ms / 1000.0)
    v_hat = v_hat - v_hat[0]
    r2, g = r2_g(v_hat[mask], vm[mask])
    return dict(tau_ms=int(tau_ms), R2=float(r2), g=float(g))


def plot_panels(
    df: pd.DataFrame,
    st: pd.DataFrame,
    bh: pd.DataFrame,
    t_grid: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(9, 12), sharex=True)
    for ax, row in zip(axes.ravel(), df.itertuples(index=False)):
        ca = load_strother_ca(st, row.cell, row.polarity, t_grid)
        f, to = map(float, row.step.split("->"))
        vm = load_behnia_vm(bh, row.cell, row.polarity, (f, to), t_grid)
        v_hat = deconv_gcamp(ca, row.tau_ms / 1000.0)
        v_hat = v_hat - v_hat[0]
        before = row.g * ca
        after = row.g * v_hat
        color = COLORS[row.cell]
        ax.plot(t_grid, vm, color=color, lw=1.8, label="Behnia Vm")
        ax.plot(
            t_grid,
            before,
            color=color,
            lw=1.2,
            ls=":",
            alpha=0.85,
            label="Strother before filter (×g)",
        )
        ax.plot(
            t_grid,
            after,
            color="0.1",
            lw=1.6,
            ls="--",
            label="Strother after filter (×g)",
        )
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axvline(T_FIT, color="0.6", lw=0.8, ls=":")
        ax.set_title(
            f"{row.cell} {row.polarity}  {row.step}\n"
            f"τ = {row.tau_ms} ms    R²(0–{T_FIT:g}s) = {row.R2:.3f}    g = {row.g:.3g}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel("Vm (mV)")
        ax.legend(fontsize=7, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    st = pd.read_csv(ST_CSV)
    bh = pd.read_csv(BH_CSV)
    t_grid = np.arange(0.0, T_MAX + 1e-9, DT)

    rows = []
    rows_350 = []
    for cell in CELLS:
        for pol in POLARITIES:
            ca = load_strother_ca(st, cell, pol, t_grid)
            for step in STEPS:
                vm = load_behnia_vm(bh, cell, pol, step, t_grid)
                meta = dict(
                    cell=cell,
                    polarity=pol,
                    step=f"{step[0]}->{step[1]}",
                    branch=BRANCH[pol],
                )
                rows.append({**meta, **fit_one(ca, vm, t_grid)})
                rows_350.append(
                    {**meta, **fit_fixed_tau(ca, vm, t_grid, TAU_ARENZ_MS)}
                )

    df = pd.DataFrame(rows)
    df_350 = pd.DataFrame(rows_350)
    df.to_csv(OUT_CSV, index=False)
    print("=== free τ* (fit 0–1 s) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\n=== fixed τ = {TAU_ARENZ_MS} ms (g fit 0–1 s) ===")
    print(df_350.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    plot_panels(
        df,
        st,
        bh,
        t_grid,
        OUT_PNG,
        f"Strother before/after deconv vs Behnia — free τ*, fit 0–{T_FIT:g} s",
    )
    plot_panels(
        df_350,
        st,
        bh,
        t_grid,
        OUT_PNG_350,
        f"Strother before/after deconv vs Behnia — fixed τ = {TAU_ARENZ_MS} ms "
        f"(Arenz), g fit 0–{T_FIT:g} s",
    )

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PNG_350}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
