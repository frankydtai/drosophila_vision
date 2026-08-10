"""Arenz-style GCaMP deconv on Strother Mi1/Tm3 — clean Ca then deconv.

Why raw deconv was garbage
--------------------------
2b.py digitizes the SEM ribbon midline (median colour mask per x). That jumps
~1–1.5 px/sample (≈0.03–0.05 ΔF/F). With τ=0.5 s, τ·ΔCa/dt from one pixel
jump ≈ 0.4 ΔF/F → fake oscillations. The digitize is too jagged to differentiate.

Fix: low-pass the digitize to a paper-smooth mean (Butterworth), then
V̂ = Ca + τ dCa/dt. Abort if deconv still has digitization-scale curvature.

τ=500 ms (Arenz ≈350 ms via ÷H_LP on clean kernels).

Run:  ../.venv/bin/python filter_strother_500ms.py
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
ST_CSV = HERE.parent / "strother" / "2b_digitized.csv"
OUT_PNG = HERE / "filter_strother_500ms.png"
OUT_CSV = HERE / "filter_strother_500ms.csv"

DT = 0.025
T_MAX = 4.0
TAU_S = 0.5
FC_HZ = 0.5  # keep flash kinetics; kill px jitter (~1/0.04 s = 25 Hz)
SAVGOL_S = 0.50
DFF_PER_PX = 2.0 / 65.0
MAX_DECONV_D2_PX = 0.15  # interior only
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


def deconv_lp(ca: np.ndarray, tau_s: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(ca)
    win = max(5, int(round(SAVGOL_S / DT)) | 1)
    win = min(win, n if n % 2 == 1 else n - 1)
    ca_s = savgol_filter(ca, win, 2, mode="interp")
    dca = savgol_filter(ca, win, 2, deriv=1, delta=DT, mode="interp")
    return ca_s, ca_s + tau_s * dca


def d2_rms_px(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.diff(y, 2) ** 2)) / DFF_PER_PX)


def main() -> int:
    st = pd.read_csv(ST_CSV)
    t_grid = np.arange(0.0, T_MAX + 1e-9, DT)
    interior = (t_grid >= 0.25) & (t_grid <= 3.75)
    rows: list[dict] = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    failed = False

    for ax, (cell, pol) in zip(
        axes.ravel(), ((c, p) for c in CELLS for p in POLARITIES)
    ):
        sub = st[(st.cell == cell) & (st.polarity == pol)]
        raw = resample(sub.time_s.values, sub.dff.values, t_grid)
        raw = np.nan_to_num(raw, nan=0.0) - float(np.nan_to_num(raw, nan=0.0)[0])

        ca0 = clean_ca(raw)
        ca_s, deconv = deconv_lp(ca0, TAU_S)
        ca_s = ca_s - ca_s[0]
        deconv = deconv - deconv[0]

        raw_d2 = d2_rms_px(raw[interior])
        dec_d2 = d2_rms_px(deconv[interior])
        resid_px = float(np.sqrt(np.mean((raw - ca_s) ** 2)) / DFF_PER_PX)
        print(
            f"{cell} {pol}: raw Δ²={raw_d2:.2f}px  deconv Δ²={dec_d2:.2f}px  "
            f"resid(raw−Ca)={resid_px:.2f}px"
        )
        if dec_d2 > MAX_DECONV_D2_PX:
            print(f"  FAIL: deconv Δ²>{MAX_DECONV_D2_PX} px")
            failed = True

        for ti, r, c, d in zip(t_grid, raw, ca_s, deconv):
            rows.append(
                dict(
                    cell=cell,
                    polarity=pol,
                    time_s=float(ti),
                    dff_raw=float(r),
                    dff_ca=float(c),
                    dff_deconv=float(d),
                )
            )

        color = COLORS[cell]
        ax.plot(t_grid, raw, color=color, alpha=0.25, lw=0.9, label="raw digitize")
        ax.plot(t_grid, ca_s, color=color, lw=1.4, ls="--", label=f"Ca (LP {FC_HZ} Hz)")
        ax.plot(t_grid, deconv, color="0.1", lw=2.0, label=rf"deconv τ={TAU_S*1e3:.0f} ms")
        ax.axhline(0, color="0.75", lw=0.6)
        ax.set_title(f"{cell} {pol}", fontsize=11)
        ax.set_ylabel("ΔF/F")
        ax.legend(fontsize=8, loc="best")

    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(
        "Strother: clean digitize (Butterworth) → GCaMP deconv — no raw derivative",
        y=0.98,
    )
    fig.tight_layout()

    if failed:
        bad = HERE / "filter_strother_500ms_FAILED.png"
        fig.savefig(bad, dpi=140)
        plt.close(fig)
        raise SystemExit(f"QC failed; see {bad}")

    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
