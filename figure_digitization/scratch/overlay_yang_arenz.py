"""Overlay Yang S5 calcium with Arenz kernels (same cell per panel).

Yang light solid; Yang dark dashed; Arenz solid.
Arenz sign-flipped so the peak is positive, then scaled to Yang positive peak.

Outputs
-------
    overlay_yang_arenz.png

Run:  ../.venv/bin/python overlay_yang_arenz.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
YANG_CSV = ROOT / "yang" / "s5be_digitized.csv"
ARENZ_4_CSV = ROOT / "arenz" / "4_digitized.csv"
ARENZ_L_CSV = ROOT / "arenz" / "L_digitized.csv"
OUT_PNG = HERE / "overlay_yang_arenz.png"

ORDER_B = ["L1", "Tm3", "Mi1"]
ORDER_E = ["L2", "Tm1", "Tm2"]
CELLS = ORDER_B + ORDER_E
ARENZ_FROM_L = {"L1", "L2"}

COLORS = {
    "L1": "#2a5db0",
    "L2": "#2a5db0",
    "Tm3": "#d62728",
    "Tm1": "#d62728",
    "Mi1": "#2ca02c",
    "Tm2": "#2ca02c",
}


def load_yang_ca(df: pd.DataFrame, cell: str, stimulus: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[(df.cell == cell) & (df.modality == "calcium") & (df.stimulus == stimulus)]
    return sub.time_s.to_numpy(float), sub.dff_percent.to_numpy(float)


def load_arenz(a4: pd.DataFrame, aL: pd.DataFrame, cell: str) -> tuple[np.ndarray, np.ndarray]:
    src = aL if cell in ARENZ_FROM_L else a4
    sub = src[src.cell == cell]
    return sub.time_s.to_numpy(float), sub.amplitude.to_numpy(float)


def arenz_positive_peak_aligned(
    arenz_amp: np.ndarray, yang_light: np.ndarray, yang_dark: np.ndarray
) -> tuple[np.ndarray, float]:
    """Flip Arenz so peak > 0, scale max to Yang positive peak (light/dark)."""
    a = np.asarray(arenz_amp, float)
    if float(np.nanmax(a)) < -float(np.nanmin(a)):
        a = -a
    yang_peak = max(float(np.nanmax(yang_light)), float(np.nanmax(yang_dark)))
    a_peak = float(np.nanmax(a))
    scale = yang_peak / a_peak if a_peak > 0 else np.nan
    return a * scale, scale


def plot_overlay(yang: pd.DataFrame, a4: pd.DataFrame, aL: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True)
    for ax, cell in zip(axes.ravel(), CELLS):
        t_on, y_on = load_yang_ca(yang, cell, "light")
        t_dark, y_dark = load_yang_ca(yang, cell, "dark")
        t_a, a = load_arenz(a4, aL, cell)
        a_scaled, scale = arenz_positive_peak_aligned(a, y_on, y_dark)
        color = COLORS[cell]
        ax.plot(t_on, y_on, color=color, lw=1.5, ls="-", label="Yang light")
        ax.plot(t_dark, y_dark, color=color, lw=1.3, ls="--", label="Yang dark")
        ax.plot(t_a, a_scaled, color="0.2", lw=1.4, ls="-", label="Arenz (+peak)")
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axvline(0, color="0.75", lw=0.6)
        ax.set_title(f"{cell}  Arenz ×{scale:.3g} → Yang +peak", fontsize=10)
        ax.set_ylabel("ΔF/F (%)")
        ax.legend(fontsize=7, loc="best")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(
        "Yang S5 Ca vs Arenz (Arenz flipped +peak → Yang positive peak)",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    yang = pd.read_csv(YANG_CSV)
    a4 = pd.read_csv(ARENZ_4_CSV)
    aL = pd.read_csv(ARENZ_L_CSV)
    plot_overlay(yang, a4, aL, OUT_PNG)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
