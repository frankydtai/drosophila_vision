"""Overlay Yang 2016 S5 voltage-ON traces with spot GT ImpR (25 ms).

Six panels (L1, Tm3, Mi1, L2, Tm1, Tm2). Left axis: digitized ΔF/F %;
right axis: ``read_RecF_ImpR(..., ms_spot=25)`` × ``RF_sign`` from
``task.spot.gt`` (L1/L2/Tm1/Tm2 flip; Mi1/Tm3 do not).

Usage (from ``simulation/``):

    ../.venv/bin/python test/plot_yang_s5be_impr.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import import_bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MODEL

DELTA_MS = float(MODEL["delta_ms"])
from task.spot.gt import GT_CELLS, read_RecF_ImpR

REPO = ROOT.parent.parent
YANG_CSV = REPO / "figure_digitization" / "yang" / "s5be_digitized.csv"
OUT_PNG = HERE / "yang_s5be_impr.png"

CELLS = ("L1", "Tm3", "Mi1", "L2", "Tm1", "Tm2")
MS_SPOT = 25.0
MS_PRE = 50.0
MS_POST = 600.0  # covers Yang ~0.54 s + margin
# Same order as GT_CELLS in task.spot.gt.read_RecF_ImpR.
_RF_SIGN = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1])


def _sym_ylim(ax, y: np.ndarray) -> None:
    """Set ylim to ``[-a, a]`` with ``a = max(|y|)`` (0 at center)."""
    a = float(np.nanmax(np.abs(np.asarray(y, dtype=np.float64))))
    if not np.isfinite(a) or a <= 0:
        a = 1.0
    ax.set_ylim(-a, a)


def main() -> None:
    df = pd.read_csv(YANG_CSV)
    yang = df[(df["modality"] == "voltage") & (df["sti"] == "light")]

    delta_ms = float(DELTA_MS)
    t_onset = int(round(MS_PRE / delta_ms))
    n_t = t_onset + int(round(MS_POST / delta_ms))
    _, impr = read_RecF_ImpR(
        t_onset=t_onset, n_t=n_t, ms_spot=MS_SPOT, delta_ms=delta_ms,
    )
    t_impr_s = (np.arange(n_t) - t_onset) * delta_ms / 1000.0
    gt_idx = {name: i for i, name in enumerate(GT_CELLS)}

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    for ax, cell in zip(axes.ravel(), CELLS):
        tr = yang[yang["cell"] == cell].sort_values("time_s")
        y_yang = tr["dff_percent"].to_numpy(dtype=np.float64)
        ax.plot(tr["time_s"], y_yang, color="C0", lw=1.2, label="Yang voltage ON")
        ax.set_ylabel("ΔF/F (%)")
        ax.set_title(cell)
        ax.axvline(0.0, color="0.5", lw=0.6, ls=":")
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.axvspan(0.0, MS_SPOT / 1000.0, color="0.85", zorder=0)
        _sym_ylim(ax, y_yang)

        i = gt_idx[cell]
        y_impr = _RF_SIGN[i] * impr[i]
        ax2 = ax.twinx()
        ax2.plot(
            t_impr_s,
            y_impr,
            color="C1",
            lw=1.2,
            label=f"GT ImpR 25 ms (×{_RF_SIGN[i]:+d})",
        )
        ax2.set_ylabel("ImpR")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax.tick_params(axis="y", labelcolor="C0")
        _sym_ylim(ax2, y_impr)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")

    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Yang S5 voltage ON vs spot GT ImpR (ms_spot=25, ×RF_sign)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
