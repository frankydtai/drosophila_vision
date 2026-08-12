#!/usr/bin/env python3
"""Overlay Gruntman21 T4 +0 (green) with Gruntman18 160 ms +0 (black).

Outputs
-------
    gruntman18_gruntman21_overlay.png

Run:  ../.venv/bin/python gruntman18_gruntman21_overlay.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
G21_CSV = ROOT / "gruntman21" / "2a_digitized.csv"
G18_CSV = ROOT / "gruntman18" / "2b_digitized.csv"
OUT_PNG = HERE / "gruntman18_gruntman21_overlay.png"


def main() -> int:
    g21 = pd.read_csv(G21_CSV)
    g18 = pd.read_csv(G18_CSV)

    t4 = g21[(g21.cell_type == "T4") & (g21.position == 0) & (g21.color == "green")]
    flash = g18[
        (g18.flash_duration_ms == 160)
        & (g18.position_idx == 0)
        & (g18.color == "black")
    ]
    if t4.empty or flash.empty:
        raise SystemExit("missing requested traces in digitized CSVs")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        t4.time_ms.to_numpy(float),
        t4.vm_mv.to_numpy(float),
        color="green",
        lw=1.6,
        label="Gruntman21 2A  T4 +0 (PC)",
    )
    ax.plot(
        flash.time_ms.to_numpy(float),
        flash.vm_mv.to_numpy(float),
        color="black",
        lw=1.6,
        label="Gruntman18 2B  160 ms +0",
    )
    ax.axhline(0, color="0.75", lw=0.6)
    ax.axvline(0, color="0.75", lw=0.6)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("Vm (mV)")
    ax.set_title("Gruntman21 T4 +0 (green) vs Gruntman18 160 ms +0 (black)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
