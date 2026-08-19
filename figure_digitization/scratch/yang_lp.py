"""Forward GCaMP LP on Yang extended V (Butter 5 Hz tail + peak extrap).

Imports V from yang_extrap; plots V_extrap, LP(V), and LP(rect(V+bias))
for bias = +2.5 / -2.5.

Run:  ../.venv/bin/python yang_lp.py
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

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "yang_lp.png"
TAU_MS = 350.0
BIAS_RECT = (+2.5, -2.5)


def lowpass(x: np.ndarray, tau_s: float, dt: float) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    a = dt / tau_s
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def lp_rect_bias(v: np.ndarray, bias: float, tau_s: float, dt: float, i0: int) -> np.ndarray:
    lp = lowpass(np.maximum(0.0, v + bias), tau_s, dt)
    return lp - lp[i0]


def main() -> int:
    df = pd.read_csv(YANG_CSV)
    vdf = df[df.modality == "voltage"]
    cdf = df[df.modality == "calcium"]
    t_grid = np.arange(0.0, float(cdf.time_s.max()) + DT + 1e-9, DT)
    tail_s = TAIL450_MS / 1000.0
    tau_s = TAU_MS / 1000.0

    fig, axes = plt.subplots(6, 2, figsize=(9, 14), sharex=True)
    for r, cell in enumerate(ORDER_B + ORDER_E):
        for c, stimulus in enumerate(["light", "dark"]):
            ax = axes[r, c]
            sub = vdf[(vdf.cell == cell) & (vdf.stimulus == stimulus)]
            vm = baseline0(resample(sub.time_s.values, sub.dff_percent.values, t_grid))
            meas = np.isfinite(vm)
            v_ext, meta = build_final_v(vm, t_grid, tail_s)
            v_fill = np.nan_to_num(v_ext, nan=0.0)
            i0 = int(np.flatnonzero(np.isfinite(v_ext))[0])
            lp0 = lowpass(v_fill, tau_s, DT)
            lp = lp0 - lp0[i0]

            color = COLORS[cell]
            ax.plot(t_grid[meas], vm[meas], color=color, alpha=0.25, lw=0.9, label="raw digitize")
            m = np.isfinite(v_ext)
            ax.plot(t_grid[m], v_ext[m], color=color, lw=1.4, label="V extrap")
            ax.plot(t_grid, lp, color="0.1", lw=1.8, label=rf"LP$_\tau$ {TAU_MS:.0f} ms")
            for bias, ls in zip(BIAS_RECT, ("--", "-.")):
                ax.plot(
                    t_grid,
                    lp_rect_bias(v_fill, bias, tau_s, DT, i0),
                    color="0.35" if bias > 0 else "0.55",
                    lw=1.3,
                    ls=ls,
                    label=rf"LP rect(V{'+' if bias >= 0 else ''}{bias:g})",
                )
            ax.axvline(meta["t_peak"], color="0.35", lw=0.5, ls="-.")
            ax.axvline(meta["t_end"], color="0.45", lw=0.5, ls="--")
            ax.axhline(0, color="0.75", lw=0.5)
            if r == 0:
                ax.set_title(stimulus, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{cell}\nΔF/F %", fontsize=9)
            if r == 5:
                ax.set_xlabel("time (s)", fontsize=9)
            if r == 0 and c == 1:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Yang S5: extrap V → LP τ={TAU_MS:.0f} ms; "
        f"rect(V+bias) LP with bias {BIAS_RECT[0]:g} / {BIAS_RECT[1]:g}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
