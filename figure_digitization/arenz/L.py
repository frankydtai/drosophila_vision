"""Digitize L1–L5 black impulse-response traces from L.png.

Same method as 4.py (black mean line, max|amp|=1, t = −t_fig + T_OFFSET).
L.png is not from Arenz 2017; layout matches. Does not touch 4.py / 4b / 4e.

Method
------
1. Time on the figure: ticks −2, −1, 0 (~−2.5 … +0.2). Output:
   t = −t_fig + T_OFFSET → ~−0.15 … +2.5 s, Δt = 1 ms.
2. Amplitude: dashed horizontal = 0; scale each trace so max(|amp|) = 1.
3. Black mask + per-x continuity; blank L1–L5 title ink at panel top-left.
4. Write L_digitized.csv / L_digitized.png.

Run:  ../.venv/bin/python L.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / "L.png"
OUT_STEM = HERE / "L_digitized"

# (cell, x_at_-2, x_at_0, y_base, y_top, y_bot)
PANELS = [
    ("L1", 267, 585, 157, 50, 470),
    ("L2", 924, 1242, 157, 50, 470),
    ("L3", 1582, 1899, 157, 50, 470),
    ("L4", 2238, 2557, 158, 50, 470),
    ("L5", 2871, 3204, 320, 50, 470),
]

# Blank "L1"…"L5" title glyphs (top-left of each panel).
TITLE_BLANK = {
    "L1": (70, 100),
    "L2": (70, 100),
    "L3": (70, 100),
    "L4": (70, 100),
    "L5": (70, 100),
}

MAX_DY_PX = 25
T_LEFT_PAD_S = 0.40  # left of −2 tick (~−2.5 on figure)
T_RIGHT_PAD_S = 0.25  # past 0 tick
DT_S = 0.001  # output time step
T_OFFSET_S = 0.1  # added after sign-flip (−t_fig)


def black_mask(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(2)
    mx = rgb.max(2)
    mn = rgb.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    return (gray < 90) & (sat < 0.45)


def extract_trace(
    rgb: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    y_base: int,
    title_blank: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    mask = black_mask(rgb)
    mask[:y0, :] = False
    mask[y1:, :] = False
    if title_blank is not None:
        y_hi, x_w = title_blank
        mask[:y_hi, x0 : x0 + x_w] = False
    xs: list[float] = []
    ys: list[float] = []
    prev: float | None = None
    for x in range(x0, x1):
        col = np.where(mask[:, x])[0]
        if not len(col):
            continue
        other = col[np.abs(col - y_base) > 3]
        # Drop dashed-zero flecks only while the stroke is away from baseline.
        # Near y=0 the solid line sits on the dashes — excluding them punches gaps.
        if prev is not None and abs(prev - y_base) <= MAX_DY_PX:
            use = col
        elif len(other):
            use = other
        else:
            use = col
        if prev is None:
            y = float(np.median(use))
        else:
            near = use[np.abs(use.astype(float) - prev) <= MAX_DY_PX]
            if not len(near):
                continue
            y = float(np.median(near))
        xs.append(float(x))
        ys.append(y)
        prev = y
    if len(xs) < 20:
        return None, None
    return np.asarray(xs), np.asarray(ys)


def digitize(img_path: Path = DEFAULT_IMAGE) -> pd.DataFrame:
    rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float64)
    rows: list[dict] = []
    for cell, x_m2, x_0, y_base, y_top, y_bot in PANELS:
        px_per_s = (x_0 - x_m2) / 2.0
        x_left = max(0, int(round(x_m2 - T_LEFT_PAD_S * px_per_s)))
        x_right = min(rgb.shape[1], int(round(x_0 + T_RIGHT_PAD_S * px_per_s)))
        xs, ys = extract_trace(
            rgb, x_left, x_right, y_top, y_bot, y_base, TITLE_BLANK.get(cell)
        )
        if xs is None:
            print(f"  !! no black trace for {cell}")
            continue
        t_fig = -2.0 + (xs - x_m2) / px_per_s
        amp = (y_base - ys).astype(float)
        amp /= np.max(np.abs(amp))
        t = (-t_fig)[::-1] + T_OFFSET_S
        amp = amp[::-1]
        t_lo = np.floor(t[0] / DT_S + 1e-12) * DT_S
        t_hi = np.ceil(t[-1] / DT_S - 1e-12) * DT_S
        t_grid = np.arange(t_lo, t_hi + 1e-12, DT_S)
        amp = np.interp(t_grid, t, amp)
        amp /= np.max(np.abs(amp))
        t = t_grid
        for ti, ai in zip(t, amp):
            rows.append(dict(cell=cell, time_s=float(ti), amplitude=float(ai)))
        peak_i = int(np.argmax(np.abs(amp)))
        print(
            f"  {cell:4s} n={len(t):3d}  t={t[0]:.2f}..{t[-1]:.2f}  "
            f"peak_t={t[peak_i]:.3f}  peak={amp[peak_i]:+.2f}"
        )
    return pd.DataFrame(rows)


def plot_check(df: pd.DataFrame, path: Path) -> None:
    order = ["L1", "L2", "L3", "L4", "L5"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3), sharex=True, sharey=True)
    for ax, cell in zip(axes, order):
        tr = df[df.cell == cell].sort_values("time_s")
        if not tr.empty:
            ax.plot(tr.time_s, tr.amplitude, color="k", lw=1.1)
        ax.axhline(0, color="0.75", lw=0.5)
        ax.axvline(0, color="0.75", lw=0.5)
        ax.set_title(cell, fontsize=10)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude (norm.)")
    fig.suptitle("L1–L5 impulse responses (digitized, t=−t_fig+0.1, max|amp|=1)", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    df = digitize()
    csv_path = OUT_STEM.with_suffix(".csv")
    png_path = OUT_STEM.with_suffix(".png")
    df.to_csv(csv_path, index=False)
    plot_check(df, png_path)
    print(f"wrote {csv_path}  ({len(df)} rows, {df.cell.nunique()} cells)")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
