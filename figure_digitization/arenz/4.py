"""Digitize Arenz et al. 2017 Fig. 4B/E control (black) temporal kernels.

4B: Mi1, Tm3, Mi4, Mi9   4E: Tm1, Tm2, Tm4, Tm9
Only the solid black control mean (±SEM wash ignored; magenta CDM ignored).

Method
------
1. Time on the figure runs −1.5 … ~+0.2 (tick 0 = figure origin). Output is
   sign-flipped then offset: t = −t_fig + T_OFFSET → ~−0.1 … +1.6 s, Δt = 1 ms.
2. Amplitude: dashed horizontal = 0; no y-axis numbers in the figure → scale
   each trace so max(|amp|) = 1 (sign preserved).
3. Black mask = dark + low-saturation pixels; per-x continuity tracking;
   dashed baseline flecks dropped when other ink exists in the column.
4. Write long CSV + verification PNG as 4_digitized.*.

Run:  ../.venv/bin/python 4.py
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
OUT_STEM = HERE / "4_digitized"

# (cell, image, x_at_-1.5, x_at_0, y_base, y_top, y_bot)
# x = centers of tick labels −1.5 and 0; y_base = dashed zero line.
PANELS = [
    ("Mi1", "4b.png", 47, 534, 421, 95, 550),
    ("Tm3", "4b.png", 697, 1184, 421, 95, 550),
    ("Mi4", "4b.png", 1346, 1833, 423, 95, 550),
    ("Mi9", "4b.png", 1996, 2483, 179, 95, 550),
    ("Tm1", "4e.png", 80, 684, 187, 100, 680),
    ("Tm2", "4e.png", 885, 1489, 189, 100, 680),
    ("Tm4", "4e.png", 1690, 2294, 186, 100, 680),
    ("Tm9", "4e.png", 2495, 3099, 186, 100, 680),
]

# Mi1/Tm3/Mi4 only: peak tip is above y=95 (was clipped flat). Open y_top to 55
# and blank the Bi/Bii/Biii title so label ink is not tracked.
PEAK_TOP_PATCH = {
    "Mi1": (55, 95, 130),  # y_top, title_y_bot, title_x_width from x_left
    "Tm3": (55, 95, 140),
    "Mi4": (55, 95, 150),
}

MAX_DY_PX = 25
T_PAD_S = 0.25  # past t=0
DT_S = 0.001  # output time step
T_OFFSET_S = 0.1  # added after sign-flip (−t_fig)


def black_mask(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(2)
    mx = rgb.max(2)
    mn = rgb.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    magenta = (rgb[:, :, 0] > rgb[:, :, 1] + 25) & (rgb[:, :, 0] > 110) & (rgb[:, :, 2] > 70)
    return (gray < 90) & (sat < 0.45) & ~magenta


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


def digitize() -> pd.DataFrame:
    cache: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for cell, image, x_m15, x_0, y_base, y_top, y_bot in PANELS:
        if image not in cache:
            cache[image] = np.array(Image.open(HERE / image).convert("RGB")).astype(np.float64)
        rgb = cache[image]
        px_per_s = (x_0 - x_m15) / 1.5
        x_left = max(0, x_m15 - 5)
        x_right = min(rgb.shape[1], int(round(x_0 + T_PAD_S * px_per_s)))
        title_blank = None
        if cell in PEAK_TOP_PATCH:
            y_top, title_y, title_w = PEAK_TOP_PATCH[cell]
            title_blank = (title_y, title_w)
        xs, ys = extract_trace(rgb, x_left, x_right, y_top, y_bot, y_base, title_blank)
        if xs is None:
            print(f"  !! no black trace for {cell}")
            continue
        t_fig = -1.5 + (xs - x_m15) / px_per_s
        amp = (y_base - ys).astype(float)
        amp /= np.max(np.abs(amp))
        # Sign-flip then offset: t_fig −1.5…+0.2 → t −0.1…+1.6.
        t = (-t_fig)[::-1] + T_OFFSET_S
        amp = amp[::-1]
        t_lo = np.floor(t[0] / DT_S + 1e-12) * DT_S
        t_hi = np.ceil(t[-1] / DT_S - 1e-12) * DT_S
        t_grid = np.arange(t_lo, t_hi + 1e-12, DT_S)
        amp = np.interp(t_grid, t, amp)
        amp /= np.max(np.abs(amp))
        t = t_grid
        panel = "4B" if image.startswith("4b") else "4E"
        for ti, ai in zip(t, amp):
            rows.append(dict(panel=panel, cell=cell, time_s=float(ti), amplitude=float(ai)))
        peak_i = int(np.argmax(np.abs(amp)))
        print(
            f"  {cell:4s} n={len(t):3d}  t={t[0]:.2f}..{t[-1]:.2f}  "
            f"peak_t={t[peak_i]:.3f}  peak={amp[peak_i]:+.2f}"
        )
    return pd.DataFrame(rows)


def plot_check(df: pd.DataFrame, path: Path) -> None:
    order = ["Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey=True)
    for i, cell in enumerate(order):
        ax = axes[i // 4, i % 4]
        tr = df[df.cell == cell].sort_values("time_s")
        if not tr.empty:
            ax.plot(tr.time_s, tr.amplitude, color="k", lw=1.1)
        ax.axhline(0, color="0.75", lw=0.5)
        ax.axvline(0, color="0.75", lw=0.5)
        ax.set_title(cell, fontsize=10)
        if i // 4 == 1:
            ax.set_xlabel("time (s)")
        if i % 4 == 0:
            ax.set_ylabel("amplitude (norm.)")
    fig.suptitle(
        "Arenz 2017 Fig. 4B/E control kernels (digitized, t=−t_fig+0.1, max|amp|=1)",
        y=0.98,
    )
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
