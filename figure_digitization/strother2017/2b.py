"""Digitize Strother et al. 2017 Fig. 3B Flash-column calcium traces from 2b.png.

2b.png is the Flash column of Fig. 3B (caption in paper.txt): median ± SEM
ΔF/F for Mi1, Tm3, Mi4, Mi9 and T4, each with ON (white) and OFF (gray) panels.

Method
------
1. Calibrate time with the bottom L-bar: 2 s = 49 px.
2. Calibrate ΔF/F with the per-row scale bars measured on paper.pdf Fig. 3B
   (between Flash and Moving-edge columns): Mi1/Tm3/Mi4 65 px = 2.0;
   Mi9 64 px = 0.5; T4 bottom L-bar in 2b.png 50 px = 0.5.
3. Per panel: colour-mask the cell's trace (+ SEM ribbon); per x-column take
   the median y of mask pixels (ribbon midline ≈ mean); baseline = panel
   bottom black rule; up = positive ΔF/F; then subtract the first sample so
   each trace starts at 0. First sample is t = 0 s.
4. Write long CSV + verification PNG as 2b_digitized.*.

Run:  ../.venv/bin/python 2b.py
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / "2b.png"
FIGURE = "2b"
OUT_STEM = HERE / f"{FIGURE}_digitized"

# Bottom L-bar in 2b.png: horizontal stem y=1631, x=170..218 → 49 px = 2 s
PX_PER_S = 49.0 / 2.0
# Per-row ΔF/F bars (Fig. 3B gutter / T4 L-bar); see module docstring
PX_PER_DFF = {
    "Mi1": 65.0 / 2.0,
    "Tm3": 65.0 / 2.0,
    "Mi4": 65.0 / 2.0,
    "Mi9": 64.0 / 0.5,
    "T4": 50.0 / 0.5,
}

# (cell, polarity, y_top, y_bottom); y_bottom ≈ ΔF/F = 0 black rule
PANELS = [
    ("Mi1", "ON", 180, 270),
    ("Mi1", "OFF", 273, 392),
    ("Tm3", "ON", 400, 560),
    ("Tm3", "OFF", 562, 680),
    ("Mi4", "ON", 690, 848),
    ("Mi4", "OFF", 850, 970),
    ("Mi9", "ON", 980, 1137),
    ("Mi9", "OFF", 1139, 1257),
    ("T4", "ON", 1265, 1426),
    ("T4", "OFF", 1429, 1547),
]

X_ONSET = 98  # first stimulus dashed line
X0, X1 = 72, 200  # trim right edge (scale-bar / label ink)

# Modal left edge of detected ink (most panels). T4 OFF sits on the black
# baseline with no coloured pixels for x=96..99; pad those with y_bottom.
PAD_LEFT_X0 = {
    ("T4", "OFF"): 96,
}

HUE = {
    "Mi1": ((0.93, 1.0), (0.0, 0.06)),
    "Tm3": ((0.07, 0.14),),
    "Mi4": ((0.52, 0.65),),
    "Mi9": ((0.42, 0.55),),
    "T4": ((0.05, 0.13),),
}
SAT_MIN = {"Mi1": 0.25, "Tm3": 0.25, "Mi4": 0.25, "Mi9": 0.20, "T4": 0.10}
MX_MAX = {"Mi1": 230, "Tm3": 240, "Mi4": 230, "Mi9": 230, "T4": 200}


def _hue_ok(cell: str, hh: float) -> bool:
    return any(lo <= hh < hi for lo, hi in HUE[cell])


def cell_mask(rgb: np.ndarray, cell: str) -> np.ndarray:
    mx = rgb.max(2)
    mn = rgb.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    out = np.zeros(rgb.shape[:2], dtype=bool)
    ys, xs = np.where((sat >= SAT_MIN[cell]) & (mx <= MX_MAX[cell]) & (mx > 70))
    if cell == "T4":
        ys, xs = np.where(
            (sat >= SAT_MIN[cell])
            & (sat < 0.40)
            & (mx <= MX_MAX[cell])
            & (mx > 70)
        )
    for y, x in zip(ys, xs):
        hh = colorsys.rgb_to_hsv(*(rgb[y, x] / 255.0))[0]
        if _hue_ok(cell, hh):
            out[y, x] = True
    return out


def extract_panel(
    rgb: np.ndarray, gray: np.ndarray, cell: str, y0: int, y1: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    mask = cell_mask(rgb, cell) & (gray > 50)
    xs, ys = [], []
    for x in range(X0, X1):
        col = np.where(mask[y0 : y1 + 1, x])[0]
        if not len(col):
            continue
        xs.append(x)
        ys.append(y0 + float(np.median(col)))
    if len(xs) < 8:
        return None, None
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    full_x = np.arange(xs.min(), xs.max() + 1)
    full_y = np.interp(full_x, xs, ys)
    return full_x, full_y


def pad_left_baseline(
    xs: np.ndarray, ys: np.ndarray, x0: int, y_base: float
) -> tuple[np.ndarray, np.ndarray]:
    """Prepend columns [x0 .. xs[0]) at baseline. No-op if already at/left of x0."""
    if xs[0] <= x0:
        return xs, ys
    pre_x = np.arange(x0, xs[0], dtype=float)
    pre_y = np.full(len(pre_x), y_base, dtype=float)
    return np.concatenate([pre_x, xs]), np.concatenate([pre_y, ys])


def digitize(img: np.ndarray) -> pd.DataFrame:
    rgb = img.astype(np.float64)
    gray = rgb.mean(2)
    rows: list[dict] = []
    for cell, polarity, y0, y1 in PANELS:
        xs, ys = extract_panel(rgb, gray, cell, y0, y1)
        if xs is None:
            print(f"  !! no trace for {cell} / {polarity}")
            continue
        if (cell, polarity) in PAD_LEFT_X0:
            xs, ys = pad_left_baseline(xs, ys, PAD_LEFT_X0[(cell, polarity)], float(y1))
        t = (xs - X_ONSET) / PX_PER_S
        t = t - t[0]
        dff = (y1 - ys) / PX_PER_DFF[cell]
        dff = dff - dff[0]
        for ti, vi in zip(t, dff):
            rows.append(
                dict(cell=cell, polarity=polarity, time_s=float(ti), dff=float(vi))
            )
        print(
            f"  {cell:4s} {polarity:3s} n={len(t):3d} "
            f"peak={dff.max():6.2f}  end={dff[-1]:6.2f}"
        )
    return pd.DataFrame(rows)


def plot_check(df: pd.DataFrame, path: Path) -> None:
    cells = ["Mi1", "Tm3", "Mi4", "Mi9", "T4"]
    colors = {
        "Mi1": "#a32f34",
        "Tm3": "#c89030",
        "Mi4": "#4080bd",
        "Mi9": "#5ab1a8",
        "T4": "#998877",
    }
    fig, axes = plt.subplots(5, 2, figsize=(6, 10), sharex=True)
    for r, cell in enumerate(cells):
        for c, polarity in enumerate(["ON", "OFF"]):
            ax = axes[r, c]
            tr = df[(df.cell == cell) & (df.polarity == polarity)].sort_values("time_s")
            if not tr.empty:
                ax.plot(tr.time_s, tr.dff, color=colors[cell], lw=1.1)
            ax.axhline(0, color="0.75", lw=0.5)
            ax.axvline(0, color="0.75", lw=0.5)
            if c == 0:
                ax.set_ylabel(cell)
            if r == 0:
                ax.set_title(polarity)
            if r == 4:
                ax.set_xlabel("time (s)")
    fig.suptitle("Strother 2017 Fig. 3B Flash column (digitized)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    img = np.array(Image.open(DEFAULT_IMAGE).convert("RGB"))
    print(f"loaded {DEFAULT_IMAGE.name}  {img.shape[1]}x{img.shape[0]}")
    df = digitize(img)
    csv_path = OUT_STEM.with_suffix(".csv")
    png_path = OUT_STEM.with_suffix(".png")
    df.to_csv(csv_path, index=False)
    plot_check(df, png_path)
    print(f"wrote {csv_path}  ({len(df)} rows, {df.cell.nunique()} cells)")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
