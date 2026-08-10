"""Digitize Behnia et al. 2014 Extended Data Figure 2 voltage traces from ex2.png.

Extended Data Fig. 2a–d excerpts: averaged Vm (±s.e.m.) of Mi1, Tm3, Tm1 and
Tm2 to 5 s luminance steps on a grey background (caption in paper.txt). This
crop is one column of eight panels (two contrast steps × four cell types).

Method
------
1. Calibrate with the top-panel L bar: 10 mV = 37 px, 1 s = 40 px
   (not the dotted baseline; not averaged with the blue-panel L).
2. Per panel: find the dotted baseline (Vm = 0) and the stimulus-step onset.
3. For each x column, take high-saturation cell-colour pixels above / below
   baseline (highest-sat y) → positive / negative branches.
4. Tm1/Tm2 0.5→1 only: after hardcoded cross times, swap branch labels.
5. Write long CSV + verification PNG as ex2_digitized.*.

Run:  ../.venv/bin/python ex2.py
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
DEFAULT_IMAGE = HERE / "ex2.png"
FIGURE = "ex2"
OUT_STEM = HERE / f"{FIGURE}_digitized"

# Top-panel L bar only (not averaged with the blue-panel copy):
#   10 mV vertical stem at x=205, y=23..59 inclusive → 37 px
#   1 s  horizontal foot at y=56, x=203..242 inclusive → 40 px
PX_PER_MV = 37 / 10
PX_PER_S = 40 / 1

HUE = {
    "Mi1": (0.82, 0.92),
    "Tm3": (0.25, 0.35),
    "Tm1": (0.60, 0.70),
    "Tm2": (0.00, 0.05),
}

PANELS = [
    ("Mi1", 0.1, 0.5, 40, 131),
    ("Mi1", 0.5, 0.9, 220, 301),
    ("Tm3", 0.1, 0.5, 420, 506),
    ("Tm3", 0.5, 0.9, 620, 687),
    ("Tm1", 0.0, 0.5, 860, 949),
    ("Tm1", 0.5, 1.0, 1035, 1126),
    ("Tm2", 0.0, 0.5, 1255, 1331),
    ("Tm2", 0.5, 1.0, 1440, 1512),
]

# Hardcoded: swap branch labels after these times (s), with corrected PX_PER_S.
FLIP_AFTER_S = {
    ("Tm1", 0.5, 1.0): 1.18,
    ("Tm2", 0.0, 0.5): 1.30,
    ("Tm2", 0.5, 1.0): 0.90,
}

X0, X1 = 30, 250
SAT_MIN = 0.45


def _hue(rgb: np.ndarray, y: int, x: int) -> float:
    return colorsys.rgb_to_hsv(*(rgb[y, x] / 255.0))[0]


def color_mask(rgb: np.ndarray, cell: str) -> tuple[np.ndarray, np.ndarray]:
    hlo, hhi = HUE[cell]
    mx = rgb.max(2)
    mn = rgb.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    ys, xs = np.where((sat >= SAT_MIN) & (mx >= 40) & (mx <= 250))
    out = np.zeros(rgb.shape[:2], dtype=bool)
    for y, x in zip(ys, xs):
        hh = _hue(rgb, y, x)
        if hlo <= hh < hhi or (cell == "Tm2" and hh >= 0.95):
            out[y, x] = True
    return out, sat


def stim_onset_x(gray: np.ndarray, y_base: int, x0: int = X0, x1: int = X1) -> int:
    y0, y1 = y_base + 25, min(gray.shape[0], y_base + 70)
    stim = gray[y0:y1, x0:x1] < 70
    return int(x0 + np.argmax(np.diff(stim.mean(0))))


def _trace_y(mask: np.ndarray, sat: np.ndarray, y0: int, y1: int, x: int) -> float | None:
    ys = np.where(mask[y0:y1, x])[0] + y0
    if not len(ys):
        return None
    return float(ys[np.argmax(sat[ys, x])])


def _track(
    mask: np.ndarray,
    sat: np.ndarray,
    y0: int,
    y1: int,
    xs: range,
    y_seed: float,
    dist_w: float = 0.02,
) -> dict[int, float]:
    out: dict[int, float] = {}
    y_prev = y_seed
    for x in xs:
        ys = np.where(mask[y0:y1, x])[0] + y0
        if not len(ys):
            continue
        score = sat[ys, x] - dist_w * np.abs(ys.astype(float) - y_prev)
        y = float(ys[int(np.argmax(score))])
        out[x] = y
        y_prev = y
    return out


def extract_tm1_0_05(
    rgb: np.ndarray, gray: np.ndarray, y_top: int, y_base: int
) -> list[dict]:
    """Tm1 0→0.5 only: dashed gaps + post-zero cross need continuity tracking."""
    mask, sat = color_mask(rgb, "Tm1")
    x_on = stim_onset_x(gray, y_base)
    y_bot = min(gray.shape[0], y_base + (y_base - y_top))
    x_seed = min(X1 - 1, x_on + 12)
    y_hi = _trace_y(mask, sat, y_top, y_base, x_seed)
    y_lo = _trace_y(mask, sat, y_base + 1, y_bot, x_seed)
    if y_hi is None or y_lo is None:
        return []
    xs_fwd = range(x_seed, X1)
    xs_back = range(x_seed, X0 - 1, -1)
    w = 0.05  # stronger continuity: dashed gaps otherwise latch onto baseline flecks
    hi = {
        **_track(mask, sat, y_top, y_bot, xs_back, y_hi, dist_w=w),
        **_track(mask, sat, y_top, y_bot, xs_fwd, y_hi, dist_w=w),
    }
    lo = {
        **_track(mask, sat, y_top, y_bot, xs_back, y_lo, dist_w=w),
        **_track(mask, sat, y_top, y_bot, xs_fwd, y_lo, dist_w=w),
    }
    rows = []
    for branch, trace in (("positive", hi), ("negative", lo)):
        for x, y in sorted(trace.items()):
            rows.append(
                dict(
                    cell="Tm1",
                    intensity_from=0.0,
                    intensity_to=0.5,
                    branch=branch,
                    time_s=(x - x_on) / PX_PER_S,
                    vm_mv=(y_base - y) / PX_PER_MV,
                )
            )
    return rows


def extract_panel(
    rgb: np.ndarray,
    gray: np.ndarray,
    cell: str,
    intensity_from: float,
    intensity_to: float,
    y_top: int,
    y_base: int,
) -> list[dict]:
    if cell == "Tm1" and intensity_from == 0.0 and intensity_to == 0.5:
        return extract_tm1_0_05(rgb, gray, y_top, y_base)
    mask, sat = color_mask(rgb, cell)
    x_on = stim_onset_x(gray, y_base)
    y_bot = min(gray.shape[0], y_base + (y_base - y_top))
    rows = []
    for x in range(X0, X1):
        t = (x - x_on) / PX_PER_S
        y_pos = _trace_y(mask, sat, y_top, y_base, x)
        y_neg = _trace_y(mask, sat, y_base + 1, y_bot, x)
        if y_pos is not None:
            rows.append(
                dict(
                    cell=cell,
                    intensity_from=intensity_from,
                    intensity_to=intensity_to,
                    branch="positive",
                    time_s=t,
                    vm_mv=(y_base - y_pos) / PX_PER_MV,
                )
            )
        if y_neg is not None:
            rows.append(
                dict(
                    cell=cell,
                    intensity_from=intensity_from,
                    intensity_to=intensity_to,
                    branch="negative",
                    time_s=t,
                    vm_mv=(y_base - y_neg) / PX_PER_MV,
                )
            )
    return rows


def flip_after_cross(df: pd.DataFrame) -> pd.DataFrame:
    """Swap branch labels after hardcoded cross times (Tm1/Tm2 0.5→1 only)."""
    out = df.copy()
    for (cell, lo, hi), t_cross in FLIP_AFTER_S.items():
        after = (
            (out.cell == cell)
            & (out.intensity_from == lo)
            & (out.intensity_to == hi)
            & (out.time_s > t_cross)
        )
        out.loc[after & (out.branch == "positive"), "branch"] = "_tmp"
        out.loc[after & (out.branch == "negative"), "branch"] = "positive"
        out.loc[after & (out.branch == "_tmp"), "branch"] = "negative"
    return out


def digitize(img: np.ndarray) -> pd.DataFrame:
    rgb = img.astype(np.float64)
    gray = rgb.mean(2)
    rows: list[dict] = []
    for cell, lo, hi, y_top, y_base in PANELS:
        rows.extend(extract_panel(rgb, gray, cell, lo, hi, y_top, y_base))
    return flip_after_cross(pd.DataFrame(rows))


def plot_check(df: pd.DataFrame, path: Path) -> None:
    cells = ["Mi1", "Tm3", "Tm1", "Tm2"]
    colors = {"Mi1": "#e039c6", "Tm3": "#5bc53a", "Tm1": "#0020f5", "Tm2": "#eb3323"}
    fig, axes = plt.subplots(4, 2, figsize=(8, 12), sharex=True)
    for r, cell in enumerate(cells):
        sub = df[df.cell == cell]
        steps = sorted(
            sub[["intensity_from", "intensity_to"]].drop_duplicates().itertuples(
                index=False
            )
        )
        for c, (lo, hi) in enumerate(steps):
            ax = axes[r, c]
            panel = sub[(sub.intensity_from == lo) & (sub.intensity_to == hi)]
            for branch, ls in (("positive", "-"), ("negative", "--")):
                tr = panel[panel.branch == branch].sort_values("time_s")
                if tr.empty:
                    continue
                ax.plot(tr.time_s, tr.vm_mv, ls, color=colors[cell], lw=1.2)
            ax.axhline(0, color="0.7", lw=0.6)
            ax.axvline(0, color="0.7", lw=0.6)
            ax.set_title(f"{cell}  {lo:g}→{hi:g}", fontsize=9)
            if c == 0:
                ax.set_ylabel("Vm (mV)")
            if r == 3:
                ax.set_xlabel("time (s)")
    fig.suptitle("Behnia 2014 Ext. Data Fig. 2 (digitized)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    img = np.array(Image.open(DEFAULT_IMAGE).convert("RGB"))
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
