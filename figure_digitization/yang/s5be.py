"""Digitize Yang et al. 2016 Figure S5 panels B/E (ON/OFF voltage & calcium) from s5be.png.

Crop layout (see paper.txt Figure S5 caption):

  B (ON):  L1, Tm3, Mi1
  E (OFF): L2, Tm1, Tm2

Each row × Voltage/Calcium × Light/Dark mean ΔF/F (±SEM wash ignored).
t=0 is the left edge of each column (flash onset); column x-ranges use the
full ink extent so the rising flank is not cropped.

Scale bars measured on s5be.png:
  0.2 s → 46 px
  Voltage  −2% (L1/L2) → 47 px;  −5% (others) → 35 px
  Calcium   5% (L1/L2) → 59 px;  20% (Tm3/Tm1) → 52 px;  10% (Mi1/Tm2) → 48 px

Run:  ../.venv/bin/python s5be.py
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
DEFAULT_IMAGE = HERE / "s5be.png"
FIGURE = "s5be"
OUT_STEM = HERE / f"{FIGURE}_digitized"

PX_PER_S = 46 / 0.2

# cell → (hue_lo, hue_hi), voltage %/px denom, calcium %/px denom, baseline_y, y_top, y_bot
# px_per_pct = bar_px / bar_pct
CELLS = {
    "L1":  dict(panel="B", hue=(0.54, 0.62), v_pct=2.0, v_px=47, ca_pct=5.0, ca_px=59,
                y_base=202, y_top=102, y_bot=288),
    "Tm3": dict(panel="B", hue=(0.00, 0.05), v_pct=5.0, v_px=35, ca_pct=20.0, ca_px=52,
                y_base=476, y_top=280, y_bot=514),
    "Mi1": dict(panel="B", hue=(0.28, 0.36), v_pct=5.0, v_px=35, ca_pct=10.0, ca_px=48,
                y_base=640, y_top=524, y_bot=718),
    "L2":  dict(panel="E", hue=(0.54, 0.62), v_pct=2.0, v_px=47, ca_pct=5.0, ca_px=59,
                y_base=1014, y_top=920, y_bot=1102),
    "Tm1": dict(panel="E", hue=(0.00, 0.05), v_pct=5.0, v_px=35, ca_pct=20.0, ca_px=52,
                y_base=1217, y_top=1100, y_bot=1302),
    "Tm2": dict(panel="E", hue=(0.28, 0.36), v_pct=5.0, v_px=35, ca_pct=10.0, ca_px=48,
                y_base=1422, y_top=1300, y_bot=1529),
}

# (modality, stimulus, x0, x1); t=0 at column left = flash onset.
# Full ink extents (old x0 cut the rising flank).
COLUMNS = [
    ("voltage", "light", 64, 190),
    ("voltage", "dark", 209, 336),
    ("calcium", "light", 372, 736),
    ("calcium", "dark", 763, 1127),
]

SAT_MIN = 0.22

# L1/L2 voltage-dark only: panel legend (blue "L1"/"L2" swatch) sits above the
# traces and steals max-sat → square pulse. Ignore ink above these y cuts.
VOLTAGE_DARK_LEGEND_CUT = {"L1": 130, "L2": 950}


def _hue(rgb: np.ndarray, y: int, x: int) -> float:
    return colorsys.rgb_to_hsv(*(rgb[y, x] / 255.0))[0]


def color_mask(rgb: np.ndarray, hue: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    hlo, hhi = hue
    mx = rgb.max(2)
    mn = rgb.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    ys, xs = np.where((sat >= SAT_MIN) & (mx >= 30) & (mx <= 245))
    out = np.zeros(rgb.shape[:2], dtype=bool)
    for y, x in zip(ys, xs):
        hh = _hue(rgb, y, x)
        if hlo <= hh < hhi or (hlo == 0.0 and hh >= 0.95):
            out[y, x] = True
    return out, sat


def extract_trace(
    mask: np.ndarray,
    sat: np.ndarray,
    y_top: int,
    y_bot: int,
    y_base: int,
    x0: int,
    x1: int,
    px_per_pct: float,
) -> list[tuple[float, float]]:
    """Mean line = highest-saturation pixel per x (skip SEM wash)."""
    pts = []
    for x in range(x0, x1):
        ys = np.where(mask[y_top:y_bot, x])[0] + y_top
        if not len(ys):
            continue
        y = float(ys[np.argmax(sat[ys, x])])
        pts.append(((x - x0) / PX_PER_S, (y_base - y) / px_per_pct))
    return pts


def extract_trace_l1_l2_voltage_dark(
    mask: np.ndarray,
    sat: np.ndarray,
    y_top: int,
    y_bot: int,
    y_base: int,
    x0: int,
    x1: int,
    px_per_pct: float,
    legend_cut: int,
) -> list[tuple[float, float]]:
    """L1/L2 voltage-dark only: drop legend swatch ink above legend_cut."""
    pts = []
    for x in range(x0, x1):
        ys = np.where(mask[y_top:y_bot, x])[0] + y_top
        ys = ys[ys >= legend_cut]
        if not len(ys):
            continue
        y = float(ys[np.argmax(sat[ys, x])])
        pts.append(((x - x0) / PX_PER_S, (y_base - y) / px_per_pct))
    return pts


def digitize(img: np.ndarray) -> pd.DataFrame:
    rgb = img.astype(np.float64)
    rows: list[dict] = []
    masks = {c: color_mask(rgb, meta["hue"]) for c, meta in CELLS.items()}
    for cell, meta in CELLS.items():
        mask, sat = masks[cell]
        for modality, stimulus, x0, x1 in COLUMNS:
            if modality == "voltage":
                px_per_pct = meta["v_px"] / meta["v_pct"]
            else:
                px_per_pct = meta["ca_px"] / meta["ca_pct"]
            if (
                cell in VOLTAGE_DARK_LEGEND_CUT
                and modality == "voltage"
                and stimulus == "dark"
            ):
                pts = extract_trace_l1_l2_voltage_dark(
                    mask,
                    sat,
                    meta["y_top"],
                    meta["y_bot"],
                    meta["y_base"],
                    x0,
                    x1,
                    px_per_pct,
                    VOLTAGE_DARK_LEGEND_CUT[cell],
                )
            else:
                pts = extract_trace(
                    mask,
                    sat,
                    meta["y_top"],
                    meta["y_bot"],
                    meta["y_base"],
                    x0,
                    x1,
                    px_per_pct,
                )
            for t, dff in pts:
                rows.append(
                    dict(
                        panel=meta["panel"],
                        cell=cell,
                        modality=modality,
                        stimulus=stimulus,
                        time_s=t,
                        dff_percent=dff,
                    )
                )
    return pd.DataFrame(rows)


def plot_check(df: pd.DataFrame, path: Path) -> None:
    order_b = ["L1", "Tm3", "Mi1"]
    order_e = ["L2", "Tm1", "Tm2"]
    colors = {
        "L1": "#2a5db0", "L2": "#2a5db0",
        "Tm3": "#d62728", "Tm1": "#d62728",
        "Mi1": "#2ca02c", "Tm2": "#2ca02c",
    }
    fig, axes = plt.subplots(6, 4, figsize=(12, 14), sharex=False)
    for r, cell in enumerate(order_b + order_e):
        for c, (modality, stimulus, _, _) in enumerate(COLUMNS):
            ax = axes[r, c]
            tr = df[
                (df.cell == cell)
                & (df.modality == modality)
                & (df.stimulus == stimulus)
            ].sort_values("time_s")
            if not tr.empty:
                ax.plot(tr.time_s, tr.dff_percent, color=colors[cell], lw=1.0)
            ax.axhline(0, color="0.75", lw=0.5)
            ax.axvline(0, color="0.75", lw=0.5)
            if r == 0:
                ax.set_title(f"{modality}\n{stimulus}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{cell}\nΔF/F %", fontsize=8)
            if r == 5:
                ax.set_xlabel("time (s)", fontsize=8)
    fig.suptitle("Yang 2016 Fig. S5 B/E (digitized)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
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
