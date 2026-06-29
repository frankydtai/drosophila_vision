#!/usr/bin/env python3
"""Digitize Figure 1 Ci/Cii traces from Gruntman et al. 2021 paper.pdf.

Extracts 16 population traces (PD=red, ND=blue) by reading the published
figure. Values are approximate — digitized from raster, not raw data.

Usage:
  .venv/bin/python digitize_fig1_ci.py
  .venv/bin/python digitize_fig1_ci.py --debug
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
DEFAULT_PDF = HERE / "paper.pdf"
DEFAULT_PAGE = 4
DEFAULT_DPI = 400

# Trace window length (ms). Time axis starts at 0 at the first extracted sample.
TIME_MIN_MS = 0.0
TIME_MAX_MS = 900.0
# 10 mV scale bar ≈ 100 px on 400 dpi render (measured on w1 panels).
PX_PER_MV = 10.0


# Subplot column bounds measured on 400 dpi render (page 3350 px wide).
# Old hand-tuned fractions used 0.25 for T4 w1 but 0.15 for T5 w1 — wrong.
T4_W1_X = (0.1591, 0.3024)  # ~480 px
T4_W4_X = (0.3224, 0.4681)  # ~488 px
T5_W1_X = (0.5499, 0.6955)  # ~487 px
T5_W4_X = (0.7185, 0.8630)  # ~484 px
PC_ROW_Y = (0.284, 0.396)  # was 0.310 — too low; clipped w4 PD peaks at crop top
NC_ROW_Y = (0.395, 0.500)


@dataclass(frozen=True)
class PanelSpec:
    key: str
    cell_type: str
    panel: str
    contrast: str
    width_led: int
    y0: float
    y1: float
    x0: float
    x1: float


# (y0, y1, x0, x1) — x from measured Ci/Cii columns (see fig1_panel_crop_diagnosis.txt)
PANELS: tuple[PanelSpec, ...] = (
    PanelSpec("T4_PC_w1", "T4", "Ci", "PC", 1, *PC_ROW_Y, *T4_W1_X),
    PanelSpec("T4_PC_w4", "T4", "Ci", "PC", 4, *PC_ROW_Y, *T4_W4_X),
    PanelSpec("T4_NC_w1", "T4", "Ci", "NC", 1, *NC_ROW_Y, *T4_W1_X),
    PanelSpec("T4_NC_w4", "T4", "Ci", "NC", 4, *NC_ROW_Y, *T4_W4_X),
    PanelSpec("T5_PC_w1", "T5", "Cii", "PC", 1, *PC_ROW_Y, *T5_W1_X),
    PanelSpec("T5_PC_w4", "T5", "Cii", "PC", 4, *PC_ROW_Y, *T5_W4_X),
    PanelSpec("T5_NC_w1", "T5", "Cii", "NC", 1, *NC_ROW_Y, *T5_W1_X),
    PanelSpec("T5_NC_w4", "T5", "Cii", "NC", 4, *NC_ROW_Y, *T5_W4_X),
)


@dataclass
class PanelCalib:
    left: int
    right: int
    top: int
    bottom: int
    trace_left: int
    trace_right: int


def render_pdf_page(pdf: Path, page: int, dpi: int, out_prefix: Path) -> Path:
    png = Path(f"{out_prefix}-{page:02d}.png")
    if png.exists() and png.stat().st_mtime >= pdf.stat().st_mtime:
        return png
    subprocess.run(
        ["pdftoppm", "-f", str(page), "-l", str(page), "-png", "-r", str(dpi), str(pdf), str(out_prefix)],
        check=True,
    )
    return png


def crop_panel(img: np.ndarray, spec: PanelSpec) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(spec.y0 * h) : int(spec.y1 * h), int(spec.x0 * w) : int(spec.x1 * w)]


def red_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0].astype(int), rgb[:, :, 1].astype(int), rgb[:, :, 2].astype(int)
    return (r > 200) & (g < 110) & (b < 110) & (r > g + 40) & (r > b + 40)


def blue_mask(rgb: np.ndarray) -> np.ndarray:
    """ND traces are gray-blue in the published figure."""
    r, g, b = rgb[:, :, 0].astype(int), rgb[:, :, 1].astype(int), rgb[:, :, 2].astype(int)
    not_red = ~red_mask(rgb)
    return (
        not_red
        & (r > 105)
        & (r < 200)
        & (g > 95)
        & (b > 95)
        & (b >= r - 8)
        & (b >= g - 12)
        & ((b - r) + (b - g) > 3)
    )


def line_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    return red_mask(rgb) if color == "red" else blue_mask(rgb)


# Max row step between adjacent columns (~2.5 mV); skips SEM double-blobs / gaps.
MAX_DY_PX = 25


def trace_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    ch, cw = rgb.shape[:2]
    red = red_mask(rgb)
    blue = blue_mask(rgb)
    mask = red | blue
    mask[int(ch * 0.78) :, :] = False

    # x: ignore sparse baseline speckle at plot margins (was forcing xl=0 on w4).
    strength = np.maximum(red.sum(axis=0), blue.sum(axis=0))
    thr = max(3, int(strength.max() * 0.08))
    active_x = strength >= thr
    if int(active_x.sum()) < 20:
        active_x = mask.sum(axis=0) > 2
    cols = np.where(active_x)[0]

    rows = np.where(mask.sum(axis=1) > 2)[0]
    if len(cols) < 20 or len(rows) < 5:
        raise ValueError("could not find trace pixels in panel crop")

    return int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])


def calibrate_panel(crop: np.ndarray) -> PanelCalib:
    ch, cw = crop.shape[:2]
    xl, xr, yt, yb = trace_bbox(crop)
    color_mask = red_mask(crop) | blue_mask(crop)
    color_mask[int(ch * 0.78) :, :] = False
    ys, _ = np.where(color_mask)
    if len(ys):
        yt = min(yt, int(ys.min()))
        yb = max(yb, int(ys.max()))

    pad_x = max(4, int((xr - xl) * 0.015))
    pad_y = max(8, int((yb - yt) * 0.15))

    left = max(0, xl - pad_x)
    right = min(cw, xr + pad_x)
    top = max(0, yt - pad_y)
    bottom = min(ch, yb + pad_y)

    return PanelCalib(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        trace_left=xl,
        trace_right=xr,
    )


def pixel_to_time_ms(x_crop: float, calib: PanelCalib) -> float:
    span = calib.trace_right - calib.trace_left
    if span <= 0:
        return TIME_MIN_MS
    frac = (x_crop - calib.trace_left) / span
    return TIME_MIN_MS + frac * (TIME_MAX_MS - TIME_MIN_MS)


def extract_trace(
    crop: np.ndarray,
    color: str,
    calib: PanelCalib,
) -> tuple[np.ndarray, np.ndarray]:
    sub = crop[calib.top : calib.bottom, calib.left : calib.right]
    mask = line_mask(sub, color)
    pw = sub.shape[1]
    xs, ys, prev = [], [], None
    for x in range(pw):
        rows = np.where(mask[:, x])[0]
        if len(rows) == 0:
            continue
        if prev is None:
            y = float(np.median(rows))
        else:
            near = rows[np.abs(rows - prev) <= MAX_DY_PX]
            if len(near) == 0:
                continue  # gap or second blob — do not jump
            y = float(np.median(near))
        xs.append(float(x))
        ys.append(y)
        prev = y

    if len(xs) < 50:
        return np.array([]), np.array([])

    xs_a = np.asarray(xs)
    ys_a = median_filter(np.asarray(ys, dtype=float), size=9)
    x_crop = calib.left + xs_a
    time_ms = np.array([pixel_to_time_ms(x, calib) for x in x_crop])
    time_ms = time_ms - time_ms[0]  # first sample = 0 ms
    # Baseline-subtracted traces: first sample = 0 mV (matches the paper).
    vm_mv = (ys_a[0] - ys_a) / PX_PER_MV
    return time_ms, vm_mv


def digitize(img: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in PANELS:
        crop = crop_panel(img, spec)
        calib = calibrate_panel(crop)
        for direction, color in (("PD", "red"), ("ND", "blue")):
            t, v = extract_trace(crop, color, calib)
            if len(t) == 0:
                continue
            tid = f"{spec.cell_type}_{spec.contrast}_w{spec.width_led}_{direction}"
            for ti, vi in zip(t, v):
                rows.append(
                    {
                        "trace_id": tid,
                        "cell_type": spec.cell_type,
                        "panel": spec.panel,
                        "contrast": spec.contrast,
                        "width_led": spec.width_led,
                        "direction": direction,
                        "color": color,
                        "time_ms": float(ti),
                        "vm_mv": float(vi),
                    }
                )
    return pd.DataFrame(rows)


def vm_ylim(df: pd.DataFrame, margin_mv: float = 2.0) -> tuple[float, float]:
    bound = max(abs(df.vm_mv.min()), abs(df.vm_mv.max())) + margin_mv
    bound = float(np.ceil(bound / 5.0) * 5.0)
    return -bound, bound


def plot_check(df: pd.DataFrame, path: Path) -> None:
    layout = [
        ("T4", "PC", 1, 0, 0),
        ("T4", "PC", 4, 0, 1),
        ("T4", "NC", 1, 1, 0),
        ("T4", "NC", 4, 1, 1),
        ("T5", "PC", 1, 0, 2),
        ("T5", "PC", 4, 0, 3),
        ("T5", "NC", 1, 1, 2),
        ("T5", "NC", 4, 1, 3),
    ]
    ylo, yhi = vm_ylim(df)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    for ct, contrast, width, row, col in layout:
        ax = axes[row, col]
        for direction, color in (("PD", "red"), ("ND", "blue")):
            tid = f"{ct}_{contrast}_w{width}_{direction}"
            sub = df[df.trace_id == tid].sort_values("time_ms")
            if sub.empty:
                continue
            ax.plot(sub.time_ms, sub.vm_mv, color=color, lw=2, label=direction)
        ax.axhline(0, color="0.8", lw=0.7)
        ax.set_title(f"{ct} {contrast} w{width}", fontsize=9)
        ax.set_xlabel("time (ms)")
        if col == 0:
            ax.set_ylabel("Vm (mV)")
        ax.set_xlim(0, TIME_MAX_MS)
        ax.set_ylim(ylo, yhi)
        ax.legend(fontsize=7)
    fig.suptitle(f"Digitized Figure 1 Ci/Cii  (Vm: {ylo:.0f}..{yhi:.0f} mV)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_debug(img: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    idx = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    for spec, (row, col) in zip(PANELS, idx):
        crop = crop_panel(img, spec)
        calib = calibrate_panel(crop)
        ax = axes[row, col]
        ax.imshow(crop)
        l, r, t, b = calib.left, calib.right, calib.top, calib.bottom
        ax.plot([l, r, r, l, l], [t, t, b, b, t], "y-", lw=1.2)
        ax.plot([calib.trace_left, calib.trace_right, calib.trace_right, calib.trace_left, calib.trace_left],
                [t, t, b, b, t], "m--", lw=1.0)
        ax.set_title(spec.key, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Yellow=extract box, magenta=trace span ({TIME_MIN_MS:.0f}..{TIME_MAX_MS:.0f} ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_npz(df: pd.DataFrame, path: Path) -> None:
    payload = {}
    for tid, grp in df.groupby("trace_id"):
        grp = grp.sort_values("time_ms")
        payload[f"{tid}__time_ms"] = grp.time_ms.to_numpy()
        payload[f"{tid}__vm_mv"] = grp.vm_mv.to_numpy()
    np.savez_compressed(path, **payload)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    p.add_argument("--page", type=int, default=DEFAULT_PAGE)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--out", type=Path, default=HERE / "fig1_ci_digitized")
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    img_path = args.image or render_pdf_page(args.pdf, args.page, args.dpi, HERE / "_fig1_digitize")
    img = np.array(Image.open(img_path).convert("RGB"))

    df = digitize(img)
    n = df.trace_id.nunique() if not df.empty else 0
    if n < 16:
        print(f"warning: only {n}/16 traces extracted", file=sys.stderr)

    df.to_csv(args.out.with_suffix(".csv"), index=False)
    save_npz(df, args.out.with_suffix(".npz"))
    plot_check(df, args.out.with_suffix(".png"))
    if args.debug:
        save_debug(img, args.out.with_name(args.out.name + "_debug.png"))

    print(f"Wrote {n} traces -> {args.out}.csv  (time={TIME_MIN_MS:.0f}..{TIME_MAX_MS:.0f} ms)")
    if not df.empty:
        peak_idx = df.groupby("trace_id")["vm_mv"].idxmax()
        print(
            df.groupby("trace_id")
            .agg(n=("time_ms", "size"), t=("time_ms", "max"), v_peak=("vm_mv", "max"))
            .assign(t_peak=df.loc[peak_idx].set_index("trace_id")["time_ms"])
            .loc[:, ["n", "t", "t_peak", "v_peak"]]
            .to_string()
        )
    return 0 if n >= 14 else 1


if __name__ == "__main__":
    raise SystemExit(main())
