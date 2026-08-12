"""Digitize Gruntman et al. 2018 Fig. 1c population Vm traces from 1c.png.

Fig. 1c is a 4 (step duration / speed) x 8 (motion direction, PD-aligned to
rightward) grid of mean +/- SEM membrane-potential traces.

Method
------
1. Panel geometry is hand-measured on 1c.png around the 32 traces.  In
   particular, the first row starts above the tall 45/90-degree peaks; the
   previous crop clipped both peaks and shifted every direction one column.
2. Scale bars bottom-left: 500 ms = 41 px, 10 mV = 122 px.
3. Each row has a repeatedly printed core trace colour.  Per panel, retain the
   largest connected component of that colour, then take its median y in each
   x-column.  This intentionally figure-specific patch excludes direction text
   and the solid stimulus-duration bars without building a general digitizer.
4. Baseline = gray horizontal rule; up = positive Vm; subtract the first
   sample so each trace starts at 0.
4. Write long CSV + verification PNG as 1c_digitized.*.

Run:  ../.venv/bin/python 1c.py
      ../.venv/bin/python 1c.py --debug
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label, median_filter

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / "1c.png"
FIGURE = "1c"
OUT_STEM = HERE / f"{FIGURE}_digitized"

# Scale bars in 1c.png (bottom-left).  Pixel coordinates are measured between
# the centre lines of the black bars: x=112..153 and y=1207..1329.
PX_PER_MS = 41.0 / 500.0
PX_PER_MV = 122.0 / 10.0
MIN_TRACE_POINTS = 30

# Hand-measured trace panels in 1c.png.  These are deliberately not equal-sized
# generic grid cells: they surround the actual traces and exclude labels.
PANEL_X = (
    (120, 320),
    (320, 520),
    (520, 720),
    (720, 920),
    (920, 1128),
    (1128, 1328),
    (1328, 1528),
    (1528, 1732),
)
ROW_Y = (80, 360, 720, 1060, 1310)

# Exact core ink colours repeated across every trace in a row.  The SEM ribbon
# and anti-aliasing have other colours; using the core keeps the extracted
# values on the printed mean line rather than on a ribbon edge.
TRACE_CORE_RGB = (
    np.array((50, 4, 2), dtype=float),
    np.array((107, 17, 10), dtype=float),
    np.array((157, 30, 20), dtype=float),
    np.array((197, 41, 28), dtype=float),
)
TRACE_CORE_MAX_DISTANCE = 22.0

DIRECTIONS_DEG = ("315", "0", "45", "90", "135", "180", "225", "270")
ROWS = (
    (160, 14),
    (80, 28),
    (40, 56),
    (20, 112),
)


def trace_mask(rgb: np.ndarray, row_idx: int) -> np.ndarray:
    delta = rgb.astype(float) - TRACE_CORE_RGB[row_idx]
    return np.sqrt(np.sum(delta * delta, axis=2)) <= TRACE_CORE_MAX_DISTANCE


def largest_trace_component(mask: np.ndarray) -> np.ndarray:
    """Keep the widest 1c ink component and discard labels/stimulus bars.

    The stimulus bars contain more pixels than some traces, but are always
    horizontally shorter.  Selecting by x span also preserves the genuine
    below-baseline tails of the 160-ms traces.
    """
    labels, n_label = label(mask, structure=np.ones((3, 3), dtype=int))
    if n_label == 0:
        return np.zeros_like(mask)
    best_label = 0
    best_score = (-1, -1)
    for component_id in range(1, n_label + 1):
        ys, xs = np.where(labels == component_id)
        if len(xs) == 0:
            continue
        score = (int(xs.max() - xs.min()), len(xs))
        if score > best_score:
            best_label = component_id
            best_score = score
    return labels == best_label


def baseline_y(panel: np.ndarray) -> float:
    gray = panel.mean(2)
    red = panel[:, :, 0]
    green = panel[:, :, 1]
    blue = panel[:, :, 2]
    mx = np.maximum(np.maximum(red, green), blue)
    mn = np.minimum(np.minimum(red, green), blue)
    sat = (mx - mn).astype(float) / (mx + 1e-6)
    rule = (sat < 0.08) & (gray > 140) & (gray < 210)
    ys = np.where(rule.sum(1) > panel.shape[1] * 0.30)[0]
    if len(ys):
        return float(np.median(ys))
    return float(panel.shape[0] * 0.75)


def trace_x_span(mask: np.ndarray) -> tuple[int, int] | tuple[None, None]:
    active = np.where(mask.any(0))[0]
    if len(active) < 40:
        return None, None
    return int(active[0]), int(active[-1])


def extract_panel(
    panel: np.ndarray,
    row_idx: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    mask = largest_trace_component(trace_mask(panel, row_idx))
    xl, xr = trace_x_span(mask)
    if xl is None:
        return None, None
    xs, ys = [], []
    for x in range(xl, xr + 1):
        rows = np.where(mask[:, x])[0]
        if len(rows) < 1:
            continue
        y = float(np.median(rows))
        xs.append(float(x))
        ys.append(y)
    if len(xs) < MIN_TRACE_POINTS:
        return None, None
    xs_arr = np.asarray(xs)
    ys_arr = median_filter(np.asarray(ys, dtype=float), size=5)
    base = baseline_y(panel)
    vm_mv = (base - ys_arr) / PX_PER_MV
    vm_mv = vm_mv - vm_mv[0]
    time_ms = (xs_arr - xl) / PX_PER_MS
    return time_ms, vm_mv


def digitize(img: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for row_idx, (step_ms, speed_dps) in enumerate(ROWS):
        y0, y1 = ROW_Y[row_idx], ROW_Y[row_idx + 1]
        for col_idx, direction_deg in enumerate(DIRECTIONS_DEG):
            x0, x1 = PANEL_X[col_idx]
            panel = img[y0:y1, x0:x1]
            time_ms, vm_mv = extract_panel(panel, row_idx)
            if time_ms is None:
                print(
                    f"  !! no trace for step={step_ms}ms dir={direction_deg}°"
                )
                continue
            trace_id = f"step{step_ms}_dir{direction_deg}"
            for time_i, vm_i in zip(time_ms, vm_mv):
                rows.append(
                    {
                        "trace_id": trace_id,
                        "step_ms": step_ms,
                        "speed_dps": speed_dps,
                        "direction_deg": direction_deg,
                        "time_ms": float(time_i),
                        "vm_mv": float(vm_i),
                    }
                )
            print(
                f"  step={step_ms:3d}ms dir={direction_deg:>3s}° "
                f"n={len(time_ms):3d} peak={vm_mv.max():5.1f} mV"
            )
    return pd.DataFrame(rows)


def plot_check(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(4, 8, figsize=(16, 8), sharex=True, sharey=True)
    ymax = max(5.0, float(df.vm_mv.max()) * 1.1) if not df.empty else 10.0
    ymin = min(-1.0, float(df.vm_mv.min()) * 1.1) if not df.empty else -1.0
    xmax = max(500.0, float(df.time_ms.max())) if not df.empty else 2500.0
    x_stop = float(np.ceil(xmax / 500.0) * 500.0)
    x_ticks = np.arange(0.0, x_stop + 1.0, 500.0)
    y_start = float(np.floor(ymin / 2.0) * 2.0)
    y_stop = float(np.ceil(ymax / 2.0) * 2.0)
    y_ticks = np.arange(y_start, y_stop + 1.0, 2.0)
    for row_idx, (step_ms, speed_dps) in enumerate(ROWS):
        for col_idx, direction_deg in enumerate(DIRECTIONS_DEG):
            ax = axes[row_idx, col_idx]
            trace_id = f"step{step_ms}_dir{direction_deg}"
            sub = df[df.trace_id == trace_id].sort_values("time_ms")
            if not sub.empty:
                color = (
                    "red"
                    if direction_deg == "90"
                    else "blue"
                    if direction_deg == "270"
                    else "0.45"
                )
                ax.plot(sub.time_ms, sub.vm_mv, color=color, lw=1.0)
            ax.axhline(0, color="0.85", lw=0.4)
            if row_idx == 0:
                ax.set_title(f"{direction_deg}°", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{step_ms} ms\n{speed_dps}°/s", fontsize=7)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(0.0, x_stop)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.tick_params(axis="both", labelsize=6, length=2)
    fig.suptitle("Gruntman 2018 Fig. 1c (digitized population Vm)", y=0.98)
    fig.supxlabel("Time (ms)", fontsize=9)
    fig.supylabel("Vm (mV)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_debug(img: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(img)
    for x0, x1 in PANEL_X:
        ax.axvline(x0, color="lime", lw=1.0)
        ax.axvline(x1, color="lime", lw=1.0)
    for y in ROW_Y:
        ax.axhline(y, color="magenta", lw=1.0)
    ax.set_title("lime=column bounds, magenta=row bounds")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--out", type=Path, default=OUT_STEM)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    img = np.array(Image.open(args.image).convert("RGB"))
    print(f"loaded {args.image.name}  {img.shape[1]}x{img.shape[0]}")
    df = digitize(img)
    csv_path = args.out.with_suffix(".csv")
    png_path = args.out.with_suffix(".png")
    df.to_csv(csv_path, index=False)
    plot_check(df, png_path)
    if args.debug:
        save_debug(img, args.out.with_name(args.out.name + "_debug.png"))
    n_trace = df.trace_id.nunique() if not df.empty else 0
    print(f"wrote {csv_path}  ({len(df)} rows, {n_trace} traces)")
    print(f"wrote {png_path}")
    return 0 if n_trace >= 30 else 1


if __name__ == "__main__":
    raise SystemExit(main())
