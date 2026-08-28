#!/usr/bin/env python3
"""Digitize Figure 1 Ci/Cii traces from a rendered Figure 1 PNG.

Extracts 16 population traces (PD=red, ND=blue) by reading the published
figure. Values are approximate — digitized from raster, not raw data.

Usage:
  ../.venv/bin/python 1ci.py
  ../.venv/bin/python 1ci.py --image 1ci.png --debug
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d, median_filter

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / "1ci.png"
FIGURE = "1ci"

# Digitized panel x-axis span (ms); pixel frac maps to 0 .. FIG1_PANEL_SPAN_MS.
FIG1_PANEL_SPAN_MS = 900.0
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

# The T5 NC w1 mean lines are visibly coarsely rasterized: single-pixel stairs
# and narrow print spikes dominate at the native ~1.85 ms sampling interval.
# Keep this correction panel-local so the other seven verified panels are
# byte-for-byte unchanged in the regenerated CSV.
TRACE_SMOOTH_SIGMA_PX = {"T5_NC_w1": 3.0}


@dataclass
class PanelCalib:
    left: int
    right: int
    top: int
    bottom: int
    trace_left: int
    trace_right: int


def crop_panel(img: np.ndarray, spec: PanelSpec) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(spec.y0 * h) : int(spec.y1 * h), int(spec.x0 * w) : int(spec.x1 * w)]


# Core mean-line colours measured from the supplied raster.  The old blue mask
# deliberately excluded the actual dark-blue ink (30, 120, 179) and therefore
# followed the pale SEM ribbon instead.  Tight colour distances retain the
# printed mean while allowing its antialiased edge pixels.
CORE_RGB = {
    "red": np.array((226, 26, 27), dtype=float),
    "blue": np.array((30, 120, 179), dtype=float),
}
CORE_MAX_DISTANCE = 18.0


def core_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    delta = rgb.astype(float) - CORE_RGB[color]
    return np.sqrt(np.sum(delta * delta, axis=2)) <= CORE_MAX_DISTANCE


def red_mask(rgb: np.ndarray) -> np.ndarray:
    return core_mask(rgb, "red")


def blue_mask(rgb: np.ndarray) -> np.ndarray:
    return core_mask(rgb, "blue")


def line_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    return red_mask(rgb) if color == "red" else blue_mask(rgb)


def trace_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    ch, cw = rgb.shape[:2]
    red = red_mask(rgb)
    blue = blue_mask(rgb)
    mask = red | blue
    mask[int(ch * 0.78) :, :] = False

    # Core-colour masks exclude SEM and annotations, so even a one-pixel-thick
    # visible endpoint is a valid trace boundary.
    cols = np.where(mask.any(axis=0))[0]

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
        return 0.0
    frac = (x_crop - calib.trace_left) / span
    return frac * FIG1_PANEL_SPAN_MS


def extract_trace(
    crop: np.ndarray,
    color: str,
    calib: PanelCalib,
    *,
    smooth_sigma_px: float = 0.0,
    baseline_y: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_crop, y_crop = extract_trace_pixels(
        crop, color, calib, smooth_sigma_px=smooth_sigma_px
    )
    if len(x_crop) < 50:
        return np.array([]), np.array([])

    time_ms = np.array([pixel_to_time_ms(x, calib) for x in x_crop])
    if baseline_y is None:
        baseline_y = float(y_crop[0])
    vm_mv = (baseline_y - y_crop) / PX_PER_MV
    return time_ms, vm_mv


def extract_trace_pixels(
    crop: np.ndarray,
    color: str,
    calib: PanelCalib,
    *,
    smooth_sigma_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the core mean line in source-crop pixel coordinates.

    A column can contain several antialiased fragments, so contiguous runs are
    reduced to centres before selecting the run nearest the preceding point.
    Missing columns are interpolated only between visible mean-line pixels;
    nothing is extrapolated beyond the printed trace.
    """
    sub = crop[calib.top : calib.bottom, calib.left : calib.right]
    mask = line_mask(sub, color)
    pw = sub.shape[1]
    xs: list[float] = []
    ys: list[float] = []
    prev: float | None = None
    for x in range(pw):
        rows = np.where(mask[:, x])[0]
        if len(rows) == 0:
            continue
        runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
        centres = np.asarray([np.median(run) for run in runs], dtype=float)
        y = float(np.median(rows)) if prev is None else float(
            centres[np.argmin(np.abs(centres - prev))]
        )
        xs.append(float(x))
        ys.append(y)
        prev = y

    if len(xs) < 50:
        return np.array([]), np.array([])

    xs_a = np.asarray(xs)
    ys_a = np.asarray(ys, dtype=float)
    full_x = np.arange(int(xs_a[0]), int(xs_a[-1]) + 1, dtype=float)
    full_y = np.interp(full_x, xs_a, ys_a)
    full_y = median_filter(full_y, size=3, mode="nearest")
    if smooth_sigma_px > 0:
        full_y = gaussian_filter1d(full_y, sigma=smooth_sigma_px, mode="nearest")
    return calib.left + full_x, calib.top + full_y


def digitize(img: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in PANELS:
        crop = crop_panel(img, spec)
        calib = calibrate_panel(crop)
        shared_baseline_y: float | None = None
        if spec.key == "T5_NC_w1":
            baseline_pixels = []
            for color in ("red", "blue"):
                _, y_pixels = extract_trace_pixels(
                    crop,
                    color,
                    calib,
                    smooth_sigma_px=TRACE_SMOOTH_SIGMA_PX[spec.key],
                )
                baseline_pixels.append(y_pixels[:60])
            # One panel-local zero preserves the printed vertical separation
            # between PD and ND.  Separate per-trace zeros falsely made the
            # first ~400 ms overlap in the digitized-only plot.
            shared_baseline_y = float(np.median(np.concatenate(baseline_pixels)))
        for direction, color in (("PD", "red"), ("ND", "blue")):
            t, v = extract_trace(
                crop,
                color,
                calib,
                smooth_sigma_px=TRACE_SMOOTH_SIGMA_PX.get(spec.key, 0.0),
                baseline_y=shared_baseline_y,
            )
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
        ax.set_xlim(0.0, FIG1_PANEL_SPAN_MS)
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
    fig.suptitle(f"Yellow=extract box, magenta=trace span (0..{FIG1_PANEL_SPAN_MS:.0f} ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_overlay(img: np.ndarray, path: Path) -> None:
    """Draw every extracted path over its original panel for visual review."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    idx = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    for spec, (row, col) in zip(PANELS, idx):
        crop = crop_panel(img, spec)
        calib = calibrate_panel(crop)
        ax = axes[row, col]
        ax.imshow(crop)
        for color, diagnostic_color in (("red", "lime"), ("blue", "yellow")):
            x, y = extract_trace_pixels(
                crop,
                color,
                calib,
                smooth_sigma_px=TRACE_SMOOTH_SIGMA_PX.get(spec.key, 0.0),
            )
            ax.plot(x, y, color=diagnostic_color, lw=0.8)
        ax.set_title(spec.key, fontsize=8)
        ax.axis("off")
    fig.suptitle("Mean-line pixel overlay: lime=PD/red, yellow=ND/blue")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_t5_nc_w1_review(img: np.ndarray, df: pd.DataFrame, path: Path) -> None:
    """Save an enlarged source overlay and calibrated T5 NC w1 comparison."""
    spec = next(panel for panel in PANELS if panel.key == "T5_NC_w1")
    crop = crop_panel(img, spec)
    calib = calibrate_panel(crop)
    fig, (source_ax, trace_ax) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
    )
    source_ax.imshow(crop)
    for color, diagnostic_color in (("red", "lime"), ("blue", "yellow")):
        x, y = extract_trace_pixels(
            crop,
            color,
            calib,
            smooth_sigma_px=TRACE_SMOOTH_SIGMA_PX[spec.key],
        )
        source_ax.plot(x, y, color=diagnostic_color, lw=0.8)
    source_ax.set_xlim(0, crop.shape[1] - 1)
    source_ax.set_ylim(360, 220)
    source_ax.set_title("T5 NC w1 source-pixel overlay (lime=PD, yellow=ND)")
    source_ax.axis("off")

    for direction, color in (("PD", "red"), ("ND", "blue")):
        trace_id = f"T5_NC_w1_{direction}"
        group = df[df.trace_id == trace_id].sort_values("time_ms")
        trace_ax.plot(group.time_ms, group.vm_mv, color=color, lw=1.5, label=direction)
    trace_ax.axhline(0, color="0.8", lw=0.7)
    trace_ax.set_xlim(0, FIG1_PANEL_SPAN_MS)
    trace_ax.set_xlabel("time (ms)")
    trace_ax.set_ylabel("Vm (mV)")
    trace_ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def validate(df: pd.DataFrame) -> None:
    expected = {
        f"{spec.cell_type}_{spec.contrast}_w{spec.width_led}_{direction}"
        for spec in PANELS
        for direction in ("PD", "ND")
    }
    actual = set(df.trace_id.unique())
    if actual != expected:
        raise ValueError(f"trace IDs differ: missing={expected-actual}, extra={actual-expected}")
    numeric = df[["time_ms", "vm_mv"]].to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite digitized values")
    if df.duplicated(["trace_id", "time_ms"]).any():
        raise ValueError("duplicate (trace_id, time_ms) rows")
    for trace_id, group in df.groupby("trace_id"):
        if not np.all(np.diff(group.time_ms.to_numpy()) > 0):
            raise ValueError(f"non-monotonic time for {trace_id}")


def save_npz(df: pd.DataFrame, path: Path) -> None:
    payload = {}
    for tid, grp in df.groupby("trace_id"):
        grp = grp.sort_values("time_ms")
        payload[f"{tid}__time_ms"] = grp.time_ms.to_numpy()
        payload[f"{tid}__vm_mv"] = grp.vm_mv.to_numpy()
    np.savez_compressed(path, **payload)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE / f"{FIGURE}_digitized")
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    img_path = args.image
    if not img_path.exists():
        print(f"error: image not found: {img_path}", file=sys.stderr)
        return 2
    img = np.array(Image.open(img_path).convert("RGB"))

    df = digitize(img)
    validate(df)
    n = df.trace_id.nunique() if not df.empty else 0
    if n < 16:
        print(f"warning: only {n}/16 traces extracted", file=sys.stderr)

    df.to_csv(args.out.with_suffix(".csv"), index=False)
    save_npz(df, args.out.with_suffix(".npz"))
    plot_check(df, args.out.with_suffix(".png"))
    save_overlay(img, args.out.with_name(args.out.name + "_overlay.png"))
    save_t5_nc_w1_review(
        img, df, args.out.with_name(args.out.name + "_T5_NC_w1_review.png")
    )
    if args.debug:
        save_debug(img, args.out.with_name(args.out.name + "_debug.png"))

    print(f"Wrote {n} traces -> {args.out}.csv  (time=0..{FIG1_PANEL_SPAN_MS:.0f} ms)")
    if not df.empty:
        peak_idx = df.groupby("trace_id")["vm_mv"].idxmax()
        trough_idx = df.groupby("trace_id")["vm_mv"].idxmin()
        summary = df.groupby("trace_id").agg(
            n=("time_ms", "size"),
            t_start=("time_ms", "min"),
            t_end=("time_ms", "max"),
            v_min=("vm_mv", "min"),
            v_max=("vm_mv", "max"),
        )
        summary["t_min"] = df.loc[trough_idx].set_index("trace_id")["time_ms"]
        summary["t_max"] = df.loc[peak_idx].set_index("trace_id")["time_ms"]
        print(
            summary.loc[:, ["n", "t_start", "t_end", "v_min", "t_min", "v_max", "t_max"]]
            .to_string()
        )
    return 0 if n >= 14 else 1


if __name__ == "__main__":
    raise SystemExit(main())
