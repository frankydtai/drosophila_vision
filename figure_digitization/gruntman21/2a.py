#!/usr/bin/env python3
"""Digitize Gruntman et al. 2021 Figure 2A traces from ``2a.png``.

Figure 2A shows baseline-subtracted T4 and T5 responses to 160 ms,
4-LED-wide bar flashes at 12 positions along the PD--ND axis.  Values are
approximate raster digitizations of the published mean traces, not raw data,
resampled at 5 ms intervals.

Outputs ``2a_digitized.csv`` and ``2a_digitized.png`` beside this script.

Run:
    ../.venv/bin/python 2a.py
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
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / "2a.png"
OUT_STEM = HERE / "2a_digitized"

# Measurements from the printed scale bars and stimulus markers.
STIMULUS_MS = 160.0
STIMULUS_PX = 27.0
PX_PER_MS = STIMULUS_PX / STIMULUS_MS
PX_PER_MV = 108.0 / 10.0
SAMPLE_INTERVAL_MS = 5.0

# The 12 small multiples are evenly spaced in the supplied raster.
PANEL_X0 = 187
PANEL_STEP = 174
PANEL_WIDTH = 186
TRACE_ROWS = {"T4": (145, 520), "T5": (520, 990)}
STIM_ONSET_LOCAL_X = 60
POSITION_ZERO_COLUMN = 6

# T4 negative-going pixels extend below the old y=440 crop. Black captions
# below the row overlap some panels, so keep NC extraction inside panel-local
# bounds while allowing green PC traces to use the full extended crop.
T4_NC_MAX_Y = {
    -6: 310,
    -5: 310,
    -4: 315,
    -3: 325,
    -2: 345,
    -1: 335,
    0: 310,
    1: 310,
    2: 310,
    3: 310,
    4: 310,
    5: 310,
}

# Core line colours.  Lighter colours are SEM bands and are intentionally
# excluded; following the core ink gives the published mean response.
CORE_RGB = {
    "green": np.array((84, 159, 92), dtype=float),
    "black": np.array((48, 48, 48), dtype=float),
}
OVERPRINT_GREEN_RGB = np.array((57, 99, 64), dtype=float)
MAX_RGB_DISTANCE = 35.0


def line_mask(panel: np.ndarray, color: str) -> np.ndarray:
    delta = panel.astype(float) - CORE_RGB[color]
    mask = np.sqrt(np.sum(delta * delta, axis=2)) <= MAX_RGB_DISTANCE
    if color == "green":
        # Where the green mean crosses the dark trace/SEM, raster overprinting
        # produces a second, darker green ink colour.
        overprint = panel.astype(float) - OVERPRINT_GREEN_RGB
        mask |= np.sqrt(np.sum(overprint * overprint, axis=2)) <= 24.0
    return mask


def extract_trace(
    panel: np.ndarray,
    color: str,
    *,
    min_x: int = 0,
    min_y: int = 0,
    max_y: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local x/y pixels for one coloured mean trace."""
    mask = line_mask(panel, color)
    mask[:min_y] = False
    if max_y is not None:
        mask[max_y:] = False
    mask[:, :min_x] = False

    def centers_at(x: int) -> np.ndarray:
        rows = np.where(mask[:, x])[0]
        if not len(rows):
            return np.array([])
        runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
        return np.asarray([np.median(run) for run in runs], dtype=float)

    # Seed on the trace baseline just before onset, then track by continuity in
    # both directions.  This rejects scale bars, labels, and isolated dark ink.
    seed_candidates: list[tuple[int, float]] = []
    for sx in range(panel.shape[1]):
        seed_candidates.extend((sx, y) for y in centers_at(sx))
    if not seed_candidates:
        raise ValueError(f"could not seed {color} trace")
    seed_target = STIM_ONSET_LOCAL_X - 10
    seed_x, seed_y = min(
        seed_candidates, key=lambda seed_candidate: (
            abs(seed_candidate[0] - seed_target), -seed_candidate[1]
        )
    )

    points: dict[int, float] = {seed_x: seed_y}
    for direction, stop in ((-1, -1), (1, panel.shape[1])):
        prev = seed_y
        for x in range(seed_x + direction, stop, direction):
            centers = centers_at(x)
            if not len(centers):
                continue
            nearest = float(centers[np.argmin(np.abs(centers - prev))])
            if abs(nearest - prev) <= 80:
                points[x] = nearest
                prev = nearest

    xs = [float(x) for x in sorted(points)]
    ys = [points[int(x)] for x in xs]

    if len(xs) < 10:
        raise ValueError(f"too few {color} trace pixels ({len(xs)})")

    x = np.asarray(xs)
    y = np.asarray(ys)
    # Fill raster/antialiasing gaps but do not extrapolate beyond visible ink.
    full_x = np.arange(int(x[0]), int(x[-1]) + 1, dtype=float)
    full_y = np.interp(full_x, x, y)
    full_y = median_filter(full_y, size=3, mode="nearest")
    return full_x, full_y


def digitize(image: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_type, (y0, y1) in TRACE_ROWS.items():
        colors = {
            "T4": {"PC": "green", "NC": "black"},
            "T5": {"PC": "black", "NC": "green"},
        }[cell_type]
        for column in range(12):
            position = column - POSITION_ZERO_COLUMN
            x0 = PANEL_X0 + column * PANEL_STEP
            panel = image[y0:y1, x0:x0 + PANEL_WIDTH, :3]

            # The coloured T4/T5 legend overlaps the final small multiple.
            legend_floor = 105 if (cell_type == "T4" and column == 11) else 0
            # Adjacent traces overlap by about 12 px in the rendered layout.
            # Ignore the preceding panel's tail and pad with this trace's own
            # pre-stimulus baseline below.
            trace_left = 12 if column > 0 else 0
            trace_bottom = None
            if cell_type == "T5" and column == 0:
                trace_left = 20
                trace_bottom = 400
            if cell_type == "T5" and column == 0:
                # The left-side caption overlaps the first T5 crop.
                legend_floor = 180
            elif cell_type == "T5" and column == 2:
                # The "leading" caption overlaps this crop.
                legend_floor = 100
            elif cell_type == "T5" and column == 6:
                # The "center (position 0)" label overlaps this crop.
                legend_floor = 70
            elif cell_type == "T5" and column == 10:
                # The "trailing" caption overlaps this crop.
                legend_floor = 100
            elif cell_type == "T5" and column == 11:
                # The T5/PC/NC legend overlaps the final small multiple.
                legend_floor = 175

            trace_pixels = {
                contrast: extract_trace(
                    panel,
                    color,
                    min_x=trace_left,
                    min_y=legend_floor,
                    max_y=(
                        T4_NC_MAX_Y[position]
                        if cell_type == "T4" and contrast == "NC"
                        else trace_bottom
                    ),
                )
                for contrast, color in colors.items()
            }
            # PC and NC are overplotted before the flash, so one core colour
            # can be hidden completely.  Use their shared visible baseline
            # rather than treating the first post-stimulus point as rest.
            pre_y = np.concatenate(
                [
                    py[px < STIM_ONSET_LOCAL_X]
                    for px, py in trace_pixels.values()
                    if np.any(px < STIM_ONSET_LOCAL_X)
                ]
            )
            baseline_y = float(np.median(pre_y))

            for contrast, color in colors.items():
                px, py = trace_pixels[contrast]
                time_ms = (px - STIM_ONSET_LOCAL_X) / PX_PER_MS
                if px[0] > 0:
                    pad_x = np.arange(0, px[0], dtype=float)
                    px = np.concatenate((pad_x, px))
                    py = np.concatenate((np.full(len(pad_x), baseline_y), py))
                    time_ms = (px - STIM_ONSET_LOCAL_X) / PX_PER_MS
                vm_mv = (baseline_y - py) / PX_PER_MV
                sample_start_ms = (
                    np.ceil(time_ms[0] / SAMPLE_INTERVAL_MS) * SAMPLE_INTERVAL_MS
                )
                sample_stop_ms = (
                    np.floor(time_ms[-1] / SAMPLE_INTERVAL_MS) * SAMPLE_INTERVAL_MS
                )
                sample_time_ms = np.arange(
                    sample_start_ms,
                    sample_stop_ms + SAMPLE_INTERVAL_MS / 2.0,
                    SAMPLE_INTERVAL_MS,
                )
                vm_mv = np.interp(sample_time_ms, time_ms, vm_mv)
                time_ms = sample_time_ms
                trace_id = f"{cell_type}_{contrast}_pos{position:+d}"
                rf_side = "leading" if position < 0 else (
                    "center" if position == 0 else "trailing"
                )
                for t, v in zip(time_ms, vm_mv):
                    rows.append({
                        "trace_id": trace_id,
                        "cell_type": cell_type,
                        "contrast": contrast,
                        "position": position,
                        "rf_side": rf_side,
                        "color": color,
                        "time_ms": float(t),
                        "vm_mv": float(v),
                    })
    return pd.DataFrame(rows)


def plot_check(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 12, figsize=(18, 5.5), sharex=True, sharey=True)
    palette = {"green": "#549f5c", "black": "#303030"}
    for row, cell_type in enumerate(("T4", "T5")):
        for column, position in enumerate(
            range(-POSITION_ZERO_COLUMN, 12 - POSITION_ZERO_COLUMN)
        ):
            ax = axes[row, column]
            subset = df[(df.cell_type == cell_type) & (df.position == position)]
            for contrast in ("PC", "NC"):
                trace = subset[subset.contrast == contrast]
                color = str(trace.color.iloc[0])
                ax.plot(trace.time_ms, trace.vm_mv, color=palette[color], lw=1.4,
                        label=contrast)
            ax.axvspan(0, STIMULUS_MS, color="0.9", zorder=-1)
            ax.axhline(0, color="0.85", lw=0.6, zorder=-1)
            ax.set_title(f"{position:+d}", fontsize=8)
            if column == 0:
                ax.set_ylabel(f"{cell_type}\nVm (mV)")
            ax.tick_params(labelsize=7)
    axes[0, -1].legend(frameon=False, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("ms", fontsize=8)
    fig.suptitle("Gruntman et al. 2021, Figure 2A raster digitization")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=OUT_STEM)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    df = digitize(image)
    csv_path = args.output.with_suffix(".csv")
    png_path = args.output.with_suffix(".png")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    plot_check(df, png_path)
    print(f"Wrote {csv_path} ({len(df):,} points, {df.trace_id.nunique()} traces)")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
