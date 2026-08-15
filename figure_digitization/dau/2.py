"""Digitize Dau et al. (2016) long-pulse R1--R6 responses in ``2.png``.

The source raster is Figure 4A of the paper: mean in-vivo photoreceptor
voltage responses (grey regions are SEM) to Bright, Mid, and Dim 1 s light
pulses.  Wild type is the black mean line and ``hdcJK910`` is red.

Calibration is measured in this raster.  The stimulus marker rises at
x=171.5 and falls at x=1059.5; its captioned 1 s duration gives 888 px/s.
The printed 15 mV vertical bar spans y=540..665 (125 px).  The printed
200 ms horizontal bar spans x=699..882 (184 px), a 3.6% raster-level check
on the stimulus-marker calibration.

The script extracts only the coloured mean strokes, follows them through
short antialiasing/overlap gaps, writes a tidy CSV, and creates both a
digitized-only check plot and a source-pixel overlay.

Run:  ../.venv/bin/python 2.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "2.png"
OUT_STEM = HERE / "2_digitized"
OVERLAY = HERE / "2_overlay.png"

# Measured from the raster.  The onset/offset marker is used because the
# caption states that the pulse lasted 1 s; the printed 200 ms bar is an
# independent calibration check (184 px versus 177.6 px expected).
STIM_ON_X = 171.5
STIM_OFF_X = 1059.5
STIMULUS_DURATION_S = 1.0
PX_PER_S = (STIM_OFF_X - STIM_ON_X) / STIMULUS_DURATION_S
VOLTAGE_BAR_TOP_Y = 540.0
VOLTAGE_BAR_BOTTOM_Y = 665.0
VOLTAGE_BAR_MV = 15.0
PX_PER_MV = (VOLTAGE_BAR_BOTTOM_Y - VOLTAGE_BAR_TOP_Y) / VOLTAGE_BAR_MV

X_LEFT = 83
X_RIGHT = 1680  # exclusive; last visible mean pixels are at x=1679
ANCHOR_X = 300
BASELINE_X = (90, 155)
MAX_DY_PX = 65.0
MAX_SHORT_GAP_PX = 12


@dataclass(frozen=True)
class TraceSpec:
    intensity: str
    genotype: str
    color: str
    y_top: int
    y_bottom: int
    anchor_y: float
    baseline_seed_y: float

    @property
    def trace_id(self) -> str:
        genotype = "wild_type" if self.genotype == "Wild type" else "hdcJK910"
        return f"{self.intensity.lower()}_{genotype}"


# Panel-local bounds and anchors are measured independently in 2.png.  They
# keep titles, cartoons, the scale bar, and the legend out of the tracked path.
TRACES = [
    TraceSpec("Bright", "Wild type", "black", 100, 610, 379.0, 548.0),
    TraceSpec("Bright", "hdcJK910", "red", 100, 610, 431.0, 560.0),
    TraceSpec("Mid", "Wild type", "black", 600, 1090, 743.0, 1036.0),
    TraceSpec("Mid", "hdcJK910", "red", 600, 1090, 841.0, 1027.0),
    TraceSpec("Dim", "Wild type", "black", 1090, 1410, 1170.0, 1297.0),
    TraceSpec("Dim", "hdcJK910", "red", 1090, 1410, 1224.0, 1291.0),
]


def mean_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Return masks for the black and red printed mean lines, excluding SEM."""
    arr = rgb.astype(np.float64)
    red, green, blue = np.moveaxis(arr, -1, 0)
    maximum = arr.max(axis=2)
    minimum = arr.min(axis=2)
    saturation = (maximum - minimum) / (maximum + 1e-6)

    black = (arr.mean(axis=2) < 100) & (saturation < 0.40)
    red_line = (red > 120) & (red - green > 40) & (red - blue > 25)
    return {"black": black, "red": red_line}


def column_candidates(mask: np.ndarray, x: int, y_top: int, y_bottom: int) -> np.ndarray:
    """Return centers of antialiased stroke fragments in one image column."""
    ys = np.flatnonzero(mask[y_top:y_bottom, x]) + y_top
    if not len(ys):
        return np.empty(0, dtype=float)

    runs: list[tuple[int, int]] = []
    start = previous = int(ys[0])
    for value in ys[1:]:
        y = int(value)
        # A one-pixel antialiasing hole still belongs to the same stroke.
        if y > previous + 2:
            runs.append((start, previous))
            start = y
        previous = y
    runs.append((start, previous))
    return np.asarray([(start + end) / 2 for start, end in runs], dtype=float)


def track_direction(
    mask: np.ndarray,
    x_values: range,
    spec: TraceSpec,
    seed_y: float,
    max_dy: float,
) -> dict[int, float]:
    """Greedily follow the nearest same-colour stroke from a measured anchor."""
    result: dict[int, float] = {}
    previous_y = seed_y
    for x in x_values:
        candidates = column_candidates(mask, x, spec.y_top, spec.y_bottom)
        if not len(candidates):
            continue
        nearest = int(np.argmin(np.abs(candidates - previous_y)))
        y = float(candidates[nearest])
        if abs(y - previous_y) > max_dy:
            continue
        result[x] = y
        previous_y = y
    return result


def fill_short_gaps(points: dict[int, float], max_gap: int = MAX_SHORT_GAP_PX) -> dict[int, float]:
    """Linearly bridge only short colour gaps caused by crossings/antialiasing."""
    filled = dict(points)
    xs = sorted(points)
    for left, right in zip(xs[:-1], xs[1:]):
        missing = right - left - 1
        if not (0 < missing <= max_gap):
            continue
        y_left = points[left]
        y_right = points[right]
        for x in range(left + 1, right):
            fraction = (x - left) / (right - left)
            filled[x] = y_left + fraction * (y_right - y_left)
    return filled


def extract_trace(mask: np.ndarray, spec: TraceSpec) -> tuple[np.ndarray, np.ndarray]:
    """Track a trace in both directions from its panel-local response anchor."""
    points = track_direction(
        mask, range(ANCHOR_X, X_RIGHT), spec, spec.anchor_y, MAX_DY_PX
    )
    points.update(
        track_direction(
            mask, range(ANCHOR_X - 1, 181, -1), spec, spec.anchor_y, MAX_DY_PX
        )
    )

    # The onset is nearly vertical, so track the visibly plotted pre-stimulus
    # mean independently instead of replacing missing pixels with zero.
    points.update(
        track_direction(mask, range(X_LEFT, 182), spec, spec.baseline_seed_y, 30.0)
    )
    points = fill_short_gaps(points)

    xs = np.asarray(sorted(points), dtype=float)
    ys = np.asarray([points[int(x)] for x in xs], dtype=float)
    if len(xs) < 1400:
        raise RuntimeError(f"Only {len(xs)} pixels tracked for {spec.trace_id}")
    return xs, ys


def digitize() -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rgb = np.asarray(Image.open(SOURCE).convert("RGB"))
    masks = mean_masks(rgb)
    rows: list[dict[str, object]] = []
    pixel_paths: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for spec in TRACES:
        xs, ys = extract_trace(masks[spec.color], spec)
        pre = (xs >= BASELINE_X[0]) & (xs <= BASELINE_X[1])
        if pre.sum() < 30:
            raise RuntimeError(f"Insufficient visible baseline for {spec.trace_id}")
        baseline_y = float(np.median(ys[pre]))
        times = (xs - STIM_ON_X) / PX_PER_S
        voltage = (baseline_y - ys) / PX_PER_MV
        pixel_paths[spec.trace_id] = (xs, ys)

        for x, y, time_s, voltage_mv in zip(xs, ys, times, voltage):
            rows.append(
                {
                    "panel": "4A",
                    "intensity": spec.intensity,
                    "genotype": spec.genotype,
                    "trace_id": spec.trace_id,
                    "time_s": float(time_s),
                    "voltage_mV": float(voltage_mv),
                    "source_x_px": float(x),
                    "source_y_px": float(y),
                    "baseline_y_px": baseline_y,
                    "stimulus_onset_s": 0.0,
                    "stimulus_offset_s": STIMULUS_DURATION_S,
                    "stimulus_duration_s": STIMULUS_DURATION_S,
                    "stimulus_level": spec.intensity,
                    "stimulus_wavelength_nm": 525,
                    "recording_temperature_C": 19,
                    "summary_statistic": "mean",
                    "sem_band_present": True,
                }
            )

    return pd.DataFrame(rows), pixel_paths


def validate(df: pd.DataFrame) -> None:
    expected = {spec.trace_id for spec in TRACES}
    found = set(df["trace_id"].unique())
    if found != expected:
        raise AssertionError(f"Expected traces {sorted(expected)}, found {sorted(found)}")
    numeric = df[["time_s", "voltage_mV", "source_x_px", "source_y_px"]]
    if not np.isfinite(numeric.to_numpy()).all():
        raise AssertionError("Non-finite numeric value in digitized data")
    if df.duplicated(["trace_id", "time_s"]).any():
        raise AssertionError("Duplicate (trace_id, time_s) rows")

    print("trace statistics:")
    for spec in TRACES:
        trace = df[df.trace_id == spec.trace_id].sort_values("time_s")
        times = trace.time_s.to_numpy()
        values = trace.voltage_mV.to_numpy()
        if not np.all(np.diff(times) > 0):
            raise AssertionError(f"Non-monotonic time in {spec.trace_id}")
        peak = int(np.argmax(values))
        trough = int(np.argmin(values))
        print(
            f"  {spec.trace_id:22s} n={len(trace):4d}  "
            f"t={times[0]:+.4f}..{times[-1]:+.4f} s  "
            f"min={values[trough]:+.2f} mV @ {times[trough]:+.4f} s  "
            f"max={values[peak]:+.2f} mV @ {times[peak]:+.4f} s"
        )


def plot_check(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8), sharex=True, sharey=True)
    colors = {"Wild type": "black", "hdcJK910": "#e13b2f"}
    for ax, intensity in zip(axes, ("Bright", "Mid", "Dim")):
        for genotype in ("Wild type", "hdcJK910"):
            trace = df[(df.intensity == intensity) & (df.genotype == genotype)]
            trace = trace.sort_values("time_s")
            ax.plot(
                trace.time_s,
                trace.voltage_mV,
                color=colors[genotype],
                lw=1.25,
                label=genotype,
            )
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axvspan(0, STIMULUS_DURATION_S, color="#f4d03f", alpha=0.13)
        ax.set_title(intensity, loc="left")
        ax.set_ylabel("voltage (mV)")
        ax.grid(alpha=0.14, linewidth=0.5)
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("time from stimulus onset (s)")
    axes[-1].set_xlim(-0.11, 1.71)
    axes[-1].set_ylim(-6, 52)
    fig.suptitle("Dau et al. 2016 Fig. 4A mean R1–R6 responses (digitized)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_overlay(
    pixel_paths: dict[str, tuple[np.ndarray, np.ndarray]], path: Path
) -> None:
    image = Image.open(SOURCE).convert("RGB")
    draw = ImageDraw.Draw(image)
    diagnostic = {"Wild type": (0, 235, 90), "hdcJK910": (0, 210, 255)}
    for spec in TRACES:
        xs, ys = pixel_paths[spec.trace_id]
        points = [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]
        draw.line(points, fill=diagnostic[spec.genotype], width=2)
    image.save(path)


def main() -> int:
    df, pixel_paths = digitize()
    validate(df)
    csv_path = OUT_STEM.with_suffix(".csv")
    check_path = OUT_STEM.with_suffix(".png")
    df.to_csv(csv_path, index=False)
    plot_check(df, check_path)
    plot_overlay(pixel_paths, OVERLAY)
    print(f"wrote {csv_path} ({len(df)} rows, {df.trace_id.nunique()} traces)")
    print(f"wrote {check_path}")
    print(f"wrote {OVERLAY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
