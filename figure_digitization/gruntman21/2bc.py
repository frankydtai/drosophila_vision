#!/usr/bin/env python3
"""Digitize Gruntman et al. 2021 Figure 2B/C mean markers.

Panels 2B and 2C show the maximal depolarizing and hyperpolarizing responses
to 160 ms bar flashes as a function of receptive-field position.  This script
extracts only the large mean markers.  The small individual-cell markers in
panel B and every connecting/reference line are intentionally excluded.

Values are approximate raster digitizations of the published figure, not raw
data.  Outputs ``2bc_digitized.csv``, ``2bc_digitized.png``, and
``2bc_digitized_overlay.png`` beside this script.

Run:
    ../.venv/bin/python 2bc.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.signal import convolve2d

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / "2bc.png"
OUT_STEM = HERE / "2bc_digitized"

FLASH_DURATION_MS = 160.0
DEGREES_PER_LED = 2.25

# Measured directly from the printed axes in 2bc.png.  Tick-pair centers are
# x=(270.5, 392.5, 514.5), (695.5, 817.5, 939.5),
# (1242.5, 1364.5, 1486.5), and (1597.5, 1719.5, 1841.5) for -5, 0, +5 LED.
PX_PER_LED = 122.0 / 5.0
PANEL_X_ZERO = {
    "B_NC": 392.5,
    "B_PC": 817.5,
    "C_W1": 1364.5,
    "C_W2": 1719.5,
}

# The T4 axis has y=(53.5, 260.5, 329.5) at (30, 0, -10) mV; the T5 axis is
# shifted down by 359 px and has the same printed scale.
PX_PER_MV = 207.0 / 30.0
Y_ZERO = {"T4": 260.5, "T5": 619.5}

GREEN = "green"
BLACK = "black"
OVERLAY_RGB = (255, 0, 255)

TEMPLATE_RADIUS_PX = 8
SEARCH_HALF_WIDTH_PX = 2
ZERO_EXCLUSION_PX = 5
EXPECTED_MARKER_COUNT = 182
EXPECTED_TRACE_COUNT = 23

# These black PC depolarization means are completely covered by green NC
# markers at the same coordinates.  Their values are therefore the visible
# green-marker values, not separately detectable black pixels.  T5 width-2
# position +6 is also fully covered, but is intentionally omitted because that
# position is excluded from the downstream Figure 2A scaling output.
OCCLUDED_COPY_RULES = (
    (
        "C_T5_NC_w1_depolarization",
        "C_T5_PC_w1_depolarization",
        (-3, 3, 4),
    ),
    (
        "C_T5_NC_w2_depolarization",
        "C_T5_PC_w2_depolarization",
        (5,),
    ),
)

# For partially covered black markers, a template matched to only the visible
# crescent shifts toward that crescent.  These centers instead come from the
# intact upper circular arc: isolated markers have an 8.5-pixel center-to-top
# distance, giving y=561+8.5 at -5 and y=552+8.5 at +4.  Position -6 is not
# overridden because it is excluded from the downstream scaling output.
PARTIALLY_OCCLUDED_PIXEL_Y = {
    ("C_T5_PC_w2_depolarization", -5): 569.5,
    ("C_T5_PC_w2_depolarization", 4): 560.5,
}


def inclusive(start: int, stop: int) -> tuple[int, ...]:
    return tuple(range(start, stop + 1))


@dataclass(frozen=True)
class ProfileSpec:
    panel: str
    panel_key: str
    cell_type: str
    contrast: str
    color: str
    width_led: int
    extremum: str
    positions: tuple[int, ...]

    @property
    def trace_id(self) -> str:
        return (
            f"{self.panel}_{self.cell_type}_{self.contrast}_"
            f"w{self.width_led}_{self.extremum}"
        )


# Marker presence was counted at full raster resolution.  Listing the visible
# positions explicitly is important for panel B, where means are omitted when
# fewer than four cells responded, and for panel C, where curves cover unequal
# position ranges.  Pure reference/connecting lines are not represented here.
PROFILES: tuple[ProfileSpec, ...] = (
    ProfileSpec("B", "B_NC", "T4", "NC", BLACK, 4, "depolarization", (-7, *inclusive(-4, 6))),
    ProfileSpec("B", "B_NC", "T4", "NC", BLACK, 4, "hyperpolarization", inclusive(-3, 0)),
    ProfileSpec("B", "B_PC", "T4", "PC", GREEN, 4, "depolarization", inclusive(-6, 4)),
    ProfileSpec("B", "B_PC", "T4", "PC", GREEN, 4, "hyperpolarization", inclusive(-2, 6)),
    ProfileSpec("B", "B_NC", "T5", "NC", GREEN, 4, "depolarization", inclusive(-6, 7)),
    ProfileSpec("B", "B_NC", "T5", "NC", GREEN, 4, "hyperpolarization", inclusive(-3, 0)),
    ProfileSpec("B", "B_PC", "T5", "PC", BLACK, 4, "depolarization", inclusive(-6, 5)),
    ProfileSpec("B", "B_PC", "T5", "PC", BLACK, 4, "hyperpolarization", inclusive(1, 7)),
    ProfileSpec("C", "C_W1", "T4", "PC", GREEN, 1, "depolarization", inclusive(-4, 4)),
    ProfileSpec("C", "C_W1", "T4", "PC", GREEN, 1, "hyperpolarization", inclusive(1, 5)),
    ProfileSpec("C", "C_W1", "T4", "NC", BLACK, 1, "depolarization", inclusive(1, 5)),
    ProfileSpec("C", "C_W1", "T4", "NC", BLACK, 1, "hyperpolarization", (-1, 0)),
    ProfileSpec("C", "C_W2", "T4", "PC", GREEN, 2, "depolarization", inclusive(-5, 4)),
    ProfileSpec("C", "C_W2", "T4", "PC", GREEN, 2, "hyperpolarization", inclusive(0, 5)),
    ProfileSpec("C", "C_W2", "T4", "NC", BLACK, 2, "depolarization", (-6, *inclusive(-2, 5))),
    ProfileSpec("C", "C_W2", "T4", "NC", BLACK, 2, "hyperpolarization", inclusive(-3, 0)),
    ProfileSpec("C", "C_W1", "T5", "NC", GREEN, 1, "depolarization", (-3, -2, *inclusive(1, 4))),
    ProfileSpec("C", "C_W1", "T5", "PC", BLACK, 1, "depolarization", (-4, -2, -1, 0, 1, 2, 5)),
    ProfileSpec("C", "C_W1", "T5", "PC", BLACK, 1, "hyperpolarization", inclusive(2, 5)),
    ProfileSpec("C", "C_W2", "T5", "NC", GREEN, 2, "depolarization", inclusive(-6, 7)),
    ProfileSpec("C", "C_W2", "T5", "NC", GREEN, 2, "hyperpolarization", inclusive(-3, 3)),
    ProfileSpec("C", "C_W2", "T5", "PC", BLACK, 2, "depolarization", inclusive(-6, 4)),
    ProfileSpec("C", "C_W2", "T5", "PC", BLACK, 2, "hyperpolarization", inclusive(1, 6)),
)


def disk_template(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((xx * xx + yy * yy) <= radius * radius).astype(np.uint8)


def color_mask(image: np.ndarray, color: str) -> np.ndarray:
    rgb = image.astype(int)
    if color == GREEN:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        return (g > 105) & (g > r + 35) & (g > b + 25)
    if color == BLACK:
        spread = rgb.max(axis=2) - rgb.min(axis=2)
        return (rgb.max(axis=2) < 100) & (spread < 10)
    raise ValueError(f"unknown marker color: {color}")


def marker_score(image: np.ndarray, color: str) -> np.ndarray:
    """Return local marker-ink density for a large circular template.

    A large filled marker nearly saturates the 197-pixel radius-8 template.
    Thin connecting lines and the small panel-B cell markers have much smaller
    support.  Lines are therefore not traced or interpolated; they only remain
    as negligible pixels inside a marker-sized candidate window.
    """
    mask = color_mask(image, color)
    return convolve2d(
        mask.astype(np.uint8), disk_template(TEMPLATE_RADIUS_PX), mode="same"
    )


def search_y_bounds(spec: ProfileSpec) -> tuple[int, int]:
    zero = Y_ZERO[spec.cell_type]
    row_top, row_bottom = {
        "T4": (70, 320),
        "T5": (425, 680),
    }[spec.cell_type]
    if spec.extremum == "depolarization":
        return row_top, int(np.floor(zero - ZERO_EXCLUSION_PX))
    return int(np.ceil(zero + ZERO_EXCLUSION_PX)), row_bottom


def locate_marker(
    score: np.ndarray, spec: ProfileSpec, position: int
) -> tuple[float, float, int]:
    expected_x = PANEL_X_ZERO[spec.panel_key] + position * PX_PER_LED
    x0 = max(0, int(np.floor(expected_x - SEARCH_HALF_WIDTH_PX)))
    x1 = min(score.shape[1], int(np.ceil(expected_x + SEARCH_HALF_WIDTH_PX)) + 1)
    y0, y1 = search_y_bounds(spec)
    window = score[y0:y1, x0:x1]
    if window.size == 0:
        raise ValueError(f"empty search window for {spec.trace_id} at {position:+d}")

    best = int(window.max())
    # Full markers generally score 130--197.  A lower threshold is required
    # only for panel-C markers partially hidden by an overlapping other-colour
    # mean marker; the expected-position table prevents a line-only detection.
    threshold = 100 if spec.panel == "B" else 45
    if best < threshold:
        raise ValueError(
            f"weak marker for {spec.trace_id} at {position:+d}: "
            f"score {best} < {threshold}"
        )

    yy, xx = np.where(window == best)
    # The plotted means sit on integer-LED bins.  Using the calibrated bin
    # center avoids shifting partially occluded green/black markers toward the
    # visible crescent of their fill.
    pixel_x = float(expected_x)
    pixel_y = float(y0 + yy.mean())
    return pixel_x, pixel_y, best


def digitize(image: np.ndarray) -> pd.DataFrame:
    scores = {color: marker_score(image, color) for color in (GREEN, BLACK)}
    rows: list[dict[str, object]] = []
    for spec in PROFILES:
        for position in spec.positions:
            pixel_x, pixel_y, score = locate_marker(
                scores[spec.color], spec, position
            )
            pixel_y = PARTIALLY_OCCLUDED_PIXEL_Y.get(
                (spec.trace_id, position), pixel_y
            )
            response_mv = (Y_ZERO[spec.cell_type] - pixel_y) / PX_PER_MV
            rows.append(
                {
                    "trace_id": spec.trace_id,
                    "panel": spec.panel,
                    "cell_type": spec.cell_type,
                    "contrast": spec.contrast,
                    "color": spec.color,
                    "width_led": spec.width_led,
                    "width_deg": spec.width_led * DEGREES_PER_LED,
                    "duration_ms": FLASH_DURATION_MS,
                    "extremum": spec.extremum,
                    "statistic": "mean",
                    "position_led": position,
                    "position_deg": position * DEGREES_PER_LED,
                    "response_mv": float(response_mv),
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "detection_score": score,
                }
            )
    visible = pd.DataFrame(rows)
    for source_trace_id, target_trace_id, positions in OCCLUDED_COPY_RULES:
        source_markers = visible[
            visible.trace_id == source_trace_id
        ].set_index("position_led")
        for position in positions:
            if position not in source_markers.index:
                raise ValueError(
                    "missing visible marker for occluded target "
                    f"{target_trace_id} at {position:+d}"
                )
            copied = source_markers.loc[position].to_dict()
            copied.update(
                {
                    "trace_id": target_trace_id,
                    "contrast": "PC",
                    "color": BLACK,
                    "position_led": position,
                }
            )
            rows.append(copied)

    return pd.DataFrame(rows).sort_values(
        ["panel", "cell_type", "width_led", "contrast", "extremum", "position_led"]
    ).reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    if len(df) != EXPECTED_MARKER_COUNT:
        raise ValueError(f"expected {EXPECTED_MARKER_COUNT} markers, got {len(df)}")
    if df.trace_id.nunique() != EXPECTED_TRACE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TRACE_COUNT} profiles, got {df.trace_id.nunique()}"
        )
    numeric = df[["position_led", "position_deg", "response_mv", "pixel_x", "pixel_y"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("non-finite numeric value in digitized data")
    if df.duplicated(["trace_id", "position_led"]).any():
        raise ValueError("duplicate (trace_id, position_led) rows")
    for trace_id, trace in df.groupby("trace_id", sort=False):
        positions = trace.position_led.to_numpy()
        if np.any(np.diff(positions) <= 0):
            raise ValueError(f"positions are not strictly increasing for {trace_id}")
    depol = df[df.extremum == "depolarization"].response_mv
    hyper = df[df.extremum == "hyperpolarization"].response_mv
    if not (depol > 0).all():
        raise ValueError("depolarization marker at or below zero")
    if not (hyper < 0).all():
        raise ValueError("hyperpolarization marker at or above zero")


def overlay_path(output: Path) -> Path:
    return output.with_name(f"{output.name}_overlay").with_suffix(".png")


def plot_overlay(image: np.ndarray, df: pd.DataFrame, path: Path) -> None:
    canvas = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    radius = 11
    for row in df.itertuples(index=False):
        x, y = float(row.pixel_x), float(row.pixel_y)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=OVERLAY_RGB,
            width=2,
        )
        draw.line((x - 4, y, x + 4, y), fill=OVERLAY_RGB, width=1)
        draw.line((x, y - 4, x, y + 4), fill=OVERLAY_RGB, width=1)
    canvas.save(path)


def plot_check(df: pd.DataFrame, path: Path) -> None:
    layout = (
        ("B", "T4", 4, "NC"),
        ("B", "T4", 4, "PC"),
        ("C", "T4", 1, None),
        ("C", "T4", 2, None),
        ("B", "T5", 4, "NC"),
        ("B", "T5", 4, "PC"),
        ("C", "T5", 1, None),
        ("C", "T5", 2, None),
    )
    palette = {GREEN: "#549f5c", BLACK: "#000000"}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), sharex=True, sharey=True)
    for ax, (panel, cell_type, width, contrast) in zip(axes.flat, layout):
        subset = df[
            (df.panel == panel)
            & (df.cell_type == cell_type)
            & (df.width_led == width)
        ]
        if contrast is not None:
            subset = subset[subset.contrast == contrast]
        for (profile_contrast, color, extremum), trace in subset.groupby(
            ["contrast", "color", "extremum"], sort=False
        ):
            ax.scatter(
                trace.position_led,
                trace.response_mv,
                s=38,
                facecolor=palette[color],
                edgecolor="black",
                linewidth=0.6,
                zorder=2 if color == GREEN else 1,
                label=f"{profile_contrast} {extremum}",
            )
        ax.axhline(0, color="0.8", lw=0.8, zorder=-1)
        title_contrast = f" {contrast}" if contrast is not None else ""
        ax.set_title(f"{panel} {cell_type}{title_contrast}, width {width}")
        ax.set_xlim(-7.7, 7.7)
        ax.set_ylim(-11, 31)
        ax.set_xticks((-5, 0, 5))
        if panel == "C":
            ax.legend(frameon=False, fontsize=7, loc="upper right")
    for ax in axes[:, 0]:
        ax.set_ylabel("Response extremum (mV)")
    for ax in axes[-1]:
        ax.set_xlabel("Position from center (LED)")
    fig.suptitle("Gruntman et al. 2021 Figure 2B/C — large mean markers only")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    for trace_id, trace in df.groupby("trace_id", sort=False):
        peak = trace.loc[trace.response_mv.idxmax()]
        trough = trace.loc[trace.response_mv.idxmin()]
        print(
            f"{trace_id}: n={len(trace):2d}, "
            f"position={trace.position_led.min():+d}..{trace.position_led.max():+d} LED, "
            f"response={trace.response_mv.min():.2f}..{trace.response_mv.max():.2f} mV, "
            f"peak@{int(peak.position_led):+d}, trough@{int(trough.position_led):+d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=OUT_STEM)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    df = digitize(image)
    validate(df)

    csv_path = args.output.with_suffix(".csv")
    check_path = args.output.with_suffix(".png")
    source_overlay_path = overlay_path(args.output)
    df.to_csv(csv_path, index=False, float_format="%.6f")
    plot_check(df, check_path)
    plot_overlay(image, df, source_overlay_path)
    print_summary(df)
    print(f"Wrote {csv_path} ({len(df)} markers, {df.trace_id.nunique()} profiles)")
    print(f"Wrote {check_path}")
    print(f"Wrote {source_overlay_path}")


if __name__ == "__main__":
    main()
