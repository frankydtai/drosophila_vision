#!/usr/bin/env python3
"""Shift and linearly sum Gruntman21 T4/T5 width-2 position traces.

The nine flash positions are separated by 4.5 degrees.  At the Gruntman fast
moving-bar speed of 56.25 degrees/s, neighbouring positions are therefore
separated by 80 ms.  T4/T5 and PC/NC traces are processed independently in
both scan directions.

The traces are 160-ms width-2 cell averages loaded from the Figure 2 MAT
files, matching the W2 discrete shifted sum in ``gruntman21_mbar.py``.

Outputs:
  gruntman21_shift_overlay.csv/.png      width-2 (backward-compatible names)
  gruntman21_shift_overlay_w1.csv/.png   width-1
  gruntman21_shift_overlay_w4.csv/.png   width-4

Run:  ../.venv/bin/python gruntman21_shift_overlay.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_CSV = HERE / "gruntman21_shift_overlay.csv"
OUT_PNG = HERE / "gruntman21_shift_overlay.png"
WIDTH_OUTPUTS = {
    1: (
        HERE / "gruntman21_shift_overlay_w1.csv",
        HERE / "gruntman21_shift_overlay_w1.png",
    ),
    2: (OUT_CSV, OUT_PNG),
    4: (
        HERE / "gruntman21_shift_overlay_w4.csv",
        HERE / "gruntman21_shift_overlay_w4.png",
    ),
}
ADDITIONAL_OUT_PNGS = {
    160.0: HERE / "gruntman21_shift_overlay_160ms.png",
    40.0: HERE / "gruntman21_shift_overlay_40ms.png",
    10.0: HERE / "gruntman21_shift_overlay_10ms.png",
}

CELL_TYPES = ("T4", "T5")
WIDTHS_LED = (1, 2, 4)
POSITIONS = tuple(range(-4, 5))
CONTRASTS = ("PC", "NC")
SAMPLE_INTERVAL_MS = 5.0
POSITION_STEP_DEG = 4.5
SPEED_DEG_OVER_S = 56.25
POSITION_DELAY_MS = POSITION_STEP_DEG / SPEED_DEG_OVER_S * 1000.0

DIRECTIONS = {
    "PD": POSITIONS,
    "ND": tuple(reversed(POSITIONS)),
}


def load_traces(width_led: int) -> dict[tuple[str, str, int], pd.DataFrame]:
    """Load and validate one uniformly sampled trace per condition."""
    from gruntman21_fit_lp import load_cell_averages

    data = load_cell_averages(width_led)
    data = data[data["cell_type"].isin(CELL_TYPES)].copy()
    traces: dict[tuple[str, str, int], pd.DataFrame] = {}

    for cell_type in CELL_TYPES:
        for contrast in CONTRASTS:
            for position in POSITIONS:
                trace = data[
                    (data["cell_type"] == cell_type)
                    & (data["contrast"] == contrast)
                    & (data["position"] == position)
                ].sort_values("time_ms")
                if trace.empty:
                    raise ValueError(
                        f"missing {cell_type} {contrast} position {position:+d}"
                    )

                time_ms = trace["time_ms"].to_numpy(dtype=float)
                if not np.allclose(np.diff(time_ms), SAMPLE_INTERVAL_MS):
                    raise ValueError(
                        f"{cell_type} {contrast} position {position:+d} is not "
                        f"sampled every {SAMPLE_INTERVAL_MS:g} ms"
                    )
                traces[(cell_type, contrast, position)] = trace
    return traces


def shifted_components(
    traces: dict[tuple[str, str, int], pd.DataFrame],
    position_delay_ms: float,
) -> pd.DataFrame:
    """Place all shifted traces and sums on one common 5 ms time grid."""
    min_time_ms = min(
        float(trace["time_ms"].min()) for trace in traces.values()
    )
    max_time_ms = max(
        float(trace["time_ms"].max()) for trace in traces.values()
    ) + (len(POSITIONS) - 1) * position_delay_ms
    common_time_ms = np.arange(
        min_time_ms,
        max_time_ms + SAMPLE_INTERVAL_MS / 2.0,
        SAMPLE_INTERVAL_MS,
    )

    frames: list[pd.DataFrame] = []
    for direction, scan_positions in DIRECTIONS.items():
        delays_ms = {
            position: index * position_delay_ms
            for index, position in enumerate(scan_positions)
        }
        for cell_type in CELL_TYPES:
            for contrast in CONTRASTS:
                total_mv = np.zeros_like(common_time_ms)
                for position in POSITIONS:
                    trace = traces[(cell_type, contrast, position)]
                    source_time_ms = trace["time_ms"].to_numpy(dtype=float)
                    source_vm_mv = trace["vm_mv"].to_numpy(dtype=float)
                    delay_ms = delays_ms[position]
                    shifted_mv = np.interp(
                        common_time_ms - delay_ms,
                        source_time_ms,
                        source_vm_mv,
                        left=0.0,
                        right=0.0,
                    )
                    total_mv += shifted_mv
                    frames.append(
                        pd.DataFrame(
                            {
                                "cell_type": cell_type,
                                "direction": direction,
                                "contrast": contrast,
                                "component": "position",
                                "position": position,
                                "delay_ms": delay_ms,
                                "time_ms": common_time_ms,
                                "vm_mv": shifted_mv,
                            }
                        )
                    )

                frames.append(
                    pd.DataFrame(
                        {
                            "cell_type": cell_type,
                            "direction": direction,
                            "contrast": contrast,
                            "component": "sum",
                            "position": np.nan,
                            "delay_ms": np.nan,
                            "time_ms": common_time_ms,
                            "vm_mv": total_mv,
                        }
                    )
                )

    return pd.concat(frames, ignore_index=True)


def plot_overlay(
    data: pd.DataFrame,
    output_path: Path,
    position_delay_ms: float,
    width_led: int,
) -> None:
    """Plot eight shifted-trace panels with one shared y-axis scale."""
    fig, axes = plt.subplots(
        len(CONTRASTS) * len(DIRECTIONS),
        len(CELL_TYPES),
        figsize=(15, 14),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    position_colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.95, len(POSITIONS))
    )
    row_conditions = [
        (contrast, direction)
        for contrast in CONTRASTS
        for direction in DIRECTIONS
    ]
    for row, (contrast, direction) in enumerate(row_conditions):
        for column, cell_type in enumerate(CELL_TYPES):
            ax = axes[row, column]
            panel = data[
                (data["cell_type"] == cell_type)
                & (data["direction"] == direction)
                & (data["contrast"] == contrast)
            ]
            for color, position in zip(position_colors, POSITIONS):
                trace = panel[
                    (panel["component"] == "position")
                    & (panel["position"] == position)
                ]
                delay_ms = float(trace["delay_ms"].iloc[0])
                ax.plot(
                    trace["time_ms"],
                    trace["vm_mv"],
                    color=color,
                    lw=0.9,
                    alpha=0.65,
                    label=f"{position:+d} ({delay_ms:.0f} ms)",
                )

            total = panel[panel["component"] == "sum"]
            ax.plot(
                total["time_ms"],
                total["vm_mv"],
                color="black",
                lw=2.2,
                label="linear sum",
            )
            ax.axhline(0.0, color="0.75", lw=0.6, zorder=-1)
            ax.axvline(0.0, color="0.75", lw=0.6, zorder=-1)
            ax.set_title(
                f"{cell_type} {contrast}: {direction}"
            )
            ax.set_ylabel("Vm (mV)")
            ax.grid(alpha=0.15)

    for ax in axes[-1]:
        ax.set_xlabel("time from first position onset (ms)")
    axes[0, 1].legend(
        fontsize=7,
        ncol=2,
        frameon=False,
        loc="upper right",
    )
    fig.suptitle(
        f"Gruntman21 T4/T5 W{width_led} shifted-position overlays\n"
        f"160-ms flashes; {position_delay_ms:g} ms per position; shared y-axis"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    traces_by_width = {}
    for width_led in WIDTHS_LED:
        traces = load_traces(width_led)
        traces_by_width[width_led] = traces
        output = shifted_components(traces, POSITION_DELAY_MS)
        output_csv, output_png = WIDTH_OUTPUTS[width_led]
        output.to_csv(output_csv, index=False, float_format="%.6f")
        plot_overlay(output, output_png, POSITION_DELAY_MS, width_led)
        print(f"wrote {output_csv} ({len(output):,} rows)")
        print(f"wrote {output_png}")

    for delay_ms, output_path in ADDITIONAL_OUT_PNGS.items():
        delayed_output = shifted_components(traces_by_width[2], delay_ms)
        plot_overlay(delayed_output, output_path, delay_ms, 2)
        print(f"wrote {output_path} (position delay: {delay_ms:g} ms)")

    print(f"CSV position delay: {POSITION_DELAY_MS:g} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
