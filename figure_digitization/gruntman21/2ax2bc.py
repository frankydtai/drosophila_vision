#!/usr/bin/env python3
"""Scale Figure 2A pulse responses by Figure 2C/B width ratios.

For each T4/T5, PC/NC, and receptive-field position, calculate the ratio of
the Figure 2C width-1 or width-2 response extremum to the Figure 2B width-4
response extremum.  If both depolarization and hyperpolarization ratios exist, use their
arithmetic mean; if only one exists, use it.  If a PC or NC condition has no
native ratio, use the opposite contrast's scale for the same cell type and
position.  The resulting position-specific scalar multiplies every time sample
in the matching Figure 2A pulse-response trace.  Width-1 T5 NC depolarization
ratios at positions -1 and 0 are the specified exception: they are linearly interpolated
between the measured ratios at positions -2 and +1.  Positions -6 and -5 are
excluded entirely from the outputs, as is position +5.  Output positions are
the retained source positions divided by two, spanning -2 to +2 in 0.5 steps.
T4 width-2 NC depolarization responses at positions -4 and -3 are linearly
interpolated between the visible width-2 markers at -6 and -2 before division
by the matching width-4 response.

Outputs ``2ax2bc_digitized.csv`` and ``2ax2bc_digitized.png`` beside this
script.  The CSV preserves the original voltage as ``source_vm_mv`` and stores
both component ratios, the final scale, and the scale-selection rule.

Run:
    ../.venv/bin/python 2ax2bc.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_TRACES = HERE / "2a_digitized.csv"
DEFAULT_EXTREMA = HERE / "2bc_digitized.csv"
OUT_STEM = HERE / "2ax2bc_digitized"

SOURCE_WIDTH_LED = 4
TARGET_WIDTHS_LED = (1, 2)
DEGREES_PER_LED = 2.25
FLASH_DURATION_MS = 160.0
INPUT_TRACE_COUNT = 48
OUTPUT_SOURCE_POSITIONS = tuple(range(-4, 5))
OUTPUT_POSITIONS = tuple(position / 2 for position in OUTPUT_SOURCE_POSITIONS)
EXPECTED_SOURCE_TRACE_COUNT = 36
EXPECTED_OUTPUT_TRACE_COUNT = 72
EXPECTED_POINT_COUNT = 15_590
EXPECTED_FALLBACK_COUNTS = {1: 4}
T5_NC_INTERPOLATION_POSITIONS = (-1, 0)
T5_NC_INTERPOLATION_ANCHORS = (-2, 1)
T4_NC_W2_INTERPOLATION_POSITIONS = (-4, -3)
T4_NC_W2_INTERPOLATION_ANCHORS = (-6, -2)

SCALE_KEYS = ["cell_type", "contrast", "position"]
RATIO_KEYS = ["cell_type", "contrast", "extremum", "position_led"]


def require_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def format_position(position: float) -> str:
    """Format an integer or half-step position with an explicit sign."""
    if float(position).is_integer():
        return f"{int(position):+d}"
    return f"{position:+.1f}"


def ratio_table(extrema: pd.DataFrame, target_width_led: int) -> pd.DataFrame:
    """Return one target-width/width-4 ratio per extremum and position."""
    require_columns(
        extrema,
        {
            "cell_type",
            "contrast",
            "extremum",
            "position_led",
            "width_led",
            "response_mv",
        },
        "extrema CSV",
    )
    target = extrema[extrema.width_led == target_width_led][
        [*RATIO_KEYS, "response_mv"]
    ].rename(columns={"response_mv": "target_response_mv"})
    width4 = extrema[extrema.width_led == SOURCE_WIDTH_LED][
        [*RATIO_KEYS, "response_mv"]
    ].rename(columns={"response_mv": "width4_response_mv"})

    paired = target.merge(width4, on=RATIO_KEYS, how="inner", validate="one_to_one")
    if (paired.width4_response_mv == 0).any():
        raise ValueError("cannot divide by a zero width-4 response")
    paired["ratio"] = paired.target_response_mv / paired.width4_response_mv
    if not np.isfinite(paired.ratio.to_numpy()).all():
        raise ValueError("non-finite width-1/width-4 ratio")
    return paired


def t4_nc_w2_interpolated_ratios(extrema: pd.DataFrame) -> pd.DataFrame:
    """Return missing T4 NC width-2 depolarization ratios from response space."""
    target = extrema[
        (extrema.cell_type == "T4")
        & (extrema.contrast == "NC")
        & (extrema.extremum == "depolarization")
        & (extrema.width_led == 2)
        & extrema.position_led.isin(T4_NC_W2_INTERPOLATION_ANCHORS)
    ].set_index("position_led")["response_mv"]
    if set(target.index) != set(T4_NC_W2_INTERPOLATION_ANCHORS):
        raise ValueError("missing T4 NC width-2 interpolation response anchor")

    width4 = extrema[
        (extrema.cell_type == "T4")
        & (extrema.contrast == "NC")
        & (extrema.extremum == "depolarization")
        & (extrema.width_led == SOURCE_WIDTH_LED)
        & extrema.position_led.isin(T4_NC_W2_INTERPOLATION_POSITIONS)
    ].set_index("position_led")["response_mv"]
    if set(width4.index) != set(T4_NC_W2_INTERPOLATION_POSITIONS):
        raise ValueError("missing T4 NC width-4 interpolation denominator")

    positions = np.asarray(T4_NC_W2_INTERPOLATION_POSITIONS, dtype=float)
    responses = np.interp(
        positions,
        np.asarray(T4_NC_W2_INTERPOLATION_ANCHORS, dtype=float),
        target.loc[list(T4_NC_W2_INTERPOLATION_ANCHORS)].to_numpy(dtype=float),
    )
    rows = []
    for position, response in zip(T4_NC_W2_INTERPOLATION_POSITIONS, responses):
        denominator = float(width4.loc[position])
        rows.append(
            {
                "cell_type": "T4",
                "contrast": "NC",
                "extremum": "depolarization",
                "position_led": position,
                "target_response_mv": response,
                "width4_response_mv": denominator,
                "ratio": response / denominator,
            }
        )
    return pd.DataFrame(rows)


def build_scales(
    traces: pd.DataFrame,
    extrema: pd.DataFrame,
    target_width_led: int,
) -> pd.DataFrame:
    """Build one scalar for every condition/position present in Figure 2A."""
    require_columns(
        traces,
        {"trace_id", "cell_type", "contrast", "position", "time_ms", "vm_mv"},
        "trace CSV",
    )
    trace_conditions = traces[["trace_id", *SCALE_KEYS]].drop_duplicates()
    if trace_conditions.trace_id.duplicated().any():
        raise ValueError("a trace_id maps to multiple experimental conditions")

    ratios = ratio_table(extrema, target_width_led)
    if target_width_led == 2:
        ratios = pd.concat(
            [ratios, t4_nc_w2_interpolated_ratios(extrema)], ignore_index=True
        )
    ratio_wide = ratios.pivot(
        index=["cell_type", "contrast", "position_led"],
        columns="extremum",
        values="ratio",
    ).reset_index()
    ratio_wide.columns.name = None
    ratio_wide = ratio_wide.rename(
        columns={
            "position_led": "position",
            "depolarization": "depolarization_ratio",
            "hyperpolarization": "hyperpolarization_ratio",
        }
    )
    for column in ("depolarization_ratio", "hyperpolarization_ratio"):
        if column not in ratio_wide:
            ratio_wide[column] = np.nan

    scales = trace_conditions.merge(
        ratio_wide,
        on=SCALE_KEYS,
        how="left",
        validate="one_to_one",
    )
    interpolation_mask = pd.Series(False, index=scales.index)
    if target_width_led == 1:
        interpolation_mask = (
            (scales.cell_type == "T5")
            & (scales.contrast == "NC")
            & scales.position.isin(T5_NC_INTERPOLATION_POSITIONS)
        )
        anchor_ratios = ratio_wide[
            (ratio_wide.cell_type == "T5")
            & (ratio_wide.contrast == "NC")
            & ratio_wide.position.isin(T5_NC_INTERPOLATION_ANCHORS)
        ].set_index("position")["depolarization_ratio"]
        if set(anchor_ratios.index) != set(T5_NC_INTERPOLATION_ANCHORS):
            raise ValueError("missing T5 NC depolarization interpolation anchor")
        if scales.loc[interpolation_mask, "depolarization_ratio"].notna().any():
            raise ValueError(
                "T5 NC interpolation target already has a measured ratio"
            )
        scales.loc[interpolation_mask, "depolarization_ratio"] = np.interp(
            scales.loc[interpolation_mask, "position"],
            np.asarray(T5_NC_INTERPOLATION_ANCHORS, dtype=float),
            anchor_ratios.loc[
                list(T5_NC_INTERPOLATION_ANCHORS)
            ].to_numpy(dtype=float),
        )
    elif target_width_led == 2:
        interpolation_mask = (
            (scales.cell_type == "T4")
            & (scales.contrast == "NC")
            & scales.position.isin(T4_NC_W2_INTERPOLATION_POSITIONS)
        )

    has_depolarization = scales.depolarization_ratio.notna()
    has_hyperpolarization = scales.hyperpolarization_ratio.notna()
    ratio_count = has_depolarization.astype(int) + has_hyperpolarization.astype(int)
    scales["scale_factor"] = scales[
        ["depolarization_ratio", "hyperpolarization_ratio"]
    ].mean(axis=1, skipna=True)
    scales["scale_source"] = np.select(
        [
            interpolation_mask,
            ratio_count == 2,
            has_depolarization,
            has_hyperpolarization,
        ],
        [
            "interpolated_depolarization",
            "mean_depolarization_hyperpolarization",
            "depolarization",
            "hyperpolarization",
        ],
        default="",
    )

    # A missing PC/NC scale inherits the native scale of the opposite contrast
    # at the same cell type and position.  Fallbacks never chain: the source
    # table below contains the other condition's native scale before filling.
    fallback = scales[
        ["cell_type", "position", "contrast", "scale_factor"]
    ].rename(
        columns={
            "contrast": "fallback_contrast",
            "scale_factor": "fallback_scale_factor",
        }
    )
    fallback["contrast"] = fallback.fallback_contrast.map(
        {"PC": "NC", "NC": "PC"}
    )
    scales = scales.merge(
        fallback,
        on=["cell_type", "position", "contrast"],
        how="left",
        validate="one_to_one",
    )
    missing_native = scales.scale_factor.isna()
    if scales.loc[missing_native, "fallback_scale_factor"].isna().any():
        missing = scales.loc[
            missing_native & scales.fallback_scale_factor.isna(), SCALE_KEYS
        ]
        raise ValueError(
            "missing both native and opposite-contrast scale:\n"
            f"{missing.to_string(index=False)}"
        )
    scales.loc[missing_native, "scale_factor"] = scales.loc[
        missing_native, "fallback_scale_factor"
    ]
    scales.loc[missing_native, "scale_source"] = (
        "fallback_" + scales.loc[missing_native, "fallback_contrast"]
    )
    scales = scales.drop(columns=["fallback_contrast", "fallback_scale_factor"])
    scales["target_width_led"] = target_width_led
    scales["target_width_deg"] = target_width_led * DEGREES_PER_LED
    return scales.sort_values(SCALE_KEYS).reset_index(drop=True)


def scale_traces(traces: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    target_widths = scales.target_width_led.unique()
    if len(target_widths) != 1:
        raise ValueError("scale_traces requires exactly one target width")
    target_width_led = int(target_widths[0])

    output = traces.rename(
        columns={"trace_id": "source_trace_id", "vm_mv": "source_vm_mv"}
    ).merge(
        scales[
            [
                "trace_id",
                "depolarization_ratio",
                "hyperpolarization_ratio",
                "scale_factor",
                "scale_source",
            ]
        ].rename(columns={"trace_id": "source_trace_id"}),
        on="source_trace_id",
        how="left",
        validate="many_to_one",
    )
    if output.scale_factor.isna().any():
        raise ValueError("at least one Figure 2A row has no scale assignment")
    output["source_width_led"] = SOURCE_WIDTH_LED
    output["source_width_deg"] = SOURCE_WIDTH_LED * DEGREES_PER_LED
    output["target_width_led"] = target_width_led
    output["target_width_deg"] = target_width_led * DEGREES_PER_LED
    output["source_position"] = output.position / 2
    output["position"] = output.source_position
    output["trace_id"] = (
        output.cell_type
        + "_"
        + output.contrast
        + "_pos"
        + output.position.map(format_position)
        + f"_w{target_width_led}"
    )
    output["vm_mv"] = output.source_vm_mv * output.scale_factor

    column_order = [
        "trace_id",
        "source_trace_id",
        "cell_type",
        "contrast",
        "source_position",
        "position",
        "rf_side",
        "color",
        "source_width_led",
        "source_width_deg",
        "target_width_led",
        "target_width_deg",
        "depolarization_ratio",
        "hyperpolarization_ratio",
        "scale_factor",
        "scale_source",
        "time_ms",
        "source_vm_mv",
        "vm_mv",
    ]
    return output[column_order]


def validate(
    source: pd.DataFrame,
    scales: pd.DataFrame,
    output: pd.DataFrame,
) -> None:
    if source.trace_id.nunique() != EXPECTED_SOURCE_TRACE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_SOURCE_TRACE_COUNT} source traces, "
            f"got {source.trace_id.nunique()}"
        )
    if scales.duplicated(["trace_id", "target_width_led"]).any():
        raise ValueError("duplicate source trace/target width in scale table")
    for target_width_led in TARGET_WIDTHS_LED:
        target_scales = scales[scales.target_width_led == target_width_led]
        if target_scales.trace_id.nunique() != EXPECTED_SOURCE_TRACE_COUNT:
            raise ValueError(
                f"width {target_width_led} scales do not cover every source trace"
            )
    if len(output) != EXPECTED_POINT_COUNT:
        raise ValueError(
            f"expected {EXPECTED_POINT_COUNT} output points, got {len(output)}"
        )
    if output.trace_id.nunique() != EXPECTED_OUTPUT_TRACE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_OUTPUT_TRACE_COUNT} output traces, "
            f"got {output.trace_id.nunique()}"
        )
    if set(output.target_width_led.unique()) != set(TARGET_WIDTHS_LED):
        raise ValueError("output does not contain both requested target widths")
    per_source = output.groupby("source_trace_id").trace_id.nunique()
    if not (per_source == len(TARGET_WIDTHS_LED)).all():
        raise ValueError("a source trace is missing a target-width result")

    numeric = output.select_dtypes(include=[np.number])
    # Ratio columns are intentionally NaN when the corresponding extremum is
    # unavailable; all other numeric output must be finite.
    finite_columns = numeric.columns.difference(
        ["depolarization_ratio", "hyperpolarization_ratio"]
    )
    if not np.isfinite(numeric[finite_columns].to_numpy()).all():
        raise ValueError("non-finite numeric value in scaled traces")

    expected_vm = output.source_vm_mv * output.scale_factor
    if not np.allclose(output.vm_mv, expected_vm, rtol=0, atol=1e-12):
        raise ValueError("scaled voltage does not equal source voltage times scale")
    if output.duplicated(["trace_id", "time_ms"]).any():
        raise ValueError("duplicate (trace_id, time_ms) rows")
    for trace_id, trace in output.groupby("trace_id", sort=False):
        if not trace.time_ms.is_monotonic_increasing:
            raise ValueError(f"time is not monotonic for {trace_id}")
        if trace.scale_factor.nunique() != 1:
            raise ValueError(f"scale is not constant within {trace_id}")
    if not (scales.scale_factor > 0).all():
        raise ValueError("every retained trace must have a positive scale")
    fallback_scales = scales[scales.scale_source.str.startswith("fallback_")]
    fallback_counts = fallback_scales.groupby("target_width_led").size().to_dict()
    if fallback_counts != EXPECTED_FALLBACK_COUNTS:
        raise ValueError(
            f"expected fallback counts {EXPECTED_FALLBACK_COUNTS}, "
            f"got {fallback_counts}"
        )
    if set(output.source_position.unique()) != set(OUTPUT_POSITIONS):
        raise ValueError("output contains an unexpected source-position column")
    if set(output.position.unique()) != set(OUTPUT_POSITIONS):
        raise ValueError("output contains an unexpected position column")
    if not np.array_equal(
        output.position.to_numpy(), output.source_position.to_numpy()
    ):
        raise ValueError("output position and source_position do not match")


def plot_check(output: pd.DataFrame, scales: pd.DataFrame, path: Path) -> None:
    row_specs = ((1, "T4"), (1, "T5"), (2, "T4"), (2, "T5"))
    fig, axes = plt.subplots(4, 9, figsize=(14.2, 10.2), sharex=True, sharey=True)
    palette = {"green": "#549f5c", "black": "#303030"}
    scale_lookup = scales.set_index(
        ["target_width_led", "cell_type", "contrast", "position"]
    )

    for row, (target_width_led, cell_type) in enumerate(row_specs):
        for column, scale_position in enumerate(OUTPUT_SOURCE_POSITIONS):
            ax = axes[row, column]
            position = scale_position / 2
            for contrast in ("PC", "NC"):
                trace = output[
                    (output.target_width_led == target_width_led)
                    & (output.cell_type == cell_type)
                    & (output.contrast == contrast)
                    & (output.source_position == position)
                ]
                if trace.empty:
                    raise ValueError(
                        "missing plotted trace: "
                        f"{cell_type} {contrast} {format_position(position)}"
                    )
                color = str(trace.color.iloc[0])
                ax.plot(
                    trace.time_ms,
                    trace.vm_mv,
                    color=palette[color],
                    lw=1.35,
                    label=contrast,
                )
            pc_scale = scale_lookup.loc[
                (target_width_led, cell_type, "PC", scale_position),
                "scale_factor",
            ]
            nc_scale = scale_lookup.loc[
                (target_width_led, cell_type, "NC", scale_position),
                "scale_factor",
            ]
            ax.axvspan(0, FLASH_DURATION_MS, color="0.9", zorder=-1)
            ax.axhline(0, color="0.82", lw=0.6, zorder=-1)
            ax.set_title(
                f"{format_position(position)}\n"
                f"PC×{pc_scale:.2f}  NC×{nc_scale:.2f}",
                fontsize=7,
            )
            if column == 0:
                ax.set_ylabel(
                    f"{cell_type}\nwidth {target_width_led}/4\nVm (mV)"
                )
            ax.tick_params(labelsize=7)

    axes[0, -1].legend(frameon=False, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("ms", fontsize=8)
    fig.suptitle(
        "Gruntman et al. 2021 Figure 2A × position-specific width-1/4 and width-2/4 scales"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_scale_summary(scales: pd.DataFrame) -> None:
    for row in scales.itertuples(index=False):
        dep = (
            "--"
            if pd.isna(row.depolarization_ratio)
            else f"{row.depolarization_ratio:.6f}"
        )
        hyp = (
            "--"
            if pd.isna(row.hyperpolarization_ratio)
            else f"{row.hyperpolarization_ratio:.6f}"
        )
        print(
            f"w{row.target_width_led} {row.trace_id}: dep={dep}, hyper={hyp}, "
            f"scale={row.scale_factor:.6f} ({row.scale_source})"
        )


def print_trace_summary(output: pd.DataFrame) -> None:
    for trace_id, trace in output.groupby("trace_id", sort=False):
        peak = trace.loc[trace.vm_mv.idxmax()]
        trough = trace.loc[trace.vm_mv.idxmin()]
        print(
            f"{trace_id}: n={len(trace):3d}, "
            f"time={trace.time_ms.min():.1f}..{trace.time_ms.max():.1f} ms, "
            f"Vm={trace.vm_mv.min():.3f}..{trace.vm_mv.max():.3f} mV, "
            f"peak@{peak.time_ms:.1f} ms, trough@{trough.time_ms:.1f} ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--extrema", type=Path, default=DEFAULT_EXTREMA)
    parser.add_argument("--output", type=Path, default=OUT_STEM)
    args = parser.parse_args()

    source_all = pd.read_csv(args.traces)
    if source_all.trace_id.nunique() != INPUT_TRACE_COUNT:
        raise ValueError(
            f"expected {INPUT_TRACE_COUNT} input traces, "
            f"got {source_all.trace_id.nunique()}"
        )
    source = source_all[
        source_all.position.isin(OUTPUT_SOURCE_POSITIONS)
    ].copy()
    extrema = pd.read_csv(args.extrema)
    scale_tables = [
        build_scales(source, extrema, target_width_led)
        for target_width_led in TARGET_WIDTHS_LED
    ]
    scales = pd.concat(scale_tables, ignore_index=True)
    output = pd.concat(
        [
            scale_traces(source, target_scales)
            for target_scales in scale_tables
        ],
        ignore_index=True,
    )
    validate(source, scales, output)

    csv_path = args.output.with_suffix(".csv")
    png_path = args.output.with_suffix(".png")
    output.to_csv(csv_path, index=False, float_format="%.9f")
    plot_check(output, scales, png_path)
    print_scale_summary(scales)
    print_trace_summary(output)
    print(
        f"Wrote {csv_path} ({len(output):,} points, "
        f"{output.trace_id.nunique()} traces)"
    )
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
