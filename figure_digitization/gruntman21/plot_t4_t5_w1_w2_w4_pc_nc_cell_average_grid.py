#!/usr/bin/env python3
"""Plot aligned T4/T5 cell-average traces for 40/160-ms PC/NC flashes."""

import argparse
from pathlib import Path
from collections import Counter

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RAW_DIR = Path(
    "/data/scratch/projects/punim0477/yitai/drosophila_vision/"
    "gruntman21/Figure2"
)
RELATIVE_POSITIONS = tuple(range(-4, 5))
TIMES_MS = np.arange(-200.0, 750.0 + 5.0, 5.0)

# Match plot_t4_160ms_w1_pc_raw_grid.py and the original Figure 2 value colors.
TRACE_COLOR = "0.45"
TRACE_ALPHA = 0.55
GREEN = "#159447"  # val=1: T4 PC / T5 NC
DARK = "#252525"   # val=0: T4 NC / T5 PC
SD_ALPHA = 0.20
STIMULUS_COLOR = "0.92"
ZERO_COLOR = "0.82"
SPINE_COLOR = "0.55"

CONDITIONS = (
    ("T4", "PC", 1, GREEN),
    ("T4", "NC", 0, DARK),
    ("T5", "PC", 0, DARK),
    ("T5", "NC", 1, GREEN),
)
WIDTHS = (1, 2, 4)


def _object(h5, value):
    array = np.asarray(value)
    return h5[array.flat[0]] if array.dtype == h5py.ref_dtype else array


def _values(h5, dataset, cell_index):
    return np.atleast_1d(
        np.asarray(_object(h5, dataset[cell_index, 0])).squeeze()
    )


def _scalar(h5, dataset, index):
    return float(np.asarray(_object(h5, dataset[index])).squeeze())


def _condition_index(result, value_index, width_index, duration_index, position_index):
    if result["empty"].ndim == 4:
        return value_index, width_index, duration_index, position_index
    if result["empty"].ndim == 3:
        return width_index, duration_index, position_index
    raise ValueError(f"unsupported result rank {result['empty'].ndim}")


def _repeat_count(h5, result, index):
    data = _object(h5, result["data"][index])
    if "numReps" in data:
        return int(np.asarray(data["numReps"]).squeeze())
    return int(data["align"]["rep"]["data"].size)


def _repeat_label(repeat_counts):
    composition = Counter(repeat_counts)
    ordered = sorted(composition.items(), key=lambda item: (-item[1], item[0]))
    return "+".join(
        rf"{n_cells}\times{n_repeats}"
        for n_repeats, n_cells in ordered
    )


def _load_condition(cell_type, value, width, duration_ms):
    path = RAW_DIR / f"singleBarSt{cell_type}.mat"
    root_name = f"singleBarSt{cell_type}"
    panels = []
    with h5py.File(path, "r") as h5:
        root = h5[root_name]
        cells = []
        for cell_index in range(root["result"].shape[0]):
            result = _object(h5, root["result"][cell_index, 0])
            if not isinstance(result, h5py.Group):
                continue
            positions = _values(h5, root["positions"], cell_index)
            durations = _values(h5, root["durations"], cell_index)
            widths = _values(h5, root["widths"], cell_index)
            values = _values(h5, root["vals"], cell_index)
            duration_hits = np.where(
                np.isclose(durations, float(duration_ms) / 1000.0)
            )[0]
            width_hits = np.where(np.isclose(widths, float(width)))[0]
            value_hits = np.where(values == value)[0]
            if not duration_hits.size or not width_hits.size or not value_hits.size:
                continue
            summary_index = tuple(-1 for _ in range(result["empty"].ndim))
            center = _scalar(h5, result["maxExtPosVal"], summary_index)
            min_position = _scalar(h5, result["minInhPosVal"], summary_index)
            relative_pd = int(np.sign(min_position - center))
            cells.append(
                (
                    result,
                    positions,
                    int(duration_hits[0]),
                    int(width_hits[0]),
                    int(value_hits[0]),
                    center,
                    relative_pd,
                )
            )

        for relative_position in RELATIVE_POSITIONS:
            traces = []
            repeat_counts = []
            for (
                result,
                positions,
                duration_index,
                width_index,
                value_index,
                center,
                relative_pd,
            ) in cells:
                # Match Figure2Code.m's width-aware leading-edge correction.
                if relative_pd == 1:
                    original_position = center + relative_position
                else:
                    original_position = (
                        center + float(width) - 1.0 - relative_position
                    )
                position_hits = np.where(
                    np.isclose(positions, original_position)
                )[0]
                if not position_hits.size:
                    continue
                index = _condition_index(
                    result,
                    value_index,
                    width_index,
                    duration_index,
                    int(position_hits[0]),
                )
                if float(
                    np.asarray(_object(h5, result["empty"][index])).squeeze()
                ):
                    continue
                sub_data = _object(h5, result["subData"][index])
                base_sub = np.asarray(sub_data["baseSub"], dtype=float)
                traces.append(
                    np.interp(TIMES_MS, base_sub[0], base_sub[1])
                )
                repeat_counts.append(_repeat_count(h5, result, index))

            traces = np.asarray(traces)
            if traces.shape[0] < 2:
                raise ValueError(
                    f"{cell_type} val={value}, width={width}, "
                    f"duration={duration_ms} ms, "
                    f"position {relative_position:+d}: "
                    f"need at least 2 cell traces, got {traces.shape[0]}"
                )
            panels.append(
                (
                    relative_position,
                    traces,
                    traces.mean(axis=0),
                    traces.std(axis=0, ddof=1),
                    _repeat_label(repeat_counts),
                )
            )
    return panels


def _ylim(rows):
    values = np.concatenate(
        [
            np.concatenate((traces.ravel(), mean - sd, mean + sd))
            for panels in rows
            for _, traces, mean, sd, _repeat_label_text in panels
        ]
    )
    y_low = float(np.floor(np.nanpercentile(values, 0.1) / 5.0) * 5.0)
    y_high = float(np.ceil(np.nanpercentile(values, 99.9) / 5.0) * 5.0)
    return y_low, y_high


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration-ms", type=int, choices=(40, 160), default=160,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = _parse_args()
    duration_ms = int(args.duration_ms)
    output = args.output or Path(__file__).with_name(
        f"t4_t5_{duration_ms}ms_w1_w2_w4_pc_nc_pos_minus4_plus4.png"
    )
    row_specs = [
        (width, cell_type, contrast, value, color)
        for width in WIDTHS
        for cell_type, contrast, value, color in CONDITIONS
    ]
    rows = [
        _load_condition(cell_type, value, width, duration_ms)
        for width, cell_type, _contrast, value, _color in row_specs
    ]
    y_low, y_high = _ylim(rows)

    fig, axes = plt.subplots(
        len(row_specs),
        9,
        figsize=(32, 40),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
    )
    for row_index, (
        (width, cell_type, contrast, _value, color), panels
    ) in enumerate(
        zip(row_specs, rows)
    ):
        for column_index, (
            relative_position, traces, mean, sd, repeat_label
        ) in enumerate(panels):
            ax = axes[row_index, column_index]
            ax.axvspan(
                0.0, float(duration_ms),
                color=STIMULUS_COLOR, linewidth=0, zorder=0,
            )
            ax.axhline(0.0, color=ZERO_COLOR, linewidth=0.35, zorder=0)
            for trace in traces:
                ax.plot(
                    TIMES_MS,
                    trace,
                    color=TRACE_COLOR,
                    alpha=TRACE_ALPHA,
                    linewidth=0.45,
                    zorder=1,
                )
            ax.fill_between(
                TIMES_MS,
                mean - sd,
                mean + sd,
                color=color,
                alpha=SD_ALPHA,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                TIMES_MS,
                mean,
                color=color,
                linewidth=1.5,
                zorder=3,
            )
            ax.text(
                0.98,
                0.94,
                rf"$n={repeat_label}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
                color="0.3",
            )
            ax.set_xlim(-200.0, 750.0)
            ax.set_ylim(y_low, y_high)
            ax.set_xticks([0, duration_ms, 750])
            ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color(SPINE_COLOR)
            if row_index == 0:
                ax.set_title(f"{relative_position:+d}", fontsize=10, pad=3)
            if column_index == 0:
                ax.set_ylabel(
                    f"w{width} · {cell_type} {contrast}\nΔVm (mV)",
                    fontsize=9,
                )
            if row_index == len(row_specs) - 1:
                ax.set_xlabel("time (ms)", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if column_index != 0:
                ax.tick_params(labelleft=False)

    handles = [
        plt.Line2D(
            [0], [0], color=TRACE_COLOR, alpha=TRACE_ALPHA,
            linewidth=0.8, label="cell-average traces",
        ),
        plt.Line2D(
            [0], [0], color=GREEN, linewidth=1.5,
            label="val=1 mean (T4 PC / T5 NC)",
        ),
        plt.Line2D(
            [0], [0], color=DARK, linewidth=1.5,
            label="val=0 mean (T4 NC / T5 PC)",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=GREEN, alpha=SD_ALPHA,
            label="mean ± 1 SD",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.992),
        fontsize=9,
    )
    fig.suptitle(
        f"T4/T5: {duration_ms} ms, widths 1/2/4, PC and NC",
        fontsize=13,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.04, right=0.995, bottom=0.025, top=0.965,
        hspace=0.12, wspace=0.08,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    print(output)
    for (
        width, cell_type, contrast, _value, _color
    ), panels in zip(row_specs, rows):
        print(
            f"w{width} {cell_type} {contrast}: "
            f"{[repeat_label for _, _traces, _mean, _sd, repeat_label in panels]}"
        )
    print(f"ylim={y_low:g},{y_high:g}")


if __name__ == "__main__":
    main()
