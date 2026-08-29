#!/usr/bin/env python3
"""Plot aligned T4 cell-average traces for w1, 160-ms PC flashes.

For each biological cell, the stored ``subData.baseSub`` trace is already the
repeat-averaged, baseline-subtracted response. Cells are aligned with the same
RF-center / preferred-direction convention as ``Figure2Code.m``.
"""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RAW_MAT = Path(
    "/data/scratch/projects/punim0477/yitai/drosophila_vision/"
    "gruntman21/Figure2/singleBarStT4.mat"
)
OUTPUT = Path(__file__).with_name("t4_w1_pc_160ms_pos_minus4_plus4.png")
RELATIVE_POSITIONS = tuple(range(-4, 5))
TIMES_MS = np.arange(-200.0, 750.0 + 5.0, 5.0)

# Match plot_t4_160ms_w1_pc_raw_grid.py exactly where colors overlap.
TRACE_COLOR = "0.45"
TRACE_ALPHA = 0.55
MEAN_COLOR = "#159447"
SD_COLOR = MEAN_COLOR
SD_ALPHA = 0.20
STIMULUS_COLOR = "0.92"
ZERO_COLOR = "0.82"
SPINE_COLOR = "0.55"


def _object(h5, value):
    array = np.asarray(value)
    return h5[array.flat[0]] if array.dtype == h5py.ref_dtype else array


def _values(h5, dataset, cell_index):
    return np.atleast_1d(
        np.asarray(_object(h5, dataset[cell_index, 0])).squeeze()
    )


def _scalar(h5, dataset, index):
    return float(np.asarray(_object(h5, dataset[index])).squeeze())


def _load_panels():
    panels = []
    with h5py.File(RAW_MAT, "r") as h5:
        root = h5["singleBarStT4"]
        cells = []
        for cell_index in range(root["result"].shape[0]):
            result = _object(h5, root["result"][cell_index, 0])
            if not isinstance(result, h5py.Group):
                continue
            positions = _values(h5, root["positions"], cell_index)
            durations = _values(h5, root["durations"], cell_index)
            widths = _values(h5, root["widths"], cell_index)
            values = _values(h5, root["vals"], cell_index)
            summary_index = tuple(-1 for _ in range(result["empty"].ndim))
            center = _scalar(h5, result["maxExtPosVal"], summary_index)
            min_position = _scalar(h5, result["minInhPosVal"], summary_index)
            relative_pd = int(np.sign(min_position - center))
            cells.append(
                (
                    result,
                    positions,
                    durations,
                    widths,
                    values,
                    center,
                    relative_pd,
                )
            )

        for relative_position in RELATIVE_POSITIONS:
            traces = []
            for (
                result,
                positions,
                durations,
                widths,
                values,
                center,
                relative_pd,
            ) in cells:
                # Width 1 has no leading-edge offset under the original RF flip.
                original_position = center + relative_position * relative_pd
                position_hits = np.where(
                    np.isclose(positions, original_position)
                )[0]
                if not position_hits.size:
                    continue
                index = (
                    int(np.where(values == 1)[0][0]),
                    int(np.where(np.isclose(widths, 1.0))[0][0]),
                    int(np.where(np.isclose(durations, 0.16))[0][0]),
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

            traces = np.asarray(traces)
            if traces.shape[0] != 15:
                raise ValueError(
                    f"relative position {relative_position:+d}: "
                    f"expected 15 cell traces, got {traces.shape[0]}"
                )
            panels.append(
                (
                    relative_position,
                    traces,
                    traces.mean(axis=0),
                    traces.std(axis=0, ddof=1),
                )
            )
    return panels


def _ylim(panels):
    values = np.concatenate(
        [
            np.concatenate((traces.ravel(), mean - sd, mean + sd))
            for _, traces, mean, sd in panels
        ]
    )
    y_low = float(np.floor(np.nanpercentile(values, 0.1) / 5.0) * 5.0)
    y_high = float(np.ceil(np.nanpercentile(values, 99.9) / 5.0) * 5.0)
    return y_low, y_high


def main():
    panels = _load_panels()
    y_low, y_high = _ylim(panels)

    fig, axes = plt.subplots(
        1,
        9,
        figsize=(32, 4.8),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for ax, (relative_position, traces, mean, sd) in zip(axes, panels):
        ax.axvspan(
            0.0, 160.0, color=STIMULUS_COLOR, linewidth=0, zorder=0
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
            color=SD_COLOR,
            alpha=SD_ALPHA,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            TIMES_MS,
            mean,
            color=MEAN_COLOR,
            linewidth=1.5,
            zorder=3,
        )
        ax.set_xlim(-200.0, 750.0)
        ax.set_ylim(y_low, y_high)
        ax.set_title(f"{relative_position:+d}", fontsize=10, pad=3)
        ax.set_xticks([0, 160, 750])
        ax.set_xlabel("time (ms)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color(SPINE_COLOR)
    axes[0].set_ylabel("T4 ΔVm (mV)", fontsize=9)

    handles = [
        plt.Line2D(
            [0], [0], color=TRACE_COLOR, alpha=TRACE_ALPHA,
            linewidth=0.8, label="15 cell-average traces",
        ),
        plt.Line2D(
            [0], [0], color=MEAN_COLOR, linewidth=1.5,
            label="population mean",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=SD_COLOR, alpha=SD_ALPHA,
            label="mean ± 1 SD",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=9,
    )
    fig.suptitle("T4: 160 ms, width 1, PC (val=1)", fontsize=13, y=1.05)
    fig.subplots_adjust(
        left=0.035, right=0.995, bottom=0.20, top=0.80, wspace=0.08
    )
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    print(OUTPUT)
    print(f"ylim={y_low:g},{y_high:g}")


if __name__ == "__main__":
    main()
