#!/usr/bin/env python3
"""Plot T4 w1/160-ms PC repetitions by cell and aligned position."""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_t4_160ms_w1_pc_raw_grid import (
    RAW_MAT,
    TIMES_MS,
    _condition_repetitions,
    _object,
    _values,
)


OUTPUT = Path(__file__).with_name("t4_160ms_w1_pc_aligned_positions_by_cell.png")
POSITIONS = tuple(range(-5, 6))


def _scalar(h5, dataset, index):
    return float(np.asarray(_object(h5, dataset[index])).squeeze())


def main():
    cells = {}
    all_values = []
    with h5py.File(RAW_MAT, "r") as h5:
        root = h5["singleBarStT4"]
        for cell_index in range(16):
            result = _object(h5, root["result"][cell_index, 0])
            if not isinstance(result, h5py.Group):
                cells[cell_index + 1] = {}
                continue

            raw_positions = _values(h5, root["positions"], cell_index)
            durations = _values(h5, root["durations"], cell_index)
            widths = _values(h5, root["widths"], cell_index)
            values = _values(h5, root["vals"], cell_index)
            summary_index = tuple(-1 for _ in range(result["empty"].ndim))
            center = _scalar(h5, result["maxExtPosVal"], summary_index)
            inhibition = _scalar(h5, result["minInhPosVal"], summary_index)
            relative_pd = int(np.sign(inhibition - center))

            duration_index = int(np.where(np.isclose(durations, 0.16))[0][0])
            width_index = int(np.where(np.isclose(widths, 1.0))[0][0])
            pc_index = int(np.where(values == 1)[0][0])
            cell_traces = {}
            for raw_index, raw_position in enumerate(raw_positions):
                aligned_position = int(round((float(raw_position) - center) * relative_pd))
                if aligned_position not in POSITIONS:
                    continue
                index = (pc_index, width_index, duration_index, raw_index)
                traces = _condition_repetitions(h5, result, index)
                if traces is not None:
                    cell_traces[aligned_position] = traces
                    all_values.append(traces)
            cells[cell_index + 1] = cell_traces

    values = np.concatenate([x.ravel() for x in all_values])
    y_lo = float(np.floor(np.nanpercentile(values, 0.1) / 5.0) * 5.0)
    y_hi = float(np.ceil(np.nanpercentile(values, 99.9) / 5.0) * 5.0)

    fig, axes = plt.subplots(
        16, 11, figsize=(24, 30), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.10, "wspace": 0.08},
    )
    for row, cell in enumerate(range(1, 17)):
        for col, position in enumerate(POSITIONS):
            ax = axes[row, col]
            ax.axvspan(0.0, 160.0, color="0.92", linewidth=0, zorder=0)
            ax.axhline(0.0, color="0.82", linewidth=0.35, zorder=0)
            traces = cells[cell].get(position)
            if traces is not None:
                for trace in traces:
                    ax.plot(TIMES_MS, trace, color="0.45", alpha=0.55, linewidth=0.45)
                ax.plot(
                    TIMES_MS, traces.mean(axis=0), color="#159447",
                    linewidth=1.0, zorder=3,
                )
                ax.text(
                    0.98, 0.94, f"n={len(traces)}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=5.5, color="0.3",
                )
            ax.set_xlim(-200.0, 750.0)
            ax.set_ylim(y_lo, y_hi)
            ax.tick_params(axis="both", labelsize=5, length=1.5, pad=1)
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("0.55")
            if row == 0:
                ax.set_title(f"aligned pos {position:+d}", fontsize=7, pad=3)
            if col == 0:
                ax.set_ylabel(f"Cell {cell}\nΔVm (mV)", fontsize=6)
            if row == 15:
                ax.set_xticks([0, 160, 750])
                ax.set_xlabel("time (ms)", fontsize=6)
            else:
                ax.tick_params(labelbottom=False)
            if col != 0:
                ax.tick_params(labelleft=False)

    fig.suptitle(
        "T4 raw repetitions: 160 ms, width 1, PC (val=1)\n"
        "columns = aligned position; gray = recordings, green = mean; "
        "each recording baseline-centered over −200–0 ms",
        fontsize=13, y=0.998,
    )
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    print(OUTPUT)
    print(f"ylim={y_lo:g},{y_hi:g}")


if __name__ == "__main__":
    main()
