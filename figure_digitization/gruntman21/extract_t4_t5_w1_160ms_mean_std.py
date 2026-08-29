#!/usr/bin/env python3
"""Export raw Gruntman T4/T5 width-1, 160-ms mean and SD traces.

Each input trace is ``subData.baseSub``: the repeat-averaged,
baseline-subtracted response of one biological cell.  The exported mean and
sample SD (ddof=1) are calculated across biological cells, not by treating
repeat recordings from the same cell as independent samples.
"""

from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


RAW_DIR = Path(
    "/data/scratch/projects/punim0477/yitai/drosophila_vision/"
    "gruntman21/Figure2"
)
OUTPUT = Path(__file__).with_name("t4_t5_w1_160ms_mean_std.csv")
POSITIONS = tuple(range(-4, 5))
TIMES_MS = np.arange(-200.0, 750.0 + 5.0, 5.0)
WIDTH_LED = 1
DURATION_MS = 160
CONDITIONS = (
    ("T4", "PC", 1),
    ("T4", "NC", 0),
    ("T5", "PC", 0),
    ("T5", "NC", 1),
)


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


def _repeat_composition(repeat_counts):
    counts = Counter(repeat_counts)
    return "+".join(
        f"{n_cells}x{n_repeats}"
        for n_repeats, n_cells in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    )


def _condition_rows(cell_type, contrast, value):
    path = RAW_DIR / f"singleBarSt{cell_type}.mat"
    root_name = f"singleBarSt{cell_type}"
    rows = []
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
                np.isclose(durations, DURATION_MS / 1000.0)
            )[0]
            width_hits = np.where(np.isclose(widths, WIDTH_LED))[0]
            value_hits = np.where(values == value)[0]
            if not duration_hits.size or not width_hits.size or not value_hits.size:
                continue
            summary_index = tuple(-1 for _ in range(result["empty"].ndim))
            center = _scalar(h5, result["maxExtPosVal"], summary_index)
            min_position = _scalar(h5, result["minInhPosVal"], summary_index)
            relative_pd = int(np.sign(min_position - center))
            if relative_pd == 0:
                raise ValueError(f"cell {cell_index}: undefined PD alignment")
            cells.append((
                result,
                positions,
                int(duration_hits[0]),
                int(width_hits[0]),
                int(value_hits[0]),
                center,
                relative_pd,
            ))

        for relative_position in POSITIONS:
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
                # Figure2Code.m leading-edge alignment; for width 1 this is
                # center + relative_position in PD coordinates.
                if relative_pd == 1:
                    original_position = center + relative_position
                else:
                    original_position = center + WIDTH_LED - 1 - relative_position
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
                if float(np.asarray(_object(h5, result["empty"][index])).squeeze()):
                    continue
                sub_data = _object(h5, result["subData"][index])
                base_sub = np.asarray(sub_data["baseSub"], dtype=np.float64)
                traces.append(np.interp(TIMES_MS, base_sub[0], base_sub[1]))
                repeat_counts.append(_repeat_count(h5, result, index))

            traces = np.asarray(traces, dtype=np.float64)
            if traces.shape[0] < 2:
                raise ValueError(
                    f"{cell_type} {contrast} position {relative_position:+d}: "
                    f"need >=2 biological cells, got {traces.shape[0]}"
                )
            mean = traces.mean(axis=0)
            std = traces.std(axis=0, ddof=1)
            trace_id = f"{cell_type}_{contrast}_pos{relative_position:+d}_w1"
            composition = _repeat_composition(repeat_counts)
            n_cells = int(traces.shape[0])
            n_recordings = int(sum(repeat_counts))
            for time_ms, mean_mv, std_mv in zip(TIMES_MS, mean, std):
                rows.append({
                    "trace_id": trace_id,
                    "cell_type": cell_type,
                    "contrast": contrast,
                    "position": relative_position,
                    "duration_ms": DURATION_MS,
                    "width_led": WIDTH_LED,
                    "time_ms": time_ms,
                    "vm_mean_mv": mean_mv,
                    "vm_std_mv": std_mv,
                    "n_cells": n_cells,
                    "n_recordings": n_recordings,
                    "repeat_composition": composition,
                })
    return rows


def main():
    rows = []
    for condition in CONDITIONS:
        rows.extend(_condition_rows(*condition))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT, index=False, float_format="%.10g")
    print(OUTPUT)
    print(
        frame[[
            "trace_id", "n_cells", "n_recordings", "repeat_composition"
        ]].drop_duplicates().to_string(index=False)
    )


if __name__ == "__main__":
    main()
