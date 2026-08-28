#!/usr/bin/env python3
"""Plot Gruntman et al. 2021 Figure 2A from the Figshare single-bar recordings.

Uses selective h5py reads for width-4, 160 ms flashes only (no full-file load).
Reimplements the Figure2Code.m alignment and panel layout for panel A.

Outputs ``2a_data.png`` and ``2a_data.csv`` beside this script.

Run:
    ../.venv/bin/python 2a_data.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRATCH_ROOT = Path("/data/scratch/projects/punim0477/yitai")
DATA_DIR = SCRATCH_ROOT / "drosophila_vision/gruntman21/Figure2"
OUT_PNG = HERE / "2a_data.png"
OUT_CSV = HERE / "2a_data.csv"

ALL_DURATIONS_S = (0.02, 0.04, 0.08, 0.16, 0.32)
ALL_WIDTHS_LED = (1, 2, 4)
PLOT_DURATION_S = 0.16
PLOT_WIDTH_LED = 4
REL_LEN = 45000
NEW_ZERO = REL_LEN // 5 * 2
CHOP_EDGES = 9
N_POSITIONS = 31
CENTER_INDEX = 16
DOWN_SAMP = 100
STIMULUS_MS = 160.0
SKIP_T4_CELL = 4

COLORS = {
    "green": ("#add88e", "#319f54"),
    "dark": ("#969696", "#252525"),
}
CONTRAST_BY_CELL = {
    "T4": {0: "NC", 1: "PC"},
    "T5": {0: "PC", 1: "NC"},
}


def deref(file_handle: h5py.File, value):
    if isinstance(value, h5py.Group):
        return value
    if isinstance(value, h5py.Dataset):
        if value.dtype == h5py.ref_dtype:
            value = value[()]
        else:
            return np.array(value)
    if isinstance(value, np.ndarray):
        if value.dtype == h5py.ref_dtype:
            return file_handle[value.flat[0]]
        return value
    return file_handle[value]


def is_empty(file_handle: h5py.File, result_group: h5py.Group, index: tuple[int, ...]) -> bool:
    try:
        flag = np.array(deref(file_handle, result_group["empty"][index])).squeeze()
    except (IndexError, KeyError, TypeError):
        return True
    if flag.size != 1:
        return True
    return bool(flag)


def pad_resp_vec(vm: np.ndarray, old_zero: int, new_zero: int, new_len: int) -> np.ndarray:
    vm = np.asarray(vm, dtype=float).ravel()
    pre_diff = new_zero - old_zero
    padded = np.concatenate([np.full(max(pre_diff, 0), np.nan), vm[abs(min(pre_diff, 0)) :]])
    post_diff = new_len - len(padded)
    if post_diff > 0:
        padded = np.concatenate([padded, np.full(post_diff, np.nan)])
    return padded[:new_len]


def read_trace(file_handle: h5py.File, result_group: h5py.Group, index: tuple[int, ...]) -> np.ndarray | None:
    if is_empty(file_handle, result_group, index):
        return None
    sub_data = deref(file_handle, result_group["subData"][index])
    base_sub = np.array(deref(file_handle, sub_data["baseSub"]))
    zero_ind = int(np.array(deref(file_handle, sub_data["zeroInd"])).squeeze())
    vm = base_sub[1] if base_sub.shape[0] == 2 else base_sub[:, 1]
    return pad_resp_vec(vm, zero_ind, NEW_ZERO, REL_LEN)


def align_index(orig_pos: float, width_led: float, rel_max_ext: float, rel_pd: float) -> int:
    pre_rel_pos = orig_pos - rel_max_ext
    if rel_pd == 1:
        rel_pos = pre_rel_pos * rel_pd + CENTER_INDEX
    else:
        rel_pos = (pre_rel_pos - width_led + 1.0) * rel_pd + CENTER_INDEX
    return int(round(rel_pos))


def result_index(
    result_group: h5py.Group,
    duration_index: int,
    width_index: int,
    val_index: int,
    position_index: int,
    n_vals: int,
) -> tuple[int, ...] | None:
    empty = result_group["empty"]
    if empty.ndim == 4:
        if (
            duration_index >= empty.shape[0]
            or width_index >= empty.shape[1]
            or val_index >= min(n_vals, empty.shape[2])
            or position_index >= empty.shape[3]
        ):
            return None
        return (duration_index, width_index, val_index, position_index)
    if empty.ndim == 3 and n_vals == 1:
        if (
            duration_index >= empty.shape[0]
            or width_index >= empty.shape[1]
            or position_index >= empty.shape[2]
        ):
            return None
        return (duration_index, width_index, position_index)
    return None


def organize_cell_array(mat_path: Path, skip_cell: int | None) -> dict[tuple[int, int], list[np.ndarray]]:
    organized: dict[tuple[int, int], list[np.ndarray]] = {}
    with h5py.File(mat_path, "r") as file_handle:
        root = file_handle["singleBarStT4" if "T4" in mat_path.name else "singleBarStT5"]
        for cell_index in range(root["positions"].shape[0]):
            if cell_index == skip_cell:
                continue
            result_group = deref(file_handle, root["result"][cell_index, 0])
            summary_index = tuple(-1 for _ in range(result_group["maxExtPosVal"].ndim))
            rel_max_ext = float(
                np.array(deref(file_handle, result_group["maxExtPosVal"][summary_index])).squeeze()
            )
            rel_min_inh = float(
                np.array(deref(file_handle, result_group["minInhPosVal"][summary_index])).squeeze()
            )
            rel_pd = float(np.sign(rel_min_inh - rel_max_ext))
            positions = np.atleast_1d(np.array(deref(file_handle, root["positions"][cell_index, 0])).squeeze())
            durations = np.atleast_1d(np.array(deref(file_handle, root["durations"][cell_index, 0])).squeeze())
            widths = np.atleast_1d(np.array(deref(file_handle, root["widths"][cell_index, 0])).squeeze())
            vals = np.atleast_1d(np.array(deref(file_handle, root["vals"][cell_index, 0])).squeeze())

            duration_index = None
            for slot_index, duration in enumerate(durations):
                if slot_index >= result_group["empty"].shape[0]:
                    break
                if np.isclose(duration, PLOT_DURATION_S):
                    duration_index = slot_index
                    break
            width_index = None
            for slot_index, width_led in enumerate(widths):
                if result_group["empty"].ndim == 4 and slot_index >= result_group["empty"].shape[1]:
                    break
                if result_group["empty"].ndim == 3 and slot_index >= result_group["empty"].shape[1]:
                    break
                if np.isclose(width_led, PLOT_WIDTH_LED):
                    width_index = slot_index
                    break
            if duration_index is None or width_index is None:
                continue

            for position_index, orig_pos in enumerate(positions):
                rel_pos = align_index(float(orig_pos), PLOT_WIDTH_LED, rel_max_ext, rel_pd)
                if not 1 <= rel_pos <= N_POSITIONS:
                    continue
                for val_index, val in enumerate(vals):
                    index = result_index(
                        result_group,
                        duration_index,
                        width_index,
                        val_index,
                        position_index,
                        len(vals),
                    )
                    if index is None:
                        continue
                    trace = read_trace(file_handle, result_group, index)
                    if trace is None:
                        continue
                    organized.setdefault((rel_pos, int(val)), []).append(trace)
    return organized


def panel_positions() -> np.ndarray:
    return np.arange(1 + CHOP_EDGES, N_POSITIONS - CHOP_EDGES + 1) + 1


def plot_panel(
    organized: dict[tuple[int, int], list[np.ndarray]],
    position_index: int,
    cell_type: str,
    ax: plt.Axes,
) -> list[dict[str, object]]:
    time_ms = (np.arange(0, REL_LEN, DOWN_SAMP) - NEW_ZERO) / 20.0
    panel_rows: list[dict[str, object]] = []
    yy_lim = (-7.5, 25.0)

    for val in (1, 0):
        traces = organized.get((position_index, val), [])
        if not traces:
            continue
        stack = np.column_stack(traces)
        mean_trace = np.nanmean(stack, axis=1)
        mean_red = mean_trace[::DOWN_SAMP]
        sem = np.std(stack[::DOWN_SAMP, :], axis=1, ddof=0) / np.sqrt(stack.shape[1])
        palette = COLORS["green"] if val == 1 else COLORS["dark"]
        ax.fill_between(time_ms, mean_red + sem, mean_red - sem, color=palette[0], linewidth=0)
        ax.plot(time_ms, mean_red, color=palette[1], linewidth=1.5)
        contrast = CONTRAST_BY_CELL[cell_type][val]
        color_name = "green" if val == 1 else "black"
        for time_value, vm_value, sem_value in zip(time_ms, mean_red, sem):
            panel_rows.append(
                {
                    "trace_id": f"{cell_type}_{contrast}_pos{position_index - CENTER_INDEX:+d}",
                    "cell_type": cell_type,
                    "contrast": contrast,
                    "position": position_index - CENTER_INDEX,
                    "color": color_name,
                    "time_ms": float(time_value),
                    "vm_mv": float(vm_value),
                    "vm_sem_mv": float(sem_value),
                }
            )

    ax.axvline(0.0, color="0.6", linewidth=0.8)
    ax.axvline(STIMULUS_MS, color="0.6", linewidth=0.8)
    ax.set_xlim(-200, 750)
    ax.set_ylim(*yy_lim)
    ax.set_title(f"{position_index - CENTER_INDEX:+d}", fontsize=8)
    ax.axhline(0.0, color="0.85", linewidth=0.5)
    ax.axvspan(0.0, STIMULUS_MS, color="0.95", zorder=-1)
    return panel_rows


def main() -> None:
    rows: list[dict[str, object]] = []
    t4_organized = organize_cell_array(DATA_DIR / "singleBarStT4.mat", SKIP_T4_CELL)
    t5_organized = organize_cell_array(DATA_DIR / "singleBarStT5.mat", None)
    position_indices = panel_positions()

    fig, axes = plt.subplots(2, len(position_indices), figsize=(18, 5.5), sharex=True, sharey=True)
    for column, position_index in enumerate(position_indices):
        rows.extend(plot_panel(t4_organized, position_index, "T4", axes[0, column]))
        rows.extend(plot_panel(t5_organized, position_index, "T5", axes[1, column]))
        if column == 0:
            axes[0, column].set_ylabel("T4\nVm (mV)")
            axes[1, column].set_ylabel("T5\nVm (mV)")
        axes[1, column].set_xlabel("ms", fontsize=8)

    axes[0, 0].plot([], [], color=COLORS["green"][1], label="bright")
    axes[0, 0].plot([], [], color=COLORS["dark"][1], label="dark")
    axes[0, -1].legend(frameon=False, fontsize=7)
    fig.suptitle("Gruntman et al. 2021, Figure 2A (Figshare data)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_CSV} ({len(df):,} points, {df.trace_id.nunique()} traces)")


if __name__ == "__main__":
    main()
