#!/usr/bin/env python3
"""Plot Gruntman et al. 2021 Figure 2B from the Figshare single-bar recordings.

Selective h5py reads of per-trial peak responses (width 4, 160 ms), matching
the respTab assembly and panel-B layout in Figure2Code.m.

Outputs ``2b_data.png`` and ``2b_data.csv`` beside this script.

Run:
    ../.venv/bin/python 2b_data.py
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
OUT_PNG = HERE / "2b_data.png"
OUT_CSV = HERE / "2b_data.csv"

PLOT_DURATION_S = 0.16
PLOT_WIDTH_LED = 4
NUM_CUTOFF = 3
SKIP_T4_CELL = 4
JITTER = 0.1
X_LIM = (-5.0, 10.0)
Y_LIM = (-15.0, 30.0)
TICK_OFFSET_LED = int(np.floor(PLOT_WIDTH_LED / 1.5))
WIDTH_DEG = PLOT_WIDTH_LED * 2.25

BLACK = "#1a1a1a"
GREEN = "#319f54"
CONTRAST_BY_CELL = {"T4": {0: "NC", 1: "PC"}, "T5": {0: "PC", 1: "NC"}}
# Paper layout: rows = T4/T5, columns = NC/PC.
PANEL_SPECS = (
    ("T4", "NC", 0, 0, BLACK, GREEN, "PC"),
    ("T4", "PC", 0, 1, GREEN, BLACK, "NC"),
    ("T5", "NC", 1, 0, GREEN, BLACK, "PC"),
    ("T5", "PC", 1, 1, BLACK, GREEN, "NC"),
)
ROW_LABEL_COLOR = {"T4": GREEN, "T5": BLACK}


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
    return flag.size != 1 or bool(flag)


def norm_position(
    orig_pos: float,
    width_led: float,
    rel_max_ext: float,
    rel_min_inh: float,
) -> float:
    rel_pd = float(np.sign(rel_min_inh - rel_max_ext))
    if rel_pd == 1:
        return orig_pos - rel_max_ext
    return (orig_pos - rel_max_ext - width_led + 1.0) * rel_pd


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


def read_resp(
    file_handle: h5py.File,
    result_group: h5py.Group,
    index: tuple[int, ...],
) -> tuple[float, float] | None:
    if is_empty(file_handle, result_group, index):
        return None
    resp = deref(file_handle, result_group["resp"][index])
    resp_max = float(np.array(deref(file_handle, resp["maxVal"])).squeeze())
    resp_min = float(np.array(deref(file_handle, resp["minVal"])).squeeze())
    if resp_max <= 0 and resp_min >= 0:
        return None
    return resp_max, resp_min


def build_resp_table(mat_path: Path, cell_type: str, skip_cell: int | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with h5py.File(mat_path, "r") as file_handle:
        root = file_handle["singleBarStT4" if "T4" in mat_path.name else "singleBarStT5"]
        for cell_index in range(root["positions"].shape[0]):
            if cell_index == skip_cell:
                continue
            result_group = deref(file_handle, root["result"][cell_index, 0])
            if len(np.atleast_1d(np.array(deref(file_handle, root["vals"][cell_index, 0])).squeeze())) == 1:
                continue
            summary_index = tuple(-1 for _ in range(result_group["maxExtPosVal"].ndim))
            rel_max_ext = float(
                np.array(deref(file_handle, result_group["maxExtPosVal"][summary_index])).squeeze()
            )
            rel_min_inh = float(
                np.array(deref(file_handle, result_group["minInhPosVal"][summary_index])).squeeze()
            )
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
                if slot_index >= result_group["empty"].shape[1]:
                    break
                if np.isclose(width_led, PLOT_WIDTH_LED):
                    width_index = slot_index
                    break
            if duration_index is None or width_index is None:
                continue

            for position_index, orig_pos in enumerate(positions):
                rel_pos = norm_position(
                    float(orig_pos), PLOT_WIDTH_LED, rel_max_ext, rel_min_inh
                )
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
                    peaks = read_resp(file_handle, result_group, index)
                    if peaks is None:
                        continue
                    resp_max, resp_min = peaks
                    contrast = CONTRAST_BY_CELL[cell_type][int(val)]
                    for extremum, response_mv in (
                        ("depolarization", resp_max),
                        ("hyperpolarization", resp_min),
                    ):
                        if extremum == "depolarization" and response_mv <= 0:
                            continue
                        if extremum == "hyperpolarization" and response_mv >= 0:
                            continue
                        rows.append(
                            {
                                "cell_type": cell_type,
                                "cell_num": cell_index + 1,
                                "contrast": contrast,
                                "bar_val": int(val),
                                "width_led": PLOT_WIDTH_LED,
                                "duration_ms": PLOT_DURATION_S * 1000.0,
                                "position_led": float(rel_pos),
                                "extremum": extremum,
                                "response_mv": float(response_mv),
                                "statistic": "cell",
                            }
                        )
    return pd.DataFrame(rows)


def mean_rows(panel_df: pd.DataFrame, cell_type: str, contrast: str, extremum: str) -> pd.DataFrame:
    subset = panel_df[
        (panel_df.cell_type == cell_type)
        & (panel_df.contrast == contrast)
        & (panel_df.extremum == extremum)
    ]
    grouped = (
        subset.groupby("position_led", as_index=False)
        .agg(response_mv=("response_mv", "mean"), n_cells=("cell_num", "count"))
        .query("n_cells > @NUM_CUTOFF")
    )
    grouped["cell_type"] = cell_type
    grouped["contrast"] = contrast
    grouped["extremum"] = extremum
    grouped["statistic"] = "mean"
    grouped["bar_val"] = subset["bar_val"].iloc[0] if len(subset) else np.nan
    grouped["width_led"] = PLOT_WIDTH_LED
    grouped["duration_ms"] = PLOT_DURATION_S * 1000.0
    grouped["cell_num"] = np.nan
    return grouped


def plot_mean_curve(
    ax: plt.Axes,
    means: pd.DataFrame,
    color: str,
    *,
    linewidth: float,
    markersize: float,
    zorder: int,
) -> None:
    if means.empty:
        return
    ordered = means.sort_values("position_led")
    ax.plot(
        ordered.position_led,
        ordered.response_mv,
        color=color,
        linewidth=linewidth,
        marker="o" if markersize else None,
        markerfacecolor=color,
        markeredgecolor="none",
        markersize=markersize,
        zorder=zorder,
    )


def plot_panel(
    ax: plt.Axes,
    cell_df: pd.DataFrame,
    cell_type: str,
    contrast: str,
    color: str,
    ref_color: str,
    ref_contrast: str,
) -> None:
    rng = np.random.default_rng(0)
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    for extremum in ("hyperpolarization", "depolarization"):
        cells = cell_df[
            (cell_df.cell_type == cell_type)
            & (cell_df.contrast == contrast)
            & (cell_df.extremum == extremum)
        ]
        if not cells.empty:
            jitter = rng.uniform(0.0, JITTER, len(cells))
            ax.plot(
                cells.position_led + jitter,
                cells.response_mv,
                ".",
                markerfacecolor="none",
                markeredgecolor=color,
                markersize=5,
                alpha=0.45,
                zorder=1,
            )
        plot_mean_curve(
            ax,
            mean_rows(cell_df, cell_type, ref_contrast, extremum),
            ref_color,
            linewidth=1.0,
            markersize=0,
            zorder=2,
        )
        plot_mean_curve(
            ax,
            mean_rows(cell_df, cell_type, contrast, extremum),
            color,
            linewidth=2.0,
            markersize=7,
            zorder=3,
        )
    tick_positions = np.arange(X_LIM[0], X_LIM[1] + 1, 5) + TICK_OFFSET_LED
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(tick - TICK_OFFSET_LED)) for tick in tick_positions])
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks(np.arange(-10, Y_LIM[1] + 1, 10))


def export_csv(cell_df: pd.DataFrame) -> pd.DataFrame:
    mean_parts = [
        mean_rows(cell_df, cell_type, contrast, extremum)
        for cell_type in ("T4", "T5")
        for contrast in ("NC", "PC")
        for extremum in ("depolarization", "hyperpolarization")
    ]
    means = pd.concat([part for part in mean_parts if not part.empty], ignore_index=True)
    out = pd.concat([cell_df, means], ignore_index=True)
    out.insert(
        0,
        "trace_id",
        out.apply(
            lambda row: (
                f"B_{row.cell_type}_{row.contrast}_w{int(row.width_led)}_"
                f"{row.extremum}_{row.statistic}"
            ),
            axis=1,
        ),
    )
    return out.sort_values(["cell_type", "contrast", "extremum", "statistic", "position_led"])


def main() -> None:
    cell_df = pd.concat(
        [
            build_resp_table(DATA_DIR / "singleBarStT4.mat", "T4", SKIP_T4_CELL),
            build_resp_table(DATA_DIR / "singleBarStT5.mat", "T5", None),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True, sharey=True)
    for cell_type, contrast, row, col, color, ref_color, ref_contrast in PANEL_SPECS:
        plot_panel(axes[row, col], cell_df, cell_type, contrast, color, ref_color, ref_contrast)
    axes[0, 0].set_ylabel("Response extremum (mV)")
    axes[1, 0].set_ylabel("Response extremum (mV)")
    for ax in axes[-1]:
        ax.set_xlabel("Position from center (LED)")
    for row, cell_type in enumerate(("T4", "T5")):
        axes[row, 0].text(
            -0.22,
            0.5,
            cell_type,
            transform=axes[row, 0].transAxes,
            fontsize=11,
            fontweight="bold",
            color=ROW_LABEL_COLOR[cell_type],
            va="center",
            ha="right",
            rotation=90,
        )
    for col, contrast in enumerate(("NC", "PC")):
        axes[0, col].text(
            0.5,
            1.04,
            contrast,
            transform=axes[0, col].transAxes,
            fontsize=9,
            ha="center",
            va="bottom",
        )
    fig.suptitle(f"Width {PLOT_WIDTH_LED} ({WIDTH_DEG:g}°)", fontsize=10, y=1.02)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.12, wspace=0.18)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    out_df = export_csv(cell_df)
    out_df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"Wrote {OUT_PNG}")
    print(
        f"Wrote {OUT_CSV} ({len(out_df):,} rows; "
        f"{len(cell_df):,} cells, {len(out_df) - len(cell_df):,} means)"
    )


if __name__ == "__main__":
    main()
