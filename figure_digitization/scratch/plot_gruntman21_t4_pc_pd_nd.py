"""Plot shifted T4 PC position responses and their PD/ND linear sums.

Reads the original 2021 Figure 2 MAT file, using the author's width-aware
RF alignment. All widths use 160-ms flashes at positions -4 through +4.
Only the time window supported by every source trace is used for totals;
baseline-subtracted values preceding the available baseline are padded by zero.
"""

from pathlib import Path
import argparse

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from gruntman21_fit_lp import RAW_DIR, _object, _values, _scalar, _condition_index

HERE = Path(__file__).resolve().parent
POSITIONS = tuple(range(-4, 5))
WIDTHS = (1, 2, 4)
FLASH_MS = 160
DT_MS = 5


def load_raw_means():
    recordings = {(w, p): [] for w in WIDTHS for p in POSITIONS}
    with h5py.File(RAW_DIR / "singleBarStT4.mat", "r") as h5:
        root = h5["singleBarStT4"]
        for cell in range(root["result"].shape[0]):
            result = _object(h5, root["result"][cell, 0])
            if not isinstance(result, h5py.Group):
                continue
            positions = _values(h5, root["positions"], cell)
            durations = _values(h5, root["durations"], cell)
            widths = _values(h5, root["widths"], cell)
            values = _values(h5, root["vals"], cell)
            dur = np.flatnonzero(np.isclose(durations, FLASH_MS / 1000))
            val = np.flatnonzero(values == 1)  # T4 PC = bright
            if not dur.size or not val.size:
                continue
            summary = (-1,) * result["empty"].ndim
            center = _scalar(h5, result["maxExtPosVal"], summary)
            side = np.sign(_scalar(h5, result["minInhPosVal"], summary) - center)
            if side == 0:
                raise ValueError(f"Undefined direction for cell {cell + 1}")
            for width in WIDTHS:
                wid = np.flatnonzero(np.isclose(widths, width))
                if not wid.size:
                    continue
                for position in POSITIONS:
                    raw_position = (center + position if side == 1
                                    else center + width - 1 - position)
                    pos = np.flatnonzero(np.isclose(positions, raw_position))
                    if not pos.size:
                        continue
                    index = _condition_index(result, int(val[0]), int(wid[0]),
                                             int(dur[0]), int(pos[0]))
                    if _scalar(h5, result["empty"], index):
                        continue
                    sub = _object(h5, result["subData"][index])
                    raw = np.asarray(sub["baseSub"], dtype=float)
                    assert raw.shape[0] == 2 and np.isfinite(raw).all()
                    assert np.all(np.diff(raw[0]) > 0)
                    recordings[width, position].append(raw)

    assert all(len(rs) >= 2 for rs in recordings.values())
    first = max(raw[0, 0] for rs in recordings.values() for raw in rs)
    last = min(raw[0, -1] for rs in recordings.values() for raw in rs)
    times = np.arange(np.ceil(first / DT_MS) * DT_MS,
                      np.floor(last / DT_MS) * DT_MS + 1, DT_MS)
    assert times[0] < 0 and times[-1] > FLASH_MS
    means, rows = {}, []
    for (width, position), rs in recordings.items():
        mean = np.mean([np.interp(times, raw[0], raw[1]) for raw in rs], axis=0)
        means[width, position] = mean
        rows.append(pd.DataFrame(dict(width_led=width, position=position,
                                     time_ms=times, vm_mv=mean, n_cells=len(rs))))
    return times, means, pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay-ms", type=float, default=40)
    args = parser.parse_args()
    assert args.delay_ms > 0
    source_t, means, source = load_raw_means()
    output_t = np.arange(-100, source_t[-1] + 1, DT_MS)
    # Every source is observed through output_t[-1]; no post-recording zeros.
    assert (len(POSITIONS) - 1) * args.delay_ms + FLASH_MS <= output_t[-1], (
        "The complete scan exceeds the common recorded time support")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True,
                             sharey="row", layout="constrained")
    colors = plt.get_cmap("viridis")(np.linspace(0.06, 0.94, len(POSITIONS)))
    direction_colors = {"PD": "#d1493f", "ND": "#2768ad"}
    frames, checks = [], []
    for col, width in enumerate(WIDTHS):
        totals = {}
        for row, (direction, order) in enumerate((
            ("PD", POSITIONS), ("ND", tuple(reversed(POSITIONS)))
        )):
            ax = axes[row, col]
            parts = []
            for position, color in zip(POSITIONS, colors):
                delay = order.index(position) * args.delay_ms
                shifted = np.interp(output_t - delay, source_t,
                                    means[width, position], left=0, right=np.nan)
                assert np.isfinite(shifted).all()
                parts.append(shifted)
                ax.plot(output_t, shifted, color=color, lw=1.05, alpha=0.85)
                ax.plot(delay, -0.08, marker="|", color=color, markersize=9,
                        transform=ax.get_xaxis_transform(), clip_on=False)
                frames.append(pd.DataFrame(dict(
                    width_led=width, direction=direction, component="position",
                    position=position, delay_ms=delay, time_ms=output_t, vm_mv=shifted)))
            total = np.sum(parts, axis=0)
            totals[direction] = total
            ax.plot(output_t, total, color=direction_colors[direction], lw=2.5)
            ax.set_title(f"W{width} ({width * 2.25:g} deg)  |  {direction}: "
                         f"{order[0]:+d} to {order[-1]:+d}", fontsize=11)
            frames.append(pd.DataFrame(dict(
                width_led=width, direction=direction, component="sum",
                position=np.nan, delay_ms=np.nan, time_ms=output_t, vm_mv=total)))
            checks.append(dict(width_led=width, direction=direction,
                               points=len(total), min_mv=float(total.min()),
                               max_mv=float(total.max()),
                               peak_time_ms=float(output_t[total.argmax()])))
        for direction, total in totals.items():
            axes[2, col].plot(output_t, total, color=direction_colors[direction],
                              lw=2.4, label=direction)
        axes[2, col].set_title(f"W{width}: PD / ND linear sums", fontsize=11)
        axes[2, col].set_xlabel("Time from first position onset (ms)")
        axes[2, col].legend(frameon=False, loc="upper right")
    for row in range(3):
        axes[row, 0].set_ylabel("Baseline-subtracted voltage (mV)")
        for ax in axes[row]:
            ax.axhline(0, color="0.65", lw=0.6)
            ax.axvline(0, color="0.65", lw=0.6)
            ax.grid(alpha=0.12)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(output_t[0], output_t[-1])
    handles = [Line2D([], [], color=c, lw=2, label=f"{p:+d}")
               for p, c in zip(POSITIONS, colors)]
    handles.append(Line2D([], [], color="0.2", lw=2.5, label="Sum (red PD / blue ND)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=10,
               frameon=False, fontsize=9, title="Aligned position (LED steps)")
    fig.suptitle("Gruntman 2021 | T4 preferred contrast (bright)\n"
                 f"160-ms flashes; {args.delay_ms:g}-ms shift per position; "
                 "9 positions (-4 to +4)\n"
                 "Original MAT recordings; equal-weight cell means; signed linear sums",
                 fontsize=14)
    stem = HERE / f"gruntman21_t4_pc_pd_nd_{args.delay_ms:g}ms"
    data = pd.concat(frames, ignore_index=True)
    assert not data.duplicated(["width_led", "direction", "component", "position", "time_ms"]).any()
    for (width, direction), group in data.groupby(["width_led", "direction"]):
        positions = group[group.component == "position"].pivot(index="time_ms", columns="position", values="vm_mv")
        total = group[group.component == "sum"].sort_values("time_ms").vm_mv.to_numpy()
        np.testing.assert_allclose(positions.sum(axis=1), total, atol=1e-12)
    fig.savefig(stem.with_suffix(".png"), dpi=170)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    data.to_csv(stem.with_suffix(".csv"), index=False, float_format="%.8f")
    source.to_csv(HERE / "gruntman21_t4_pc_position_means_raw.csv", index=False, float_format="%.8f")
    print(source.groupby(["width_led", "position"]).n_cells.first().to_string())
    print(f"Source time support: {source_t[0]} to {source_t[-1]} ms")
    print(pd.DataFrame(checks).to_string(index=False))
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
