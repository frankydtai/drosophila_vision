"""Shift and sum this run's plotted T4a bright GT and trained mean traces.

Run from vision/simulation with ../.venv/bin/python.  Saved sbar mids are
hex-axis positions: one mid = 4.5 deg = two original GT LED positions.
At 56.25 deg/s, adjacent integer mids are separated by 80 ms.
Only the common measured/simulated time window is used for totals; no
post-response values are fabricated. Original plots/checkpoints are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM.parents[0]))
os.chdir(SIM)
import import_bootstrap  # noqa: F401,E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

from build_hex import DEG
from figure.plot import load_best
from figure.sbar import network_sbar_trace_readout, resolve_sbar_gts
from figure.panel import gt_trace_affine
from task.sbar.pack import sbar_pd_axis_sign

DEFAULT_RUN = SIM / "0_runs/hp_lp/30225520-run_init_from-n_iter-500-tasks-sbar-a_lsd-1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--speed-deg-s", type=float, default=56.25)
    args = ap.parse_args()
    if args.speed_deg_s <= 0:
        raise ValueError("speed must be positive")
    run = args.run.resolve()
    protected = [p for p in (run / "data").iterdir() if p.is_file()]
    protected += [run / "sbar_gt_v.png"]
    before = {p: digest(p) for p in protected}
    torch.set_num_threads(2)
    session, z, cost = load_best(str(run), verbose=True)
    readout = network_sbar_trace_readout(session, z, "sbar", "bright")
    gts, _ = resolve_sbar_gts({"bright": session})
    opts = session.train_opts["sbar_sti_opts"]
    mids = tuple(readout.mids)
    assert mids == (-2, -1, 0, 1, 2)
    assert sbar_pd_axis_sign("T4a") == 1
    assert int(opts["sbar_gt_w"]) == 2
    dt = float(opts["delta_ms"])
    time = (np.arange(readout.n_t) - readout.t_onset) * dt
    pre = (time >= -100) & (time < 0)
    gain, bias = readout.gt_affine_by_cell["T4a"]
    means, baselines, source_rows = {}, {}, []
    for mid in mids:
        model = np.asarray(readout.v_readout["T4a"][mid], dtype=float)
        target = gt_trace_affine(readout, "T4a", gts["bright"]["T4a", mid])
        target[time < 0] = bias  # plotted GT baseline = fitted affine offset
        for name, values in (("GT", target), ("Trained T4a", model)):
            baseline = bias if name == "GT" else float(values[pre].mean())
            assert np.isfinite(values).all()
            means[name, mid] = values - baseline
            baselines[name, mid] = baseline
            source_rows.append(pd.DataFrame(dict(
                source=name, mid=mid, raw_led_position=2 * mid,
                relative_angle_deg=float(DEG) * mid, time_ms=time,
                absolute_voltage_mv=values, baseline_mv=baseline,
                delta_voltage_mv=values - baseline,
                n_model_nodes=readout.n_by_cell_mid["T4a"][mid])))
    source_data = pd.concat(source_rows, ignore_index=True)

    # Positive bar-minus-cell mid is positive on the T4a PD axis.
    pd_delays = {mid: (mid - min(mids)) * float(DEG) / args.speed_deg_s * 1000
                 for mid in mids}
    nd_delays = {mid: (max(mids) - mid) * float(DEG) / args.speed_deg_s * 1000
                 for mid in mids}
    mapping = pd.DataFrame(dict(
        mid=mids, raw_led_position=[2 * mid for mid in mids],
        relative_angle_deg=[mid * float(DEG) for mid in mids],
        pd_delay_ms=[pd_delays[mid] for mid in mids],
        nd_delay_ms=[nd_delays[mid] for mid in mids]))
    output_t = time[time >= -100]
    assert max(pd_delays.values()) + float(opts["ms_sti"]) <= output_t[-1]
    colors = plt.get_cmap("viridis")(np.linspace(0.06, 0.94, len(mids)))
    direction_colors = {"PD": "#d1493f", "ND": "#2768ad"}
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11), sharex=True,
                             sharey=True, constrained_layout=True)
    frames, metrics = [], []
    for col, (direction, delays) in enumerate((("PD", pd_delays), ("ND", nd_delays))):
        totals = {}
        for row, name in enumerate(("GT", "Trained T4a")):
            ax = axes[row, col]
            parts = []
            for mid, color in zip(mids, colors):
                shifted = np.interp(output_t - delays[mid], time, means[name, mid],
                                    left=0.0, right=np.nan)
                assert np.isfinite(shifted).all()
                parts.append(shifted)
                ax.plot(output_t, shifted, color=color, lw=1.25, alpha=0.9)
                ax.plot(delays[mid], -0.055, "|", color=color, markersize=8,
                        transform=ax.get_xaxis_transform(), clip_on=False)
                frames.append(pd.DataFrame(dict(
                    source=name, direction=direction, component="position", mid=mid,
                    raw_led_position=2 * mid, delay_ms=delays[mid],
                    time_ms=output_t, delta_voltage_mv=shifted)))
            total = np.sum(parts, axis=0)
            totals[name] = total
            ax.plot(output_t, total, lw=2.6,
                    color="0.15" if name == "GT" else direction_colors[direction],
                    label="Linear sum")
            ax.legend(loc="upper right", frameon=False, fontsize=9)
            arrow = "-2 to +2" if direction == "PD" else "+2 to -2"
            ax.set_title(f"{name} | {direction}: mid {arrow}")
            frames.append(pd.DataFrame(dict(
                source=name, direction=direction, component="sum", mid=np.nan,
                raw_led_position=np.nan, delay_ms=np.nan, time_ms=output_t,
                delta_voltage_mv=total)))
            metrics.append(dict(source=name, direction=direction,
                peak_mv=float(total.max()), peak_time_ms=float(output_t[total.argmax()]),
                trough_mv=float(total.min()), trough_time_ms=float(output_t[total.argmin()])))
        axes[2, col].plot(output_t, totals["GT"], "--", color="0.15", lw=2.5, label="GT sum")
        axes[2, col].plot(output_t, totals["Trained T4a"],
                         color=direction_colors[direction], lw=2.5, label="Trained T4a sum")
        axes[2, col].set_title(f"{direction}: GT vs trained T4a")
        axes[2, col].legend(frameon=False, loc="upper right")
        axes[2, col].set_xlabel("Time from first position onset (ms)")
    for row in range(3):
        axes[row, 0].set_ylabel("Baseline-subtracted voltage (mV)")
        for ax in axes[row]:
            ax.axhline(0, color="0.65", lw=0.7)
            ax.axvline(0, color="0.65", lw=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.14)
            ax.set_xlim(output_t[0], output_t[-1])
    handles = [Line2D([], [], color=c, lw=2,
                     label=f"mid {m:+g} / LED {2*m:+g}") for m, c in zip(mids, colors)]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    step_delay = float(DEG) / args.speed_deg_s * 1000
    run_id = run.name.split("-", 1)[0]
    node_counts = sorted(set(readout.n_by_cell_mid["T4a"].values()))
    node_label = "/".join(str(n) for n in node_counts)
    fig.suptitle(f"Run {run_id} | T4a preferred contrast (bright)\n"
                 f"160-ms flash; 4.5 deg per mid; {args.speed_deg_s:g} deg/s; "
                 f"{step_delay:g}-ms shift per panel\n"
                 f"GT: original W2 recordings; model: mean of {node_label} T4a neurons",
                 fontsize=14)
    stem = run / f"sbar_t4a_pc_pd_nd_{step_delay:g}ms"
    fig.savefig(stem.with_suffix(".png"), dpi=170)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    # Source-panel check: absolute GT/model traces match the supplied top row.
    check, axs = plt.subplots(1, 5, figsize=(15, 3.3), sharex=True, sharey=True,
                               constrained_layout=True)
    for ax, mid in zip(axs, mids):
        for name, color in (("GT", "0.4"), ("Trained T4a", "red")):
            source = source_data[(source_data.source == name) & (source_data.mid == mid)]
            ax.plot(source.time_ms + float(opts["ms_pre"]), source.absolute_voltage_mv,
                    color=color, label=name, lw=1.6)
        ax.axvspan(float(opts["ms_pre"]), float(opts["ms_pre"]) + float(opts["ms_sti"]),
                   color="0.85", alpha=0.4)
        ax.set_title(f"mid {mid:+g} / LED {2 * mid:+g}")
        ax.set_xlabel("Original simulation time (ms)")
    axs[0].set_ylabel("Absolute voltage (mV)")
    axs[0].legend(frameon=False)
    check.suptitle("Source check: sbar_gt_v.png, T4a bright (solid curves)")
    check.savefig(run / "sbar_t4a_pc_source_check.png", dpi=150)
    plt.close(check)

    result = pd.concat(frames, ignore_index=True)
    assert not result.duplicated(["source", "direction", "component", "mid", "time_ms"]).any()
    for _, group in result.groupby(["source", "direction"]):
        components = group[group.component == "position"].pivot(
            index="time_ms", columns="mid", values="delta_voltage_mv")
        total = group[group.component == "sum"].sort_values("time_ms").delta_voltage_mv
        np.testing.assert_allclose(components.sum(axis=1), total, atol=1e-12)
    result.to_csv(stem.with_suffix(".csv"), index=False)
    source_data.to_csv(run / "sbar_t4a_pc_sources.csv", index=False)
    mapping.to_csv(run / "sbar_t4a_pc_position_delays.csv", index=False)
    meta = dict(run=str(run), cell="T4a", contrast="bright", source_gt_width_led=2,
                flash_ms=float(opts["ms_sti"]), speed_deg_s=args.speed_deg_s,
                degree_per_mid=float(DEG), delay_per_mid_ms=step_delay,
                simulation_delta_ms=dt, original_onset_ms=float(opts["ms_pre"]),
                output_end_ms=float(output_t[-1]), gt_affine_gain=float(gain),
                gt_affine_bias_mv=float(bias), model_baseline_window_ms=[-100, 0],
                source_run_cost=cost, sum_metrics=metrics,
                tail_policy="Stop totals at 600 ms; no post-recording extrapolation or padding.")
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    assert all(digest(p) == before[p] for p in protected), "Original run files changed"
    print(mapping.to_string(index=False))
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"GT affine: gain={gain}, bias={bias}; output ends at {output_t[-1]} ms")
    print(f"Validated component sums; preserved {len(protected)} original files")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
