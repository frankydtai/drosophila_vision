#!/usr/bin/env python3
"""Second-by-second timeline for 512-panel model_all matplotlib path."""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training_config import COST_WINDOW_STEPS, PARAMETER_DIR
from plot.moving_bar import (
    MOVING_BAR_DPI,
    _compute_moving_bar_all_type_traces,
    _moving_bar_ylim,
    _plot_moving_bar_cell,
    _set_moving_bar_xticks,
)
from plot_trained import load_session
import FiveCol_MedSim_Pytorch as fc
import torch


def _moving_bar_right_ticks(ax):
    _set_moving_bar_xticks(ax)


def _moving_bar_hide_ticks(ax):
    ax.tick_params(labelbottom=False, labelleft=False)


class Timeline:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last_sec = -1
        self.events: list[tuple[float, str]] = []

    def mark(self, msg: str):
        t = time.perf_counter() - self.t0
        self.events.append((t, msg))
        sec = int(t)
        if sec > self.last_sec:
            self.last_sec = sec
            print(f"t={sec:4d}s  {msg}", flush=True)

    def phase(self, name: str, dt: float):
        self.mark(f"{name} done  +{dt:.2f}s")


def load_z_from_table(rundir: str, session) -> torch.Tensor:
    import csv

    table_path = os.path.join(rundir, "training_with_Ih_table.csv")
    with open(table_path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_type = {row["ctype"]: row for row in rows}
    type_names = list(session.backend.network.type_names)
    inp_gain = [float(by_type[t]["inp_gain"]) for t in type_names]
    out_gain = [float(by_type[t]["out_gain"]) for t in type_names]
    out_scale = [float(by_type[t]["out_scale"]) for t in type_names]
    Ih_gmax = [float(by_type[t]["Ih_gmax"]) for t in ["L1", "L2", "L3", "L4", "L5"]]
    Ih_midv = float(rows[0]["Ih_midv"])
    Ih_slope = float(rows[0]["Ih_slope"])
    tau_midv = float(rows[0]["tau_midv"])
    z = np.array(
        inp_gain + out_gain + Ih_gmax + [Ih_midv, Ih_slope, tau_midv] + out_scale,
        dtype=np.float64,
    )
    return torch.tensor(z, dtype=torch.float64, device=session.device)


def plot_512_instrumented(session, z, path: str, tl: Timeline):
    """512-panel path: 32 types x 16 specs (all directions)."""
    tl.mark("start traces")
    t0 = time.perf_counter()
    center_only = session.moving_bar_center_column
    types, spec_names, model_mean, model_sem, data_mean = _compute_moving_bar_all_type_traces(session, z)
    tl.phase("traces (_compute_moving_bar_all_type_traces)", time.perf_counter() - t0)

    keys = list(model_mean.keys())
    show_sem = not center_only
    t0 = time.perf_counter()
    ylim = _moving_bar_ylim(model_mean, model_sem, data_mean, keys, show_sem=show_sem)
    tl.phase("ylim", time.perf_counter() - t0)

    nrows = len(types)
    ncols = len(spec_names)
    assert nrows * ncols == 512, f"expected 512 panels, got {nrows}x{ncols}"

    tl.mark(f"subplots begin ({nrows}x{ncols}, figsize={1.4*ncols:.1f}x{0.85*nrows:.1f} in)")
    t0 = time.perf_counter()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(1.4 * ncols, 0.85 * nrows), sharex=True, sharey=True,
    )
    if nrows == 1:
        axes = np.asarray([axes])
    if ncols == 1:
        axes = axes[:, None]
    tl.phase("subplots create", time.perf_counter() - t0)

    tl.mark("plot cells begin")
    t_plot0 = time.perf_counter()
    for ri, tname in enumerate(types):
        t_row = time.perf_counter()
        for ci, sname in enumerate(spec_names):
            ax = axes[ri, ci]
            key = (tname, sname)
            if key not in model_mean:
                ax.axis("off")
                continue
            _plot_moving_bar_cell(
                ax, model_mean[key], model_sem[key], data_mean.get(key),
                sname if ri == 0 else sname,
                show_ylabel=(ci == 0),
                show_sem=show_sem and key in model_sem and np.any(model_sem[key]),
                ylim=ylim,
                cell_ticks=False,
            )
        if ncols:
            axes[ri, 0].set_ylabel(tname, fontsize=6, labelpad=4)
        dt_row = time.perf_counter() - t_row
        tl.mark(f"row {ri+1:2d}/{nrows} plotted  +{dt_row:.2f}s  (cum plot {time.perf_counter()-t_plot0:.1f}s)")

    tl.mark("tick styling begin")
    t0 = time.perf_counter()
    for ri in range(nrows):
        for ci in range(ncols):
            key = (types[ri], spec_names[ci])
            if key not in model_mean:
                continue
            ax = axes[ri, ci]
            if ci == ncols - 1:
                _moving_bar_right_ticks(ax)
            else:
                _moving_bar_hide_ticks(ax)
    tl.phase("tick styling (512x tick_params)", time.perf_counter() - t0)

    tl.mark("suptitle + subplots_adjust")
    t0 = time.perf_counter()
    from network.stimulus import sti_columns
    C = session.backend.network
    scope = f"avg over {len(sti_columns(C))} sti columns"
    fig.suptitle("Moving-bar model-all  [" + scope + ", t_center ± 0.45 s]", fontsize=10)
    fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.55, wspace=0.3)
    tl.phase("suptitle+adjust", time.perf_counter() - t0)

    draw_total = time.perf_counter() - t_plot0
    tl.mark(f"draw loop total {draw_total:.1f}s — savefig next")

    tl.mark("set_rasterized(True) on 512 axes")
    t0 = time.perf_counter()
    for ax in fig.axes:
        ax.set_rasterized(True)
    tl.phase("set_rasterized", time.perf_counter() - t0)

    tl.mark(f"savefig begin dpi={MOVING_BAR_DPI}")
    t0 = time.perf_counter()

    # heartbeat during savefig (runs in main thread; coarse)
    done = {"flag": False}

    def heartbeat():
        while not done["flag"]:
            tl.mark("savefig in progress (Agg draw+rasterize+PNG encode)")
            time.sleep(1.0)

    import threading
    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    fig.savefig(path, dpi=MOVING_BAR_DPI, bbox_inches='tight')
    done["flag"] = True
    hb.join(timeout=0.1)
    plt.close(fig)
    tl.phase("savefig", time.perf_counter() - t0)
    tl.mark("done")


def main():
    rundir = str(PARAMETER_DIR / "conductance" / "run_26693975")
    out_png = os.path.join(rundir, "model_all_512_profile.png")
    session = load_session(rundir, "conductance")
    z = load_z_from_table(rundir, session)

    print(f"matplotlib {matplotlib.__version__}  dpi={MOVING_BAR_DPI}", flush=True)
    print(f"panels=32x16=512  pixels~{int(22.4*MOVING_BAR_DPI)}x{int(27.2*MOVING_BAR_DPI)}", flush=True)
    print("=== per-second timeline ===", flush=True)

    tl = Timeline()
    pr = cProfile.Profile()
    pr.enable()
    plot_512_instrumented(session, z, out_png, tl)
    pr.disable()

    total = tl.events[-1][0] if tl.events else 0
    print("\n=== phase summary ===", flush=True)
    # collapse into phases
    phases = []
    for t, msg in tl.events:
        if "done" in msg or msg.startswith("row") or msg.startswith("draw loop"):
            phases.append((t, msg))
    prev = 0.0
    for t, msg in phases:
        if "done" in msg or msg.startswith("draw loop"):
            print(f"  {msg:55s}  cum={t:7.1f}s  delta={t-prev:7.1f}s", flush=True)
            prev = t

    print(f"\nTOTAL {total:.1f}s", flush=True)

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(25)
    print("\n=== cProfile top 25 (cumtime) ===", flush=True)
    print(s.getvalue(), flush=True)


if __name__ == "__main__":
    main()
