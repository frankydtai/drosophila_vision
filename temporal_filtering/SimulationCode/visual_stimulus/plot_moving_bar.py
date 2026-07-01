#!/usr/bin/env python3
"""Visualise moving-bar column coverage on a connectome hex field (demo only).

Snapshots pick entry / peak-spread / exit from the generated column current.
Geometry/timing from ``moving_bar_stimulus``; unit currents from ``network.stimulus``.

Usage (from SimulationCode/, uses project .venv):

    ../.venv/bin/python visual_stimulus/plot_moving_bar.py
    ../.venv/bin/python visual_stimulus/plot_moving_bar.py --gif
    ../.venv/bin/python visual_stimulus/plot_moving_bar.py --network right_min_neuron1_extent2 --direction down --gif
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLOT_DIR = os.path.join(HERE, "plotted_moving_bar")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

from Medulla_Library import SIGNAL_BASELINE, SIGNAL_BRIGHT
from network.construction import load_network
from network.stimulus import build_moving_bar_signals
from column_mapper import DEFAULT_KERNEL_SIZE, HEX_PATCH_RADIUS, hex_to_pixel, set_axis_labels
from visual_stimulus.moving_bar_stimulus import (
    bar_rect_at_step,
    gruntman_moving_bar_specs,
    moving_bar_transit_times,
)
from connectome_io import resolve_network_json

PLOT_BG = "#F5F0DC"  # axes background (beige), not column baseline color
DEFAULT_NETWORK = "right_min_neuron1"


def _run_tag(network_path: str, meta: dict) -> str:
    """``right`` or ``left``; append ``_extentN`` only when the run folder has it."""
    run_name = Path(network_path).resolve().parent.name
    side = str(meta.get("side") or run_name.split("_")[0])
    m = re.search(r"_extent(\d+)$", run_name)
    if m:
        return f"{side}_extent{m.group(1)}"
    return side


def _output_tag(network_path: str, meta: dict, direction: str) -> str:
    """``2{direction}_{side}`` or ``2{direction}_{side}_extentN``."""
    return f"2{direction}_{_run_tag(network_path, meta)}"


def _default_outputs(network_path: str, meta: dict, direction: str) -> tuple[str, str]:
    tag = _output_tag(network_path, meta, direction)
    return (
        os.path.join(PLOT_DIR, f"moving_bar_{tag}.png"),
        os.path.join(PLOT_DIR, f"moving_bar_{tag}.gif"),
    )


def _photo_columns_for_plot(C):
    """Demo helper: column centres for hex drawing (not part of network.stimulus)."""
    cols = {}
    u_in = C.u[C.is_input]
    v_in = C.v[C.is_input]
    for u, v in zip(u_in.tolist(), v_in.tolist()):
        key = (int(u), int(v))
        if key in cols:
            continue
        if len(C.input_units_at(key[0], key[1])) == 0:
            continue
        x, y = hex_to_pixel(key[0], key[1], DEFAULT_KERNEL_SIZE)
        cols[key] = (float(x), float(y))
    return [(cols[k][0], cols[k][1]) for k in sorted(cols)]


def _hex_vertices_for_plot(cx: float, cy: float) -> np.ndarray:
    angles = np.deg2rad(30.0 + 60.0 * np.arange(6, dtype=np.float64))
    vx = cx + HEX_PATCH_RADIUS * np.cos(angles)
    vy = cy + HEX_PATCH_RADIUS * np.sin(angles)
    return np.column_stack([vx, vy])


def _field_bounds_from_columns(columns) -> tuple[float, float, float, float]:
    verts = [_hex_vertices_for_plot(x, y) for x, y in columns]
    return (
        float(min(v[:, 0].min() for v in verts)),
        float(min(v[:, 1].min() for v in verts)),
        float(max(v[:, 0].max() for v in verts)),
        float(max(v[:, 1].max() for v in verts)),
    )


def _field_limits(columns):
    x0, y0, x1, y1 = _field_bounds_from_columns(columns)
    pad = 2.0
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad


def _transit_frame_times(spec, field_deg, t_on, maxtime, frame_step: int) -> list[int]:
    t0, _, t1 = moving_bar_transit_times(spec, field_deg, t_on=t_on, maxtime=maxtime)
    return list(range(t0, t1 + 1, max(1, frame_step)))


def _draw_bar_outline(ax, spec, field_deg, t: int, t_on: int):
    xmin, ymin, xmax, ymax = bar_rect_at_step(spec, field_deg, t, t_on=t_on)
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="red",
            linewidth=1.8,
            zorder=10,
        )
    )


def _current_cmap(i_max: float, i_baseline: float):
    """Column face: 0 pA black (dark bar), baseline gray, 40 pA white (bright bar)."""
    mid = i_baseline / i_max if i_max > 0 else 0.5
    return mcolors.LinearSegmentedColormap.from_list(
        "bar_current",
        [(0.0, "#000000"), (mid, "#888888"), (1.0, "#FFFFFF")],
    )


def _style_axes(ax):
    ax.set_facecolor(PLOT_BG)


def _val_to_color(val: float, cmap, i_max: float) -> tuple:
    t = float(np.clip(val / i_max if i_max > 0 else 0.0, 0.0, 1.0))
    return cmap(t)


def _draw_hex_field(ax, columns, vals, i_max, i_baseline, xlim, ylim):
    cmap = _current_cmap(i_max, i_baseline)
    verts = [_hex_vertices_for_plot(x, y) for x, y in columns]
    colors = [_val_to_color(val, cmap, i_max) for val in vals]
    ax.add_collection(
        PolyCollection(
            verts,
            facecolors=colors,
            edgecolors="0.35",
            linewidths=0.15,
            alpha=0.95,
        )
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    _style_axes(ax)
    set_axis_labels(ax, fontsize=8)


def plot_snapshot(ax, columns, column_current, t, spec, spec_name, i_max, i_baseline, xlim, ylim, t_on, field_deg):
    _draw_hex_field(ax, columns, column_current[t], i_max, i_baseline, xlim, ylim)
    _draw_bar_outline(ax, spec, field_deg, t, t_on)
    ax.set_title(f"{spec_name}  t={t} ({t * 0.01:.2f} s)", fontsize=9)


def write_snapshots(columns, showcase, column_current, i_max, i_baseline, output, side, t_on, maxtime, field_deg):
    xlim = _field_limits(columns)[:2]
    ylim = _field_limits(columns)[2:]
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    panel_h = max(2.4, 3.0 * yspan / max(xspan / 3.0, 1.0))
    fig, axes = plt.subplots(
        len(showcase), 3,
        figsize=(14.0, panel_h * len(showcase)),
        facecolor=PLOT_BG,
    )
    if len(showcase) == 1:
        axes = np.expand_dims(axes, 0)

    for i, spec in enumerate(showcase):
        times = moving_bar_transit_times(spec, field_deg, t_on=t_on, maxtime=maxtime)
        for j, (t, label) in enumerate(zip(times, ("entry", "center", "exit"))):
            plot_snapshot(
                axes[i, j], columns, column_current[i], t, spec,
                f"{spec.name} ({label})", i_max, i_baseline, xlim, ylim, t_on, field_deg,
            )
        spread = float(np.ptp(column_current[i, times[1]]))
        print(f"  {spec.name}: entry/center/exit t={times}  center ptp={spread:.1f} pA")

    fig.suptitle(
        f"Moving-bar column current (pA)  side={side}  "
        f"{len(columns)} photo columns  I_baseline={i_baseline}  I_max={i_max}",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def write_animation(columns, showcase, column_current, i_max, i_baseline, output, side, t_on, maxtime, field_deg, frame_step):
    times = sorted({t for spec in showcase for t in _transit_frame_times(spec, field_deg, t_on, maxtime, frame_step)})
    if not times:
        print("no animation frames")
        return

    xlim = _field_limits(columns)[:2]
    ylim = _field_limits(columns)[2:]
    fig, axes = plt.subplots(len(showcase), 1, figsize=(4.5, 2.8 * len(showcase)), squeeze=False, facecolor=PLOT_BG)
    title = fig.suptitle("", fontsize=11)

    def update(frame_idx):
        t = times[frame_idx]
        title.set_text(
            f"Moving-bar column current (pA)  side={side}  "
            f"{len(columns)} photo columns  I_baseline={i_baseline}  I_max={i_max}  t={t} ({t * 0.01:.2f} s)"
        )
        for i, spec in enumerate(showcase):
            axes[i, 0].clear()
            plot_snapshot(
                axes[i, 0], columns, column_current[i], t, spec,
                spec.name, i_max, i_baseline, xlim, ylim, t_on, field_deg,
            )
        return [title]

    anim = FuncAnimation(fig, update, frames=len(times), interval=80, blit=False)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    anim.save(output, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"wrote {output}  ({len(times)} frames, t={times[0]}..{times[-1]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", type=str, default=DEFAULT_NETWORK,
                    help=f"built_network run folder name (default: {DEFAULT_NETWORK})")
    ap.add_argument("-o", "--output", type=str, default=None,
                    help="snapshot PNG (default: visual_stimulus/plotted_moving_bar/moving_bar_2<dir>_<side>[_extentN].png)")
    ap.add_argument("--gif", nargs="?", const="", default=None,
                    help="write GIF; default path if flag alone, or pass a path")
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--direction", type=str, default="right", choices=("right", "left", "up", "down"))
    ap.add_argument("--i-bright", type=float, default=SIGNAL_BRIGHT)
    args = ap.parse_args()

    network_json = str(resolve_network_json(args.network))
    C = load_network(network_json, device="cpu")
    default_png, default_gif = _default_outputs(network_json, C.meta, args.direction)
    output = args.output or default_png
    showcase = [s for s in gruntman_moving_bar_specs() if s.direction == args.direction]
    T = build_moving_bar_signals(C, specs=showcase)
    columns = _photo_columns_for_plot(C)
    t_on = int(T.info["t_on"])
    maxtime = int(T.info["maxtime"])
    field_deg = tuple(T.info["field_deg"])
    i_baseline = float(T.info["i_baseline"])
    i_bright = args.i_bright
    print(
        f"maxtime={maxtime} steps ({maxtime * 0.01:.2f} s)  "
        f"sweep={T.info['sweep_steps']} steps ({T.info['sweep_time_s']:.2f} s after t_on)"
    )

    write_snapshots(
        columns, showcase, T.column_current, i_bright, i_baseline,
        output, C.meta.get("side", "?"), t_on, maxtime, field_deg,
    )
    if args.gif is not None:
        gif = default_gif if args.gif == "" else args.gif
        write_animation(
            columns, showcase, T.column_current, i_bright, i_baseline, gif,
            C.meta.get("side", "?"), t_on, maxtime, field_deg, args.frame_step,
        )
    print(f"signal shape {tuple(T.signal.shape)}  specs={T.info['spec_names']}")


if __name__ == "__main__":
    main()
