"""Visualise moving-bar hex coverage (demo only).

Connectome: hex sti field from ``task.moving_bar.sti_geo``.

Usage (from simulation/, project .venv):

    ../.venv/bin/python 5_figure/plot_sti/moving_bar.py
    ../.venv/bin/python 5_figure/plot_sti/moving_bar.py --gif
    ../.venv/bin/python 5_figure/plot_sti/moving_bar.py --network right_min_neuron1_r2 --direction down --gif
    ../.venv/bin/python 5_figure/plot_sti/moving_bar.py --network right_min_neuron1_r2 --bar-radius 2
"""
from __future__ import annotations

from default_params import (
    NETWORK_CONSTRUCTION,
    NEURON_PARAM,
    NEURON_SCHEMA,
)

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PLOT_DIR = os.path.join(HERE, "plotted_moving_bar")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import import_bootstrap  # noqa: F401
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from train.param import SIM_DTYPE
from network.construction import load_network
from import_bootstrap import parse_bool, parse_comma_list
from build_hex import (
    FIELD_VIEW_PAD_DEG,
    draw_hex_patches,
    draw_hex_patches_uv,
    view_bounds_from_vertices,
    set_axis_labels,
    xy_deg_from_uv,
)
from task.moving_bar.sti_geo import (
    DEFAULT_BAR_RADIUS,
    sti_hexes,
)
from task.moving_bar.sti_spec import (
    GRUNTMAN_CONTRASTS,
    GRUNTMAN_DIRECTIONS,
    bar_lane_rects_at_t,
    build_moving_bar_signals,
    gruntman_moving_bar_specs,
    moving_bar_transit_times,
)
from path import DEFAULT_NETWORK_RUN, network_run_tag, resolve_network_json

PLOT_BG = "#F5F0DC"  # axes background (beige), not hex baseline color
_STI_CLI_CONTRASTS = ",".join(GRUNTMAN_CONTRASTS)


def _field_limits(hexes, *, hexes_are_xy_deg: bool = False):
    if hexes_are_xy_deg:
        x_deg = [x for x, _ in hexes]
        y_deg = [y for _, y in hexes]
    else:
        x_deg, y_deg = xy_deg_from_uv(
            [u for u, _ in hexes],
            [v for _, v in hexes],
        )
    x0, y0, x1, y1 = view_bounds_from_vertices(x_deg, y_deg)
    pad = FIELD_VIEW_PAD_DEG
    return (x0 - pad, x1 + pad), (y0 - pad, y1 + pad)


def _draw_bar_outline(ax, spec, view_deg, t: int, t_onset: int, *, bar_radius: int, multi_bar: bool = True):
    rects = bar_lane_rects_at_t(spec, view_deg, bar_radius, t, multi_bar=bool(multi_bar), t_onset=t_onset)
    for xmin, ymin, xmax, ymax in rects:
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


def _draw_hex_field(ax, hexes, vals, i_max, i_baseline, xlim, ylim, *, hexes_are_xy_deg: bool = False):
    cmap = _current_cmap(i_max, i_baseline)
    colors = [
        cmap(float(np.clip(val / i_max if i_max > 0 else 0.0, 0.0, 1.0)))
        for val in vals
    ]
    if hexes_are_xy_deg:
        x_deg = [x for x, _ in hexes]
        y_deg = [y for _, y in hexes]
        draw_hex_patches(ax, x_deg, y_deg, colors, linewidth=0.15, alpha=0.95)
    else:
        u = [uv[0] for uv in hexes]
        v = [uv[1] for uv in hexes]
        draw_hex_patches_uv(ax, u, v, colors, linewidth=0.15, alpha=0.95)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_facecolor(PLOT_BG)
    set_axis_labels(ax, fontsize=8)


def plot_snapshot(
    ax, hexes, i_sti_hex, t, spec, spec_name, i_max, i_baseline, xlim, ylim, t_onset, view_deg, *,
    hexes_are_xy_deg: bool = False,
    bar_radius=None,
    multi_bar: bool = True,
):
    _draw_hex_field(
        ax, hexes, i_sti_hex[t], i_max, i_baseline, xlim, ylim,
        hexes_are_xy_deg=hexes_are_xy_deg,
    )
    _draw_bar_outline(ax, spec, view_deg, t, t_onset, bar_radius=bar_radius, multi_bar=bool(multi_bar))
    ax.set_title(f"{spec_name}  t={t} ({t * NEURON_PARAM['delta_ms'] / 1000.0:.2f} s)", fontsize=9)


def save_snapshots(
    plot_hexes,
    showcase,
    i_sti_hex,
    i_max,
    i_baseline,
    output,
    side,
    t_onset,
    n_t,
    view_deg,
    snapshot_t=None,
    hexes_are_xy_deg: bool = False,
    bar_radius=None,
    multi_bar: bool = True,
):
    snapshot_t = list(snapshot_t or [])
    if snapshot_t:
        if any(t < 0 for t in snapshot_t):
            raise SystemExit("--t must be non-negative t indices")
        bad = [t for t in snapshot_t if t >= n_t]
        if bad:
            raise SystemExit(f"--t out of range (n_t={n_t}): {bad}")
    xlim, ylim = _field_limits(plot_hexes, hexes_are_xy_deg=hexes_are_xy_deg)
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    panel_h = max(2.4, 3.0 * yspan / max(xspan / 3.0, 1.0))
    ncols = len(snapshot_t) if snapshot_t else 3
    fig, axes = plt.subplots(
        len(showcase), ncols,
        figsize=(14.0, panel_h * len(showcase)),
        facecolor=PLOT_BG,
    )
    if len(showcase) == 1:
        axes = np.expand_dims(axes, 0)

    for i, spec in enumerate(showcase):
        if snapshot_t:
            times = snapshot_t
            labels = [f"t={t}" for t in times]
        else:
            start_t, mid_t, exit_t = moving_bar_transit_times(
                spec, view_deg, bar_radius, multi_bar=bool(multi_bar), t_onset=t_onset, n_t=n_t,
            )
            times = [start_t, mid_t, exit_t]
            labels = ("start", "mid", "exit")
        for j, (t, label) in enumerate(zip(times, labels)):
            plot_snapshot(
                axes[i, j], plot_hexes, i_sti_hex[i], t, spec,
                f"{spec.name} ({label})", i_max, i_baseline, xlim, ylim, t_onset, view_deg,
                hexes_are_xy_deg=hexes_are_xy_deg,
                bar_radius=bar_radius,
                multi_bar=bool(multi_bar),
            )
        if len(times) >= 3 and not snapshot_t:
            spread = float(np.ptp(i_sti_hex[i, times[1]]))
            print(f"  {spec.name}: start/mid/exit t={times}  mid ptp={spread:.1f} pA")
        else:
            print(f"  {spec.name}: snapshot t={times}")

    fig.suptitle(
        f"Moving-bar i_sti_hex (pA)  side={side}  "
        f"{len(plot_hexes)} sti hexes  I_baseline={i_baseline}  I_max={i_max}",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def save_animation(
    plot_hexes, showcase, i_sti_hex, i_max, i_baseline, output, side, t_onset, n_t, view_deg, t_stride,
    *, hexes_are_xy_deg: bool = False,
    bar_radius=None,
    multi_bar: bool = True,
):
    stride = max(1, t_stride)
    times = set()
    for spec in showcase:
        t0, _, t1 = moving_bar_transit_times(
            spec, view_deg, bar_radius, multi_bar=bool(multi_bar), t_onset=t_onset, n_t=n_t,
        )
        times.update(range(t0, t1 + 1, stride))
    times = sorted(times)
    if not times:
        print("no animation frames")
        return

    xlim, ylim = _field_limits(plot_hexes, hexes_are_xy_deg=hexes_are_xy_deg)
    fig, axes = plt.subplots(len(showcase), 1, figsize=(4.5, 2.8 * len(showcase)), squeeze=False, facecolor=PLOT_BG)
    title = fig.suptitle("", fontsize=11)

    def update(frame_idx):
        t = times[frame_idx]
        title.set_text(
            f"Moving-bar i_sti_hex (pA)  side={side}  "
            f"{len(plot_hexes)} sti hexes  I_baseline={i_baseline}  I_max={i_max}  t={t} ({t * NEURON_PARAM['delta_ms'] / 1000.0:.2f} s)"
        )
        for i, spec in enumerate(showcase):
            axes[i, 0].clear()
            plot_snapshot(
                axes[i, 0], plot_hexes, i_sti_hex[i], t, spec,
                spec.name, i_max, i_baseline, xlim, ylim, t_onset, view_deg,
                hexes_are_xy_deg=hexes_are_xy_deg,
                bar_radius=bar_radius,
                multi_bar=bool(multi_bar),
            )
        return [title]

    anim = FuncAnimation(fig, update, frames=len(times), interval=80, blit=False)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    anim.save(output, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"wrote {output}  ({len(times)} frames, t={times[0]}..{times[-1]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", type=str, default=DEFAULT_NETWORK_RUN,
                    help=f"4_built_networks run folder name (default: {DEFAULT_NETWORK_RUN})")
    ap.add_argument("--output", type=str, default=None,
                    help="snapshot PNG (default: moving_bar_2<dir>_<side>)")
    ap.add_argument("--gif", nargs="?", const="", default=None,
                    help="write GIF; default path if flag alone, or pass a path")
    ap.add_argument("--t-stride", type=int, default=2,
                    help="GIF frame stride in t (default 2)")
    ap.add_argument("--t", type=str, default="",
                    help="comma-separated t indices for snapshot hexes, e.g. 50,60,72,90")
    ap.add_argument("--sti", type=str, default=_STI_CLI_CONTRASTS,
                    help=f"comma-separated moving-bar contrasts to plot: "
                         f"{_STI_CLI_CONTRASTS} (default: {_STI_CLI_CONTRASTS})")
    ap.add_argument("--direction", type=str, default=GRUNTMAN_DIRECTIONS[0],
                    choices=GRUNTMAN_DIRECTIONS)
    ap.add_argument(
        "--multi-bar",
        type=parse_bool,
        default=True,
        metavar="BOOL",
        help="tile simultaneous lane-clipped bars (default true); "
             "false → whole-view single bar over the full network view",
    )
    ap.add_argument("--bar-radius", type=int, default=DEFAULT_BAR_RADIUS,
                    help="per-lane spacing width in hex nodes (default 2)")
    ap.add_argument("--i-bright", type=float, default=NETWORK_CONSTRUCTION['i_bright'])
    args = ap.parse_args()
    snapshot_t = [int(tok) for tok in parse_comma_list(args.t)]
    sti = parse_comma_list(args.sti)
    if not sti:
        raise SystemExit(f"--sti must include at least one of {_STI_CLI_CONTRASTS}")
    bad_sti = sorted(set(sti) - set(GRUNTMAN_CONTRASTS))
    if bad_sti:
        raise SystemExit(f"--sti supports only {_STI_CLI_CONTRASTS}; got {bad_sti}")
    sti_set = set(sti)

    showcase = [
        s for s in gruntman_moving_bar_specs()
        if s.direction == args.direction and s.contrast in sti_set
    ]
    i_bright = args.i_bright

    network_json = str(resolve_network_json(args.network))
    connectome = load_network(
        network_json, device="cpu",
        a_syn_exc=NEURON_PARAM['a_syn_exc'], a_syn_inh=NEURON_PARAM['a_syn_inh'],
        syn_mode=NEURON_SCHEMA['syn_mode'], dtype=SIM_DTYPE,
    )
    tag = f"2{args.direction}_{network_run_tag(network_json, connectome.meta)}"
    default_png = os.path.join(PLOT_DIR, f"moving_bar_{tag}.png")
    default_gif = os.path.join(PLOT_DIR, f"moving_bar_{tag}.gif")
    output = args.output or default_png
    T = build_moving_bar_signals(
        connectome,
        specs=showcase,
        bar_radius=args.bar_radius,
        multi_bar=bool(args.multi_bar),
        delta_ms=NEURON_PARAM['delta_ms'],
        i_baseline=NETWORK_CONSTRUCTION['i_baseline'],
        i_bright_moving_bar=i_bright,
        sim_dtype=SIM_DTYPE,
    )
    plot_hexes = [(c.u, c.v) for c in sti_hexes(connectome)]
    i_sti_hex = T.i_sti_hex
    t_onset = int(T.info["t_onset"])
    n_t = int(T.info["n_t"])
    view_deg = tuple(T.info["view_deg"])
    i_baseline = float(T.info["i_baseline_moving_bar"])
    side = connectome.meta.get("side", "?")
    bar_radius = int(args.bar_radius)
    print(
        f"bar_radius={bar_radius}  "
        f"n_t={n_t} ({n_t * NEURON_PARAM['delta_ms'] / 1000.0:.2f} s)  "
        f"sweep_t={T.info['sweep_t']} ({T.info['sweep_time_s']:.2f} s after t_onset)"
    )

    save_snapshots(
        plot_hexes, showcase, i_sti_hex, i_bright, i_baseline,
        output, side, t_onset, n_t, view_deg, snapshot_t=snapshot_t,
        hexes_are_xy_deg=False,
        bar_radius=bar_radius,
        multi_bar=bool(args.multi_bar),
    )
    if args.gif is not None:
        gif = default_gif if args.gif == "" else args.gif
        save_animation(
            plot_hexes, showcase, i_sti_hex, i_bright, i_baseline, gif,
            side, t_onset, n_t, view_deg, args.t_stride,
            hexes_are_xy_deg=False,
            bar_radius=bar_radius,
            multi_bar=bool(args.multi_bar),
        )
    print(f"i_sti shape {tuple(T.i_sti.shape)}  specs={T.info['spec_names']}")


if __name__ == "__main__":
    main()
