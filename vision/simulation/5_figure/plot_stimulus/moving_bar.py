"""Visualise moving-bar column coverage (demo only).

Connectome: hex sti field from ``task.moving_bar.input``.

Usage (from simulation/, project .venv):

    ../.venv/bin/python 5_figure/plot_stimulus/moving_bar.py
    ../.venv/bin/python 5_figure/plot_stimulus/moving_bar.py --gif
    ../.venv/bin/python 5_figure/plot_stimulus/moving_bar.py --network right_min_neuron1_extent2 --direction down --gif
    ../.venv/bin/python 5_figure/plot_stimulus/moving_bar.py --network right_min_neuron1_extent2 --bar-extent 2
"""
from __future__ import annotations

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

from training.defaults import (
    I_BASELINE,
    I_BRIGHT,
    DELTA_MS, EXC_SYNWEIGHT, INH_SYNWEIGHT,
    SYN_MODE,
)
from training.target_pack import SIM_DTYPE
from network.construction import load_network
from training.driver import parse_bool
from connectome_io import parse_comma_list
from column_mapper import (
    FIELD_VIEW_PAD_DEG,
    draw_hex_patches,
    draw_hex_patches_uv,
    field_bounds_centers,
    set_axis_labels,
    uv_to_xy_deg,
)
from task.moving_bar.input import (
    DEFAULT_BAR_EXTENT,
    GRUNTMAN_CONTRASTS,
    GRUNTMAN_DIRECTIONS,
    bar_lane_rects_at_t,
    build_moving_bar_signals,
    field_bounds,
    gruntman_moving_bar_specs,
    moving_bar_transit_times,
    sti_columns,
)
from connectome_io import DEFAULT_NETWORK_RUN, network_run_tag, resolve_network_json

PLOT_BG = "#F5F0DC"  # axes background (beige), not column baseline color
_STI_CLI_DEFAULT = ",".join(GRUNTMAN_CONTRASTS)


def _output_tag(network_path: str, meta: dict, direction: str) -> str:
    """``2{direction}_{side}`` or ``2{direction}_{side}_extentN``."""
    return f"2{direction}_{network_run_tag(network_path, meta)}"


def _default_outputs(network_path: str, meta: dict, direction: str) -> tuple[str, str]:
    tag = _output_tag(network_path, meta, direction)
    return (
        os.path.join(PLOT_DIR, f"moving_bar_{tag}.png"),
        os.path.join(PLOT_DIR, f"moving_bar_{tag}.gif"),
    )





def _field_limits(columns, *, columns_are_xy_deg: bool = False):
    if columns_are_xy_deg:
        x_deg = [x for x, _ in columns]
        y_deg = [y for _, y in columns]
    else:
        x_deg, y_deg = uv_to_xy_deg(
            [u for u, _ in columns],
            [v for _, v in columns],
        )
    x0, y0, x1, y1 = field_bounds_centers(x_deg, y_deg)
    pad = FIELD_VIEW_PAD_DEG
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad


def _transit_frame_times(
    spec,
    field_deg,
    t_onset,
    n_t,
    t_stride: int,
    *,
    bar_extent: int,
    multi_bar: bool = True,
) -> list[int]:
    t0, _, t1 = moving_bar_transit_times(
        spec, field_deg, bar_extent, multi_bar=bool(multi_bar), t_onset=t_onset, n_t=n_t,
    )
    return list(range(t0, t1 + 1, max(1, t_stride)))


def _draw_bar_outline(ax, spec, field_deg, t: int, t_onset: int, *, bar_extent: int, multi_bar: bool = True):
    rects = bar_lane_rects_at_t(spec, field_deg, bar_extent, t, multi_bar=bool(multi_bar), t_onset=t_onset)
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


def _style_axes(ax):
    ax.set_facecolor(PLOT_BG)


def _val_to_color(val: float, cmap, i_max: float) -> tuple:
    t = float(np.clip(val / i_max if i_max > 0 else 0.0, 0.0, 1.0))
    return cmap(t)


def _draw_hex_field(ax, columns, vals, i_max, i_baseline, xlim, ylim, *, columns_are_xy_deg: bool = False):
    cmap = _current_cmap(i_max, i_baseline)
    colors = [_val_to_color(val, cmap, i_max) for val in vals]
    if columns_are_xy_deg:
        x_deg = [x for x, _ in columns]
        y_deg = [y for _, y in columns]
        draw_hex_patches(ax, x_deg, y_deg, colors, linewidth=0.15, alpha=0.95)
    else:
        u = [uv[0] for uv in columns]
        v = [uv[1] for uv in columns]
        draw_hex_patches_uv(ax, u, v, colors, linewidth=0.15, alpha=0.95)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    _style_axes(ax)
    set_axis_labels(ax, fontsize=8)


def plot_snapshot(
    ax, columns, column_current, t, spec, spec_name, i_max, i_baseline, xlim, ylim, t_onset, field_deg, *,
    columns_are_xy_deg: bool = False,
    bar_extent=None,
    multi_bar: bool = True,
):
    _draw_hex_field(
        ax, columns, column_current[t], i_max, i_baseline, xlim, ylim,
        columns_are_xy_deg=columns_are_xy_deg,
    )
    _draw_bar_outline(ax, spec, field_deg, t, t_onset, bar_extent=bar_extent, multi_bar=bool(multi_bar))
    ax.set_title(f"{spec_name}  t={t} ({t * DELTA_MS / 1000.0:.2f} s)", fontsize=9)


def write_snapshots(
    plot_columns,
    showcase,
    column_current,
    i_max,
    i_baseline,
    output,
    side,
    t_onset,
    n_t,
    field_deg,
    snapshot_t=None,
    columns_are_xy_deg: bool = False,
    bar_extent=None,
    multi_bar: bool = True,
):
    snapshot_t = list(snapshot_t or [])
    if snapshot_t:
        if any(t < 0 for t in snapshot_t):
            raise SystemExit("--t must be non-negative t indices")
        bad = [t for t in snapshot_t if t >= n_t]
        if bad:
            raise SystemExit(f"--t out of range (n_t={n_t}): {bad}")
    xlim = _field_limits(plot_columns, columns_are_xy_deg=columns_are_xy_deg)[:2]
    ylim = _field_limits(plot_columns, columns_are_xy_deg=columns_are_xy_deg)[2:]
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
            start_t, center_t, exit_t = moving_bar_transit_times(
                spec, field_deg, bar_extent, multi_bar=bool(multi_bar), t_onset=t_onset, n_t=n_t,
            )
            times = [start_t, center_t, exit_t]
            labels = ("start", "center", "exit")
        for j, (t, label) in enumerate(zip(times, labels)):
            plot_snapshot(
                axes[i, j], plot_columns, column_current[i], t, spec,
                f"{spec.name} ({label})", i_max, i_baseline, xlim, ylim, t_onset, field_deg,
                columns_are_xy_deg=columns_are_xy_deg,
                bar_extent=bar_extent,
                multi_bar=bool(multi_bar),
            )
        if len(times) >= 3 and not snapshot_t:
            spread = float(np.ptp(column_current[i, times[1]]))
            print(f"  {spec.name}: start/center/exit t={times}  center ptp={spread:.1f} pA")
        else:
            print(f"  {spec.name}: snapshot t={times}")

    fig.suptitle(
        f"Moving-bar column current (pA)  side={side}  "
        f"{len(plot_columns)} sti columns  I_baseline={i_baseline}  I_max={i_max}",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def write_animation(
    plot_columns, showcase, column_current, i_max, i_baseline, output, side, t_onset, n_t, field_deg, t_stride,
    *, columns_are_xy_deg: bool = False,
    bar_extent=None,
    multi_bar: bool = True,
):
    times = sorted({
        t for spec in showcase
        for t in _transit_frame_times(
            spec, field_deg, t_onset, n_t, t_stride, bar_extent=bar_extent, multi_bar=bool(multi_bar),
        )
    })
    if not times:
        print("no animation frames")
        return

    xlim = _field_limits(plot_columns, columns_are_xy_deg=columns_are_xy_deg)[:2]
    ylim = _field_limits(plot_columns, columns_are_xy_deg=columns_are_xy_deg)[2:]
    fig, axes = plt.subplots(len(showcase), 1, figsize=(4.5, 2.8 * len(showcase)), squeeze=False, facecolor=PLOT_BG)
    title = fig.suptitle("", fontsize=11)

    def update(frame_idx):
        t = times[frame_idx]
        title.set_text(
            f"Moving-bar column current (pA)  side={side}  "
            f"{len(plot_columns)} sti columns  I_baseline={i_baseline}  I_max={i_max}  t={t} ({t * DELTA_MS / 1000.0:.2f} s)"
        )
        for i, spec in enumerate(showcase):
            axes[i, 0].clear()
            plot_snapshot(
                axes[i, 0], plot_columns, column_current[i], t, spec,
                spec.name, i_max, i_baseline, xlim, ylim, t_onset, field_deg,
                columns_are_xy_deg=columns_are_xy_deg,
                bar_extent=bar_extent,
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
                    help=f"built_network run folder name (default: {DEFAULT_NETWORK_RUN})")
    ap.add_argument("-o", "--output", type=str, default=None,
                    help="snapshot PNG (default: moving_bar_2<dir>_<side>)")
    ap.add_argument("--gif", nargs="?", const="", default=None,
                    help="write GIF; default path if flag alone, or pass a path")
    ap.add_argument("--t-stride", type=int, default=2,
                    help="GIF frame stride in t (default 2)")
    ap.add_argument("--t", type=str, default="",
                    help="comma-separated t indices for snapshot columns, e.g. 50,60,72,90")
    ap.add_argument("--sti", type=str, default=_STI_CLI_DEFAULT,
                    help=f"comma-separated moving-bar contrasts to plot: "
                         f"{_STI_CLI_DEFAULT} (default: {_STI_CLI_DEFAULT})")
    ap.add_argument("--direction", type=str, default=GRUNTMAN_DIRECTIONS[0],
                    choices=GRUNTMAN_DIRECTIONS)
    ap.add_argument(
        "--multi-bar",
        type=parse_bool,
        default=True,
        metavar="BOOL",
        help="tile simultaneous lane-clipped bars (default true); "
             "false → whole-field single bar over the full network field",
    )
    ap.add_argument("--bar-extent", type=int, default=DEFAULT_BAR_EXTENT,
                    help="per-lane spacing width in hex-column units (default 2)")
    ap.add_argument("--i-bright", type=float, default=I_BRIGHT)
    args = ap.parse_args()
    snapshot_t = [int(tok) for tok in parse_comma_list(args.t)]
    sti = parse_comma_list(args.sti)
    if not sti:
        raise SystemExit(f"--sti must include at least one of {_STI_CLI_DEFAULT}")
    bad_sti = sorted(set(sti) - set(GRUNTMAN_CONTRASTS))
    if bad_sti:
        raise SystemExit(f"--sti supports only {_STI_CLI_DEFAULT}; got {bad_sti}")
    sti_set = set(sti)

    showcase = [
        s for s in gruntman_moving_bar_specs()
        if s.direction == args.direction and s.contrast in sti_set
    ]
    i_bright = args.i_bright

    network_json = str(resolve_network_json(args.network))
    C = load_network(
        network_json, device="cpu",
        exc_synweight=EXC_SYNWEIGHT, inh_synweight=INH_SYNWEIGHT,
        syn_mode=SYN_MODE, dtype=SIM_DTYPE,
    )
    default_png, default_gif = _default_outputs(network_json, C.meta, args.direction)
    output = args.output or default_png
    T = build_moving_bar_signals(
        C,
        specs=showcase,
        bar_extent=args.bar_extent,
        multi_bar=bool(args.multi_bar),
        delta_ms=DELTA_MS,
        i_baseline=I_BASELINE,
        i_bright_bar=i_bright,
        sim_dtype=SIM_DTYPE,
    )
    plot_columns = [(c.u, c.v) for c in sti_columns(C)]
    column_current = T.column_current
    t_onset = int(T.info["t_onset"])
    n_t = int(T.info["n_t"])
    field_deg = tuple(T.info["field_deg"])
    i_baseline = float(T.info["i_baseline"])
    side = C.meta.get("side", "?")
    columns_are_xy_deg = False
    bar_extent = int(args.bar_extent)
    info = T.info
    print(
        f"bar_extent={bar_extent}  "
        f"n_t={n_t} ({n_t * DELTA_MS / 1000.0:.2f} s)  "
        f"sweep_t={T.info['sweep_t']} ({T.info['sweep_time_s']:.2f} s after t_onset)"
    )

    write_snapshots(
        plot_columns, showcase, column_current, i_bright, i_baseline,
        output, side, t_onset, n_t, field_deg, snapshot_t=snapshot_t,
        columns_are_xy_deg=columns_are_xy_deg,
        bar_extent=bar_extent,
        multi_bar=bool(args.multi_bar),
    )
    if args.gif is not None:
        gif = default_gif if args.gif == "" else args.gif
        write_animation(
            plot_columns, showcase, column_current, i_bright, i_baseline, gif,
            side, t_onset, n_t, field_deg, args.t_stride,
            columns_are_xy_deg=columns_are_xy_deg,
            bar_extent=bar_extent,
            multi_bar=bool(args.multi_bar),
        )
    print(f"signal shape {tuple(T.signal.shape)}  specs={info['spec_names']}")


if __name__ == "__main__":
    main()
