"""Visualise moving-bar hex i_sti (demo only).

Connectome: hex sti from ``task.sbar.sti_geo``.

Usage (from simulation/, project .venv):

    ../.venv/bin/python 5_figure/plot_sti/mbar.py
    ../.venv/bin/python 5_figure/plot_sti/mbar.py mbar_plot_gif=true
    ../.venv/bin/python 5_figure/plot_sti/mbar.py mbar_plot_direction=down
"""
from __future__ import annotations

import os
import sys

import hydra

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PLOT_DIR = os.path.join(HERE, "plotted_mbar")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import import_bootstrap  # noqa: F401
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from config import (
    FIGURE_PLOT_STI_MBAR,
    MBAR_INPUT_SPEC,
    MODEL,
    SBAR_INPUT_GEO,
    NEURON_SCHEMA,
    NETWORK_PATH,
    TRAIN_CONFIG,
    resolve_config,
)
from train.param import SIM_DTYPE
from network.construction import load_network
from build_hex import (
    FIELD_VIEW_PAD_DEG,
    plot_hex_patches_uv,
    view_bounds_from_vertices,
    set_axis_labels,
    xy_deg_from_uv,
)
from task.mbar.sti_geo import bar_bound0_bar_bound1s, bar_bounds
from task.sbar.sti_geo import sti_hexes
from task.mbar.sti_spec import (
    build_mbar_signals,
    gruntman_mbar_specs,
    i_baseline_from_i_sti,
    mbar_transit_times,
)
from task.spread.sti_spec import CONTRASTS
from path import network_run_token, resolve_network_json

PLOT_BG = "#F5F0DC"  # axes background (beige), not hex baseline color


def _plot_bar_outline(ax, spec, view_deg, t: int, t_onset: int, *, bar_dist: int, multi_bar: bool = True):
    delta_ms = MODEL["delta_ms"]
    shift_deg = float(spec.speed_deg_over_s) * (float(delta_ms) / 1000.0)
    if spec.direction in ("left", "down"):
        shift_deg = -shift_deg
    elif spec.direction not in ("right", "up"):
        raise ValueError(f"unknown direction {spec.direction!r}")
    bar_w_deg = float(spec.bar_w_deg)
    for bar_bound0, bar_bound1 in bar_bound0_bar_bound1s(
        spec, view_deg, bar_dist, multi_bar=bool(multi_bar),
    ):
        if spec.direction in ("right", "up"):
            bar_bound = float(bar_bound0) - bar_w_deg
        else:
            bar_bound = float(bar_bound1) + bar_w_deg
        bar_bound = bar_bound + (t - t_onset) * shift_deg
        visible_bar = bar_bounds(spec, bar_bound, view_deg, bar_bound0, bar_bound1)
        if visible_bar is None:
            continue
        x0, y0, x1, y1 = visible_bar
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
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


def _plot_i_sti_hex(ax, hexes, i_sti, i_max, i_baseline, xlim, ylim):
    cmap = _current_cmap(i_max, i_baseline)
    colors = [
        cmap(float(np.clip(val / i_max if i_max > 0 else 0.0, 0.0, 1.0)))
        for val in i_sti
    ]
    u = [uv[0] for uv in hexes]
    v = [uv[1] for uv in hexes]
    plot_hex_patches_uv(ax, u, v, colors, linewidth=0.15, alpha=0.95)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_facecolor(PLOT_BG)
    set_axis_labels(ax, fontsize=8)


def _plot_mbar_sti_t(
    ax, hexes, i_sti_hex, t, spec, label, i_max, i_baseline, xlim, ylim, t_onset, view_deg, *,
    bar_dist=None,
    multi_bar: bool = True,
):
    _plot_i_sti_hex(
        ax, hexes, i_sti_hex[t], i_max, i_baseline, xlim, ylim,
    )
    _plot_bar_outline(ax, spec, view_deg, t, t_onset, bar_dist=bar_dist, multi_bar=bool(multi_bar))
    ax.set_title(f"{label}  t={t} ({t * MODEL['delta_ms'] / 1000.0:.2f} s)", fontsize=9)


def save_mbar_sti_png(
    figure_hexes,
    showcase,
    i_sti_hex,
    i_max,
    i_baseline,
    path,
    side,
    t_onset,
    n_t,
    view_deg,
    ts=None,
    bar_dist=None,
    multi_bar: bool = True,
):
    ts = list(ts or [])
    if ts:
        if any(t < 0 for t in ts):
            raise SystemExit("mbar_plot_t must be non-negative t idxs")
        bad = [t for t in ts if t >= n_t]
        if bad:
            raise SystemExit(f"mbar_plot_t out of range (n_t={n_t}): {bad}")
    x_deg, y_deg = xy_deg_from_uv(
        [u for u, _ in figure_hexes],
        [v for _, v in figure_hexes],
    )
    x0, y0, x1, y1 = view_bounds_from_vertices(x_deg, y_deg)
    pad = FIELD_VIEW_PAD_DEG
    xlim = (x0 - pad, x1 + pad)
    ylim = (y0 - pad, y1 + pad)
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    panel_h = max(2.4, 3.0 * yspan / max(xspan / 3.0, 1.0))
    n_col = len(ts) if ts else 3
    fig, axes = plt.subplots(
        len(showcase), n_col,
        figsize=(14.0, panel_h * len(showcase)),
        facecolor=PLOT_BG,
    )
    if len(showcase) == 1:
        axes = np.expand_dims(axes, 0)

    for row, spec in enumerate(showcase):
        if ts:
            times = ts
            labels = [f"{spec.token} (t={t})" for t in times]
        else:
            start_t, mid_t, exit_t = mbar_transit_times(
                spec, view_deg, bar_dist, multi_bar=bool(multi_bar), t_onset=t_onset, n_t=n_t,
            )
            times = [start_t, mid_t, exit_t]
            labels = [f"{spec.token} ({label})" for label in ("start", "mid", "exit")]
        for col, (t, label) in enumerate(zip(times, labels)):
            _plot_mbar_sti_t(
                axes[row, col], figure_hexes, i_sti_hex[row], t, spec,
                label, i_max, i_baseline, xlim, ylim, t_onset, view_deg,
                bar_dist=bar_dist,
                multi_bar=bool(multi_bar),
            )
        if len(times) >= 3 and not ts:
            spread = float(np.ptp(i_sti_hex[row, times[1]]))
            print(f"  {spec.token}: start/mid/exit t={times}  mid ptp={spread:.1f} pA")
        else:
            print(f"  {spec.token}: t={times}")

    fig.suptitle(
        f"Moving-bar i_sti_hex (pA)  side={side}  "
        f"{len(figure_hexes)} sti hexes  I_baseline={i_baseline}  I_max={i_max}",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def save_animation(
    figure_hexes, showcase, i_sti_hex, i_max, i_baseline, path, side, t_onset, n_t, view_deg, t_stride,
    *, bar_dist=None,
    multi_bar: bool = True,
):
    stride = max(1, t_stride)
    times = set()
    for spec in showcase:
        t0, _, t1 = mbar_transit_times(
            spec, view_deg, bar_dist, multi_bar=bool(multi_bar), t_onset=t_onset, n_t=n_t,
        )
        times.update(range(t0, t1 + 1, stride))
    times = sorted(times)
    if not times:
        print("no animation frames")
        return

    x_deg, y_deg = xy_deg_from_uv(
        [u for u, _ in figure_hexes],
        [v for _, v in figure_hexes],
    )
    x0, y0, x1, y1 = view_bounds_from_vertices(x_deg, y_deg)
    pad = FIELD_VIEW_PAD_DEG
    xlim = (x0 - pad, x1 + pad)
    ylim = (y0 - pad, y1 + pad)
    fig, axes = plt.subplots(len(showcase), 1, figsize=(4.5, 2.8 * len(showcase)), squeeze=False, facecolor=PLOT_BG)
    title = fig.suptitle("", fontsize=11)

    def update(frame):
        t = times[frame]
        title.set_text(
            f"Moving-bar i_sti_hex (pA)  side={side}  "
            f"{len(figure_hexes)} sti hexes  I_baseline={i_baseline}  I_max={i_max}  t={t} ({t * MODEL['delta_ms'] / 1000.0:.2f} s)"
        )
        for row, spec in enumerate(showcase):
            axes[row, 0].clear()
            _plot_mbar_sti_t(
                axes[row, 0], figure_hexes, i_sti_hex[row], t, spec,
                spec.token, i_max, i_baseline, xlim, ylim, t_onset, view_deg,
                bar_dist=bar_dist,
                multi_bar=bool(multi_bar),
            )
        return [title]

    anim = FuncAnimation(fig, update, frames=len(times), interval=80, blit=False)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"wrote {path}  ({len(times)} frames, t={times[0]}..{times[-1]})")


def plot_mbar_sti(
    *,
    network: str,
    direction: str,
    contrasts,
    path: str | None,
    gif: bool,
    gif_output: str | None,
    t_stride: int,
    ts,
    bar_dist: int,
    multi_bar: bool,
) -> None:
    direction = str(direction)
    bar_directions = MBAR_INPUT_SPEC["bar_directions"]
    if direction not in bar_directions:
        raise SystemExit(
            f"mbar_plot_direction must be one of {bar_directions}; got {direction!r}"
        )
    sti = list(contrasts)
    if not sti:
        raise SystemExit("contrasts must list at least one moving-bar contrast")
    bad_sti = sorted(set(sti) - set(CONTRASTS))
    if bad_sti:
        raise SystemExit(f"contrasts supports only {CONTRASTS}; got {bad_sti}")

    showcase = [
        spec for spec in gruntman_mbar_specs(
            contrasts=tuple(sti),
            bar_ws_deg=MBAR_INPUT_SPEC["bar_ws_deg"],
            bar_directions=bar_directions,
        )
        if spec.direction == direction
    ]
    i_sti_spec = TRAIN_CONFIG["i_sti"]
    i_baseline = i_baseline_from_i_sti(i_sti_spec)

    network_json = str(resolve_network_json(network))
    connectome = load_network(
        network_json, device="cpu",
        a_syn_exc=MODEL['a_syn_exc'], a_syn_inh=MODEL['a_syn_inh'],
        syn_mode=NEURON_SCHEMA['syn_mode'], dtype=SIM_DTYPE,
    )
    token = f"2{direction}_{network_run_token(network_json, connectome.meta)}"
    fallback_png = os.path.join(PLOT_DIR, f"mbar_{token}.png")
    fallback_gif = os.path.join(PLOT_DIR, f"mbar_{token}.gif")
    path = path or fallback_png
    bar_dist = int(bar_dist)
    i_sti_hex_parts = []
    specs = []
    T = None
    for contrast in sti:
        contrast_specs = [spec for spec in showcase if spec.contrast == contrast]
        if not contrast_specs:
            continue
        T = build_mbar_signals(
            connectome,
            specs=contrast_specs,
            bar_dist=bar_dist,
            multi_bar=multi_bar,
            delta_ms=MODEL['delta_ms'],
            i_baseline=i_baseline,
            i_sti=float(i_sti_spec[contrast]),
        )
        i_sti_hex_parts.append(T.i_sti_hex)
        specs.extend(contrast_specs)
    if not i_sti_hex_parts:
        raise SystemExit("no moving-bar specs to plot")
    i_sti_hex = np.concatenate(i_sti_hex_parts, axis=0)
    i_max = max(float(i_sti_spec[contrast]) for contrast in sti)
    figure_hexes = [(hex.u, hex.v) for hex in sti_hexes(connectome)]
    t_onset = int(T.t_onset)
    n_t = int(T.n_t)
    view_deg = tuple(T.view_deg)
    i_baseline = float(T.i_baseline)
    side = connectome.meta.get("side", "?")
    print(
        f"bar_dist={bar_dist}  "
        f"n_t={n_t} ({n_t * MODEL['delta_ms'] / 1000.0:.2f} s)  "
        f"sweep_t={T.sweep_t} ({T.sweep_s:.2f} s after t_onset)"
    )

    save_mbar_sti_png(
        figure_hexes, specs, i_sti_hex, i_max, i_baseline,
        path, side, t_onset, n_t, view_deg, ts=ts,
        bar_dist=bar_dist,
        multi_bar=multi_bar,
    )
    if gif:
        gif_path = gif_output or fallback_gif
        save_animation(
            figure_hexes, specs, i_sti_hex, i_max, i_baseline, gif_path,
            side, t_onset, n_t, view_deg, t_stride,
            bar_dist=bar_dist,
            multi_bar=multi_bar,
        )
    print(f"i_sti_hex shape {tuple(i_sti_hex.shape)}  specs={[spec.token for spec in specs]}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(hydra_config) -> None:
    resolve_config(hydra_config)
    plot_mbar_sti(
        network=str(NETWORK_PATH["network"]),
        direction=str(FIGURE_PLOT_STI_MBAR["direction"]),
        contrasts=TRAIN_CONFIG["contrasts"],
        path=FIGURE_PLOT_STI_MBAR.get("output"),
        gif=bool(FIGURE_PLOT_STI_MBAR["gif"]),
        gif_output=FIGURE_PLOT_STI_MBAR.get("gif_output"),
        t_stride=int(FIGURE_PLOT_STI_MBAR["t_stride"]),
        ts=FIGURE_PLOT_STI_MBAR.get("t"),
        bar_dist=int(SBAR_INPUT_GEO["bar_dist"]),
        multi_bar=bool(SBAR_INPUT_GEO["multi_bar"]),
    )


if __name__ == "__main__":
    main()
