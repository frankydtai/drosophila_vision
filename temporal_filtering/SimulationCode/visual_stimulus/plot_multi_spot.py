"""Visualise multi-spot centre tiling on a connectome column field.

Marks spot centres (crimson) on :func:`column_mapper.draw_fafb_columns` for
network columns only. Spot centres from :func:`network.spot_target.build_spotting`.

Usage (from SimulationCode/, project .venv):

    ../.venv/bin/python visual_stimulus/plot_multi_spot.py
    ../.venv/bin/python visual_stimulus/plot_multi_spot.py --spot-extents 0.5,1,1.5,2
    ../.venv/bin/python visual_stimulus/plot_multi_spot.py --network right_min_neuron1_extent2
    ../.venv/bin/python visual_stimulus/plot_multi_spot.py --fully-inside false
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLOT_DIR = os.path.join(HERE, "plotted_multi_spot")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import network_bootstrap  # noqa: F401
from column_mapper import (
    HEX_PATCH_RADIUS,
    draw_fafb_columns,
    field_bounds_centers,
    set_axis_labels,
    uv_to_xy_deg,
)
from connectome_io import resolve_network_json
from network.construction import Network, load_network
from network.spot_target import build_spotting, spot_dist, spot_extent_half_steps
from train import parse_bool, parse_comma_list
from visual_stimulus.plot_moving_bar_stimulus import _run_tag

DEFAULT_NETWORK = "right_min_neuron1"


def _default_output(network_path: str, meta: dict) -> str:
    return os.path.join(PLOT_DIR, f"plotted_multi_spot_{_run_tag(network_path, meta)}.png")


def _network_columns_df(C: Network) -> pd.DataFrame:
    """One row per unique ``(u, v)`` on connectome ``C``."""
    uv = sorted({(int(u), int(v)) for u, v in zip(C.u, C.v)})
    return pd.DataFrame({"column_id": -1, "u": [u for u, _ in uv], "v": [v for _, v in uv]})


def _panel_grid(n: int) -> tuple[int, int]:
    ncol = max(1, int(math.ceil(math.sqrt(n))))
    nrow = int(math.ceil(n / ncol))
    return nrow, ncol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        type=str,
        default=DEFAULT_NETWORK,
        help=f"built_network run folder name (default: {DEFAULT_NETWORK})",
    )
    parser.add_argument(
        "--spot-extents",
        default="0.5,1,1.5,2",
        metavar="E,...",
        help="comma-separated spot_extent values per panel, 0.5 multiples (default: 0.5,1,1.5,2)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="output PNG (default: plotted_multi_spot/plotted_multi_spot_<side>_extentN.png)",
    )
    parser.add_argument(
        "--fully-inside",
        type=parse_bool,
        default=True,
        metavar="BOOL",
        help="keep only centres whose spot footprint lies inside connectome extent (default: true)",
    )
    args = parser.parse_args()
    spot_extents = [float(x) for x in parse_comma_list(args.spot_extents)]
    if not spot_extents:
        raise SystemExit("--spot-extents must list at least one value")
    for spot_extent in spot_extents:
        try:
            spot_extent_half_steps(spot_extent)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    network_json = str(resolve_network_json(args.network))
    C = load_network(network_json, device="cpu")
    run_tag = _run_tag(network_json, C.meta)
    output = args.output or _default_output(network_json, C.meta)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    df_columns = _network_columns_df(C)
    x_deg, y_deg = uv_to_xy_deg(df_columns["u"].values, df_columns["v"].values)
    x0, y0, x1, y1 = field_bounds_centers(x_deg, y_deg)
    pad = 2.0
    xlim = (x0 - pad, x1 + pad)
    ylim = (y0 - pad, y1 + pad)

    nrow, ncol = _panel_grid(len(spot_extents))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 6.5 * nrow), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    counts = {}
    for ax, spot_extent in zip(axes_flat, spot_extents):
        draw_fafb_columns(ax, df_columns, hex_radius_px=HEX_PATCH_RADIUS, label=False)
        centers = build_spotting(
            C, spot_extent=spot_extent, fully_inside=args.fully_inside,
        ).centers
        counts[spot_extent] = len(centers)
        print(
            f"network={run_tag}  spot_extent={spot_extent}  "
            f"spot_dist={spot_dist(spot_extent)}  n_spots={counts[spot_extent]}",
        )
        if centers:
            cu = np.array([c[0] for c in centers], dtype=np.int64)
            cv = np.array([c[1] for c in centers], dtype=np.int64)
            sx, sy = uv_to_xy_deg(cu, cv)
            ax.plot(
                sx, sy, "o", color="crimson", markersize=5,
                markeredgecolor="black", markeredgewidth=0.4,
            )
        ax.set_title(
            f"spot_extent={spot_extent}  spot_dist={spot_dist(spot_extent)}  n={len(centers)}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_aspect("equal")
        set_axis_labels(ax)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    for ax in axes_flat[len(spot_extents):]:
        ax.set_visible(False)

    fig.suptitle(f"Spot centres vs spot_extent ({run_tag})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")
    print("counts:", counts)


if __name__ == "__main__":
    main()
