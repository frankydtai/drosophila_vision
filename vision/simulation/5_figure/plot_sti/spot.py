"""Visualise multi-spot center tiling on a connectome hex view.

Marks spot centers (crimson) and draws each spot's axial-radius hex
(straight edges through ``(spot_radius + 0.5) * _HEX_DIRECTIONS``, via
``xy_deg_from_uv`` — not a Euclidean RegularPolygon) on
:func:`build_hex.plot_fafb_columns` for network hexes only.
Spot centers from :func:`task.spot.sti_geo.build_spot`.

Usage (from simulation/, project .venv):

    ../.venv/bin/python 5_figure/plot_sti/spot.py
    ../.venv/bin/python 5_figure/plot_sti/spot.py --spot-radii 0.5,1,1.5,2
    ../.venv/bin/python 5_figure/plot_sti/spot.py --network right_min_neuron1_r2
    ../.venv/bin/python 5_figure/plot_sti/spot.py --fully-inside false
    ../.venv/bin/python 5_figure/plot_sti/spot.py --multi-spot false
"""
from __future__ import annotations

from default_params import (
    NEURON_PARAM,
    NEURON_SCHEMA,
)

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PLOT_DIR = os.path.join(HERE, "plotted_multi_spot")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import import_bootstrap  # noqa: F401
import network.path  # noqa: F401
from build_hex import (
    FIELD_VIEW_PAD_DEG,
    HEX_PATCH_RADIUS,
    _HEX_DIRECTIONS,
    plot_fafb_columns,
    view_bounds_from_vertices,
    set_axis_labels,
    xy_deg_from_uv,
)
from path import DEFAULT_NETWORK_RUN, network_run_tag, resolve_network_json
from network.construction import Network, load_network
from train.param import SIM_DTYPE
from task.spot.sti_geo import (
    build_spot,
    spot_radius_dist,
    spot_radius_half_steps,
)
from import_bootstrap import parse_comma_list
from train.cli import add_multi_spot_arguments


def _network_hexes_df(connectome: Network) -> pd.DataFrame:
    """One unique ``(u, v)`` hex per connectome position."""
    uv = sorted({(int(u), int(v)) for u, v in zip(connectome.us, connectome.vs)})
    return pd.DataFrame({"column_id": -1, "u": [u for u, _ in uv], "v": [v for _, v in uv]})


def _draw_spot_radius_hexes(ax, centers_u, centers_v, spot_radius: float) -> None:
    """Straight axial-radius hex about each center (vertices along ``_HEX_DIRECTIONS``).

    Vertex axial distance is ``spot_radius + 0.5`` (= ``spot_radius_dist/2``): outer
    boundary of the footprint / halfway to neighboring spot centers.
    """
    e = float(spot_radius) + 0.5
    du = np.array([d[0] for d in _HEX_DIRECTIONS], dtype=float)
    dv = np.array([d[1] for d in _HEX_DIRECTIONS], dtype=float)
    for cu, cv in zip(np.atleast_1d(centers_u), np.atleast_1d(centers_v)):
        xs, ys = xy_deg_from_uv(float(cu) + e * du, float(cv) + e * dv)
        ax.add_patch(
            Polygon(
                np.column_stack([xs, ys]),
                closed=True,
                fill=False,
                edgecolor="crimson",
                linewidth=1.0,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        type=str,
        default=DEFAULT_NETWORK_RUN,
        help=f"4_built_networks run folder name (default: {DEFAULT_NETWORK_RUN})",
    )
    parser.add_argument(
        "--spot-radii",
        default="0.5,1,1.5,2",
        metavar="E,...",
        help="comma-separated spot_radius values per panel, 0.5 multiples "
             "(default: 0.5,1,1.5,2)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output PNG (default: plotted_multi_spot/plotted_multi_spot_<side>_rN.png)",
    )
    add_multi_spot_arguments(parser)
    args = parser.parse_args()
    spot_radii = [float(x) for x in parse_comma_list(args.spot_radii)]
    if not spot_radii:
        raise SystemExit("--spot-radii must list at least one value")
    for spot_radius in spot_radii:
        try:
            spot_radius_half_steps(spot_radius)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    network_json = str(resolve_network_json(args.network))
    connectome = load_network(
        network_json, device="cpu",
        a_syn_exc=NEURON_PARAM['a_syn_exc'], a_syn_inh=NEURON_PARAM['a_syn_inh'],
        syn_mode=NEURON_SCHEMA['syn_mode'], dtype=SIM_DTYPE,
    )
    run_tag = network_run_tag(network_json, connectome.meta)
    output = args.output or os.path.join(
        PLOT_DIR, f"plotted_multi_spot_{run_tag}.png",
    )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    df_hexes = _network_hexes_df(connectome)
    x_deg, y_deg = xy_deg_from_uv(df_hexes["u"].values, df_hexes["v"].values)
    x0, y0, x1, y1 = view_bounds_from_vertices(x_deg, y_deg)
    pad = FIELD_VIEW_PAD_DEG
    xlim = (x0 - pad, x1 + pad)
    ylim = (y0 - pad, y1 + pad)

    n = len(spot_radii)
    ncol = max(1, int(math.ceil(math.sqrt(n))))
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 6.5 * nrow), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    n_by_spot_radius = {}
    for ax, spot_radius in zip(axes_flat, spot_radii):
        plot_fafb_columns(ax, df_hexes, hex_radius_px=HEX_PATCH_RADIUS, label=False)
        centers = build_spot(
            connectome,
            spot_radius=spot_radius,
            multi_spot=args.multi_spot,
            fully_inside=args.fully_inside,
        ).centers
        n_spots = len(centers)
        n_by_spot_radius[spot_radius] = n_spots
        dist = spot_radius_dist(spot_radius)
        print(
            f"network={run_tag}  spot_radius={spot_radius}  "
            f"spot_radius_dist={dist}  n_spots={n_spots}",
        )
        if centers:
            cu = np.array([center[0] for center in centers], dtype=np.int64)
            cv = np.array([center[1] for center in centers], dtype=np.int64)
            sx, sy = xy_deg_from_uv(cu, cv)
            _draw_spot_radius_hexes(ax, cu, cv, spot_radius)
            ax.plot(
                sx, sy, "o", color="crimson", markersize=5,
                markeredgecolor="black", markeredgewidth=0.4,
            )
        ax.set_title(
            f"spot_radius={spot_radius}  spot_radius_dist={dist}  n={n_spots}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_aspect("equal")
        set_axis_labels(ax)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    for ax in axes_flat[len(spot_radii):]:
        ax.set_visible(False)

    fig.suptitle(f"Spot centers vs spot_radius ({run_tag})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")
    print("n_by_spot_radius:", n_by_spot_radius)


if __name__ == "__main__":
    main()
