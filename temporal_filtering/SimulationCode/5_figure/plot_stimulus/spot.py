"""Visualise multi-spot center tiling on a connectome column field.

Marks spot centers (crimson) and draws each spot's axial-extent hex
(straight edges through ``(spot_extent + 0.5) * _HEX_DIRECTIONS``, via
``uv_to_xy_deg`` — not a Euclidean RegularPolygon) on
:func:`column_mapper.draw_fafb_columns` for network columns only.
Spot centers from :func:`task.spot.input.build_spot`.

Usage (from SimulationCode/, project .venv):

    ../.venv/bin/python 5_figure/plot_stimulus/2_plot_multi_spot.py
    ../.venv/bin/python 5_figure/plot_stimulus/2_plot_multi_spot.py --spot-extents 0.5,1,1.5,2
    ../.venv/bin/python 5_figure/plot_stimulus/2_plot_multi_spot.py --network right_min_neuron1_extent2
    ../.venv/bin/python 5_figure/plot_stimulus/2_plot_multi_spot.py --fully-inside false
    ../.venv/bin/python 5_figure/plot_stimulus/2_plot_multi_spot.py --multi-spot false
"""
from __future__ import annotations

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
from column_mapper import (
    FIELD_VIEW_PAD_DEG,
    HEX_PATCH_RADIUS,
    _HEX_DIRECTIONS,
    draw_fafb_columns,
    field_bounds_centers,
    set_axis_labels,
    uv_to_xy_deg,
)
from connectome_io import DEFAULT_NETWORK_RUN, network_run_tag, resolve_network_json
from network.construction import Network, load_network
from training.defaults import (
    PHYSICS,
    SPOT_EXTENTS,
    SYN_MODE,
)
from training.target_pack import SIM_DTYPE
from task.spot.input import (
    build_spot,
    spot_dist,
    spot_extent_half_steps,
)
from connectome_io import parse_comma_list
from training.driver import add_spot_layout_arguments

_SPOT_EXTENTS_CLI_DEFAULT = ",".join(
    str(int(x)) if float(x) == int(x) else str(x) for x in SPOT_EXTENTS
)


def _default_output(network_path: str, meta: dict) -> str:
    return os.path.join(PLOT_DIR, f"plotted_multi_spot_{network_run_tag(network_path, meta)}.png")


def _network_columns_df(C: Network) -> pd.DataFrame:
    """One row per unique ``(u, v)`` on connectome ``C``."""
    uv = sorted({(int(u), int(v)) for u, v in zip(C.u, C.v)})
    return pd.DataFrame({"column_id": -1, "u": [u for u, _ in uv], "v": [v for _, v in uv]})


def _panel_grid(n: int) -> tuple[int, int]:
    ncol = max(1, int(math.ceil(math.sqrt(n))))
    nrow = int(math.ceil(n / ncol))
    return nrow, ncol


def _draw_spot_extent_hexes(ax, centers_u, centers_v, spot_extent: float) -> None:
    """Straight axial-extent hex about each center (vertices along ``_HEX_DIRECTIONS``).

    Vertex axial distance is ``spot_extent + 0.5`` (= ``spot_dist/2``): outer
    boundary of the footprint / halfway to neighboring spot centers.
    """
    e = float(spot_extent) + 0.5
    du = np.array([d[0] for d in _HEX_DIRECTIONS], dtype=float)
    dv = np.array([d[1] for d in _HEX_DIRECTIONS], dtype=float)
    for cu, cv in zip(np.atleast_1d(centers_u), np.atleast_1d(centers_v)):
        xs, ys = uv_to_xy_deg(float(cu) + e * du, float(cv) + e * dv)
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
        help=f"built_network run folder name (default: {DEFAULT_NETWORK_RUN})",
    )
    parser.add_argument(
        "--spot-extents",
        default=_SPOT_EXTENTS_CLI_DEFAULT,
        metavar="E,...",
        help=f"comma-separated spot_extent values per panel, 0.5 multiples "
             f"(default: {_SPOT_EXTENTS_CLI_DEFAULT})",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="output PNG (default: plotted_multi_spot/plotted_multi_spot_<side>_extentN.png)",
    )
    add_spot_layout_arguments(parser)
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
    C = load_network(
        network_json, device="cpu",
        exc_synweight=PHYSICS.exc_synweight, inh_synweight=PHYSICS.inh_synweight,
        syn_mode=SYN_MODE, dtype=SIM_DTYPE,
    )
    run_tag = network_run_tag(network_json, C.meta)
    output = args.output or _default_output(network_json, C.meta)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    df_columns = _network_columns_df(C)
    x_deg, y_deg = uv_to_xy_deg(df_columns["u"].values, df_columns["v"].values)
    x0, y0, x1, y1 = field_bounds_centers(x_deg, y_deg)
    pad = FIELD_VIEW_PAD_DEG
    xlim = (x0 - pad, x1 + pad)
    ylim = (y0 - pad, y1 + pad)

    nrow, ncol = _panel_grid(len(spot_extents))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 6.5 * nrow), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    counts = {}
    for ax, spot_extent in zip(axes_flat, spot_extents):
        draw_fafb_columns(ax, df_columns, hex_radius_px=HEX_PATCH_RADIUS, label=False)
        centers = build_spot(
            C,
            spot_extent=spot_extent,
            multi_spot=args.multi_spot,
            fully_inside=args.fully_inside,
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
            _draw_spot_extent_hexes(ax, cu, cv, spot_extent)
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

    fig.suptitle(f"Spot centers vs spot_extent ({run_tag})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")
    print("counts:", counts)


if __name__ == "__main__":
    main()
