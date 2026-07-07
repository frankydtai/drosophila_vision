"""Visualise the extent-2 hexagon spottings on the FAFB right optic lobe.

The multi-column / shifted-spotting training covers the optic lobe with extent-2
hexagons (19 columns each), stimulates all spot centres at once and batches over
the 7 sub-spot shifts. There are two spotting layouts (see ``column_mapper.spot_basis``):

  - Disjoint (default): centres spaced 2k+1, gap-free, no overlap -> 31 spots.
  - Edge-sharing: centres spaced 2k, neighbouring spots share their boundary
    ring -> 43 spots (the denser, overlapping cover).

This script renders both so the geometry can be eyeballed:

  - Left panel:   the 31 disjoint spots, each filled with its own colour.
  - Middle panel: the 43 edge-sharing spots drawn as outlines + centres
                  (cells are shared between spots, so outlines read better).
  - Right panel:  a single spot with its 7 sub-spot shift positions.

It only draws geometry; all hex math is reused from :mod:`column_mapper`.

Run with the project venv:

    .venv/bin/python "Connectome/FAFBv783/spot_extent2_hexagons.py"
"""

from __future__ import annotations

import logging

import numpy as np

from connectome_io import DATA_DIR
from column_mapper import (
    DEFAULT_EXTENT,
    HEX_PATCH_RADIUS,
    HexGrid,
    columns_with_uv,
    draw_fafb_columns,
    draw_hex_patches,
    set_axis_labels,
    uv_to_xy_deg,
    shift_offsets,
    spot_centers,
    spot_offsets,
)

logger = logging.getLogger(__name__)

# Radius of each spot (extent-2 -> 19-cell hexagons).
SPOT_EXTENT = 2
# Output image filename (written next to this script).
OUTPUT_FILE = "extent2_spotting.png"


def _draw_spot_cells(ax, cells, facecolor, edgecolor, hex_radius_px, alpha=0.55):
    """Fill the given axial cells with one colour (a single spot)."""
    xs, ys = uv_to_xy_deg(
        np.array([c[0] for c in cells]), np.array([c[1] for c in cells]),
    )
    draw_hex_patches(
        ax, xs, ys, facecolor,
        edgecolor=edgecolor,
        hex_radius_px=hex_radius_px,
        linewidth=0.8,
        alpha=alpha,
    )


def _axis_limits(grid: HexGrid, margin: float = 2.0):
    x, y = uv_to_xy_deg(grid.u, grid.v)
    return (x.min() - margin, x.max() + margin), (y.min() - margin, y.max() + margin)


def draw_spotting_panel(
    ax, grid: HexGrid, df_right, hex_radius_px: float, share_edges: bool
):
    """Draw every extent-2 spot coloured over the FAFB right columns.

    ``share_edges=False`` -> 31 disjoint spots (opaque fill, no overlap).
    ``share_edges=True``  -> 43 edge-sharing spots (translucent fill so the
    shared boundary cells read as blended/overlapping regions).
    """
    import matplotlib.pyplot as plt

    # Light FAFB background so the spots read as the foreground structure.
    draw_fafb_columns(
        ax, df_right, extent=grid.extent, hex_radius_px=hex_radius_px, label=False,
        inside_color=("whitesmoke", "lightgrey"),
        outside_color=("white", "lightgrey"),
    )

    centers = spot_centers(grid.extent, SPOT_EXTENT, share_edges=share_edges)
    offsets = spot_offsets(SPOT_EXTENT)
    cmap = plt.get_cmap("tab20")
    alpha = 0.35 if share_edges else 0.6
    for i, (cu, cv) in enumerate(centers):
        cells = [(cu + du, cv + dv) for du, dv in offsets]
        color = cmap(i % cmap.N)
        _draw_spot_cells(ax, cells, color, "black", hex_radius_px, alpha=alpha)
        cx, cy = uv_to_xy_deg(cu, cv)
        ax.plot(float(cx), float(cy), ".", color="black", markersize=4)
        ax.text(
            float(cx), float(cy), str(i), ha="center", va="center",
            fontsize=5.5, fontweight="bold", color="black",
        )
    layout = "edge-sharing (spacing 2k)" if share_edges else "disjoint (spacing 2k+1)"
    ax.set_title(
        f"{len(centers)} extent-{SPOT_EXTENT} spots - {layout}\n"
        f"{len(offsets)} columns each, over FAFB right (extent={grid.extent})",
        fontsize=11, fontweight="bold",
    )


def draw_shift_panel(ax, grid: HexGrid, df_right, hex_radius_px: float):
    """Right panel: the centre spot and its 7 sub-spot shift positions."""
    draw_fafb_columns(
        ax, df_right, extent=grid.extent, hex_radius_px=hex_radius_px, label=False,
        inside_color=("whitesmoke", "lightgrey"),
        outside_color=("white", "lightgrey"),
    )
    offsets = spot_offsets(SPOT_EXTENT)
    cells = [(du, dv) for du, dv in offsets]
    _draw_spot_cells(ax, cells, "lightskyblue", "navy", hex_radius_px, alpha=0.45)

    shifts = shift_offsets()
    for j, (su, sv) in enumerate(shifts):
        sx, sy = uv_to_xy_deg(su, sv)
        ax.plot(float(sx), float(sy), "o", color="crimson", markersize=8)
        ax.text(
            float(sx), float(sy), str(j), ha="center", va="center",
            fontsize=6, fontweight="bold", color="white",
        )
    ax.set_title(
        f"Centre spot + {len(shifts)} sub-spot shifts",
        fontsize=12, fontweight="bold",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = HexGrid(DEFAULT_EXTENT)
    df_right = columns_with_uv("right")
    hex_radius_px = HEX_PATCH_RADIUS

    fig, axes = plt.subplots(1, 3, figsize=(30, 11), sharex=True, sharey=True)
    draw_spotting_panel(axes[0], grid, df_right, hex_radius_px, share_edges=False)
    draw_spotting_panel(axes[1], grid, df_right, hex_radius_px, share_edges=True)
    draw_shift_panel(axes[2], grid, df_right, hex_radius_px)

    xlim, ylim = _axis_limits(grid)
    for ax in axes:
        ax.set_aspect("equal")
        set_axis_labels(ax)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    plt.tight_layout()
    out_path = DATA_DIR / OUTPUT_FILE
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    n_disjoint = len(spot_centers(grid.extent, SPOT_EXTENT, share_edges=False))
    n_shared = len(spot_centers(grid.extent, SPOT_EXTENT, share_edges=True))
    print(
        f"Wrote {out_path}  (disjoint={n_disjoint} spots, "
        f"edge-sharing={n_shared} spots, "
        f"{len(spot_offsets(SPOT_EXTENT))} columns/spot, "
        f"{len(shift_offsets())} shifts)"
    )


if __name__ == "__main__":
    main()
