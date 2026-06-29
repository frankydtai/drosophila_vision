"""Visualise the extent-2 hexagon tilings on the FAFB right optic lobe.

The multi-column / shifted-tiling training covers the optic lobe with extent-2
hexagons (19 columns each), stimulates all tile centres at once and batches over
the 7 sub-tile shifts. There are two tiling layouts (see ``hex_grid.tile_basis``):

  - Disjoint (default): centres spaced 2k+1, gap-free, no overlap -> 31 tiles.
  - Edge-sharing: centres spaced 2k, neighbouring tiles share their boundary
    ring -> 43 tiles (the denser, overlapping cover).

This script renders both so the geometry can be eyeballed:

  - Left panel:   the 31 disjoint tiles, each filled with its own colour.
  - Middle panel: the 43 edge-sharing tiles drawn as outlines + centres
                  (cells are shared between tiles, so outlines read better).
  - Right panel:  a single tile with its 7 sub-tile shift positions.

It only draws geometry; all hex math is reused from :mod:`hex_grid`.

Run with the project venv:

    .venv/bin/python "Connectome/FAFB v783/tile_extent2_hexagons.py"
"""

from __future__ import annotations

import logging

import numpy as np

from fafb_io import DATA_DIR
from hex_grid import (
    DEFAULT_EXTENT,
    DEFAULT_KERNEL_SIZE,
    HexGrid,
    draw_fafb_columns,
    hex_to_pixel,
    shift_offsets,
    tile_centers,
    tile_offsets,
    unique_columns,
)

logger = logging.getLogger(__name__)

# Radius of each tile (extent-2 -> 19-cell hexagons).
TILE_EXTENT = 2
# Output image filename (written next to this script).
OUTPUT_FILE = "extent2_tiling.png"


def _draw_tile_cells(ax, cells, facecolor, edgecolor, hex_radius_px, alpha=0.55):
    """Fill the given axial cells with one colour (a single tile)."""
    from matplotlib.patches import RegularPolygon

    xs, ys = hex_to_pixel(
        np.array([c[0] for c in cells]), np.array([c[1] for c in cells])
    )
    for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
        ax.add_patch(
            RegularPolygon(
                (x, y), numVertices=6, radius=hex_radius_px,
                orientation=np.radians(30),
                facecolor=facecolor, edgecolor=edgecolor,
                linewidth=0.8, alpha=alpha,
            )
        )


def _axis_limits(grid: HexGrid, margin: float = 2.0):
    x, y = hex_to_pixel(grid.u, grid.v)
    return (x.min() - margin, x.max() + margin), (y.min() - margin, y.max() + margin)


def draw_tiling_panel(
    ax, grid: HexGrid, df_right, hex_radius_px: float, share_edges: bool
):
    """Draw every extent-2 tile coloured over the FAFB right columns.

    ``share_edges=False`` -> 31 disjoint tiles (opaque fill, no overlap).
    ``share_edges=True``  -> 43 edge-sharing tiles (translucent fill so the
    shared boundary cells read as blended/overlapping regions).
    """
    import matplotlib.pyplot as plt

    # Light FAFB background so the tiles read as the foreground structure.
    draw_fafb_columns(
        ax, df_right, hex_radius_px=hex_radius_px, label=False,
        inside_color=("whitesmoke", "lightgrey"),
        outside_color=("white", "lightgrey"),
    )

    centers = tile_centers(grid.extent, TILE_EXTENT, share_edges=share_edges)
    offsets = tile_offsets(TILE_EXTENT)
    cmap = plt.get_cmap("tab20")
    alpha = 0.35 if share_edges else 0.6
    for i, (cu, cv) in enumerate(centers):
        cells = [(cu + du, cv + dv) for du, dv in offsets]
        color = cmap(i % cmap.N)
        _draw_tile_cells(ax, cells, color, "black", hex_radius_px, alpha=alpha)
        cx, cy = hex_to_pixel(cu, cv)
        ax.plot(float(cx), float(cy), ".", color="black", markersize=4)
        ax.text(
            float(cx), float(cy), str(i), ha="center", va="center",
            fontsize=5.5, fontweight="bold", color="black",
        )
    layout = "edge-sharing (spacing 2k)" if share_edges else "disjoint (spacing 2k+1)"
    ax.set_title(
        f"{len(centers)} extent-{TILE_EXTENT} tiles - {layout}\n"
        f"{len(offsets)} columns each, over FAFB right (extent={grid.extent})",
        fontsize=11, fontweight="bold",
    )


def draw_shift_panel(ax, grid: HexGrid, df_right, hex_radius_px: float):
    """Right panel: the centre tile and its 7 sub-tile shift positions."""
    draw_fafb_columns(
        ax, df_right, hex_radius_px=hex_radius_px, label=False,
        inside_color=("whitesmoke", "lightgrey"),
        outside_color=("white", "lightgrey"),
    )
    offsets = tile_offsets(TILE_EXTENT)
    cells = [(du, dv) for du, dv in offsets]
    _draw_tile_cells(ax, cells, "lightskyblue", "navy", hex_radius_px, alpha=0.45)

    shifts = shift_offsets()
    for j, (su, sv) in enumerate(shifts):
        sx, sy = hex_to_pixel(su, sv)
        ax.plot(float(sx), float(sy), "o", color="crimson", markersize=8)
        ax.text(
            float(sx), float(sy), str(j), ha="center", va="center",
            fontsize=6, fontweight="bold", color="white",
        )
    ax.set_title(
        f"Centre tile + {len(shifts)} sub-tile shifts",
        fontsize=12, fontweight="bold",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = HexGrid(DEFAULT_EXTENT)
    df_right = grid.assign_columns(unique_columns("right"), "right")
    hex_radius_px = 0.5 * float(DEFAULT_KERNEL_SIZE)

    fig, axes = plt.subplots(1, 3, figsize=(30, 11), sharex=True, sharey=True)
    draw_tiling_panel(axes[0], grid, df_right, hex_radius_px, share_edges=False)
    draw_tiling_panel(axes[1], grid, df_right, hex_radius_px, share_edges=True)
    draw_shift_panel(axes[2], grid, df_right, hex_radius_px)

    xlim, ylim = _axis_limits(grid)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    axes[0].invert_yaxis()

    plt.tight_layout()
    out_path = DATA_DIR / OUTPUT_FILE
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    n_disjoint = len(tile_centers(grid.extent, TILE_EXTENT, share_edges=False))
    n_shared = len(tile_centers(grid.extent, TILE_EXTENT, share_edges=True))
    print(
        f"Wrote {out_path}  (disjoint={n_disjoint} tiles, "
        f"edge-sharing={n_shared} tiles, "
        f"{len(tile_offsets(TILE_EXTENT))} columns/tile, "
        f"{len(shift_offsets())} shifts)"
    )


if __name__ == "__main__":
    main()
