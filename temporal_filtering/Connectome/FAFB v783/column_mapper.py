"""Hex-grid geometry for the 721-column FAFB construction.

This module owns all hex-lattice math so the rest of the pipeline never restates
coordinate formulas:

  - ``get_hex_coords(extent)`` enumerates the (u, v) axial coordinates of a hex
    disc.
  - ``pq_to_uv(p, q, side)`` converts FAFB ``column_assignment`` (p, q) indices to
    axial (u, v), which differs per hemisphere.
  - ``inside_mask(u, v, extent)`` is the shared inside/outside-the-disc predicate.
  - :class:`HexGrid` holds an ideal disc's (u, v) coordinates (the plot reference
    / tiling extent); ``columns_with_uv(side)`` gives FAFB columns' (u, v).

Run a sanity summary with the project venv:

    .venv/bin/python "Connectome/FAFB v783/column_mapper.py"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import fafb_io
from fafb_io import COLUMN_HEX_DIR

logger = logging.getLogger(__name__)

# -- Single source of truth: grid size ----------------------------------------

# DEFAULT_EXTENT is the radius of the IDEAL reference hex disc (the left figure
# panel and the default for HexGrid / tiling). extent=10 -> 3*10*11+1 = 331 cells.
DEFAULT_EXTENT = 10

# EXTENT is the single shared spatial knob (crop in build_network; inside/outside
# colouring in the figure). < 0 (default) = no cap / no outside; >= 0 = that disc
# radius. build_network imports this so both scripts share ONE default.
EXTENT = -1

# Hex-cell spacing used when drawing the column map (degrees per cell;
# 4.5 deg is the fly inter-ommatidial angle).
DEFAULT_KERNEL_SIZE = 4.5
# Drawn hex patch radius in pixels (half the cell spacing). Single source so every
# hex plot (draw_fafb_columns, plot_column_map, lc_columns) shares the same size.
HEX_PATCH_RADIUS = 0.5 * DEFAULT_KERNEL_SIZE
# Single source of truth for the plot axis unit / labels (pixel spacing above is
# in degrees, so every hex map shares these). Use set_axis_labels(ax) to apply.
AXIS_UNIT = "degree"
X_AXIS_LABEL = f"X ({AXIS_UNIT})"
Y_AXIS_LABEL = f"Y ({AXIS_UNIT})"
# RegularPolygon orientation (radians) for pointy-top hexes.
_HEX_PATCH_ORIENTATION = np.radians(30)

# Rendered column map: base filename (no --extent) and the --extent variant.
COLUMN_MAP_FILE = "column_map.png"
COLUMN_MAP_EXTENT_FILE = "column_map_extent{extent}.png"

# Single source of truth for FAFB column colors (fill, edge), reused by every
# plot so column_hex_map.png and lc_columns_right.png stay consistent.
INSIDE_COLOR: Tuple[str, str] = ("lightgreen", "darkgreen")
OUTSIDE_COLOR: Tuple[str, str] = ("lightcoral", "darkred")
EMPTY_COLOR: Tuple[str, str] = ("whitesmoke", "lightgrey")


def get_hex_coords(extent: int) -> Tuple[np.ndarray, np.ndarray]:
    """Axial (u, v) coordinates of a hex disc, in canonical row-major order.

    Args:
        extent: Hex-disc radius (0 returns the single center coordinate).

    Returns:
        (u, v) integer arrays of length 3*extent*(extent+1)+1.
    """
    u, v = [], []
    for q in range(-extent, extent + 1):
        for r in range(max(-extent, -extent - q), min(extent, extent - q) + 1):
            u.append(q)
            v.append(r)
    return np.array(u, dtype=np.int64), np.array(v, dtype=np.int64)


def pq_to_uv(p, q, side: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert FAFB column (p, q) indices to axial (u, v) for one hemisphere.

    - left:  u = -q, v = q - p
    - right: u = -p, v = p - q
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    p = np.asarray(p, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64)
    if side == "left":
        return -q, q - p
    return -p, p - q


# -- Pure lattice math: distance, rings, tiles, shifts ------------------------
#
# These are coordinate-only helpers (no FAFB data, no plotting). They are the
# single source of truth for the hex math reused by the multi-column / tiling
# pipeline (connectome_tiling.py, connectome_target.py, tile_extent2_hexagons.py).

# The six unit step directions in axial (u, v), counter-clockwise.
_HEX_DIRECTIONS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


def _rot60(u: int, v: int) -> Tuple[int, int]:
    """Rotate an axial (u, v) offset 60 degrees counter-clockwise about origin."""
    return -v, u + v


def hex_radius(u: int, v: int) -> int:
    """Hex-lattice distance from the origin to axial (u, v)."""
    u, v = int(u), int(v)
    return (abs(u) + abs(v) + abs(u + v)) // 2


def inside_mask(u, v, extent: int) -> np.ndarray:
    """Boolean mask: is each axial (u, v) inside the radius-``extent`` hex disc?

    extent < 0 -> no cap, everything is inside (the shared default, EXTENT). This
    is the single source of truth for the inside/outside split used by
    draw_fafb_columns, build_network's crop and the LC-column plot.
    """
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    if extent < 0:
        return np.ones(u.shape, dtype=bool)
    return (np.abs(u) + np.abs(v) + np.abs(u + v)) // 2 <= extent


def ring_offsets(radius: int) -> list:
    """Axial (u, v) offsets of the cells exactly ``radius`` steps from origin."""
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if radius == 0:
        return [(0, 0)]
    out = []
    # Start ``radius`` steps along direction 4, then walk the six edges.
    u, v = _HEX_DIRECTIONS[4][0] * radius, _HEX_DIRECTIONS[4][1] * radius
    for d in range(6):
        du, dv = _HEX_DIRECTIONS[d]
        for _ in range(radius):
            out.append((u, v))
            u, v = u + du, v + dv
    return out


def tile_offsets(extent: int) -> list:
    """Axial (u, v) offsets of every cell in a hex disc of the given radius."""
    offs: list = []
    for r in range(extent + 1):
        offs.extend(ring_offsets(r))
    return offs


def shift_offsets() -> list:
    """The 7 sub-tile shifts: the tile centre plus its 6 nearest neighbours."""
    return tile_offsets(1)


def tile_basis(
    tile_extent: int, share_edges: bool = False
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Sublattice generators for a hex tiling by radius-``tile_extent`` hexes.

    Two layouts (``k = tile_extent``):

    - ``share_edges=False`` (default, disjoint): centres are spaced ``2k+1`` apart
      on the gap-free perfect-tiling sublattice spanned by ``(2k+1, -k)`` and its
      60-degree rotation. The squared norm equals the cell count, so tiles neither
      overlap nor leave gaps (31 tiles for extent=15, tile_extent=2).
    - ``share_edges=True`` (edge-sharing): centres are spaced ``2k`` apart along the
      edge-perpendicular directions, ``(2k, -k)`` and its rotation. Each tile then
      shares its boundary ring with its 6 neighbours, giving a denser, overlapping
      cover (43 tiles for extent=15, tile_extent=2).
    """
    first = 2 * tile_extent if share_edges else 2 * tile_extent + 1
    g1 = (first, -tile_extent)
    g2 = _rot60(*g1)
    return g1, g2


def tile_centers(
    extent: int = DEFAULT_EXTENT,
    tile_extent: int = 2,
    fully_inside: bool = True,
    share_edges: bool = False,
) -> list:
    """Axial centres of the radius-``tile_extent`` hexes covering an ``extent`` disc.

    Args:
        extent: Radius of the disc to cover (the optic-lobe grid).
        tile_extent: Radius of each tile (2 -> 19-cell extent-2 hexagons).
        fully_inside: If True (default) keep only tiles whose every cell lies
            inside the disc.
        share_edges: If False (default) use the disjoint gap-free tiling (31 tiles
            for extent=15, tile_extent=2); if True use the edge-sharing overlapping
            tiling (43 tiles), where neighbouring tiles share their boundary ring.

    Returns:
        Tile-centre (u, v) tuples, ordered by radius then angle.
    """
    (a1, b1), (a2, b2) = tile_basis(tile_extent, share_edges)
    members = tile_offsets(tile_extent)
    span = 2 * (extent // max(tile_extent, 1) + 2)
    centers = []
    for m in range(-span, span + 1):
        for n in range(-span, span + 1):
            cu, cv = m * a1 + n * a2, m * b1 + n * b2
            if hex_radius(cu, cv) > extent:
                continue
            if fully_inside and any(
                hex_radius(cu + du, cv + dv) > extent for du, dv in members
            ):
                continue
            centers.append((cu, cv))
    centers.sort(key=lambda c: (hex_radius(*c), _angle(*c)))
    return centers


def _angle(u: int, v: int) -> float:
    """Pixel-space angle of (u, v), for a stable angular tie-break ordering."""
    x, y = hex_to_pixel(u, v)
    return float(np.arctan2(float(y), float(x)))


class HexGrid:
    """The (u, v) axial coordinates of an ideal hex disc of a given extent.

    A pure coordinate container (used as the reference disc for plotting and as
    the extent for tiling). FAFB column (u, v) come from :func:`columns_with_uv`.
    """

    def __init__(self, extent: int = DEFAULT_EXTENT) -> None:
        self.extent = extent
        self.u, self.v = get_hex_coords(extent)
        self.n_columns = len(self.u)
        logger.info("HexGrid extent=%d -> %d columns", extent, self.n_columns)


def hex_to_pixel(u, v, kernel_size: float = DEFAULT_KERNEL_SIZE):
    """Axial (u, v) -> pixel (x, y) for plotting (x = d*v, y = d*(u + v/2))."""
    d = float(kernel_size)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return d * v, d * (u + v / 2.0)


def set_axis_labels(ax, fontsize: Optional[int] = None) -> None:
    """Apply the shared X/Y axis labels (single source of truth) to ``ax``."""
    kw = {} if fontsize is None else {"fontsize": fontsize}
    ax.set_xlabel(X_AXIS_LABEL, **kw)
    ax.set_ylabel(Y_AXIS_LABEL, **kw)


def _draw_hexes(ax, u, v, labels, facecolor, edgecolor, hex_radius, fontsize=3):
    """Draw labeled hexagons at the given axial coordinates."""
    from matplotlib.patches import RegularPolygon

    xs, ys = hex_to_pixel(np.asarray(u), np.asarray(v))
    for x, y, label in zip(np.atleast_1d(xs), np.atleast_1d(ys), labels):
        ax.add_patch(
            RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_radius,
                orientation=_HEX_PATCH_ORIENTATION,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1,
                alpha=0.6,
            )
        )
        if label is not None:
            ax.text(
                x, y, str(label), ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=edgecolor,
            )


def draw_fafb_columns(
    ax,
    df: pd.DataFrame,
    extent: Optional[int] = None,
    hex_radius_px: Optional[float] = None,
    label: bool = True,
    fontsize: int = 3,
    inside_color: Tuple[str, str] = INSIDE_COLOR,
    outside_color: Tuple[str, str] = OUTSIDE_COLOR,
) -> None:
    """Draw one hemisphere's FAFB columns, split inside/outside a hex disc.

    Reusable drawing primitive: ``df`` carries ``u``, ``v`` and ``column_id``. The
    inside/outside split is computed here from the shared ``inside_mask(u, v,
    extent)`` -- ``extent`` None or < 0 means every column is "inside" (one colour).
    """
    if hex_radius_px is None:
        hex_radius_px = HEX_PATCH_RADIUS
    mask = inside_mask(df["u"].values, df["v"].values,
                       -1 if extent is None else extent)
    inside = df[mask]
    outside = df[~mask]
    in_labels = (
        inside["column_id"].astype(int).tolist() if label else [None] * len(inside)
    )
    out_labels = (
        outside["column_id"].astype(int).tolist() if label else [None] * len(outside)
    )
    _draw_hexes(
        ax, inside["u"].values, inside["v"].values, in_labels,
        inside_color[0], inside_color[1], hex_radius_px, fontsize,
    )
    _draw_hexes(
        ax, outside["u"].values, outside["v"].values, out_labels,
        outside_color[0], outside_color[1], hex_radius_px, fontsize,
    )


def plot_column_map(
    ideal_grid: "HexGrid",
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    extent: Optional[int] = None,
    save_path: Optional[Path] = None,
    dpi: int = 400,
):
    """Render a 1x3 column map (left to right) and save it.

    Panels:
        left:   axial (u, v) reference for the ``ideal_grid`` hex disc
        middle: FAFB left columns
        right:  FAFB right columns

    ``extent`` only controls the FAFB panels: ``< 0`` (or ``None``) draws every
    column green (no inside/outside split); ``>= 0`` colours columns inside/outside
    that disc (computed from ``df``'s ``u``/``v`` via ``inside_mask``). The left
    reference panel always uses ``ideal_grid`` (fixed extent).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    classify = extent is not None and extent >= 0
    hex_radius_px = HEX_PATCH_RADIUS
    iu, iv = ideal_grid.u, ideal_grid.v

    ix, iy = hex_to_pixel(iu, iv)
    rx, ry = hex_to_pixel(df_right["u"].values, df_right["v"].values)
    lx, ly = hex_to_pixel(df_left["u"].values, df_left["v"].values)
    all_x = np.concatenate([ix, rx, lx])
    all_y = np.concatenate([iy, ry, ly])
    margin = 2
    xlim = (all_x.min() - margin, all_x.max() + margin)
    ylim = (all_y.min() - margin, all_y.max() + margin)

    fig, axes = plt.subplots(1, 3, figsize=(24, 9), sharex=True, sharey=True)

    _draw_hexes(
        axes[0], iu, iv,
        [f"({int(a)},{int(b)})" for a, b in zip(iu, iv)],
        "lightblue", "darkblue", hex_radius_px, fontsize=3.5,
    )
    axes[0].set_title(
        f"Axial (u, v) coordinates\n{ideal_grid.n_columns} cells, "
        f"extent={ideal_grid.extent}",
        fontsize=12, fontweight="bold",
    )

    def _draw_fafb(ax, df, side_label):
        draw_fafb_columns(ax, df, extent=extent, hex_radius_px=hex_radius_px)
        if classify:
            mask = inside_mask(df["u"].values, df["v"].values, extent)
            n_in, n_out = int(mask.sum()), int((~mask).sum())
            count_line = f"{n_in} inside + {n_out} outside = {len(df)} total"
        else:
            count_line = f"{len(df)} columns"
        ax.set_title(
            f"FAFB columns ({side_label})\n{count_line}\n(labeled with column_id)",
            fontsize=12, fontweight="bold",
        )

    _draw_fafb(axes[1], df_left, "left")
    _draw_fafb(axes[2], df_right, "right")

    for ax in axes:
        ax.set_aspect("equal")
        set_axis_labels(ax, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    axes[0].invert_yaxis()

    legend_elements = [
        Patch(facecolor="lightblue", edgecolor="darkblue", label="Ideal model / (u,v)"),
        Patch(facecolor=INSIDE_COLOR[0], edgecolor=INSIDE_COLOR[1],
              label="FAFB inside" if classify else "FAFB column"),
    ]
    if classify:
        legend_elements.append(
            Patch(facecolor=OUTSIDE_COLOR[0], edgecolor=OUTSIDE_COLOR[1],
                  label="FAFB outside")
        )
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=len(legend_elements),
        bbox_to_anchor=(0.5, 0.99), fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved column map to %s", save_path)
    plt.close(fig)
    return save_path


def unique_columns(side: str) -> pd.DataFrame:
    """One row per column_id (first p, q) for a hemisphere, from raw data."""
    df = fafb_io.load_column_assignments()
    if "hemisphere" in df.columns:
        df = df[df["hemisphere"] == side]
    return df.groupby("column_id", as_index=False).first()


def columns_with_uv(side: str) -> pd.DataFrame:
    """FAFB columns for a hemisphere as ``[column_id, p, q, u, v]``.

    (u, v) is pure ``pq_to_uv`` -- no grid/extent involved. This is the single
    source for the column<->(u, v) table (the column_map CSV and any FAFB panel).
    """
    df = unique_columns(side).copy()
    df["u"], df["v"] = pq_to_uv(df["p"].values, df["q"].values, side)
    return df[["column_id", "p", "q", "u", "v"]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the column<->hex tables and render the column map."
    )
    parser.add_argument(
        "--extent", type=int, default=EXTENT,
        help=f"Hex-disc radius for colouring the FAFB panels. <0 (default {EXTENT}) "
             "means no outside: every column is green and the figure is "
             "column_map.png. >=0 colours columns inside/outside that radius and "
             "saves column_map_extent<N>.png. (The left reference panel always uses "
             f"the ideal extent={DEFAULT_EXTENT}.)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    # The CSV needs no grid: (u, v) comes purely from pq_to_uv. --extent only
    # affects the figure colouring, never the tables.
    assigned = {}
    COLUMN_HEX_DIR.mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        cols = columns_with_uv(side)
        assigned[side] = cols
        out_csv = fafb_io.column_map_path(side)
        cols.to_csv(out_csv, index=False)
        print(f"{side:>5}: columns={len(cols)} -> {out_csv.name}")

    fname = (
        COLUMN_MAP_FILE if args.extent < 0
        else COLUMN_MAP_EXTENT_FILE.format(extent=args.extent)
    )
    plot_column_map(
        HexGrid(DEFAULT_EXTENT),
        df_left=assigned["left"],
        df_right=assigned["right"],
        extent=args.extent,
        save_path=COLUMN_HEX_DIR / fname,
    )
    print(f"Column map written to: {COLUMN_HEX_DIR / fname}")


if __name__ == "__main__":
    main()
