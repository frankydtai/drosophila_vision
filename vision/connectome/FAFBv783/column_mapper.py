"""Hex-grid geometry for the 721-column FAFB construction.

This module owns all hex-lattice math so the rest of the pipeline never restates
coordinate formulas:

  - ``members_in_extent`` / ``get_hex_coords`` enumerate axial (u, v) on a hex
    disc.
  - ``pq_to_uv(p, q, side)`` converts FAFB ``column_assignment`` (p, q) indices to
    axial (u, v), which differs per hemisphere.
  - ``inside_mask(u, v, extent)`` is the shared inside/outside-the-disc predicate.
  - ``uv_to_xy`` / ``xy_to_uv`` convert axial ``(u, v)`` to hex-step ``(x, y)``;
    ``xy_to_xy_deg`` scales hex-step by :data:`DEG`; ``uv_to_xy_deg`` composes both.
  - ``hex_vertices`` / ``draw_hex_patches`` draw degree-space hex patches
    (shared by column maps, moving-bar stimulus, and plots).
  - :class:`HexGrid` holds an ideal disc's (u, v) coordinates (the plot reference
    panel); ``columns_with_uv(side)`` gives FAFB columns' (u, v).

Run a sanity summary with the project venv:

    .venv/bin/python "connectome/FAFBv783/column_mapper.py"
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import connectome_io
from connectome_io import COLUMN_MAP_DIR

logger = logging.getLogger(__name__)

# -- Single source of truth: grid size ----------------------------------------

# DEFAULT_EXTENT is the radius of the IDEAL reference hex disc (the left figure
# panel and the default for HexGrid). extent=10 -> 3*10*11+1 = 331 cells.
DEFAULT_EXTENT = 10

# EXTENT is the single shared spatial knob (crop in build_network; inside/outside
# colouring in the figure). < 0 (default) = no cap / no outside; >= 0 = that disc
# radius. build_network imports this so both scripts share ONE default.
EXTENT = -1

# FAFB inter-ommatidial angle: hex-step (x, y) -> degree via ``xy_to_xy_deg``.
DEG = 4.5
# Drawn hex patch radius in degrees (half the FAFB cell spacing).
HEX_PATCH_RADIUS = 0.5 * DEG
# Axis-limit padding (degrees) around column centres in field plots.
FIELD_VIEW_PAD_DEG = 2.0
# Single source of truth for the plot axis unit / labels.
AXIS_UNIT = "degree"
X_AXIS_LABEL = f"X ({AXIS_UNIT})"
Y_AXIS_LABEL = f"Y ({AXIS_UNIT})"
# Hex patch orientation (radians). Must match degree-space centres from
# ``uv_to_xy_deg(u,v)``: spaced ``DEG`` apart vertically (step (1,0)),
# r = DEG/2, so patches are pointy-top.
HEX_PATCH_ORIENTATION = np.radians(30)


# Rendered column map: base filename (no --extent) and the --extent variant.
COLUMN_MAP_FILE = "column_map.png"
COLUMN_MAP_EXTENT_FILE = "column_map_extent{extent}.png"

# Single source of truth for FAFB column colors (fill, edge), reused by every
# plot so column_map.png and lc_columns_right.png stay consistent.
INSIDE_COLOR: Tuple[str, str] = ("lightgreen", "darkgreen")
OUTSIDE_COLOR: Tuple[str, str] = ("lightcoral", "darkred")
EMPTY_COLOR: Tuple[str, str] = ("whitesmoke", "lightgrey")


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


def _hex_radius_arr(u, v) -> np.ndarray:
    """Vectorized hex-lattice distance from the origin to axial (u, v)."""
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    return (np.abs(u) + np.abs(v) + np.abs(u + v)) // 2


def hex_radius(u: int, v: int) -> int:
    """Hex-lattice distance from the origin to axial (u, v)."""
    return int(_hex_radius_arr(u, v))


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
    return _hex_radius_arr(u, v) <= int(extent)


# -- Hex disc members -----------------------------------------------------------

_HEX_DIRECTIONS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


def members_at_shell(shell: int) -> list:
    """Axial (u, v) members exactly ``shell`` hex steps from origin."""
    if shell < 0:
        raise ValueError(f"shell must be >= 0, got {shell}")
    if shell == 0:
        return [(0, 0)]
    out = []
    u, v = _HEX_DIRECTIONS[4][0] * shell, _HEX_DIRECTIONS[4][1] * shell
    for d in range(6):
        du, dv = _HEX_DIRECTIONS[d]
        for _ in range(shell):
            out.append((u, v))
            u, v = u + du, v + dv
    return out


def members_in_extent(extent) -> list:
    """All axial (u, v) members with :func:`hex_radius` <= ``extent``."""
    shell_max = int(math.floor(float(extent)))
    members: list = []
    for shell in range(shell_max + 1):
        members.extend(members_at_shell(shell))
    return members


def get_hex_coords(extent: int) -> Tuple[np.ndarray, np.ndarray]:
    """Axial (u, v) coordinates of a hex disc (shell order via :func:`members_in_extent`)."""
    members = members_in_extent(int(extent))
    u = np.array([m[0] for m in members], dtype=np.int64)
    v = np.array([m[1] for m in members], dtype=np.int64)
    return u, v


class HexGrid:
    """The (u, v) axial coordinates of an ideal hex disc of a given extent.

    A pure coordinate container (used as the reference disc for plotting and as
    panel); FAFB column (u, v) come from :func:`columns_with_uv`.
    """

    def __init__(self, extent: int = DEFAULT_EXTENT) -> None:
        self.extent = extent
        self.u, self.v = get_hex_coords(extent)
        self.n_columns = len(self.u)
        logger.info("HexGrid extent=%d -> %d columns", extent, self.n_columns)


def uv_to_xy(u, v) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """Axial ``(u, v)`` -> hex-step ``(x, y)`` with ``x = v``, ``y = u + v/2``."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return v, u + v / 2.0


def xy_to_uv(x, y) -> Tuple[int, int]:
    """Inverse of :func:`uv_to_xy` for hex centres (integer axial coords).

    Raises:
        ValueError: if ``(x, y)`` is not (within tolerance) a hex centre.
    """
    x = float(x)
    y = float(y)
    v = x
    u = y - v / 2.0
    iu, iv = round(u), round(v)
    if abs(u - iu) > 1e-6 or abs(v - iv) > 1e-6:
        raise ValueError(
            f"(x,y)=({x},{y}) -> (u,v)=({u},{v}) is not an integer hex centre"
        )
    return int(iu), int(iv)


def xy_to_xy_deg(
    x,
    y,
    deg: float = DEG,
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """Hex-step ``(x, y)`` -> degree ``(x_deg, y_deg)`` via ``deg * (x, y)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return deg * x, deg * y


def uv_to_xy_deg(
    u,
    v,
    deg: float = DEG,
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """Axial ``(u, v)`` -> degree ``(x_deg, y_deg)`` via :func:`uv_to_xy` + :func:`xy_to_xy_deg`."""
    return xy_to_xy_deg(*uv_to_xy(u, v), deg=deg)


def hex_vertices(
    cx: float,
    cy: float,
    radius: float = HEX_PATCH_RADIUS,
) -> np.ndarray:
    """Degree-space hex polygon vertices centred at ``(cx, cy)``.

    Matches ``matplotlib.patches.RegularPolygon`` with ``orientation`` =
    :data:`HEX_PATCH_ORIENTATION` (unit polygon includes matplotlib's ``pi/2``
    "points-up" offset, then ``scale(radius).rotate(orientation)``).
    """
    angles = (
        np.pi / 2
        + HEX_PATCH_ORIENTATION
        + (2.0 * np.pi / 6.0) * np.arange(6, dtype=np.float64)
    )
    vx = cx + radius * np.cos(angles)
    vy = cy + radius * np.sin(angles)
    return np.column_stack([vx, vy])


def field_bounds_centers(
    x,
    y,
    radius: float = HEX_PATCH_RADIUS,
) -> Tuple[float, float, float, float]:
    """Axis-aligned extent in degrees from hex patch vertices at ``(x, y)``."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    for cx, cy in zip(xs, ys):
        v = hex_vertices(float(cx), float(cy), radius)
        xmins.append(float(v[:, 0].min()))
        ymins.append(float(v[:, 1].min()))
        xmaxs.append(float(v[:, 0].max()))
        ymaxs.append(float(v[:, 1].max()))
    return min(xmins), min(ymins), max(xmaxs), max(ymaxs)


def draw_hex_patches(
    ax,
    x,
    y,
    facecolor,
    edgecolor: str = "0.35",
    hex_radius_px: Optional[float] = None,
    linewidth: float = 0.15,
    alpha: float = 0.95,
) -> None:
    """Draw hex patches at degree-space centres (same primitive as column_map)."""
    from matplotlib.patches import RegularPolygon

    if hex_radius_px is None:
        hex_radius_px = HEX_PATCH_RADIUS
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if np.ndim(facecolor) == 0 or (
        isinstance(facecolor, str) and not isinstance(facecolor, np.ndarray)
    ):
        facecolors = [facecolor] * len(xs)
    else:
        facecolors = list(facecolor)
    for xi, yi, fc in zip(xs, ys, facecolors):
        ax.add_patch(
            RegularPolygon(
                (float(xi), float(yi)),
                numVertices=6,
                radius=hex_radius_px,
                orientation=HEX_PATCH_ORIENTATION,
                facecolor=fc,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
            )
        )


def draw_hex_patches_uv(
    ax,
    u,
    v,
    facecolor,
    **kwargs,
) -> None:
    """Draw hex patches for axial ``(u, v)`` via :func:`uv_to_xy_deg`."""
    x_deg, y_deg = uv_to_xy_deg(u, v)
    draw_hex_patches(ax, x_deg, y_deg, facecolor, **kwargs)


def set_axis_labels(ax, fontsize: Optional[int] = None) -> None:
    """Apply the shared X/Y axis labels (single source of truth) to ``ax``."""
    kw = {} if fontsize is None else {"fontsize": fontsize}
    ax.set_xlabel(X_AXIS_LABEL, **kw)
    ax.set_ylabel(Y_AXIS_LABEL, **kw)


def _draw_hexes(ax, u, v, labels, facecolor, edgecolor, hex_radius, fontsize=3):
    """Draw labeled hexagons at the given axial coordinates."""
    xs, ys = uv_to_xy_deg(np.asarray(u), np.asarray(v))
    draw_hex_patches(
        ax, xs, ys, facecolor,
        edgecolor=edgecolor,
        hex_radius_px=hex_radius,
        linewidth=1,
        alpha=0.6,
    )
    if labels is None:
        return
    for x, y, label in zip(np.atleast_1d(xs), np.atleast_1d(ys), labels):
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

    ix, iy = uv_to_xy_deg(iu, iv)
    rx, ry = uv_to_xy_deg(df_right["u"].values, df_right["v"].values)
    lx, ly = uv_to_xy_deg(df_left["u"].values, df_left["v"].values)
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
    df = connectome_io.load_column_assignments()
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
    COLUMN_MAP_DIR.mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        cols = columns_with_uv(side)
        assigned[side] = cols
        out_csv = connectome_io.column_map_path(side)
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
        save_path=COLUMN_MAP_DIR / fname,
    )
    print(f"Column map written to: {COLUMN_MAP_DIR / fname}")


if __name__ == "__main__":
    main()
