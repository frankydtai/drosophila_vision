"""Hex-grid geometry for the 721-column FAFB construction.

This module owns all hex-lattice math so the rest of the pipeline never restates
coordinate formulas:

  - ``get_hex_coords(extent)`` enumerates the (u, v) axial coordinates of a hex
    disc; a coordinate's position in this list is its ``hex_index`` (the same
    ordering Flyvis uses, so indices stay compatible).
  - ``pq_to_uv(p, q, side)`` converts FAFB ``column_assignment`` (p, q) indices to
    axial (u, v), which differs per hemisphere.
  - :class:`HexGrid` builds the (u, v) -> hex_index lookup for a given extent and
    maps FAFB columns onto it, flagging whether each lands inside the disc.

Run a sanity summary with the project venv:

    .venv/bin/python "Connectome/FAFB v783/hex_grid.py"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import fafb_io
from fafb_io import DATA_DIR

logger = logging.getLogger(__name__)

# -- Single source of truth: grid size ----------------------------------------

# extent is the hex-disc radius. extent=15 gives 3*15*16+1 = 721 columns, the
# size of the reference optic-lobe model.
DEFAULT_EXTENT = 15

# Hex-cell spacing used when drawing the column map (pixels per cell).
DEFAULT_KERNEL_SIZE = 13
# RegularPolygon orientation (radians) for pointy-top hexes.
_HEX_PATCH_ORIENTATION = np.radians(30)

# Default filename for the rendered column map.
COLUMN_MAP_FILE = "column_hex_map.png"
# Per-side column -> hex_index table filename pattern.
COLUMN_INDEX_FILE = "column_id_hex_index_{side}.csv"


def get_num_hexals(extent: int) -> int:
    """Number of columns in a hex disc of the given radius."""
    return 3 * extent * (extent + 1) + 1


def get_hex_coords(extent: int) -> Tuple[np.ndarray, np.ndarray]:
    """Axial (u, v) coordinates of a hex disc, in canonical hex_index order.

    Args:
        extent: Hex-disc radius (0 returns the single center coordinate).

    Returns:
        (u, v) integer arrays of length ``get_num_hexals(extent)``.
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


class HexGrid:
    """A hex disc of a given extent with a (u, v) -> hex_index lookup."""

    def __init__(self, extent: int = DEFAULT_EXTENT) -> None:
        self.extent = extent
        self.u, self.v = get_hex_coords(extent)
        self.n_columns = len(self.u)
        self._uv_to_index = {
            (int(self.u[i]), int(self.v[i])): i for i in range(self.n_columns)
        }
        logger.info("HexGrid extent=%d -> %d columns", extent, self.n_columns)

    def uv_to_hex_index(self, u: int, v: int) -> Optional[int]:
        """Return the hex_index for (u, v), or None if outside the disc."""
        return self._uv_to_index.get((int(u), int(v)))

    def assign_columns(self, columns: pd.DataFrame, side: str) -> pd.DataFrame:
        """Map FAFB columns onto the grid.

        Args:
            columns: One row per column carrying integer ``p`` and ``q``.
            side: 'left' or 'right'.

        Returns:
            A copy with added ``u``, ``v``, ``hex_index`` (Int64, NA if outside)
            and ``hex_status`` ('inside' / 'outside').
        """
        out = columns.copy()
        u, v = pq_to_uv(out["p"].values, out["q"].values, side)
        out["u"] = u
        out["v"] = v
        hex_index = [self._uv_to_index.get((int(a), int(b))) for a, b in zip(u, v)]
        out["hex_index"] = pd.array(hex_index, dtype="Int64")
        out["hex_status"] = np.where(out["hex_index"].isna(), "outside", "inside")
        return out


def hex_to_pixel(u, v, kernel_size: float = DEFAULT_KERNEL_SIZE):
    """Axial (u, v) -> pixel (x, y) for plotting (x = d*v, y = d*(u + v/2))."""
    d = float(kernel_size)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return d * v, d * (u + v / 2.0)


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


def plot_column_map(
    grid: "HexGrid",
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    save_path: Optional[Path] = None,
    dpi: int = 400,
):
    """Render a 2x2 column map and save it.

    Panels:
        top-left:     axial (u, v) reference for the hex disc
        top-right:    the ideal hex model, labeled with hex_index
        bottom-left:  FAFB right columns (inside green / outside red)
        bottom-right: FAFB left columns
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    hex_radius = 0.5 * float(DEFAULT_KERNEL_SIZE)
    iu, iv = grid.u, grid.v

    ix, iy = hex_to_pixel(iu, iv)
    rx, ry = hex_to_pixel(df_right["u"].values, df_right["v"].values)
    lx, ly = hex_to_pixel(df_left["u"].values, df_left["v"].values)
    all_x = np.concatenate([ix, rx, lx])
    all_y = np.concatenate([iy, ry, ly])
    margin = 2
    xlim = (all_x.min() - margin, all_x.max() + margin)
    ylim = (all_y.min() - margin, all_y.max() + margin)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)

    _draw_hexes(
        axes[0, 0], iu, iv,
        [f"({int(a)},{int(b)})" for a, b in zip(iu, iv)],
        "lightblue", "darkblue", hex_radius, fontsize=3.5,
    )
    axes[0, 0].set_title(
        f"Axial (u, v) coordinates\n{grid.n_columns} cells, extent={grid.extent}",
        fontsize=12, fontweight="bold",
    )

    _draw_hexes(
        axes[0, 1], iu, iv, list(range(grid.n_columns)),
        "lightblue", "darkblue", hex_radius,
    )
    axes[0, 1].set_title(
        f"Ideal hex model\n{grid.n_columns} columns, extent={grid.extent} "
        f"(labeled with hex_index)",
        fontsize=12, fontweight="bold",
    )

    def _draw_fafb(ax, df, side_label):
        inside = df[df["hex_status"] == "inside"]
        outside = df[df["hex_status"] == "outside"]
        _draw_hexes(
            ax, inside["u"].values, inside["v"].values,
            inside["column_id"].astype(int).tolist(),
            "lightgreen", "darkgreen", hex_radius,
        )
        _draw_hexes(
            ax, outside["u"].values, outside["v"].values,
            outside["column_id"].astype(int).tolist(),
            "lightcoral", "darkred", hex_radius,
        )
        ax.set_title(
            f"FAFB columns ({side_label})\n"
            f"{len(inside)} inside + {len(outside)} outside = {len(df)} total\n"
            f"(labeled with column_id)",
            fontsize=12, fontweight="bold",
        )

    _draw_fafb(axes[1, 0], df_right, "right")
    _draw_fafb(axes[1, 1], df_left, "left")

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xlabel("X (pixel)", fontsize=11)
        ax.set_ylabel("Y (pixel)", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    axes[0, 0].invert_yaxis()

    legend_elements = [
        Patch(facecolor="lightblue", edgecolor="darkblue", label="Ideal model / (u,v)"),
        Patch(facecolor="lightgreen", edgecolor="darkgreen", label="FAFB inside"),
        Patch(facecolor="lightcoral", edgecolor="darkred", label="FAFB outside"),
    ]
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 0.99), fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved column map to %s", save_path)
    plt.close(fig)
    return save_path


def _unique_columns(side: str) -> pd.DataFrame:
    """One row per column_id (first p, q) for a hemisphere, from raw data."""
    df = fafb_io.load_column_assignments()
    if "hemisphere" in df.columns:
        df = df[df["hemisphere"] == side]
    return df.groupby("column_id", as_index=False).first()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    grid = HexGrid()
    print(f"extent={grid.extent}, grid columns={grid.n_columns}")
    assigned = {}
    for side in ("left", "right"):
        cols = grid.assign_columns(_unique_columns(side), side)
        assigned[side] = cols
        n_inside = int((cols["hex_status"] == "inside").sum())
        n_outside = int((cols["hex_status"] == "outside").sum())
        n_filled = cols.loc[cols["hex_status"] == "inside", "hex_index"].nunique()
        out_csv = DATA_DIR / COLUMN_INDEX_FILE.format(side=side)
        cols[["column_id", "p", "q", "u", "v", "hex_index", "hex_status"]].to_csv(
            out_csv, index=False
        )
        print(
            f"{side:>5}: columns={len(cols)} inside={n_inside} "
            f"outside={n_outside} grid_cells_filled={n_filled}/{grid.n_columns} "
            f"-> {out_csv.name}"
        )

    plot_column_map(
        grid,
        df_left=assigned["left"],
        df_right=assigned["right"],
        save_path=DATA_DIR / COLUMN_MAP_FILE,
    )
    print(f"Column map written to: {DATA_DIR / COLUMN_MAP_FILE}")


if __name__ == "__main__":
    main()
