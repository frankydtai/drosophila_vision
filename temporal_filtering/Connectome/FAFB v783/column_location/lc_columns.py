#!/usr/bin/env python3
"""Plot 10 panels of the FAFB right optic lobe, one per LC type, with each column
filled red when an LC neuron of that type is assigned to it.

"Assigned" means the column appears as ``majority_column_id`` in that type's
``column_location/<tag>_right_pre.csv`` (produced by column_locator.py). Nothing
is re-implemented here: the hex lattice + drawing primitive come from hex_grid,
the per-type CSV path/name from column_locator, and raw I/O from fafb_io.

Outputs are optional: nothing is (re)generated unless you pass a flag.

    .venv/bin/python "Connectome/FAFB v783/column_location/lc_columns.py" --png
    .venv/bin/python "Connectome/FAFB v783/column_location/lc_columns.py" --csv
    .venv/bin/python "Connectome/FAFB v783/column_location/lc_columns.py" --png --csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd

# This script lives in column_location/; make the package modules (one dir up)
# importable so we reuse fafb_io / hex_grid / column_locator.
_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import fafb_io  # noqa: E402
from fafb_io import DATA_DIR  # noqa: E402
from hex_grid import (  # noqa: E402
    DEFAULT_KERNEL_SIZE,
    EMPTY_COLOR,
    INSIDE_COLOR,
    OUTSIDE_COLOR,
    _draw_hexes,
    hex_to_pixel,
)
from column_locator import OUTPUT_SUBDIR, _output_name  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SIDE = "right"
DIRECTION = "pre"
# The 10 LC types in the LC18_model/paper.txt panel order (Figures 2I top row,
# 2J bottom row).
LC_TYPES: List[str] = [
    "LPLC2", "LC4", "LPLC1", "LC17", "LC12",
    "LC18", "LC21", "LC11", "LC25", "LC15",
]
OUTPUT_FILE = "lc_columns_right.png"
TABLE_FILE = "lc_columns_right.csv"


def column_counts(lc_type: str) -> pd.Series:
    """Per-column neuron count for ``lc_type`` (index=column_id, value=#neurons)."""
    csv = DATA_DIR / OUTPUT_SUBDIR / _output_name(SIDE, [lc_type], DIRECTION)
    if not csv.exists():
        logger.warning("Missing %s; run column_locator.py %s first", csv, lc_type)
        return pd.Series(dtype="int64")
    col = pd.read_csv(csv)["majority_column_id"].dropna().astype(int)
    return col.value_counts()


def occupied_columns(lc_type: str) -> Set[int]:
    """Column ids that >=1 located neuron of ``lc_type`` is assigned to."""
    return set(column_counts(lc_type).index)


def build_column_table(all_column_ids: List[int]) -> pd.DataFrame:
    """Table: rows = every FAFB right column, cols = each LC type's neuron count,
    plus a ``sum`` column; sorted by ``sum`` descending."""
    table = pd.DataFrame(index=sorted(all_column_ids))
    table.index.name = "column_id"
    for lc in LC_TYPES:
        table[lc] = column_counts(lc).reindex(table.index).fillna(0).astype(int)
    table["sum"] = table[LC_TYPES].sum(axis=1)
    table = table.sort_values("sum", ascending=False)
    # Prepend a per-type total row (sum over all columns).
    total = pd.DataFrame(table.sum(axis=0)).T
    total.index = ["total"]
    total.index.name = table.index.name
    return pd.concat([total, table])


def make_figure(cols: pd.DataFrame) -> Path:
    """Render the 10-panel FAFB right occupancy figure and save it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    col_u = dict(zip(cols["column_id"].astype(int), cols["u"].astype(int)))
    col_v = dict(zip(cols["column_id"].astype(int), cols["v"].astype(int)))
    inside_ids = set(cols[cols["hex_status"] == "inside"]["column_id"].astype(int))

    hex_radius = 0.5 * float(DEFAULT_KERNEL_SIZE)
    bg_u, bg_v = cols["u"].values, cols["v"].values
    bg_labels = [None] * len(cols)
    bx, by = hex_to_pixel(bg_u, bg_v)
    margin = 2
    xlim = (bx.min() - margin, bx.max() + margin)
    ylim = (by.min() - margin, by.max() + margin)

    fig, axes = plt.subplots(2, 5, figsize=(26, 11), sharex=True, sharey=True)

    # Per-column neuron counts for every type; color deepens with count.
    counts_by_lc = {lc: column_counts(lc) for lc in LC_TYPES}
    global_max = max(
        [int(s.max()) for s in counts_by_lc.values() if len(s)] + [1]
    )
    greens, reds = plt.get_cmap("Greens"), plt.get_cmap("Reds")

    def _shade(cmap, light, count):
        # count==1 keeps the original light color; count>=2 is much darker.
        if count <= 1:
            return light
        frac = count / global_max if global_max else 1.0
        return cmap(0.55 + 0.44 * frac)

    def _fill(ax, counts, keep, cmap, light, edge):
        """Draw occupied columns (filtered by ``keep``), shaded by neuron count."""
        n = 0
        for cnt in sorted(set(int(c) for c in counts.values)):
            ids = [c for c in counts.index[counts == cnt] if c in col_u and keep(c)]
            if not ids:
                continue
            u, v = zip(*[(col_u[c], col_v[c]) for c in ids])
            _draw_hexes(
                ax, np.array(u), np.array(v), [None] * len(ids),
                _shade(cmap, light, cnt), edge, hex_radius,
            )
            n += len(ids)
        return n

    for ax, lc in zip(axes.flat, LC_TYPES):
        # Background: every right column in light grey (reuse _draw_hexes).
        _draw_hexes(ax, bg_u, bg_v, bg_labels, EMPTY_COLOR[0], EMPTY_COLOR[1], hex_radius)
        counts = counts_by_lc[lc]
        # Colors come from hex_grid (same source as column_hex_map.png):
        # inside green, outside red; count>=2 is drawn darker.
        n_in = _fill(ax, counts, lambda c: c in inside_ids, greens, INSIDE_COLOR[0], INSIDE_COLOR[1])
        n_out = _fill(ax, counts, lambda c: c not in inside_ids, reds, OUTSIDE_COLOR[0], OUTSIDE_COLOR[1])
        ax.set_title(
            f"{lc}\n{n_in + n_out} columns ({n_out} outside)",
            fontsize=12, fontweight="bold",
        )

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("X (pixel)", fontsize=9)
        ax.set_ylabel("Y (pixel)", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[0, 0].invert_yaxis()  # shared axes -> all panels match column_hex_map

    legend = []
    for cnt in range(1, global_max + 1):
        s = "" if cnt == 1 else "s"
        legend.append(Patch(facecolor=_shade(greens, INSIDE_COLOR[0], cnt), edgecolor=INSIDE_COLOR[1],
                            label=f"inside, {cnt} neuron{s}"))
    for cnt in range(1, global_max + 1):
        s = "" if cnt == 1 else "s"
        legend.append(Patch(facecolor=_shade(reds, OUTSIDE_COLOR[0], cnt), edgecolor=OUTSIDE_COLOR[1],
                            label=f"outside, {cnt} neuron{s}"))
    legend.append(Patch(facecolor=EMPTY_COLOR[0], edgecolor=EMPTY_COLOR[1], label="empty column"))
    fig.legend(
        handles=legend, loc="upper center", ncol=len(legend), fontsize=11,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.suptitle(
        f"FAFB {SIDE} columns occupied by 10 LC types ({DIRECTION}-located)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(__file__).resolve().parent / OUTPUT_FILE
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    print(f"Saved {out}")
    return out


def make_table(cols: pd.DataFrame) -> Path:
    """Build + save the per-column occupancy table (all columns x 10 LC + sum)."""
    table = build_column_table([int(c) for c in cols["column_id"]])
    table_path = Path(__file__).resolve().parent / TABLE_FILE
    table.to_csv(table_path)
    logger.info("Saved %s", table_path)
    print(f"Saved {table_path}  ({len(table)} columns)")
    print(table.head(10).to_string())
    return table_path


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="LC column occupancy on FAFB right. Nothing is regenerated "
                    "unless --png and/or --csv is given."
    )
    parser.add_argument("--png", action="store_true", help="regenerate the 10-panel figure")
    parser.add_argument("--csv", action="store_true", help="regenerate the per-column table")
    args = parser.parse_args(argv)

    if not (args.png or args.csv):
        print("Nothing to do. Pass --png and/or --csv to (re)generate outputs.")
        return

    cols = fafb_io.load_column_hex_index(SIDE)
    if args.png:
        make_figure(cols)
    if args.csv:
        make_table(cols)


if __name__ == "__main__":
    main()
