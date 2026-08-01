"""Plot 10 panels of the FAFB right optic lobe, one per LC type, with each column
filled red when an LC neuron of that type is assigned to it.

"Assigned" means the column appears as ``majority_column_id`` in that type's
``3_assigned_columns/<tag>_right_pre.csv`` (produced by assign_column.py). Nothing
is re-implemented here: the hex lattice + drawing primitive come from build_hex,
the per-cell CSV path/name from assign_column, and raw I/O from path.

Outputs are optional: nothing is (re)generated unless you pass a flag.

    .venv/bin/python "connectome/FAFBv783/3_assigned_columns/plot_lc_column.py" --png
    .venv/bin/python "connectome/FAFBv783/3_assigned_columns/plot_lc_column.py" --csv
    .venv/bin/python "connectome/FAFBv783/3_assigned_columns/plot_lc_column.py" --png --csv

Restrict to a subset of LC types with ``--types`` (comma-separated); the table is
then named after the numeric suffixes of the chosen cells (e.g.
``LC18,LC21,LC11,LC25`` -> the file ``lc_columns_right_18_21_11_25.csv``)::

    .venv/bin/python "connectome/FAFBv783/3_assigned_columns/plot_lc_column.py" \
        --csv --types LC18,LC21,LC11,LC25
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd

from import_bootstrap import parse_comma_list

import path  # noqa: E402
from build_hex import (  # noqa: E402
    EMPTY_COLOR,
    EXTENT,
    HEX_PATCH_RADIUS,
    INSIDE_COLOR,
    OUTSIDE_COLOR,
    _draw_hexes,
    uv_to_xy_deg,
    inside_mask,
    set_axis_labels,
)
from assign_column import _output_name  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SIDE = "right"
DIRECTION = "pre"
# The 10 LC types split into the two functional groups of the paper
# (LC18_model/paper.txt). These define the FIXED two-row panel:
#   ROW 1 = Figure 2I, looming-responsive group   (always the first row)
#   ROW 2 = Figure 2J, moving-object-responsive group (always the second row)
# Order within each list is the fixed column position of that type.
ROW1_CELLS: List[str] = ["LPLC2", "LC4", "LPLC1", "LC17", "LC12"]
ROW2_CELLS: List[str] = ["LC18", "LC21", "LC11", "LC25", "LC15"]
LC_CELLS: List[str] = ROW1_CELLS + ROW2_CELLS

# Per-column fill (face, edge) keyed by the neuron count in that column. The count
# is clamped to the largest key, so >=4 neurons all use the count-4 color.
#   1: light green / light coral  (build_hex inside/outside defaults)
#   2: dark green  / dark red
#   3: light blue  / light purple
#   4: dark blue   / dark purple
INSIDE_SHADE = {
    1: (INSIDE_COLOR[0], INSIDE_COLOR[1]),
    2: ("darkgreen", "black"),
    3: ("lightblue", "steelblue"),
    4: ("darkblue", "black"),
}
OUTSIDE_SHADE = {
    1: (OUTSIDE_COLOR[0], OUTSIDE_COLOR[1]),
    2: ("darkred", "black"),
    3: ("plum", "purple"),
    4: ("purple", "black"),
}
OUTPUT_FILE = "lc_columns_right.png"
TABLE_FILE = "lc_columns_right.csv"


def _subset_name(default_file: str, lc_cells: List[str]) -> str:
    """Output filename for a chosen subset of LC types.

    The full 10-type set keeps the canonical ``default_file``; any subset is
    named after the numeric suffixes of its types, e.g. ["LC18", "LPLC1"] with
    ``lc_columns_right.csv`` -> ``lc_columns_right_18_1.csv``.
    """
    base, ext = default_file.rsplit(".", 1)
    if list(lc_cells) == LC_CELLS:
        return default_file
    nums = [re.search(r"(\d+)$", t).group(1) for t in lc_cells]
    return f"{base}_{'_'.join(nums)}.{ext}"


def table_name(lc_cells: List[str]) -> str:
    """CSV filename for a chosen subset of LC types."""
    return _subset_name(TABLE_FILE, lc_cells)


def figure_name(lc_cells: List[str]) -> str:
    """PNG filename for a chosen subset of LC types."""
    return _subset_name(OUTPUT_FILE, lc_cells)


def column_counts(lc_cell: str) -> pd.Series:
    """Per-column neuron count for ``lc_cell`` (index=column_id, value=#neurons)."""
    csv = path.ASSIGNED_COLUMNS_DIR / _output_name(SIDE, [lc_cell], DIRECTION)
    if not csv.exists():
        logger.warning("Missing %s; run assign_column.py %s first", csv, lc_cell)
        return pd.Series(dtype="int64")
    col = pd.read_csv(csv)["majority_column_id"].dropna().astype(int)
    return col.value_counts()


def occupied_columns(lc_cell: str) -> Set[int]:
    """Column ids that >=1 located neuron of ``lc_cell`` is assigned to."""
    return set(column_counts(lc_cell).index)


def unresolved_neurons(lc_cell: str) -> pd.DataFrame:
    """Rows of ``lc_cell`` neurons with no resolved column (majority_column_id NA).

    These are the neurons excluded from the occupancy table because the locator
    could not assign them to a column.
    """
    csv = path.ASSIGNED_COLUMNS_DIR / _output_name(SIDE, [lc_cell], DIRECTION)
    if not csv.exists():
        logger.warning("Missing %s; run assign_column.py %s first", csv, lc_cell)
        return pd.DataFrame()
    df = pd.read_csv(csv)
    return df[df["majority_column_id"].isna()]


def build_column_table(all_column_ids: List[int], lc_cells: List[str]) -> pd.DataFrame:
    """Table: rows = every FAFB right column, cols = each LC type's neuron count,
    plus a ``sum`` column; sorted by ``sum`` descending."""
    table = pd.DataFrame(index=sorted(all_column_ids))
    table.index.name = "column_id"
    for lc in lc_cells:
        table[lc] = column_counts(lc).reindex(table.index).fillna(0).astype(int)
    table["sum"] = table[lc_cells].sum(axis=1)
    table = table.sort_values("sum", ascending=False)
    # Prepend a per-cell total row (sum over all columns).
    total = pd.DataFrame(table.sum(axis=0)).T
    total.index = ["total"]
    total.index.name = table.index.name
    return pd.concat([total, table])


def make_figure(cols: pd.DataFrame, lc_cells: List[str] = LC_CELLS) -> Path:
    """Render the per-cell FAFB right occupancy figure.

    The two rows are FIXED by functional group: the looming-responsive types
    (``ROW1_CELLS``, paper Figure 2I) always go in the first row, the
    moving-object-responsive types (``ROW2_CELLS``, Figure 2J) in the second.
    Within each row the selected types are packed left-to-right (no blank gaps),
    keeping their canonical order. An empty group's row is dropped entirely.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    col_u = dict(zip(cols["column_id"].astype(int), cols["u"].astype(int)))
    col_v = dict(zip(cols["column_id"].astype(int), cols["v"].astype(int)))
    # Inside/outside split from the shared knob (EXTENT < 0 -> all inside).
    mask = inside_mask(cols["u"].values, cols["v"].values, EXTENT)
    inside_ids = set(cols["column_id"].astype(int)[mask])

    bg_u, bg_v = cols["u"].values, cols["v"].values
    bg_labels = [None] * len(cols)
    bx, by = uv_to_xy_deg(bg_u, bg_v)
    margin = 2
    xlim = (bx.min() - margin, bx.max() + margin)
    ylim = (by.min() - margin, by.max() + margin)

    # Fixed rows by functional group, but pack each row left-to-right (no gaps).
    row_lists = [
        [lc for lc in group if lc in lc_cells]
        for group in (ROW1_CELLS, ROW2_CELLS)
    ]
    row_lists = [r for r in row_lists if r]  # drop a group with nothing selected
    nrows = len(row_lists)
    # +1 column per row for the rightmost "sum" panel (per-column total of the row).
    ncols = max(len(r) for r in row_lists) + 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 5.5 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )

    # Per-column neuron counts for every selected type; color deepens with count.
    counts_by_lc = {lc: column_counts(lc) for lc in lc_cells}

    def _row_sum(row: List[str]) -> pd.Series:
        """Per-column neuron total summed over the types in ``row``."""
        s = pd.Series(dtype="int64")
        for lc in row:
            s = s.add(counts_by_lc[lc], fill_value=0)
        return s.astype("int64")

    row_sums = [_row_sum(r) for r in row_lists]
    global_max = max(
        [int(s.max()) for s in counts_by_lc.values() if len(s)]
        + [int(s.max()) for s in row_sums if len(s)]
        + [1]
    )
    def _shade(shade, count):
        """(face, edge) for ``count`` neurons, clamped to the largest key."""
        return shade[min(int(count), max(shade))]

    def _fill(ax, counts, keep, shade):
        """Draw occupied columns (filtered by ``keep``), colored by neuron count."""
        n = 0
        for cnt in sorted(set(int(c) for c in counts.values)):
            ids = [c for c in counts.index[counts == cnt] if c in col_u and keep(c)]
            if not ids:
                continue
            u, v = zip(*[(col_u[c], col_v[c]) for c in ids])
            face, edge = _shade(shade, cnt)
            _draw_hexes(
                ax, np.array(u), np.array(v), [None] * len(ids),
                face, edge, HEX_PATCH_RADIUS,
            )
            n += len(ids)
        return n

    def _panel(ax, counts, title):
        # Background: every right column in light grey (reuse _draw_hexes).
        _draw_hexes(ax, bg_u, bg_v, bg_labels, EMPTY_COLOR[0], EMPTY_COLOR[1], HEX_PATCH_RADIUS)
        # Discrete per-count colors (INSIDE_SHADE / OUTSIDE_SHADE).
        n_in = _fill(ax, counts, lambda c: c in inside_ids, INSIDE_SHADE)
        n_out = _fill(ax, counts, lambda c: c not in inside_ids, OUTSIDE_SHADE)
        ax.set_title(
            f"{title}\n{n_in + n_out} columns ({n_out} outside)",
            fontsize=12, fontweight="bold",
        )

    drawn_axes = []
    for r, row in enumerate(row_lists):
        for c, lc in enumerate(row):
            ax = axes[r, c]
            drawn_axes.append(ax)
            _panel(ax, counts_by_lc[lc], lc)
        # Rightmost panel of the row: per-column sum over this row's types.
        ax_sum = axes[r, ncols - 1]
        drawn_axes.append(ax_sum)
        _panel(ax_sum, row_sums[r], "sum (" + "+".join(row) + ")")

    # Hide every slot that has no selected type (empty columns / empty rows).
    for ax in axes.flat:
        if ax not in drawn_axes:
            ax.set_visible(False)

    for ax in drawn_axes:
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        set_axis_labels(ax, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

    legend = []
    n_levels = min(global_max, max(INSIDE_SHADE))

    def _level_label(cnt):
        plus = "+" if (cnt == n_levels and global_max > n_levels) else ""
        s = "" if cnt == 1 else "s"
        return f"{cnt}{plus} neuron{s}"

    for cnt in range(1, n_levels + 1):
        face, edge = _shade(INSIDE_SHADE, cnt)
        legend.append(Patch(facecolor=face, edgecolor=edge, label=f"inside, {_level_label(cnt)}"))
    for cnt in range(1, n_levels + 1):
        face, edge = _shade(OUTSIDE_SHADE, cnt)
        legend.append(Patch(facecolor=face, edgecolor=edge, label=f"outside, {_level_label(cnt)}"))
    legend.append(Patch(facecolor=EMPTY_COLOR[0], edgecolor=EMPTY_COLOR[1], label="empty column"))
    fig.legend(
        handles=legend, loc="upper center", ncol=len(legend), fontsize=11,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.suptitle(
        f"FAFB {SIDE} columns occupied by {len(lc_cells)} LC types ({DIRECTION}-located)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = path.ASSIGNED_COLUMNS_DIR / figure_name(lc_cells)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    print(f"Saved {out}")
    return out


def make_table(cols: pd.DataFrame, lc_cells: List[str] = LC_CELLS) -> Path:
    """Build + save the per-column occupancy table (all columns x LC types + sum)."""
    table = build_column_table([int(c) for c in cols["column_id"]], lc_cells)
    table_path = path.ASSIGNED_COLUMNS_DIR / table_name(lc_cells)
    table.to_csv(table_path)
    logger.info("Saved %s", table_path)
    print(f"Saved {table_path}  ({len(table)} columns)")
    report_unresolved(lc_cells)
    return table_path


def report_unresolved(lc_cells: List[str]) -> None:
    """Print per-cell neuron counts: total, located, unresolved (no ids)."""
    print(f"\n=== neuron counts ({SIDE}, {DIRECTION}) ===")
    print("  type: total located unresolved")
    for lc in lc_cells:
        csv = path.ASSIGNED_COLUMNS_DIR / _output_name(SIDE, [lc], DIRECTION)
        if not csv.exists():
            logger.warning("Missing %s; run assign_column.py %s first", csv, lc)
            continue
        col = pd.read_csv(csv)["majority_column_id"]
        total = len(col)
        located = int(col.notna().sum())
        print(f"  {lc}: {total} {located} {total - located}")


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="LC column occupancy on FAFB right. Nothing is regenerated "
                    "unless --png and/or --csv is given."
    )
    parser.add_argument("--png", action="store_true", help="regenerate the 10-panel figure")
    parser.add_argument("--csv", action="store_true", help="regenerate the per-column table")
    parser.add_argument(
        "--cells",
        default="",
        metavar="LC,...",
        help="comma-separated LC cells (default: all 10). The CSV is named after "
             "the numeric suffixes, e.g. LC18,LC21 -> lc_columns_right_18_21.csv",
    )
    args = parser.parse_args(argv)

    requested = parse_comma_list(args.cells)
    if not requested:
        lc_cells = LC_CELLS
    else:
        bad = [t for t in requested if t not in LC_CELLS]
        if bad:
            raise SystemExit(f"unknown LC cell(s) {bad!r} (expected {LC_CELLS})")
        lc_cells = requested

    if not (args.png or args.csv):
        report_unresolved(lc_cells)
        return

    cols = path.load_column_map(SIDE)
    if args.png:
        make_figure(cols, lc_cells)
    if args.csv:
        make_table(cols, lc_cells)


if __name__ == "__main__":
    main()
