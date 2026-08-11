"""Locate neurons that have no column_assignment of their own.

Some visual neurons (notably R1-6) never get a direct ``(p, q)`` column in
``column_assignment``; they must be placed by their synaptic partners. This
module is a general locator: for any cell, it infers each neuron's column
from the majority column of its partners. Neurons keep their original type
(no per-column slot splitting).

Direction matters and depends on the neuron's role:

  - ``post``  : locate by *downstream* targets' columns. Use for input neurons
                that send into the lattice (e.g. R1-6 -> its post columns).
  - ``pre``   : locate by *upstream* sources' columns. Use for output/projection
                neurons that read out of the lattice (e.g. LC/VS <- its pre columns).

``ASSIGNED_COLUMN_CELLS`` is the sole list of (cell, direction) pairs merged into
``network.json`` by ``4_build_network.py`` (and thus eligible for radius crops).
It includes R1-6, Lawf1/Lawf2, and SimulationCode-mapped FAFB types that lack
native column_assignment.

Cell types are one positional comma-separated token (like analyze_cell_syn.py); direction
is a ``--post`` flag (default ``pre``, by upstream sources). Outputs go to the
``3_assigned_columns/`` subfolder as ``<tag>_<side>_<direction>.csv`` (e.g.
``r1_6_left_post.csv``).

Run with the project venv (default: all ``ASSIGNED_COLUMN_CELLS``):

    .venv/bin/python "connectome/FAFBv783/3_assign_column.py"
    .venv/bin/python "connectome/FAFBv783/3_assign_column.py" R1-6 --post
    .venv/bin/python "connectome/FAFBv783/3_assign_column.py" TmY11,L3
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import List, Optional, Sequence, Tuple

import pandas as pd

import path
from import_bootstrap import parse_comma_list

logger = logging.getLogger(__name__)

# Sole (cell, direction) list for partner-based column placement. Consumed by
# 4_build_network.py; CSV path is <tag>_<side>_<direction>.csv under
# 3_assigned_columns/. FAFB Matsliah names only (not Borst aliases).
ASSIGNED_COLUMN_CELLS: List[Tuple[str, str]] = [
    ("R1-6", "post"),
    ("Lawf1", "pre"),
    ("Lawf2", "pre"),
    # SimulationCode extras with no native column_assignment
    ("Lai", "pre"),
    ("Mi2", "pre"),
    ("Mi10", "pre"),
    ("Mi13", "pre"),
    ("Mi14", "pre"),
    ("Mi15", "pre"),
    ("Tm5f", "pre"),
    ("Tm5a", "pre"),
    ("Tm5b", "pre"),
    ("Tm5c", "pre"),
    ("Tm16", "pre"),
    ("Dm3v", "pre"),
    ("Tm31", "pre"),
    ("TmY3", "pre"),
    ("TmY4", "pre"),
    ("TmY5a", "pre"),
    ("TmY9q", "pre"),
    ("TmY9q__perp", "pre"),
    ("TmY10", "pre"),
    ("TmY11", "pre"),
    ("TmY14", "pre"),
    ("TmY15", "pre"),
    ("Tm27", "pre"),
]

DEFAULT_DIRECTION = "pre"


def _cell_tag(cell: str) -> str:
    """Turn a cell into a filename tag, e.g. 'R1-6' -> 'r1_6'."""
    return re.sub(r"[^0-9a-z]+", "_", cell.lower()).strip("_")


def locate_neurons(
    neurons: pd.DataFrame,
    columns: pd.DataFrame,
    connections: pd.DataFrame,
    target_cells: Sequence[str],
    side: str,
    direction: str = DEFAULT_DIRECTION,
    weight_by_syn: bool = False,
    uv_from_column: Optional[dict] = None,
) -> pd.DataFrame:
    """Infer a column (and slot) for each neuron of the requested types.

    Args:
        neurons: visual neurons (root_id, type, side) already filtered to ``side``.
        columns: column_assignment rows (root_id, hemisphere, column_id) for ``side``.
        connections: connection rows (pre_root_id, post_root_id, syn_count).
        target_cells: cells to locate.
        side: 'left' or 'right' (for logging only; inputs must already match).
        direction: 'post' (by downstream targets) or 'pre' (by upstream sources).
        weight_by_syn: vote by summed syn_count instead of distinct-partner count.
        uv_from_column: optional {column_id: (u, v)} map; when given, adds max_u/min_u/
            max_v/min_v (the hex range spanned by each neuron's column partners).

    Returns:
        One row per target neuron: root_id, type, n_<dir>, n_<dir>_with_column
        (where <dir> is 'post' for direction='post' and 'pre' for
        direction='pre'), votes (descending per-column vote counts as a string,
        e.g. "5, 5, 5, 3"; sums to n_<dir>_with_column), majority_column_id
        (Int64, NA if unresolved). When ``uv_from_column`` is given, also per-coordinate
        mean/max/min for u, v (hex) and x, y (hex-step via build_hex.xy_from_uv): mean_* is the
        vote-weighted average over the column partners, max_*/min_* the range
        (all NA if unresolved). In this case ``majority_column_id`` keeps the
        top-voted column only when it has >50% of the votes; otherwise it is the
        column nearest (Euclidean in u,v) to the vote-weighted mean.
    """
    if direction not in ("post", "pre"):
        raise ValueError(f"direction must be 'post' or 'pre', got {direction!r}")

    targets = neurons[neurons["cell"].isin(list(target_cells))][["root_id", "cell"]]
    target_ids = set(targets["root_id"].astype("int64"))
    logger.info(
        "Locating %d neurons of cells %s (%s, direction=%s)",
        len(target_ids), list(target_cells), side, direction,
    )

    partner_column_id = (
        columns.drop_duplicates("root_id").set_index("root_id")["column_id"]
    )

    # self_id is the target neuron; partner_id provides the column vote.
    # partner_kind names the output count fields after the partner side.
    if direction == "post":
        self_id, partner_id = "pre_root_id", "post_root_id"
        partner_kind = "post"
    else:
        self_id, partner_id = "post_root_id", "pre_root_id"
        partner_kind = "pre"
    n_partner_field = f"n_{partner_kind}"
    n_with_column_field = f"n_{partner_kind}_with_column"

    e = connections[connections[self_id].isin(target_ids)][
        [self_id, partner_id, "syn_count"]
    ].copy()
    e["column_id"] = e[partner_id].map(partner_column_id)

    n_partners = e.groupby(self_id)[partner_id].nunique()

    with_column = e.dropna(subset=["column_id"]).copy()
    with_column["column_id"] = with_column["column_id"].astype("int64")
    n_partners_with_column = with_column.groupby(self_id)[partner_id].nunique()

    if weight_by_syn:
        votes = with_column.groupby([self_id, "column_id"])["syn_count"].sum()
    else:
        votes = with_column.groupby([self_id, "column_id"])[partner_id].nunique()
    votes = votes.reset_index(name="votes")
    # Majority: most votes; ties broken by the larger column_id (matches NIPS).
    votes = votes.sort_values(
        [self_id, "votes", "column_id"], ascending=[True, False, False]
    )
    best = votes.groupby(self_id).first()
    # All per-column vote counts (descending), e.g. "5, 5, 5, 3"; sums to n_with_column.
    votes_list = votes.groupby(self_id, sort=False)["votes"].apply(
        lambda s: ", ".join(str(int(x)) for x in s)
    )

    out = targets.rename(columns={"root_id": "_rid"}).copy()
    out["root_id"] = out["_rid"].astype("int64")
    out = out.drop(columns="_rid")
    out[n_partner_field] = (
        out["root_id"].map(n_partners).fillna(0).astype("int64")
    )
    out[n_with_column_field] = (
        out["root_id"].map(n_partners_with_column).fillna(0).astype("int64")
    )
    out["votes"] = out["root_id"].map(votes_list).fillna("").astype("string")
    out["majority_column_id"] = out["root_id"].map(best["column_id"]).astype("Int64")

    if uv_from_column is not None:
        from build_hex import xy_from_uv

        u_by_column_id = {int(c): uv[0] for c, uv in uv_from_column.items()}
        v_by_column_id = {int(c): uv[1] for c, uv in uv_from_column.items()}
        vu = votes[[self_id, "column_id", "votes"]].copy()
        vu["u"] = vu["column_id"].map(u_by_column_id)
        vu["v"] = vu["column_id"].map(v_by_column_id)
        vu["x"], vu["y"] = xy_from_uv(
            vu["u"].astype("float"), vu["v"].astype("float"),
        )
        # Vote-weighted mean position (weight = per-column vote count).
        vu["w"] = vu["votes"].astype("float")
        for coord in ("u", "v", "x", "y"):
            vu[f"_w{coord}"] = vu[coord].astype("float") * vu["w"]
        g = vu.groupby(self_id)
        w_sum = g["w"].sum()
        raw_mean = {c: g[f"_w{c}"].sum() / w_sum for c in ("u", "v", "x", "y")}
        # Per coordinate, arrange as mean (weighted), max, min.
        for coord, dtype in (("u", "Int64"), ("v", "Int64"), ("x", "Float64"), ("y", "Float64")):
            out[f"mean_{coord}"] = (
                out["root_id"].map(raw_mean[coord].round(3)).astype("Float64")
            )
            out[f"max_{coord}"] = out["root_id"].map(g[coord].max()).astype(dtype)
            out[f"min_{coord}"] = out["root_id"].map(g[coord].min()).astype(dtype)

        # Majority column: keep the top-voted column when it holds >50% of the
        # votes; otherwise use the column nearest (Euclidean in u,v) to the
        # vote-weighted mean.
        votes_sum = g["votes"].sum()
        best_frac = best["votes"] / votes_sum
        vu["d2"] = (vu["u"] - vu[self_id].map(raw_mean["u"])) ** 2 + (
            vu["v"] - vu[self_id].map(raw_mean["v"])
        ) ** 2
        nearest = (
            vu.sort_values(
                [self_id, "d2", "votes", "column_id"], ascending=[True, True, False, False]
            )
            .groupby(self_id)
            .first()["column_id"]
        )
        chosen = best["column_id"].astype("float").copy()
        use_nearest = best_frac <= 0.5
        chosen.loc[use_nearest] = nearest.reindex(chosen.index).loc[use_nearest]
        out["majority_column_id"] = out["root_id"].map(chosen).astype("Int64")

    return out.sort_values(["cell", "majority_column_id", "root_id"]).reset_index(
        drop=True
    )


def _output_name(side: str, target_cells: Sequence[str], direction: str) -> str:
    tag = "_".join(_cell_tag(t) for t in target_cells)
    return f"{tag}_{side}_{direction}.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate neurons by partner columns.")
    parser.add_argument(
        "cells",
        nargs="?",
        default=None,
        metavar="CELL[,CELL...]",
        help="Comma-separated cells to locate. Default: every entry in "
             "ASSIGNED_COLUMN_CELLS (one CSV per cell, using that entry's direction).",
    )
    parser.add_argument("--side", default="right", choices=["left", "right", "both"])
    parser.add_argument(
        "--post", action="store_true",
        help="Locate by downstream targets (post). Default is pre (by upstream "
             "sources). Ignored when running the full ASSIGNED_COLUMN_CELLS list.",
    )
    parser.add_argument(
        "--weight-by-syn", action="store_true",
        help="Vote by summed syn_count instead of distinct-partner count.",
    )
    return parser.parse_args()


def _jobs_from_args(args: argparse.Namespace) -> List[Tuple[List[str], str]]:
    """Return [(cells, direction), ...] jobs to run."""
    if args.cells is None:
        return [([cell], direction) for cell, direction in ASSIGNED_COLUMN_CELLS]
    cells = parse_comma_list(args.cells)
    if not cells:
        raise SystemExit("cells must not be empty")
    direction = "post" if args.post else "pre"
    return [(cells, direction)]


def _locate_and_write(
    *,
    cells: Sequence[str],
    direction: str,
    side: str,
    all_neurons: pd.DataFrame,
    all_columns: pd.DataFrame,
    out_dir,
    weight_by_syn: bool,
) -> None:
    neurons = all_neurons[all_neurons["side"] == side]
    columns = all_columns[all_columns["hemisphere"] == side]
    target_ids = set(
        neurons[neurons["cell"].isin(list(cells))]["root_id"].astype("int64")
    )
    # Pull all edges touching the targets on the relevant side (no syn cut).
    connections = path.load_connections(keep_neuron_ids=target_ids)

    # column_id -> (u, v) for the per-neuron hex range (max/min u/v).
    uv_from_column = None
    if path.column_map_path(side).exists():
        hex_df = path.load_column_map(side)
        uv_from_column = {
            int(r.column_id): (int(r.u), int(r.v))
            for r in hex_df.itertuples(index=False)
        }
    else:
        logger.warning(
            "Missing %s; skipping max/min u/v columns",
            path.column_map_path(side),
        )

    located = locate_neurons(
        neurons=neurons,
        columns=columns,
        connections=connections,
        target_cells=cells,
        side=side,
        direction=direction,
        weight_by_syn=weight_by_syn,
        uv_from_column=uv_from_column,
    )
    out_path = out_dir / _output_name(side, cells, direction)
    located.to_csv(out_path, index=False)

    n_total = len(located)
    n_located = int(located["majority_column_id"].notna().sum())
    print(f"\n=== locate {list(cells)} ({side}, direction={direction}) ===")
    print(f"  neurons: {n_total}  located: {n_located}  unresolved: {n_total - n_located}")
    print(f"  output: {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    jobs = _jobs_from_args(args)
    sides = ["left", "right"] if args.side == "both" else [args.side]

    all_neurons = path.load_visual_neurons()
    all_columns = path.load_column_assignments()

    out_dir = path.ASSIGNED_COLUMNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for cells, direction in jobs:
        for side in sides:
            _locate_and_write(
                cells=cells,
                direction=direction,
                side=side,
                all_neurons=all_neurons,
                all_columns=all_columns,
                out_dir=out_dir,
                weight_by_syn=args.weight_by_syn,
            )


if __name__ == "__main__":
    main()
