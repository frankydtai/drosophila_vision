"""Locate neurons that have no column_assignment of their own.

Some visual neurons (notably R1-6) never get a direct ``(p, q)`` column in
``column_assignment``; they must be placed by their synaptic partners. This
module is a general locator: for any cell type, it infers each neuron's column
from the majority column of its partners. Neurons keep their original type
(no per-column slot splitting).

Direction matters and depends on the neuron's role:

  - ``post``  : locate by *downstream* targets' columns. Use for input neurons
                that send into the lattice (e.g. R1-6 -> its post columns).
  - ``pre``   : locate by *upstream* sources' columns. Use for output/projection
                neurons that read out of the lattice (e.g. LC/VS <- its pre columns).

Run with the project venv (defaults to R1-6, post-majority, both sides):

    .venv/bin/python "Connectome/FAFB v783/column_locator.py"
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Sequence

import pandas as pd

import fafb_io
from fafb_io import DATA_DIR

logger = logging.getLogger(__name__)

# Cell types located by their downstream targets by default.
DEFAULT_TARGET_TYPES = ("R1-6",)
DEFAULT_DIRECTION = "post"


def _type_tag(cell_type: str) -> str:
    """Turn a cell type into a filename tag, e.g. 'R1-6' -> 'r1_6'."""
    return re.sub(r"[^0-9a-z]+", "_", cell_type.lower()).strip("_")


def locate_neurons(
    neurons: pd.DataFrame,
    columns: pd.DataFrame,
    connections: pd.DataFrame,
    target_types: Sequence[str],
    side: str,
    direction: str = DEFAULT_DIRECTION,
    weight_by_syn: bool = False,
) -> pd.DataFrame:
    """Infer a column (and slot) for each neuron of the requested types.

    Args:
        neurons: visual neurons (root_id, type, side) already filtered to ``side``.
        columns: column_assignment rows (root_id, hemisphere, column_id) for ``side``.
        connections: connection rows (pre_root_id, post_root_id, syn_count).
        target_types: cell types to locate.
        side: 'left' or 'right' (for logging only; inputs must already match).
        direction: 'post' (by downstream targets) or 'pre' (by upstream sources).
        weight_by_syn: vote by summed syn_count instead of distinct-partner count.

    Returns:
        One row per target neuron: root_id, type, n_partners,
        n_partners_with_column, majority_column_id (Int64, NA if unresolved),
        votes (Int64).
    """
    if direction not in ("post", "pre"):
        raise ValueError(f"direction must be 'post' or 'pre', got {direction!r}")

    targets = neurons[neurons["type"].isin(list(target_types))][["root_id", "type"]]
    target_ids = set(targets["root_id"].astype("int64"))
    logger.info(
        "Locating %d neurons of types %s (%s, direction=%s)",
        len(target_ids), list(target_types), side, direction,
    )

    partner_col = (
        columns.drop_duplicates("root_id").set_index("root_id")["column_id"]
    )

    # self_id is the target column; partner_id provides the column vote.
    if direction == "post":
        self_id, partner_id = "pre_root_id", "post_root_id"
    else:
        self_id, partner_id = "post_root_id", "pre_root_id"

    e = connections[connections[self_id].isin(target_ids)][
        [self_id, partner_id, "syn_count"]
    ].copy()
    e["col"] = e[partner_id].map(partner_col)

    n_partners = e.groupby(self_id)[partner_id].nunique()

    e_col = e.dropna(subset=["col"]).copy()
    e_col["col"] = e_col["col"].astype("int64")
    n_partners_with_column = e_col.groupby(self_id)[partner_id].nunique()

    if weight_by_syn:
        votes = e_col.groupby([self_id, "col"])["syn_count"].sum()
    else:
        votes = e_col.groupby([self_id, "col"])[partner_id].nunique()
    votes = votes.reset_index(name="votes")
    # Majority: most votes; ties broken by the larger column_id (matches NIPS).
    votes = votes.sort_values(
        [self_id, "votes", "col"], ascending=[True, False, False]
    )
    best = votes.groupby(self_id).first()

    out = targets.rename(columns={"root_id": "_rid"}).copy()
    out["root_id"] = out["_rid"].astype("int64")
    out = out.drop(columns="_rid")
    out["n_partners"] = (
        out["root_id"].map(n_partners).fillna(0).astype("int64")
    )
    out["n_partners_with_column"] = (
        out["root_id"].map(n_partners_with_column).fillna(0).astype("int64")
    )
    out["majority_column_id"] = out["root_id"].map(best["col"]).astype("Int64")
    out["votes"] = out["root_id"].map(best["votes"]).astype("Int64")
    return out.sort_values(["type", "majority_column_id", "root_id"]).reset_index(
        drop=True
    )


def _output_name(side: str, target_types: Sequence[str]) -> str:
    if list(target_types) == list(DEFAULT_TARGET_TYPES):
        tag = _type_tag(target_types[0])
    else:
        tag = "_".join(_type_tag(t) for t in target_types)
    return f"location_{tag}_{side}.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate neurons by partner columns.")
    parser.add_argument("--side", default="both", choices=["left", "right", "both"])
    parser.add_argument(
        "--types", nargs="+", default=list(DEFAULT_TARGET_TYPES),
        help="Cell types to locate (default: R1-6).",
    )
    parser.add_argument(
        "--direction", default=DEFAULT_DIRECTION, choices=["post", "pre"],
        help="post: by downstream targets; pre: by upstream sources.",
    )
    parser.add_argument(
        "--weight-by-syn", action="store_true",
        help="Vote by summed syn_count instead of distinct-partner count.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    sides = ["left", "right"] if args.side == "both" else [args.side]

    all_neurons = fafb_io.load_visual_neurons()
    all_columns = fafb_io.load_column_assignments()

    for side in sides:
        neurons = all_neurons[all_neurons["side"] == side]
        columns = all_columns[all_columns["hemisphere"] == side]
        target_ids = set(
            neurons[neurons["type"].isin(args.types)]["root_id"].astype("int64")
        )
        # Pull all edges touching the targets on the relevant side (no syn cut).
        connections = fafb_io.load_connections(keep_neuron_ids=target_ids)

        located = locate_neurons(
            neurons=neurons,
            columns=columns,
            connections=connections,
            target_types=args.types,
            side=side,
            direction=args.direction,
            weight_by_syn=args.weight_by_syn,
        )
        out_path = DATA_DIR / _output_name(side, args.types)
        located.to_csv(out_path, index=False)

        n_total = len(located)
        n_located = int(located["majority_column_id"].notna().sum())
        print(f"\n=== locate {args.types} ({side}, direction={args.direction}) ===")
        print(f"  neurons: {n_total}  located: {n_located}  unresolved: {n_total - n_located}")
        print(f"  output: {out_path}")


if __name__ == "__main__":
    main()
