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

Cell types are positional (like cell_syn.py); direction is a ``--post`` flag
(default ``pre``, by upstream sources). Outputs go to the ``column_location/``
subfolder as ``<tag>_<side>_<direction>.csv`` (e.g. ``r1_6_left_post.csv``).

Run with the project venv (defaults to R1-6, right side, pre):

    .venv/bin/python "Connectome/FAFBv783/column_locator.py" R1-6 --post
    .venv/bin/python "Connectome/FAFBv783/column_locator.py" TmY11
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Optional, Sequence

import pandas as pd

import connectome_io

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
    col_to_uv: Optional[dict] = None,
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
        col_to_uv: optional {column_id: (u, v)} map; when given, adds max_u/min_u/
            max_v/min_v (the hex extent spanned by each neuron's column partners).

    Returns:
        One row per target neuron: root_id, type, n_<dir>, n_<dir>_with_column
        (where <dir> is 'post' for direction='post' and 'pre' for
        direction='pre'), votes (descending per-column vote counts as a string,
        e.g. "5, 5, 5, 3"; sums to n_<dir>_with_column), majority_column_id
        (Int64, NA if unresolved). When ``col_to_uv`` is given, also per-coordinate
        mean/max/min for u, v (hex) and x, y (pixel; x=v, y=u+v/2): mean_* is the
        vote-weighted average over the column partners, max_*/min_* the extent
        (all NA if unresolved). In this case ``majority_column_id`` keeps the
        top-voted column only when it has >50% of the votes; otherwise it is the
        column nearest (Euclidean in u,v) to the vote-weighted mean.
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
    # partner_kind names the output count columns after the partner side.
    if direction == "post":
        self_id, partner_id = "pre_root_id", "post_root_id"
        partner_kind = "post"
    else:
        self_id, partner_id = "post_root_id", "pre_root_id"
        partner_kind = "pre"
    n_col = f"n_{partner_kind}"
    n_with_col = f"n_{partner_kind}_with_column"

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
    # All per-column vote counts (descending), e.g. "5, 5, 5, 3"; sums to n_with_column.
    votes_list = votes.groupby(self_id, sort=False)["votes"].apply(
        lambda s: ", ".join(str(int(x)) for x in s)
    )

    out = targets.rename(columns={"root_id": "_rid"}).copy()
    out["root_id"] = out["_rid"].astype("int64")
    out = out.drop(columns="_rid")
    out[n_col] = (
        out["root_id"].map(n_partners).fillna(0).astype("int64")
    )
    out[n_with_col] = (
        out["root_id"].map(n_partners_with_column).fillna(0).astype("int64")
    )
    out["votes"] = out["root_id"].map(votes_list).fillna("").astype("string")
    out["majority_column_id"] = out["root_id"].map(best["col"]).astype("Int64")

    if col_to_uv is not None:
        # Reuse the single source of truth for axial->pixel (x=v, y=u+v/2).
        from column_mapper import hex_to_pixel

        u_by_col = {int(c): uv[0] for c, uv in col_to_uv.items()}
        v_by_col = {int(c): uv[1] for c, uv in col_to_uv.items()}
        vu = votes[[self_id, "col", "votes"]].copy()
        vu["u"] = vu["col"].map(u_by_col)
        vu["v"] = vu["col"].map(v_by_col)
        vu["x"], vu["y"] = hex_to_pixel(
            vu["u"].astype("float"), vu["v"].astype("float"), kernel_size=1.0
        )
        # Vote-weighted mean position (weight = per-column vote count).
        vu["w"] = vu["votes"].astype("float")
        for coord in ("u", "v", "x", "y"):
            vu[f"_w{coord}"] = vu[coord].astype("float") * vu["w"]
        g = vu.groupby(self_id)
        wsum = g["w"].sum()
        raw_mean = {c: g[f"_w{c}"].sum() / wsum for c in ("u", "v", "x", "y")}
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
        total = g["votes"].sum()
        best_frac = best["votes"] / total
        vu["d2"] = (vu["u"] - vu[self_id].map(raw_mean["u"])) ** 2 + (
            vu["v"] - vu[self_id].map(raw_mean["v"])
        ) ** 2
        nearest = (
            vu.sort_values(
                [self_id, "d2", "votes", "col"], ascending=[True, True, False, False]
            )
            .groupby(self_id)
            .first()["col"]
        )
        chosen = best["col"].astype("float").copy()
        use_nearest = best_frac <= 0.5
        chosen.loc[use_nearest] = nearest.reindex(chosen.index).loc[use_nearest]
        out["majority_column_id"] = out["root_id"].map(chosen).astype("Int64")

    return out.sort_values(["type", "majority_column_id", "root_id"]).reset_index(
        drop=True
    )


def _output_name(side: str, target_types: Sequence[str], direction: str) -> str:
    if list(target_types) == list(DEFAULT_TARGET_TYPES):
        tag = _type_tag(target_types[0])
    else:
        tag = "_".join(_type_tag(t) for t in target_types)
    return f"{tag}_{side}_{direction}.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate neurons by partner columns.")
    parser.add_argument(
        "cell_types", nargs="*", default=list(DEFAULT_TARGET_TYPES),
        metavar="CELL_TYPE",
        help="Cell type(s) to locate as positional args (default: R1-6).",
    )
    parser.add_argument("--side", default="right", choices=["left", "right", "both"])
    parser.add_argument(
        "--post", action="store_true",
        help="Locate by downstream targets (post). Default is pre (by upstream "
             "sources).",
    )
    parser.add_argument(
        "--weight-by-syn", action="store_true",
        help="Vote by summed syn_count instead of distinct-partner count.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    direction = "post" if args.post else "pre"
    sides = ["left", "right"] if args.side == "both" else [args.side]

    all_neurons = connectome_io.load_visual_neurons()
    all_columns = connectome_io.load_column_assignments()

    out_dir = connectome_io.COLUMN_LOCATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for side in sides:
        neurons = all_neurons[all_neurons["side"] == side]
        columns = all_columns[all_columns["hemisphere"] == side]
        target_ids = set(
            neurons[neurons["type"].isin(args.cell_types)]["root_id"].astype("int64")
        )
        # Pull all edges touching the targets on the relevant side (no syn cut).
        connections = connectome_io.load_connections(keep_neuron_ids=target_ids)

        # column_id -> (u, v) for the per-neuron hex extent (max/min u/v).
        col_to_uv = None
        if connectome_io.column_map_path(side).exists():
            hex_df = connectome_io.load_column_map(side)
            col_to_uv = {
                int(r.column_id): (int(r.u), int(r.v))
                for r in hex_df.itertuples(index=False)
            }
        else:
            logger.warning(
                "Missing %s; skipping max/min u/v columns",
                connectome_io.column_map_path(side),
            )

        located = locate_neurons(
            neurons=neurons,
            columns=columns,
            connections=connections,
            target_types=args.cell_types,
            side=side,
            direction=direction,
            weight_by_syn=args.weight_by_syn,
            col_to_uv=col_to_uv,
        )
        out_path = out_dir / _output_name(side, args.cell_types, direction)
        located.to_csv(out_path, index=False)

        n_total = len(located)
        n_located = int(located["majority_column_id"].notna().sum())
        print(f"\n=== locate {args.cell_types} ({side}, direction={direction}) ===")
        print(f"  neurons: {n_total}  located: {n_located}  unresolved: {n_total - n_located}")
        print(f"  output: {out_path}")


if __name__ == "__main__":
    main()
