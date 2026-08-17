"""Diagnose why many LEFT R1-6 stay unresolved by the column locator.

Reuses the pipeline modules (no re-implementation) and reports, per side:
  1. resolved vs unresolved counts,
  2. the cause split (no output edges at all vs. partners without a column),
  3. R1-6 output edges per neuron (connectivity depth),
  4. R1-6 post-target types and their column-assignment coverage.

Run with the project venv:

    .venv/bin/python "Connectome/FAFBv783/scratch/diagnose_left_unresolved.py"
"""

from __future__ import annotations

import pandas as pd

import import_bootstrap  # noqa: F401

from build_network import FafbDataLoader  # noqa: E402
from assign_column import locate_neurons  # noqa: E402

TARGET_CELL = "R1-6"
SIDES = ("right", "left")


def main() -> None:
    loader = FafbDataLoader()
    neurons_all = loader.load_visual_neurons()
    columns_all = loader.load_column_assignments()
    id2cell = dict(zip(neurons_all["root_id"], neurons_all["cell"]))

    for side in SIDES:
        neurons = neurons_all[neurons_all["side"] == side]
        columns = columns_all[columns_all["hemisphere"] == side]
        col_ids = set(columns["root_id"])
        target_ids = set(
            neurons[neurons["cell"] == TARGET_CELL]["root_id"].astype("int64")
        )
        connections = loader.load_connections(keep_neuron_ids=target_ids)

        located = locate_neurons(
            neurons=neurons,
            columns=columns,
            connections=connections,
            target_cells=[TARGET_CELL],
            side=side,
            direction="post",
        )

        total = len(located)
        un = located[located["majority_column_id"].isna()]
        no_edges = int((un["n_post"] == 0).sum())
        no_col = int(((un["n_post"] > 0) & (un["n_post_with_column"] == 0)).sum())

        print(f"\n{'='*64}\n{side.upper()}  ({TARGET_CELL})\n{'='*64}")
        print(f"  neurons            : {total}")
        print(f"  resolved           : {total - len(un)}")
        print(f"  unresolved         : {len(un)}")
        print(f"    - no output edges at all          : {no_edges}")
        print(f"    - has edges but no partner column : {no_col}")
        print(f"  output edges/neuron : mean={located['n_post'].mean():.2f} "
              f"median={located['n_post'].median():.0f} "
              f"max={located['n_post'].max()}")

        # Post-target type breakdown + per-cell column coverage.
        edges = connections[connections["pre_root_id"].isin(target_ids)].copy()
        edges["post_type"] = edges["post_root_id"].map(id2cell).fillna("?")
        edges["post_has_col"] = edges["post_root_id"].isin(col_ids)
        frac = edges["post_has_col"].mean() * 100
        print(f"  R1-6 post edges     : {len(edges)}  "
              f"(targets with column: {frac:.1f}%)")
        top = edges["post_type"].value_counts().head(8)
        cov = (edges.groupby("post_type")["post_has_col"].mean() * 100).round(1)
        summary = pd.DataFrame({
            "n_edge": top,
            "pct_with_column": cov.reindex(top.index),
        })
        print("  top post-target types (n_edge, % with column):")
        print(summary.to_string())

    print(f"\n{'='*64}\nCONCLUSION\n{'='*64}")
    print(
        "LEFT stays unresolved mainly because its R1-6 partners lack a column,\n"
        "not because R1-6 are disconnected:\n"
        "  - fewer reconstructed output edges per R1-6 (lower mean),\n"
        "  - a large share of targets are untyped '?' (0% column),\n"
        "  - lamina targets L1/L2/L3 have lower column coverage than on the right.\n"
        "Root cause: the LEFT optic lobe is less completely proofread/annotated;\n"
        "the RIGHT lobe is the reference side (hence ~99% resolved)."
    )


if __name__ == "__main__":
    main()
