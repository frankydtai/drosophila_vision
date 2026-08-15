"""Verify the cells that HAVE recorded gt in vision, and
check whether each one is column-assigned in FAFB.

"Cells with gt" are the 13 cells returned by Medulla_Library.read_RecF_gt()
(ImpR_gt / RecF_gt, shape (13, 45)); they are listed as ``GT_CELLS`` below
(legacy Medulla_Library attribute was ``cell_list``). These are the only cells the model is fitted
against (the other 52 of the 65 cell entries are connectivity-only).

For each gt cell this reports, per hemisphere:
  - whether the cell exists in FAFB visual_neuron_types,
  - how many neurons it has,
  - how many (and what fraction) have a direct column_assignment.

Run with the project venv:

    .venv/bin/python "Connectome/FAFBv783/test/verify_gt_celltypes_columns.py"
"""

from __future__ import annotations

import pandas as pd

import import_bootstrap  # noqa: F401

from build_network import FafbDataLoader  # noqa: E402

# vision/SimulationCode/Medulla_Library.py legacy cells (the 13 cells
# with measured impulse-response gt).
GT_CELLS = [
    "L1", "L2", "L3", "L4", "L5",
    "Mi1", "Tm3", "Mi4", "Mi9",
    "Tm1", "Tm2", "Tm4", "Tm9",
]
SIDES = ("right", "left")


def main() -> None:
    loader = FafbDataLoader()
    neurons_all = loader.load_visual_neurons()
    columns_all = loader.load_column_assignments()

    for side in SIDES:
        neurons = neurons_all[neurons_all["side"] == side]
        col_ids = set(columns_all[columns_all["hemisphere"] == side]["root_id"])

        rows = []
        for cell in GT_CELLS:
            ids = set(neurons[neurons["cell"] == cell]["root_id"])
            with_col = len(ids & col_ids)
            n = len(ids)
            rows.append({
                "cell": cell,
                "in_fafb": n > 0,
                "n_neurons": n,
                "n_with_column": with_col,
                "pct_with_column": round(with_col / n * 100, 1) if n else 0.0,
            })
        table = pd.DataFrame(rows)

        print(f"\n{'='*60}\n{side.upper()}\n{'='*60}")
        print(table.to_string(index=False))

        missing = table[~table["in_fafb"]]["cell"].tolist()
        not_full = table[(table["in_fafb"]) & (table["pct_with_column"] < 100)]
        all_active = len(missing) == 0
        all_columned = all_active and (table["n_with_column"] == table["n_neurons"]).all()

        print(f"\n  all 13 active in FAFB         : {all_active}"
              + (f"  (missing: {missing})" if missing else ""))
        print(f"  all neurons column-assigned  : {all_columned}")
        if len(not_full):
            print("  types with <100% column coverage:")
            print(not_full[["cell", "n_neurons", "n_with_column",
                            "pct_with_column"]].to_string(index=False))


if __name__ == "__main__":
    main()
