"""Verify the cell types that HAVE recorded data in temporal_filtering, and
check whether each one is column-assigned in FAFB.

"Cells with data" are the 13 types returned by Medulla_Library.read_RecF_data()
(ImpR_data / RecF_data, shape (13, 45)); they are listed as ``cell_list`` in
SimulationCode/Medulla_Library.py. These are the only cells the model is fitted
against (the other 52 of the 65 ctype entries are connectivity-only).

For each data cell type this reports, per hemisphere:
  - whether the type exists in FAFB visual_neuron_types,
  - how many neurons it has,
  - how many (and what fraction) have a direct column_assignment.

Run with the project venv:

    .venv/bin/python "Connectome/FAFB v783/test/verify_data_celltypes_columns.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from build_network import FafbDataLoader  # noqa: E402

# temporal_filtering/SimulationCode/Medulla_Library.py, cell_list (the 13 cells
# with measured impulse-response data).
DATA_CELL_TYPES = [
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
        for cell_type in DATA_CELL_TYPES:
            ids = set(neurons[neurons["type"] == cell_type]["root_id"])
            with_col = len(ids & col_ids)
            n = len(ids)
            rows.append({
                "cell_type": cell_type,
                "in_fafb": n > 0,
                "n_neurons": n,
                "n_with_column": with_col,
                "pct_with_column": round(with_col / n * 100, 1) if n else 0.0,
            })
        table = pd.DataFrame(rows)

        print(f"\n{'='*60}\n{side.upper()}\n{'='*60}")
        print(table.to_string(index=False))

        missing = table[~table["in_fafb"]]["cell_type"].tolist()
        not_full = table[(table["in_fafb"]) & (table["pct_with_column"] < 100)]
        all_present = len(missing) == 0
        all_columned = all_present and (table["n_with_column"] == table["n_neurons"]).all()

        print(f"\n  all 13 present in FAFB        : {all_present}"
              + (f"  (missing: {missing})" if missing else ""))
        print(f"  all neurons column-assigned  : {all_columned}")
        if len(not_full):
            print("  types with <100% column coverage:")
            print(not_full[["cell_type", "n_neurons", "n_with_column",
                            "pct_with_column"]].to_string(index=False))


if __name__ == "__main__":
    main()
