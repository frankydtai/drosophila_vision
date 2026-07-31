"""List the predicted neurotransmitter (nt_type) of R7 and R8 in FAFB.

R7/R8 are the inner photoreceptors. This reports, per hemisphere, the nt_type
distribution of their outgoing synapses (by edge count and by synapse count),
straight from the raw connections.

Run with the project venv:

    .venv/bin/python "Connectome/FAFBv783/test/r7_r8_nt.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from build_network import FafbDataLoader  # noqa: E402

TYPES = ("R7", "R8")
SIDES = ("right", "left")


def main() -> None:
    loader = FafbDataLoader()
    neurons = loader.load_visual_neurons()

    for side in SIDES:
        side_neurons = neurons[neurons["side"] == side]
        for cell_type in TYPES:
            ids = set(side_neurons[side_neurons["type"] == cell_type]["root_id"].astype("int64"))
            conn = loader.load_connections(keep_neuron_ids=ids)
            conn = conn[conn["pre_root_id"].isin(ids)]
            print(f"\n===== {cell_type} ({side}) =====  "
                  f"neurons={len(ids)} edges={len(conn)} syn={int(conn['syn_count'].sum())}")
            if conn.empty:
                print("  (no outgoing connections)")
                continue
            by_edges = conn["nt_type"].value_counts()
            by_syn = conn.groupby("nt_type")["syn_count"].sum()
            table = pd.DataFrame({
                "n_edges": by_edges,
                "pct_edges": (by_edges / by_edges.sum() * 100).round(1),
                "n_syn": by_syn,
                "pct_syn": (by_syn / by_syn.sum() * 100).round(1),
            }).sort_values("n_syn", ascending=False)
            print(table.to_string())


if __name__ == "__main__":
    main()
