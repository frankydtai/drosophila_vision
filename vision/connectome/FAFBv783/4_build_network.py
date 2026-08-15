"""Load + filter the FAFB visual subnetwork and assemble the network JSON.

This single, self-contained module merges the data layer and the network build:

  1. Load the three raw FAFB CSVs (downloads/) and filter to one hemisphere with
     ``min_neuron_count`` (cell cut) and ``min_syn_count`` (weak-edge cut),
     writing <side>_min_neuron<N>/{neurons,columns,connections}.csv.gz etc.
  2. Assemble nodes + edges into <side>_min_neuron<N>/network.json, using the
     column map (column_map_<side>.csv from 2_build_hex.py) and located placements
     from ``ASSIGNED_COLUMN_CELLS`` in assign_column
     (3_assigned_columns/<tag>_<side>_<direction>.csv; missing CSVs are built by
     running 3_assign_column.py). Column position is OPTIONAL: neurons without a
     column become nodes with null u/v.

Spatial cropping to a central hex disc is ``5_apply_radius.py``, not this script.

Run with the project venv:

    .venv/bin/python "connectome/FAFBv783/4_build_network.py" --side right --min-neuron-count 1
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

import path
from assign_column import ASSIGNED_COLUMN_CELLS
from path import BUILT_NETWORKS_DIR

logger = logging.getLogger(__name__)

# -- Build defaults (data-layer paths/loaders live in path) -----------------

# Hemisphere to build by default.
DEFAULT_SIDE = "right"
# A cell is kept only if it has at least this many neurons (the cell cut).
DEFAULT_DEFAULT_MIN_NEURON_COUNT = 1
# Connection rows with fewer synapses than this are discarded.
DEFAULT_DEFAULT_MIN_SYN_COUNT = 5
# Optic-lobe neuropil stems; the side suffix (_L / _R) is appended at load time.
VISUAL_NEUROPIL_STEMS = ("ME", "LO", "LOP", "LA")
# ASSIGNED_COLUMN_CELLS: sole list lives in assign_column (3_assign_column.py).
_ASSIGN_COLUMN_SCRIPT = Path(__file__).resolve().parent / "3_assign_column.py"

# Neurotransmitter -> synapse sign. Glutamate is inhibitory (Drosophila GluClalpha).
SIGN_FROM_NT = {"ACH": 1.0, "GLUT": -1.0, "GABA": -1.0, "SER": 1.0, "DA": 1.0, "OCT": 1.0}
# Photoreceptors are histaminergic (inhibitory) but FAFB lacks a histamine class,
# so their sign is forced negative regardless of the predicted nt.
FORCED_NEGATIVE_PRE_CELLS = {"R1-6", "R7", "R8"}
# Cells that receive light sti (R1-6, R7, R8).
STI_CELLS = {"R1-6", "R7", "R8"}
# Per-edge sign rule: "per_edge" (dominant nt per pre/post pair) or
# "per_pre" (one sign per presynaptic neuron, Dale's principle).
SIGN_MODE = "per_edge"


# =============================================================================
# Data layer: load + filter + save
# =============================================================================


@dataclass
class VisualSystem:
    """Filtered FAFB visual subnetwork for one hemisphere."""

    neurons: pd.DataFrame
    columns: pd.DataFrame
    connections: pd.DataFrame
    # Per-cell table over the (pre-cut) side: type, count, family, subsystem, category.
    cell_table: pd.DataFrame
    metadata: Dict[str, object] = field(default_factory=dict)

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Write the filtered subnetwork to <side>_min_neuron<N>/ and return it."""
        if output_dir is not None:
            out = Path(output_dir)
        else:
            name = f"{self.metadata['side']}_min_neuron{self.metadata['min_neuron_count']}"
            out = BUILT_NETWORKS_DIR / name
        out.mkdir(parents=True, exist_ok=True)

        self.neurons.to_csv(out / "neurons.csv.gz", index=False, compression="gzip")
        self.columns.to_csv(out / "columns.csv.gz", index=False, compression="gzip")
        self.connections.to_csv(out / "connections.csv.gz", index=False, compression="gzip")

        cell_table = self.cell_table
        cell_table.sort_values("count", ascending=False, kind="stable")[
            ["cell", "count"]
        ].to_csv(out / "cell_counts.csv", index=False)
        cell_table.sort_values("cell", kind="stable").to_csv(
            out / "cell_counts_abc.csv", index=False
        )

        with open(out / "metadata.json", "w") as fh:
            json.dump(self.metadata, fh, indent=2)

        logger.info("Saved filtered visual system to %s", out)
        return out


class FafbDataLoader:
    """Filters the FAFB visual subnetwork (raw I/O delegated to path)."""

    def load_visual_neurons(self) -> pd.DataFrame:
        return path.load_visual_neurons()

    def load_column_assignments(self) -> pd.DataFrame:
        return path.load_column_assignments()

    def load_connections(
        self,
        keep_neuron_ids: Optional[set] = None,
        keep_neuropils: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return path.load_connections(keep_neuron_ids, keep_neuropils)

    def filter_visual_system(
        self,
        side: str = DEFAULT_SIDE,
        subsystems: Optional[Sequence[str]] = None,
        min_neuron_count: int = DEFAULT_MIN_NEURON_COUNT,
        min_syn_count: int = DEFAULT_MIN_SYN_COUNT,
    ) -> VisualSystem:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

        neurons = self.load_visual_neurons()
        neurons = neurons[neurons["side"] == side]
        if subsystems:
            neurons = neurons[neurons["subsystem"].isin(list(subsystems))]
        logger.info("After side=%s + subsystem filter: %d neurons", side, len(neurons))

        cell_counts_unfiltered = neurons["cell"].value_counts()
        attr_cols = ["family", "subsystem", "category"]
        cell_table = neurons.groupby("cell")[attr_cols].first()
        cell_table.insert(0, "count", cell_counts_unfiltered)
        cell_table = cell_table.rename_axis("cell").reset_index()
        n_cells_before = neurons["cell"].nunique()
        if min_neuron_count > 0:
            keep_cells = cell_counts_unfiltered[
                cell_counts_unfiltered >= min_neuron_count
            ].index
            neurons = neurons[neurons["cell"].isin(keep_cells)].copy()
        logger.info(
            "min_neuron_count=%d: cells %d -> %d, neurons -> %d",
            min_neuron_count, n_cells_before, neurons["cell"].nunique(), len(neurons),
        )

        neuron_ids = set(neurons["root_id"].astype("int64").values)

        columns = self.load_column_assignments()
        columns = columns[columns["hemisphere"] == side]
        columns = columns[columns["root_id"].isin(neuron_ids)].copy()
        logger.info(
            "Column assignments for kept neurons: %d rows, %d columns",
            len(columns), columns["column_id"].nunique(),
        )

        side_suffix = "L" if side == "left" else "R"
        neuropils = [f"{stem}_{side_suffix}" for stem in VISUAL_NEUROPIL_STEMS]
        connections = self.load_connections(
            keep_neuron_ids=neuron_ids, keep_neuropils=neuropils
        )
        connections = connections[
            connections["pre_root_id"].isin(neuron_ids)
            & connections["post_root_id"].isin(neuron_ids)
        ].copy()
        n_conn_before_syn = len(connections)
        if min_syn_count > 0:
            connections = connections[connections["syn_count"] >= min_syn_count].copy()
        logger.info(
            "Connections within kept neurons: %d, after min_syn_count=%d: %d",
            n_conn_before_syn, min_syn_count, len(connections),
        )

        subsystem_list = (
            list(subsystems) if subsystems
            else sorted(neurons["subsystem"].dropna().unique().tolist())
        )
        metadata: Dict[str, object] = {
            "side": side,
            "subsystems": subsystem_list,
            "min_neuron_count": min_neuron_count,
            "min_syn_count": min_syn_count,
            "n_neurons": len(neurons),
            "n_cells": int(neurons["cell"].nunique()),
            "n_columns": int(columns["column_id"].nunique()),
            "n_connections": len(connections),
        }
        return VisualSystem(
            neurons=neurons,
            columns=columns,
            connections=connections,
            cell_table=cell_table,
            metadata=metadata,
        )


# =============================================================================
# Network build: nodes + edges -> network.json
# =============================================================================


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}. Run 2_build_hex.py and 3_assign_column.py first, "
            "or use build_all.py."
        )
    return path


def _assigned_column_csv(side: str, cell: str, direction: str) -> Path:
    """``3_assigned_columns/<tag>_<side>_<direction>.csv`` (tag matches assign_column)."""
    tag = re.sub(r"[^0-9a-z]+", "_", cell.lower()).strip("_")
    return path.ASSIGNED_COLUMNS_DIR / f"{tag}_{side}_{direction}.csv"


def _ensure_assigned_column_csv(side: str, cell: str, direction: str) -> Path:
    """Return the located-column CSV, running ``3_assign_column.py`` if missing."""
    out = _assigned_column_csv(side, cell, direction)
    if out.exists():
        return out
    cmd = [sys.executable, str(_ASSIGN_COLUMN_SCRIPT), cell, "--side", side]
    if direction == "post":
        cmd.append("--post")
    logger.info("Missing %s; running: %s", out.name, " ".join(cmd))
    subprocess.run(cmd, check=True)
    return _require(out)


def pos_from_column(side: str) -> Dict[int, Tuple[int, int]]:
    """Map column_id -> (u, v) for every positioned FAFB column.

    The base build is always the full graph (no spatial cap). Spatial cropping is
    ``5_apply_radius.py``.
    """
    _require(path.column_map_path(side))
    df = path.load_column_map(side)
    return {
        int(r.column_id): (int(r.u), int(r.v))
        for r in df.itertuples(index=False)
    }


def _sign_per_pre(connections: pd.DataFrame) -> Dict[int, float]:
    """Synapse-weighted dominant nt -> sign, one value per presynaptic neuron."""
    w = connections.groupby(["pre_root_id", "nt_type"])["syn_count"].sum().reset_index()
    w = w.sort_values(["pre_root_id", "syn_count"], ascending=[True, False])
    dom = w.groupby("pre_root_id").first()
    return {int(rid): SIGN_FROM_NT.get(str(nt), 1.0) for rid, nt in dom["nt_type"].items()}


def _dominant_nt_per_edge(connections: pd.DataFrame) -> Dict[Tuple[int, int], str]:
    """Most frequent nt_type per (pre, post) pair (vectorized; per-edge mode)."""
    g = (
        connections.groupby(["pre_root_id", "post_root_id", "nt_type"])
        .size()
        .reset_index(name="n")
        .sort_values("n")
        .groupby(["pre_root_id", "post_root_id"], sort=False)
        .tail(1)
    )
    return {
        (int(r.pre_root_id), int(r.post_root_id)): str(r.nt_type)
        for r in g.itertuples(index=False)
    }


def _sign_per_edge(pre_id: int, post_id: int, dom_nt: Dict[Tuple[int, int], str]) -> float:
    return SIGN_FROM_NT.get(str(dom_nt.get((pre_id, post_id), "ACH")), 1.0)


def build(side: str, min_neuron_count: int) -> Path:
    """Assemble the full network.json for one (side, min_neuron_count) run folder.

    Always keeps every positioned FAFB column (no spatial cap); cropping to a
    central disc is ``5_apply_radius.py``.
    """
    run_dir = BUILT_NETWORKS_DIR / f"{side}_min_neuron{min_neuron_count}"
    neurons = pd.read_csv(_require(run_dir / "neurons.csv.gz"))
    columns = pd.read_csv(_require(run_dir / "columns.csv.gz"))
    connections = pd.read_csv(_require(run_dir / "connections.csv.gz"))
    col_pos = pos_from_column(side)

    kept_ids: Set[int] = set(neurons["root_id"].astype("int64"))
    cell_from_id = dict(zip(neurons["root_id"].astype("int64"), neurons["cell"].astype(str)))

    # Column position is OPTIONAL: column-assigned neurons + located types.
    # pos maps root_id -> (u, v, column_id). Native assignment wins over located.
    pos: Dict[int, Tuple[int, int, int]] = {}
    for r in columns.itertuples(index=False):
        rid = int(r.root_id)
        if rid not in kept_ids or rid in pos:
            continue
        cid = int(r.column_id)
        uv = col_pos.get(cid)
        if uv is not None:
            pos[rid] = (uv[0], uv[1], cid)

    for cell, direction in ASSIGNED_COLUMN_CELLS:
        loc = pd.read_csv(_ensure_assigned_column_csv(side, cell, direction))
        loc = loc[loc["majority_column_id"].notna()]
        for r in loc.itertuples(index=False):
            rid = int(r.root_id)
            if rid not in kept_ids or rid in pos:
                continue
            cid = int(r.majority_column_id)
            uv = col_pos.get(cid)
            if uv is not None:
                pos[rid] = (uv[0], uv[1], cid)

    nodes = []
    for rid in kept_ids:
        cell = cell_from_id[rid]
        u, v, cid = pos.get(rid, (None, None, None))
        nodes.append({
            "id": rid, "name": cell, "u": u, "v": v, "column_id": cid,
            "sti": cell in STI_CELLS, "output": False,
        })
    logger.info(
        "Nodes: %d (%d with column position, %d without)",
        len(nodes), len(pos), len(nodes) - len(pos),
    )

    conn = connections[
        connections["pre_root_id"].isin(kept_ids)
        & connections["post_root_id"].isin(kept_ids)
    ].copy()
    n_syn_by_pair = conn.groupby(["pre_root_id", "post_root_id"], sort=False)["syn_count"].sum()

    if SIGN_MODE == "per_pre":
        pre_sign = _sign_per_pre(conn)
        dom_nt: Dict[Tuple[int, int], str] = {}
    else:
        pre_sign = {}
        dom_nt = _dominant_nt_per_edge(conn)

    edges = []
    for (pre_id, post_id), n_syn in n_syn_by_pair.items():
        pre_id, post_id = int(pre_id), int(post_id)
        st = cell_from_id[pre_id]
        if st in FORCED_NEGATIVE_PRE_CELLS:
            syn_sign = -1.0
        elif SIGN_MODE == "per_pre":
            syn_sign = pre_sign.get(pre_id, 1.0)
        else:
            syn_sign = _sign_per_edge(pre_id, post_id, dom_nt)
        sp = pos.get(pre_id)
        tp = pos.get(post_id)
        if sp is not None and tp is not None:
            du, dv = int(tp[0] - sp[0]), int(tp[1] - sp[1])
        else:
            du, dv = None, None
        edges.append({
            "src": pre_id, "tar": post_id, "syn_sign": syn_sign, "n_syn": float(n_syn),
            "source_cell": st, "target_cell": cell_from_id[post_id],
            "du": du, "dv": dv,
        })
    logger.info("Built %d edges", len(edges))

    payload = {
        "metadata": {
            "side": side,
            "min_neuron_count": min_neuron_count,
            "sign_mode": SIGN_MODE,
            "sign_from_nt": SIGN_FROM_NT,
            "forced_negative_pre_cells": sorted(FORCED_NEGATIVE_PRE_CELLS),
            "n_nodes": len(nodes),
            "n_nodes_with_column": len(pos),
            "n_edges": len(edges),
            "n_sti_nodes": int(sum(n["sti"] for n in nodes)),
            "n_cells": int(len({n["name"] for n in nodes})),
        },
        "nodes": nodes,
        "edges": edges,
    }
    out_path = run_dir / "network.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    logger.info("Wrote %s", out_path)

    _write_summary(run_dir, payload["metadata"])
    return out_path


def _write_summary(run_dir: Path, meta: Dict[str, object]) -> Path:
    """Write a human-readable summary.txt of the node/edge/type stats."""
    # Filter-level stats (min_syn_count, columns, raw connections) from save().
    filt: Dict[str, object] = {}
    meta_json = run_dir / "metadata.json"
    if meta_json.exists():
        filt = json.load(open(meta_json))

    n_nodes = int(meta["n_nodes"])
    n_with_column = int(meta["n_nodes_with_column"])
    lines = [
        f"network summary: {run_dir.name}",
        "=" * 40,
        f"side                 : {meta['side']}",
        f"min_neuron_count     : {meta['min_neuron_count']}",
        f"min_syn_count        : {filt.get('min_syn_count')}",
        f"sign_mode            : {meta['sign_mode']}",
        "",
        f"n_nodes              : {n_nodes}",
        f"n_nodes_with_column  : {n_with_column}",
        f"n_nodes_without_column  : {n_nodes - n_with_column}",
        f"n_sti_nodes        : {meta['n_sti_nodes']}",
        f"n_edges              : {meta['n_edges']}",
        f"n_cells         : {meta['n_cells']}",
        "",
        f"n_columns (assigned) : {filt.get('n_columns')}",
        f"n_connections (raw)  : {filt.get('n_connections')}",
        f"forced_negative      : {', '.join(meta['forced_negative_pre_cells'])}",
        f"sign_from_nt           : {meta['sign_from_nt']}",
        "",
    ]
    out = run_dir / "summary.txt"
    out.write_text("\n".join(lines))
    logger.info("Wrote %s", out)
    return out


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load+filter FAFB and assemble the network JSON."
    )
    parser.add_argument("--side", default=DEFAULT_SIDE, choices=["left", "right", "both"])
    parser.add_argument("--min-neuron-count", type=int, default=DEFAULT_MIN_NEURON_COUNT)
    parser.add_argument("--min-syn-count", type=int, default=DEFAULT_MIN_SYN_COUNT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    sides = ["left", "right"] if args.side == "both" else [args.side]

    loader = FafbDataLoader()
    for side in sides:
        vs = loader.filter_visual_system(
            side=side,
            min_neuron_count=args.min_neuron_count,
            min_syn_count=args.min_syn_count,
        )
        vs.save()
        out = build(side, args.min_neuron_count)
        meta = json.load(open(out))["metadata"]
        print(f"\n=== build_network ({side}, min_neuron={args.min_neuron_count}) ===")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print(f"  output: {out}")


if __name__ == "__main__":
    main()