"""Load + filter the FAFB visual subnetwork and assemble the network JSON.

This single, self-contained module merges the data layer and the network build:

  1. Load the three raw FAFB CSVs (download/) and filter to one hemisphere with
     ``min_neuron_count`` (type cut) and ``min_syn_count`` (weak-edge cut),
     writing <side>_min_neuron<N>/{neurons,columns,connections}.csv.gz etc.
  2. Assemble nodes + edges into <side>_min_neuron<N>/network.json, using the
     hex map (column_id_hex_index_<side>.csv from hex_grid.py) and the R1-6
     placement (location_r1_6_<side>.csv from column_locator.py). Column
     position is OPTIONAL: neurons without a column become nodes with null u/v.

It does not import the other project scripts. Run with the project venv:

    .venv/bin/python "Connectome/FAFB v783/build_network.py" --side right --min-neuron-count 1
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import pandas as pd

import fafb_io
from fafb_io import DATA_DIR

logger = logging.getLogger(__name__)

# -- Build defaults (data-layer paths/loaders live in fafb_io) -----------------

# Hemisphere to build by default.
DEFAULT_SIDE = "right"
# A cell type is kept only if it has at least this many neurons (the type cut).
DEFAULT_MIN_NEURON_COUNT = 1
# Connection rows with fewer synapses than this are discarded.
DEFAULT_MIN_SYN_COUNT = 5
# Optic-lobe neuropil stems; the side suffix (_L / _R) is appended at load time.
VISUAL_NEUROPIL_STEMS = ("ME", "LO", "LOP", "LA")

# Hex-disc radius (extent=15 -> 721 columns).
EXTENT = 15
# Neurotransmitter -> synapse sign. Glutamate is inhibitory (Drosophila GluClalpha).
NT_TO_SIGN = {"ACH": 1.0, "GLUT": -1.0, "GABA": -1.0, "SER": 1.0, "DA": 1.0, "OCT": 1.0}
# Photoreceptors are histaminergic (inhibitory) but FAFB lacks a histamine class,
# so their sign is forced negative regardless of the predicted nt.
FORCED_NEGATIVE_PRE_TYPES = {"R1-6", "R7", "R8"}
# Types treated as network inputs (photoreceptors).
INPUT_TYPES = {"R1-6", "R7", "R8"}
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
    # Per-type table over the (pre-cut) side: type, count, family, subsystem, category.
    type_table: pd.DataFrame
    metadata: Dict[str, object] = field(default_factory=dict)

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Write the filtered subnetwork to <side>_min_neuron<N>/ and return it."""
        if output_dir is not None:
            out = Path(output_dir)
        else:
            name = f"{self.metadata['side']}_min_neuron{self.metadata['min_neuron_count']}"
            out = DATA_DIR / name
        out.mkdir(parents=True, exist_ok=True)

        self.neurons.to_csv(out / "neurons.csv.gz", index=False, compression="gzip")
        self.columns.to_csv(out / "columns.csv.gz", index=False, compression="gzip")
        self.connections.to_csv(out / "connections.csv.gz", index=False, compression="gzip")

        type_table = self.type_table
        type_table.sort_values("count", ascending=False, kind="stable")[
            ["type", "count"]
        ].to_csv(out / "type_counts.csv", index=False)
        type_table.sort_values("type", kind="stable").to_csv(
            out / "type_counts_abc.csv", index=False
        )

        with open(out / "metadata.json", "w") as fh:
            json.dump(self.metadata, fh, indent=2)

        logger.info("Saved filtered visual system to %s", out)
        return out


class FafbDataLoader:
    """Filters the FAFB visual subnetwork (raw I/O delegated to fafb_io)."""

    def load_visual_neurons(self) -> pd.DataFrame:
        return fafb_io.load_visual_neurons()

    def load_column_assignments(self) -> pd.DataFrame:
        return fafb_io.load_column_assignments()

    def load_connections(
        self,
        keep_neuron_ids: Optional[set] = None,
        keep_neuropils: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return fafb_io.load_connections(keep_neuron_ids, keep_neuropils)

    def filter_visual_system(
        self,
        side: str = DEFAULT_SIDE,
        subsystems: Optional[Sequence[str]] = None,
        min_neuron_count: int = DEFAULT_MIN_NEURON_COUNT,
        min_syn_count: int = DEFAULT_MIN_SYN_COUNT,
        use_cache: bool = True,
    ) -> VisualSystem:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

        # Cache the filtered subnetwork inside its run folder so the expensive
        # raw-CSV streaming runs once. Only the default-subsystem path is cached.
        cache_path: Optional[Path] = None
        if subsystems is None:
            cache_path = (
                DATA_DIR / f"{side}_min_neuron{min_neuron_count}" / ".filter_cache"
            )
            if use_cache and cache_path.exists():
                logger.info("Loading filtered visual system from cache %s", cache_path)
                with open(cache_path, "rb") as fh:
                    return pickle.load(fh)

        neurons = self.load_visual_neurons()
        neurons = neurons[neurons["side"] == side]
        if subsystems:
            neurons = neurons[neurons["subsystem"].isin(list(subsystems))]
        logger.info("After side=%s + subsystem filter: %d neurons", side, len(neurons))

        type_counts_unfiltered = neurons["type"].value_counts()
        attr_cols = ["family", "subsystem", "category"]
        type_table = neurons.groupby("type")[attr_cols].first()
        type_table.insert(0, "count", type_counts_unfiltered)
        type_table = type_table.rename_axis("type").reset_index()
        n_types_before = neurons["type"].nunique()
        if min_neuron_count > 0:
            keep_types = type_counts_unfiltered[
                type_counts_unfiltered >= min_neuron_count
            ].index
            neurons = neurons[neurons["type"].isin(keep_types)].copy()
        logger.info(
            "min_neuron_count=%d: types %d -> %d, neurons -> %d",
            min_neuron_count, n_types_before, neurons["type"].nunique(), len(neurons),
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
            "n_cell_types": int(neurons["type"].nunique()),
            "n_columns": int(columns["column_id"].nunique()),
            "n_connections": len(connections),
        }
        vs = VisualSystem(
            neurons=neurons,
            columns=columns,
            connections=connections,
            type_table=type_table,
            metadata=metadata,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(vs, fh)
            logger.info("Cached filtered visual system to %s", cache_path)
        return vs


# =============================================================================
# Network build: nodes + edges -> network.json
# =============================================================================


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}. Run hex_grid.py and column_locator.py first, "
            "or use build_all.py."
        )
    return path


def _column_to_hex(side: str) -> Dict[int, Tuple[int, int, int]]:
    """Map column_id -> (u, v, hex_index) for columns inside the hex disc."""
    df = pd.read_csv(_require(DATA_DIR / f"column_id_hex_index_{side}.csv"))
    df = df[df["hex_status"] == "inside"]
    return {
        int(r.column_id): (int(r.u), int(r.v), int(r.hex_index))
        for r in df.itertuples(index=False)
    }


def _sign_per_pre(connections: pd.DataFrame) -> Dict[int, float]:
    """Synapse-weighted dominant nt -> sign, one value per presynaptic neuron."""
    w = connections.groupby(["pre_root_id", "nt_type"])["syn_count"].sum().reset_index()
    w = w.sort_values(["pre_root_id", "syn_count"], ascending=[True, False])
    dom = w.groupby("pre_root_id").first()
    return {int(rid): NT_TO_SIGN.get(str(nt), 1.0) for rid, nt in dom["nt_type"].items()}


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
    return NT_TO_SIGN.get(str(dom_nt.get((pre_id, post_id), "ACH")), 1.0)


def build(side: str, min_neuron_count: int) -> Path:
    """Assemble network.json for one (side, min_neuron_count) run folder."""
    run_dir = DATA_DIR / f"{side}_min_neuron{min_neuron_count}"
    neurons = pd.read_csv(_require(run_dir / "neurons.csv.gz"))
    columns = pd.read_csv(_require(run_dir / "columns.csv.gz"))
    connections = pd.read_csv(_require(run_dir / "connections.csv.gz"))
    col_hex = _column_to_hex(side)

    kept_ids: Set[int] = set(neurons["root_id"].astype("int64"))
    id_to_type = dict(zip(neurons["root_id"].astype("int64"), neurons["type"].astype(str)))

    # Column position is OPTIONAL: column-assigned neurons + located R1-6.
    pos: Dict[int, Tuple[int, int, int]] = {}
    for r in columns.itertuples(index=False):
        rid = int(r.root_id)
        if rid not in kept_ids or rid in pos:
            continue
        uvh = col_hex.get(int(r.column_id))
        if uvh is not None:
            pos[rid] = uvh

    loc = pd.read_csv(_require(DATA_DIR / f"location_r1_6_{side}.csv"))
    loc = loc[loc["majority_column_id"].notna()]
    for r in loc.itertuples(index=False):
        rid = int(r.root_id)
        if rid not in kept_ids or rid in pos:
            continue
        uvh = col_hex.get(int(r.majority_column_id))
        if uvh is not None:
            pos[rid] = uvh

    nodes = []
    for rid in kept_ids:
        typ = id_to_type[rid]
        u, v, h = pos.get(rid, (None, None, None))
        nodes.append({
            "id": rid, "name": typ, "u": u, "v": v, "hex_index": h,
            "input": typ in INPUT_TYPES, "output": False,
        })
    logger.info(
        "Nodes: %d (%d with column position, %d without)",
        len(nodes), len(pos), len(nodes) - len(pos),
    )

    conn = connections[
        connections["pre_root_id"].isin(kept_ids)
        & connections["post_root_id"].isin(kept_ids)
    ].copy()
    agg_syn = conn.groupby(["pre_root_id", "post_root_id"], sort=False)["syn_count"].sum()

    if SIGN_MODE == "per_pre":
        pre_sign = _sign_per_pre(conn)
        dom_nt: Dict[Tuple[int, int], str] = {}
    else:
        pre_sign = {}
        dom_nt = _dominant_nt_per_edge(conn)

    edges = []
    for (pre_id, post_id), n_syn in agg_syn.items():
        pre_id, post_id = int(pre_id), int(post_id)
        st = id_to_type[pre_id]
        if st in FORCED_NEGATIVE_PRE_TYPES:
            sign = -1.0
        elif SIGN_MODE == "per_pre":
            sign = pre_sign.get(pre_id, 1.0)
        else:
            sign = _sign_per_edge(pre_id, post_id, dom_nt)
        sp = pos.get(pre_id)
        tp = pos.get(post_id)
        if sp is not None and tp is not None:
            du, dv = int(tp[0] - sp[0]), int(tp[1] - sp[1])
        else:
            du, dv = None, None
        edges.append({
            "src": pre_id, "tar": post_id, "sign": sign, "n_syn": float(n_syn),
            "source_type": st, "target_type": id_to_type[post_id],
            "du": du, "dv": dv,
        })
    logger.info("Built %d edges", len(edges))

    payload = {
        "metadata": {
            "side": side,
            "min_neuron_count": min_neuron_count,
            "extent": EXTENT,
            "sign_mode": SIGN_MODE,
            "nt_to_sign": NT_TO_SIGN,
            "forced_negative_pre_types": sorted(FORCED_NEGATIVE_PRE_TYPES),
            "n_nodes": len(nodes),
            "n_nodes_with_column": len(pos),
            "n_edges": len(edges),
            "n_input_nodes": int(sum(n["input"] for n in nodes)),
            "n_cell_types": int(len({n["name"] for n in nodes})),
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


def crop_network(run_dir: Path, crop_extent: int, recenter: bool = True) -> Path:
    """Crop a built network.json to the central hex disc of ``crop_extent``.

    Keeps only column-positioned nodes whose hex distance from the centre is
    ``<= crop_extent`` (and the edges between them), writing a sibling run folder
    ``<run_dir.name>_extent<crop_extent>_col/network.json``. This is the small,
    multi-column training input (extent=2 -> 19 columns).

    Args:
        run_dir: An existing run folder containing network.json.
        crop_extent: Hex-disc radius to keep around the centre (2 -> 19 columns).
        recenter: If True, re-index ``hex_index`` against a fresh
            ``HexGrid(crop_extent)`` so indices run 0..N-1 for the small disc.
    """
    # Reuse the lattice math from hex_grid (single source of truth).
    from hex_grid import HexGrid, hex_radius

    src = _require(run_dir / "network.json")
    payload = json.load(open(src))
    nodes = payload["nodes"]
    edges = payload["edges"]

    kept_nodes = [
        n for n in nodes
        if n.get("u") is not None and hex_radius(n["u"], n["v"]) <= crop_extent
    ]
    kept_ids: Set[int] = {n["id"] for n in kept_nodes}
    kept_edges = [
        e for e in edges if e["src"] in kept_ids and e["tar"] in kept_ids
    ]

    if recenter:
        grid = HexGrid(extent=crop_extent)
        for n in kept_nodes:
            n["hex_index"] = grid.uv_to_hex_index(n["u"], n["v"])

    n_with_col = sum(1 for n in kept_nodes if n.get("u") is not None)
    src_meta = payload.get("metadata", {})
    metadata: Dict[str, object] = {
        "side": src_meta.get("side"),
        "min_neuron_count": src_meta.get("min_neuron_count"),
        "extent": crop_extent,
        "cropped_from": run_dir.name,
        "source_extent": src_meta.get("extent"),
        "sign_mode": src_meta.get("sign_mode"),
        "nt_to_sign": src_meta.get("nt_to_sign"),
        "forced_negative_pre_types": src_meta.get("forced_negative_pre_types"),
        "n_nodes": len(kept_nodes),
        "n_nodes_with_column": n_with_col,
        "n_edges": len(kept_edges),
        "n_input_nodes": int(sum(bool(n.get("input")) for n in kept_nodes)),
        "n_cell_types": int(len({n["name"] for n in kept_nodes})),
    }

    out_dir = run_dir.parent / f"{run_dir.name}_extent{crop_extent}_col"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "network.json"
    with open(out_path, "w") as fh:
        json.dump({"metadata": metadata, "nodes": kept_nodes, "edges": kept_edges}, fh)
    logger.info(
        "Cropped %s -> %s (extent=%d): %d nodes, %d edges, %d types",
        run_dir.name, out_path, crop_extent,
        len(kept_nodes), len(kept_edges), metadata["n_cell_types"],
    )
    _write_summary(out_dir, metadata)
    return out_path


def _write_summary(run_dir: Path, meta: Dict[str, object]) -> Path:
    """Write a human-readable summary.txt of the node/edge/type stats."""
    # Filter-level stats (min_syn_count, columns, raw connections) from save().
    filt: Dict[str, object] = {}
    meta_json = run_dir / "metadata.json"
    if meta_json.exists():
        filt = json.load(open(meta_json))

    n_nodes = int(meta["n_nodes"])
    n_with_col = int(meta["n_nodes_with_column"])
    lines = [
        f"network summary: {run_dir.name}",
        "=" * 40,
        f"side                 : {meta['side']}",
        f"min_neuron_count     : {meta['min_neuron_count']}",
        f"min_syn_count        : {filt.get('min_syn_count')}",
        f"extent               : {meta['extent']}",
        f"sign_mode            : {meta['sign_mode']}",
        "",
        f"n_nodes              : {n_nodes}",
        f"n_nodes_with_column  : {n_with_col}",
        f"n_nodes_without_col  : {n_nodes - n_with_col}",
        f"n_input_nodes        : {meta['n_input_nodes']}",
        f"n_edges              : {meta['n_edges']}",
        f"n_cell_types         : {meta['n_cell_types']}",
        "",
        f"n_columns (assigned) : {filt.get('n_columns')}",
        f"n_connections (raw)  : {filt.get('n_connections')}",
        f"forced_negative      : {', '.join(meta['forced_negative_pre_types'])}",
        f"nt_to_sign           : {meta['nt_to_sign']}",
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
    parser.add_argument(
        "--skip-filter", action="store_true",
        help="Skip load+filter; build only from existing run folders.",
    )
    parser.add_argument(
        "--refresh-cache", action="store_true",
        help="Ignore the filter cache and recompute from the raw CSVs.",
    )
    parser.add_argument(
        "--crop-extent", type=int, default=None,
        help="After building, crop network.json to the central hex disc of this "
             "radius, writing <run>_extent<N>_col/ (e.g. 2 -> 19 columns).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    sides = ["left", "right"] if args.side == "both" else [args.side]

    loader = None if args.skip_filter else FafbDataLoader()
    for side in sides:
        if loader is not None:
            vs = loader.filter_visual_system(
                side=side,
                min_neuron_count=args.min_neuron_count,
                min_syn_count=args.min_syn_count,
                use_cache=not args.refresh_cache,
            )
            vs.save()
        out = build(side, args.min_neuron_count)
        meta = json.load(open(out))["metadata"]
        print(f"\n=== build_network ({side}, min_neuron={args.min_neuron_count}) ===")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print(f"  output: {out}")

        if args.crop_extent is not None:
            crop_out = crop_network(out.parent, args.crop_extent)
            crop_meta = json.load(open(crop_out))["metadata"]
            print(f"\n=== crop (extent={args.crop_extent}) ===")
            for k, v in crop_meta.items():
                print(f"  {k}: {v}")
            print(f"  output: {crop_out}")


if __name__ == "__main__":
    main()
