"""
Connectome Loader
=================
Unified interface for loading connectome data from four sources:

  Source              Dataset              File format
  ──────────────────  ───────────────────  ──────────────────────────────
  'banc'              BANC whole-brain     neurons.csv.gz + connections_princeton.csv.gz
  'optic_lobe'        maleCNS optic lobe   *.feather (Jaxley tutorial)
  'male_cns'          Janelia male CNS     *.feather (body annotations + connectome weights)
  'fafb'              FlyWire FAFB v783    consolidated_cell_types.csv.gz +
                                           classification.csv.gz +
                                           neurons.csv.gz +
                                           connections_princeton.csv.gz
  'flyvis'            FlyVis avg-filter    HDF5 files under flyvis/data/connectome/

All sources are normalised into two DataFrames:

  nodes  columns: root_id (int64), cell_type (str), nt_type (str),
                  super_class (str), sub_class (str), side (str),
                  node_idx (int64), [source-specific extras]

  edges  columns: pre_root_id (int64), post_root_id (int64),
                  syn_count (float32), neuropil (str),
                  pre_idx (int64), post_idx (int64)

Factory methods
---------------
    loader = ConnectomeLoader.from_banc()
    loader = ConnectomeLoader.from_optic_lobe()
    loader = ConnectomeLoader.from_fafb(data_dir='path/to/fafb/')
    loader = ConnectomeLoader.from_flyvis()
    nodes, edges = loader.load()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default data paths
# ---------------------------------------------------------------------------
_PKG_ROOT   = Path(__file__).parents[2]   # NIPS2026/
_BANC_DIR   = _PKG_ROOT / "Connectome Dataset" / "Banc"
_OL_DIR     = _PKG_ROOT / "Jaxley_notebook" / "jaxley_tutorial-sjcabs" / "tutorial"
_FLYVIS_DIR = _PKG_ROOT / "flyvis" / "data" / "connectome" / "ConnectomeFromAvgFilters_0000"

# FAFB codex files – place under Connectome Dataset/FAFB (default below).
_FAFB_DIR   = _PKG_ROOT / "Connectome Dataset" / "FAFB"

__all__ = ["ConnectomeLoader"]


@dataclass
class ConnectomeLoader:
    """
    Unified connectome loader.

    Parameters
    ----------
    source : str
        One of ``'banc'``, ``'optic_lobe'``, ``'male_cns'``, ``'fafb'``, ``'flyvis'``.
    data_dir : Path
        Root directory containing the raw files for this source.
    cell_types : list[str] or None
        Filter to these cell types only (matched against ``cell_type`` column).
    super_classes : list[str] or None
        Filter to these super_class values (e.g. ``['optic_lobe_intrinsic']``).
    neuropils : list[str] or None
        Filter edges to these neuropil regions.
    sides : list[str] or None
        Filter nodes to these sides (e.g. ``['right', 'left']``).
    min_syn_count : int
        Drop edges with fewer synapses. Default 2.
    allow_large : bool
        Opt-in to load very large edge tables (e.g. male_cns full connectome).
    """
    source: str
    data_dir: Path
    cell_types: Optional[List[str]] = None
    super_classes: Optional[List[str]] = None
    neuropils: Optional[List[str]] = None
    sides: Optional[List[str]] = None
    min_syn_count: int = 2
    allow_large: bool = False

    _nodes: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _edges: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_banc(
        cls,
        data_dir: Optional[Path] = None,
        **kwargs,
    ) -> "ConnectomeLoader":
        """
        BANC whole-brain connectome.

        Files required in data_dir:
            neurons.csv.gz
            connections_princeton.csv.gz

        Columns (neurons.csv.gz):
            Root ID, Top in/out region, Community labels,
            Predicted NT type, Predicted NT confidence,
            Verified NT type, Verified Neuropeptide,
            Body Part, Function, Flow, Super Class, Class, Sub Class,
            Hemilineage, Nerve, Soma side,
            Primary Cell Type, Alternative Cell Type(s),
            Cable length (nm), Surface area (nm^2), Volume (nm^3)

        Columns (connections_princeton.csv.gz):
            pre_root_id, post_root_id, neuropil, syn_count, nt_type
        """
        return cls(source="banc", data_dir=Path(data_dir or _BANC_DIR), **kwargs)

    @classmethod
    def from_optic_lobe(
        cls,
        data_dir: Optional[Path] = None,
        **kwargs,
    ) -> "ConnectomeLoader":
        """
        maleCNS optic-lobe subset (hex column 8, 925 neurons, 7302 edges).

        Files required in data_dir:
            malecns_09_optic_lobe_hex_08_meta.feather
            malecns_09_optic_lobe_hex_08_simple_edgelist.feather

        Node columns (key ones):
            malecns_09_id, cell_type, neurotransmitter_predicted,
            super_class, cell_class, cell_sub_class, side, flow, region

        Edge columns:
            pre, post, count, norm, total_input
        """
        return cls(source="optic_lobe", data_dir=Path(data_dir or _OL_DIR), **kwargs)

    @classmethod
    def from_male_cns(
        cls,
        data_dir: Path,
        **kwargs,
    ) -> "ConnectomeLoader":
        """
        Janelia male CNS connectome (full CNS).

        This dataset is typically too large to load without filtering.
        Use ``cell_types=...`` / ``super_classes=...`` / ``sides=...`` or
        set ``allow_large=True`` explicitly.

        Files required in data_dir:
            body-annotations-male-cns-*.feather
            connectome-weights-male-cns-*.feather

        Optional (recommended):
            body-neurotransmitters-male-cns-*.feather  – provides nt_type + nt_score

        The edge table is streamed + filtered in batches to avoid loading the
        full 150M+ edge file into memory.
        """
        return cls(source="male_cns", data_dir=Path(data_dir), **kwargs)

    @classmethod
    def from_fafb(
        cls,
        data_dir: Optional[Path] = None,
        **kwargs,
    ) -> "ConnectomeLoader":
        """
        FlyWire FAFB v783 full connectome (139,255 neurons, ~50 M synapses).

        Files required in data_dir (download from https://codex.flywire.ai):
            consolidated_cell_types.csv.gz   – root_id, primary_type, additional_type(s)
            classification.csv.gz            – root_id, flow, super_class, class, sub_class,
                                               hemilineage, side, nerve
            neurons.csv.gz                   – root_id, group, nt_type, nt_type_score,
                                               da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg
            connections_princeton.csv.gz     – pre_root_id, post_root_id, neuropil, syn_count, nt_type
            (alternatives: connections_princeton_no_threshold.csv.gz,
                           connections_buhmann_no_threshold.csv.gz)

        Node columns (after merge):
            root_id, cell_type, flow, super_class, class, sub_class,
            hemilineage, side, nerve, nt_type, nt_score,
            da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg

        Edge columns:
            pre_root_id, post_root_id, neuropil, syn_count, nt_type
        """
        return cls(source="fafb", data_dir=Path(data_dir or _FAFB_DIR), **kwargs)

    @classmethod
    def from_flyvis(
        cls,
        data_dir: Optional[Path] = None,
        **kwargs,
    ) -> "ConnectomeLoader":
        """
        FlyVis ConnectomeFromAvgFilters HDF5 files.
        Useful as a sanity-check / small reference network.

        Files required in data_dir:
            nodes/index.h5
            edges/source_index.h5, edges/target_index.h5, edges/n_syn.h5, edges/sign.h5
        """
        return cls(source="flyvis", data_dir=Path(data_dir or _FLYVIS_DIR), **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load, filter, and return ``(nodes_df, edges_df)``."""
        logger.info("[%s] Loading connectome from %s ...", self.source, self.data_dir)

        dispatch = {
            "banc":       self._load_banc,
            "optic_lobe": self._load_optic_lobe,
            "male_cns":   self._load_male_cns,
            "fafb": self._load_fafb,
            "flyvis":     self._load_flyvis,
        }
        if self.source not in dispatch:
            raise ValueError(f"Unknown source '{self.source}'. Available: {list(dispatch)}")

        nodes, edges = dispatch[self.source]()

        # ── Apply filters ────────────────────────────────────────────
        nodes = self._filter_nodes(nodes)
        edges = self._filter_edges(edges, nodes)

        # ── Build integer indices ─────────────────────────────────────
        nodes = nodes.reset_index(drop=True)
        nodes["node_idx"] = nodes.index
        id_to_idx = nodes.set_index("root_id")["node_idx"].to_dict()
        edges = edges.copy()
        edges["pre_idx"]  = edges["pre_root_id"].map(id_to_idx)
        edges["post_idx"] = edges["post_root_id"].map(id_to_idx)
        edges = edges.dropna(subset=["pre_idx", "post_idx"]).reset_index(drop=True)
        edges["pre_idx"]  = edges["pre_idx"].astype(np.int64)
        edges["post_idx"] = edges["post_idx"].astype(np.int64)

        self._nodes, self._edges = nodes, edges
        logger.info(
            "[%s] Loaded %d neurons, %d edges (min_syn=%d)",
            self.source, len(nodes), len(edges), self.min_syn_count,
        )
        return nodes, edges

    # ------------------------------------------------------------------
    # Source-specific loaders
    # ------------------------------------------------------------------

    # ── BANC ──────────────────────────────────────────────────────────
    def _load_banc(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        neu_path  = self.data_dir / "neurons.csv.gz"
        conn_path = self.data_dir / "connections_princeton.csv.gz"
        self._require_files(neu_path, conn_path)

        neu = pd.read_csv(neu_path, compression="gzip")
        neu.columns = neu.columns.str.strip()

        rename_neu = {
            "Root ID":                    "root_id",
            "Primary Cell Type":          "cell_type",
            "Predicted NT type":          "nt_type",
            "Verified NT type":           "nt_type_verified",
            "Super Class":                "super_class",
            "Class":                      "class",
            "Sub Class":                  "sub_class",
            "Soma side":                  "side",
            "Flow":                       "flow",
            "Body Part":                  "body_part",
            "Hemilineage":                "hemilineage",
            "Nerve":                      "nerve",
            "Top in/out region":          "top_region",
            "Community labels":           "community_label",
            "Cable length (nm)":          "cable_length_nm",
            "Surface area (nm^2)":        "surface_area_nm2",
            "Volume (nm^3)":              "volume_nm3",
        }
        neu = neu.rename(columns={k: v for k, v in rename_neu.items() if k in neu.columns})
        for col in ["cell_type", "nt_type", "super_class", "sub_class", "side"]:
            if col not in neu.columns:
                neu[col] = "unknown"

        conn = pd.read_csv(conn_path, compression="gzip")
        conn.columns = conn.columns.str.strip()
        # already has: pre_root_id, post_root_id, neuropil, syn_count, nt_type
        if "syn_count" not in conn.columns:
            conn["syn_count"] = 1
        return neu, conn

    # ── Optic lobe (maleCNS feather) ──────────────────────────────────
    def _load_optic_lobe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            import pyarrow.feather as feather
        except ImportError:
            raise ImportError("pyarrow required for optic_lobe source. pip install pyarrow")

        node_path = self.data_dir / "malecns_09_optic_lobe_hex_08_meta.feather"
        edge_path = self.data_dir / "malecns_09_optic_lobe_hex_08_simple_edgelist.feather"
        self._require_files(node_path, edge_path)

        neu  = feather.read_table(node_path).to_pandas()
        conn = feather.read_table(edge_path).to_pandas()

        # Rename to common schema
        rename_neu = {
            "malecns_09_id":             "root_id",
            "cell_type":                  "cell_type",
            "neurotransmitter_predicted": "nt_type",
            "super_class":                "super_class",
            "cell_class":                 "class",
            "cell_sub_class":             "sub_class",
            "side":                       "side",
            "flow":                       "flow",
            "region":                     "top_region",
            "fafb_cell_type":             "fafb_cell_type",
            "neurotransmitter_score":     "nt_score",
        }
        neu = neu.rename(columns={k: v for k, v in rename_neu.items() if k in neu.columns})
        for col in ["cell_type", "nt_type", "super_class", "sub_class", "side"]:
            if col not in neu.columns:
                neu[col] = "unknown"

        # malecns_09_id is stored as string in the feather file;
        # edge pre/post are int32 — cast nodes to int64 so the join works.
        neu["root_id"] = pd.to_numeric(neu["root_id"], errors="coerce").astype("Int64")

        rename_conn = {
            "pre":   "pre_root_id",
            "post":  "post_root_id",
            "count": "syn_count",
        }
        conn = conn.rename(columns={k: v for k, v in rename_conn.items() if k in conn.columns})
        conn["pre_root_id"]  = conn["pre_root_id"].astype("Int64")
        conn["post_root_id"] = conn["post_root_id"].astype("Int64")
        conn["neuropil"] = "optic_lobe"  # feather has no neuropil column
        return neu, conn

    # ── male CNS (Janelia feather) ───────────────────────────────────────
    def _load_male_cns(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load male CNS nodes + edges from feather files.

        Notes
        -----
        The connectome weights file can contain ~150M edges. We stream it in
        record batches and filter using the (already filtered) node set.
        """
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.feather as feather
            import pyarrow.ipc as ipc
        except ImportError as e:
            raise ImportError(
                "pyarrow required for male_cns source. pip install pyarrow"
            ) from e

        def _find_one(patterns: List[str]) -> Path:
            for pat in patterns:
                hits = sorted(self.data_dir.glob(pat))
                if hits:
                    return hits[0]
            raise FileNotFoundError(
                f"Could not find required male_cns file in {self.data_dir}. "
                f"Tried patterns: {patterns}"
            )

        # Required
        anno_path = _find_one(["body-annotations-male-cns-*.feather"])
        weights_path = _find_one(["connectome-weights-male-cns-*.feather"])

        # Optional
        nt_path = None
        nt_hits = sorted(self.data_dir.glob("body-neurotransmitters-male-cns-*.feather"))
        if nt_hits:
            nt_path = nt_hits[0]

        neu = feather.read_table(str(anno_path)).to_pandas()

        rename_neu = {
            "bodyId": "root_id",
            "type": "cell_type",
            "superclass": "super_class",
            "subclass": "sub_class",
            "rootSide": "side",
            "somaSide": "soma_side",
            "group": "group",
            "status": "status",
            "instance": "instance",
            "hemibrainType": "hemibrain_type",
            "flywireType": "flywire_type",
        }
        neu = neu.rename(columns={k: v for k, v in rename_neu.items() if k in neu.columns})

        # Ensure required columns exist.
        for col in ["cell_type", "super_class", "sub_class", "side"]:
            if col not in neu.columns:
                neu[col] = "unknown"
        neu["cell_type"] = neu["cell_type"].fillna("unknown")
        neu["super_class"] = neu["super_class"].fillna("unknown")
        neu["sub_class"] = neu["sub_class"].fillna("unknown")
        neu["side"] = neu["side"].fillna("unknown")
        neu["side"] = neu["side"].replace({"R": "right", "L": "left"})

        neu["root_id"] = pd.to_numeric(neu["root_id"], errors="coerce").astype("Int64")

        # Merge neurotransmitter consensus if available.
        if nt_path is not None:
            nt_df = feather.read_table(
                str(nt_path),
                columns=["body", "consensus_nt", "predicted_nt_confidence"],
            ).to_pandas()
            nt_df = nt_df.rename(
                columns={
                    "body": "root_id",
                    "consensus_nt": "nt_type",
                    "predicted_nt_confidence": "nt_score",
                }
            )
            nt_df["root_id"] = pd.to_numeric(nt_df["root_id"], errors="coerce").astype("Int64")
            neu = neu.merge(nt_df, on="root_id", how="left")

        if "nt_type" not in neu.columns:
            neu["nt_type"] = "unknown"
        neu["nt_type"] = neu["nt_type"].fillna("unknown")

        # Apply node filters *before* reading the huge edge table.
        neu = self._filter_nodes(neu)

        if (
            not self.allow_large
            and self.cell_types is None
            and self.super_classes is None
            and self.sides is None
        ):
            raise ValueError(
                "male_cns connectome-weights file contains ~150M edges and is too large to load "
                "without filtering. Please pass cell_types=..., super_classes=..., or sides=..., "
                "or set allow_large=True explicitly."
            )

        valid_ids = neu["root_id"].dropna().astype(np.int64).to_numpy()
        value_set = pa.array(valid_ids, type=pa.int64())

        # Stream and filter edges in batches.
        filtered_batches = []
        kept_rows = 0
        import time as _time
        t0 = _time.time()

        with pa.memory_map(str(weights_path), "r") as source:
            reader = ipc.open_file(source)
            for b_idx in range(reader.num_record_batches):
                batch = reader.get_batch(b_idx)
                pre = batch.column("body_pre")
                post = batch.column("body_post")
                w = batch.column("weight")

                mask = pc.and_(
                    pc.is_in(pre, value_set=value_set),
                    pc.is_in(post, value_set=value_set),
                )
                if self.min_syn_count is not None:
                    mask = pc.and_(mask, pc.greater_equal(w, self.min_syn_count))

                filtered = batch.filter(mask)
                if filtered.num_rows:
                    filtered_batches.append(filtered)
                    kept_rows += filtered.num_rows

                if (b_idx + 1) % 200 == 0:
                    elapsed = max(_time.time() - t0, 1e-6)
                    logger.info(
                        "[male_cns] scanned %d/%d batches (%.1f batches/s), kept %d rows so far",
                        b_idx + 1,
                        reader.num_record_batches,
                        (b_idx + 1) / elapsed,
                        kept_rows,
                    )

        if filtered_batches:
            tbl = pa.Table.from_batches(filtered_batches)
            edges = tbl.to_pandas()
        else:
            edges = pd.DataFrame(columns=["body_pre", "body_post", "weight"])

        edges = edges.rename(
            columns={"body_pre": "pre_root_id", "body_post": "post_root_id", "weight": "syn_count"}
        )
        edges["pre_root_id"] = pd.to_numeric(edges["pre_root_id"], errors="coerce").astype("Int64")
        edges["post_root_id"] = pd.to_numeric(edges["post_root_id"], errors="coerce").astype("Int64")
        edges["neuropil"] = "cns"

        return neu, edges

    # ── FAFB Codex ────────────────────────────────────────────────────
    def _load_fafb(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        d = self.data_dir

        # 1. Cell types
        ct_path = d / "consolidated_cell_types.csv.gz"
        cl_path = d / "classification.csv.gz"
        nt_path = d / "neurons.csv.gz"           # cols: root_id, group, nt_type, nt_type_score, da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg

        missing = [p for p in [ct_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"FAFB data files missing: {missing}\n"
                "Download from https://codex.flywire.ai/api/download?dataset=fafb "
                f"and place in {d}"
            )

        neu = pd.read_csv(ct_path, compression="gzip")
        neu.columns = neu.columns.str.strip()
        # cols: root_id, primary_type, additional_type(s)
        neu = neu.rename(columns={"primary_type": "cell_type"})

        # Merge classification if available
        # cols: root_id, flow, super_class, class, sub_class, hemilineage, side, nerve
        if cl_path.exists():
            cl = pd.read_csv(cl_path, compression="gzip")
            cl.columns = cl.columns.str.strip()
            neu = neu.merge(cl, on="root_id", how="left")

        # Merge NT predictions if available
        # cols: root_id, group, nt_type, nt_type_score, da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg
        if nt_path.exists():
            nt = pd.read_csv(nt_path, compression="gzip")
            nt.columns = nt.columns.str.strip()
            nt = nt.rename(columns={"nt_type_score": "nt_score"})
            nt_keep_cols = ["root_id"] + [c for c in ["nt_type", "nt_score", "da_avg", "ser_avg",
                                                        "gaba_avg", "glut_avg", "ach_avg", "oct_avg"]
                                           if c in nt.columns]
            neu = neu.merge(nt[nt_keep_cols], on="root_id", how="left")

        for col in ["nt_type", "super_class", "sub_class", "side", "flow"]:
            if col not in neu.columns:
                neu[col] = "unknown"

        # 2. Connections
        # available files: connections_princeton.csv.gz, connections_princeton_no_threshold.csv.gz,
        #                  connections_buhmann_no_threshold.csv.gz
        # cols: pre_root_id, post_root_id, neuropil, syn_count, nt_type
        conn = None
        for fname in ["connections.parquet", "connections_princeton.csv.gz",
                      "connections.csv.gz", "connections.csv"]:
            cpath = d / fname
            if cpath.exists():
                if fname.endswith(".parquet"):
                    conn = pd.read_parquet(cpath)
                else:
                    conn = pd.read_csv(cpath, compression=("gzip" if fname.endswith(".gz") else None))
                conn.columns = conn.columns.str.strip()
                break

        if conn is None:
            raise FileNotFoundError(
                f"No connections file found in {d}. "
                "Expected one of: connections.parquet, connections_princeton.csv.gz, connections.csv.gz"
            )

        # Normalise edge column names
        edge_rename = {
            "pre_root_id":  "pre_root_id",
            "post_root_id": "post_root_id",
            "neuropil":     "neuropil",
            "syn_count":    "syn_count",
            "weight":       "syn_count",
            "bodyId_pre":   "pre_root_id",
            "bodyId_post":  "post_root_id",
        }
        conn = conn.rename(columns={k: v for k, v in edge_rename.items() if k in conn.columns})
        if "syn_count" not in conn.columns:
            conn["syn_count"] = 1
        if "neuropil" not in conn.columns:
            conn["neuropil"] = "unknown"
        return neu, conn

    # ── FlyVis (HDF5) ─────────────────────────────────────────────────
    def _load_flyvis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            import h5py
        except ImportError:
            # Try to install it silently if missing
            import os
            os.system(f"{sys.executable} -m pip install h5py --quiet")
            import h5py

        node_index_path  = self.data_dir / "nodes" / "index.h5"
        src_path         = self.data_dir / "edges" / "source_index.h5"
        tgt_path         = self.data_dir / "edges" / "target_index.h5"
        syn_path         = self.data_dir / "edges" / "n_syn.h5"
        sign_path        = self.data_dir / "edges" / "sign.h5"
        src_type_path    = self.data_dir / "edges" / "source_type.h5"
        tgt_type_path    = self.data_dir / "edges" / "target_type.h5"
        self._require_files(node_index_path, src_path, tgt_path, syn_path)

        def _read_h5(path):
            with h5py.File(path, "r") as f:
                key = list(f.keys())[0]
                return f[key][:]

        node_types_raw = _read_h5(node_index_path)
        # node_index.h5 stores encoded cell type strings
        if node_types_raw.dtype.kind in ("S", "O", "U"):
            node_types = [t.decode() if isinstance(t, bytes) else str(t)
                          for t in node_types_raw]
        else:
            node_types = node_types_raw.tolist()

        neu = pd.DataFrame({
            "root_id":     np.arange(len(node_types), dtype=np.int64),
            "cell_type":   node_types,
            "nt_type":     "unknown",
            "super_class": "optic_lobe_intrinsic",
            "sub_class":   "unknown",
            "side":        "unknown",
        })

        # Optionally decode sign as nt_type
        if sign_path.exists():
            signs = _read_h5(sign_path).flatten()
        # FlyVis sign is per edge. We map it back to per-node nt_type by taking
        # the majority sign of outgoing edges for each node.
        signs = _read_h5(sign_path).flatten()
        src_idx_raw = _read_h5(src_path).flatten().astype(np.int64)

        # Map sign (+1/-1) to nt_type per node
        node_signs = np.zeros(len(node_types))
        for i, s in zip(src_idx_raw, signs):
            node_signs[i] = s # last one wins or use majority
        
        def _sign_to_nt(s):
            if s > 0: return "acetylcholine"
            if s < 0: return "gaba"
            return "unknown"
            
        neu = pd.DataFrame({
            "root_id":     np.arange(len(node_types), dtype=np.int64),
            "cell_type":   node_types,
            "nt_type":     [_sign_to_nt(s) for s in node_signs],
            "super_class": "optic_lobe_intrinsic",
            "sub_class":   "unknown",
            "side":        "unknown",
        })

        src_idx  = src_idx_raw
        tgt_idx  = _read_h5(tgt_path).flatten().astype(np.int64)
        n_syn    = _read_h5(syn_path).flatten().astype(np.float32)

        conn = pd.DataFrame({
            "pre_root_id":  src_idx,
            "post_root_id": tgt_idx,
            "syn_count":    n_syn,
            "neuropil":     "optic_lobe",
        })
        return neu, conn

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------
    def _filter_nodes(self, nodes: pd.DataFrame) -> pd.DataFrame:
        if self.cell_types is not None:
            nodes = nodes[nodes["cell_type"].isin(self.cell_types)]
        if self.super_classes is not None and "super_class" in nodes.columns:
            nodes = nodes[nodes["super_class"].isin(self.super_classes)]
        if self.sides is not None and "side" in nodes.columns:
            nodes = nodes[nodes["side"].isin(self.sides)]
        return nodes.copy()

    def _filter_edges(self, edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
        valid_ids = set(nodes["root_id"].tolist())
        edges = edges[
            edges["pre_root_id"].isin(valid_ids) &
            edges["post_root_id"].isin(valid_ids)
        ].copy()
        if self.neuropils is not None and "neuropil" in edges.columns:
            edges = edges[edges["neuropil"].isin(self.neuropils)]
        if "syn_count" in edges.columns:
            edges = edges[edges["syn_count"] >= self.min_syn_count]
        return edges.copy()

    @staticmethod
    def _require_files(*paths):
        for p in paths:
            if not Path(p).exists():
                # If it's a relative path to flyvis/banc, provide helpful error
                raise FileNotFoundError(
                    f"Required file not found: {p}\n"
                    "Please ensure you have downloaded the dataset and placed it in the correct folder."
                )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def nodes(self) -> pd.DataFrame:
        if self._nodes is None:
            self.load()
        return self._nodes

    @property
    def edges(self) -> pd.DataFrame:
        if self._edges is None:
            self.load()
        return self._edges

    def get_adjacency_tensors(self):
        """
        Return ``(pre_idx, post_idx, syn_count)`` as numpy int64/float32 arrays
        suitable for building sparse or dense connectivity matrices.
        """
        e = self.edges
        return (
            e["pre_idx"].to_numpy(dtype=np.int64),
            e["post_idx"].to_numpy(dtype=np.int64),
            e["syn_count"].to_numpy(dtype=np.float32),
        )

    def nt_sign(self) -> np.ndarray:
        """
        Return ``(n_nodes,)`` float32 array:
          +1  excitatory  (acetylcholine / cholinergic)
          -1  inhibitory  (GABA / glutamate / gabaergic)
           0  unknown
        """
        nt = self.nodes["nt_type"].str.lower().fillna("")
        sign = np.where(
            nt.isin(["acetylcholine", "ach", "cholinergic"]), 1.0,
            np.where(nt.isin(["gaba", "glutamate", "gabaergic"]), -1.0, 0.0)
        )
        return sign.astype(np.float32)

    def summary(self) -> Dict:
        """Return a dict of basic statistics about the loaded connectome."""
        n = self.nodes
        e = self.edges
        stats = {
            "source":         self.source,
            "n_neurons":      len(n),
            "n_edges":        len(e),
            "n_cell_types":   n["cell_type"].nunique(),
            "n_super_classes": n["super_class"].nunique() if "super_class" in n.columns else None,
            "top_cell_types": n["cell_type"].value_counts().head(10).to_dict(),
            "mean_syn_count": float(e["syn_count"].mean()) if len(e) else 0.0,
            "max_syn_count":  float(e["syn_count"].max()) if len(e) else 0.0,
            "nt_distribution": n["nt_type"].value_counts().to_dict(),
        }
        return stats

    def __repr__(self) -> str:
        loaded = self._nodes is not None
        s = f"ConnectomeLoader(source='{self.source}', data_dir='{self.data_dir}'"
        if loaded:
            s += f", n_nodes={len(self._nodes)}, n_edges={len(self._edges)}"
        s += ")"
        return s
