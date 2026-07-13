"""Shared I/O for the FAFB connectome build: paths and raw-CSV readers.

This is the one place that knows where the raw FAFB files live and how to read
them. ``build_network.py``, ``column_mapper.py`` and ``column_locator.py`` all import
from here (they never import each other), so the path constants and the three
CSV loaders are defined exactly once.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Build directory (outputs) and the raw downloaded CSVs (download/).
DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "download"
# Per-run network folders (<side>_min_neuron<N>/ etc.) live under here.
NETWORK_DIR = DATA_DIR / "built_network"
DEFAULT_NETWORK_RUN = "right_min_neuron1_extent2"
# Column map artifacts (per-side column tables + the column_map.png) live here.
COLUMN_MAP_DIR = DATA_DIR / "column_map"
# Located-column CSVs (r1_6_<side>_post.csv etc., from column_locator.py) live here.
COLUMN_LOCATION_DIR = DATA_DIR / "column_location"
# Per-network moving-bar column-current cache (under each built_network run folder).
MOVING_BAR_CACHE_DIRNAME = "moving_bar_cache"

# Borst 5-column simulator matrices (``FiveCol_MedSim_*`` / ``Circuits/``).
SIMULATION_CODE_DIR = DATA_DIR.parent.parent / "SimulationCode"
BORST_CIRCUITS_DIR = SIMULATION_CODE_DIR / "Circuits"
BORST_MULTI_COL_M = BORST_CIRCUITS_DIR / "multi_colM.npy"
BORST_CTYPE_NPY = BORST_CIRCUITS_DIR / "ctype.npy"


def parse_comma_list(text: str) -> List[str]:
    """Split a comma-separated token list (empty string → ``[]``)."""
    return [t.strip() for t in str(text or "").split(",") if t.strip()]


def network_json_path(side: str, min_neuron_count: int = 1) -> Path:
    """Path to a built connectome ``network.json`` (default: full FAFB per side)."""
    return NETWORK_DIR / f"{side}_min_neuron{min_neuron_count}" / "network.json"


TYPE_COUNTS_ABC_FILE = "type_counts_abc.csv"


def resolve_network_json(spec: str) -> Path:
    """Resolve ``built_network/<run_name>/network.json`` or an explicit path.

    * Run name (e.g. ``right_min_neuron1``) → ``NETWORK_DIR/<run_name>/network.json``
    * Directory path → ``<dir>/network.json``
    * ``*.json`` path → that file
    """
    p = Path(spec)
    if p.suffix == ".json":
        return p.resolve()
    if p.is_dir():
        return (p / "network.json").resolve()
    if p.is_absolute():
        raise FileNotFoundError(f"not a network run directory: {p}")
    return (NETWORK_DIR / spec / "network.json").resolve()


def network_run_tag(network_path: str, meta: dict) -> str:
    """``right`` / ``left``; append ``_extentN`` when the run folder name has it."""
    run_name = Path(network_path).resolve().parent.name
    side = str(meta.get("side") or run_name.split("_")[0])
    m = re.search(r"_extent(\d+)$", run_name)
    if m:
        return f"{side}_extent{m.group(1)}"
    return side


def type_counts_abc_path(network_json: Path) -> Path:
    """``type_counts_abc.csv`` next to a built ``network.json``."""
    return Path(network_json).resolve().parent / TYPE_COUNTS_ABC_FILE


def moving_bar_cache_dir(network_json: Path) -> Path:
    """Directory for cached moving-bar column currents for one network run."""
    return Path(network_json).resolve().parent / MOVING_BAR_CACHE_DIRNAME

VISUAL_NEURON_TYPES_FILE = "visual_neuron_types.csv.gz"
COLUMN_ASSIGNMENT_FILE = "column_assignment.csv.gz"
CONNECTIONS_FILE = "connections_princeton.csv.gz"

# Per-side column -> (u, v) table: written by column_mapper.py and read back by
# build_network.py / column_locator.py. Single source for this filename so the
# pattern is never restated. Columns: column_id, p, q, u, v.
COLUMN_MAP_FILE = "column_map_{side}.csv"

# Rows read per chunk when scanning the (large) connections file.
CONNECTIONS_CHUNK_SIZE = 500_000


def column_map_path(side: str) -> Path:
    """Path to the per-side column_id -> (u, v) table (written by column_mapper.py)."""
    return COLUMN_MAP_DIR / COLUMN_MAP_FILE.format(side=side)


def load_column_map(side: str) -> pd.DataFrame:
    """Read column_map_<side>.csv (column_id, p, q, u, v)."""
    return pd.read_csv(column_map_path(side))


def load_visual_neurons() -> pd.DataFrame:
    """visual_neuron_types: root_id, type, family, subsystem, category, side."""
    df = pd.read_csv(RAW_DIR / VISUAL_NEURON_TYPES_FILE, compression="gzip")
    logger.info("Loaded %d visual neurons, %d types", len(df), df["type"].nunique())
    return df


def load_column_assignments() -> pd.DataFrame:
    """column_assignment: root_id, hemisphere, type, column_id, x, y, p, q."""
    df = pd.read_csv(RAW_DIR / COLUMN_ASSIGNMENT_FILE, compression="gzip")
    logger.info(
        "Loaded %d column assignments, %d columns", len(df), df["column_id"].nunique()
    )
    return df


def load_connections(
    keep_neuron_ids: Optional[set] = None,
    keep_neuropils: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """connections_princeton (pre_root_id, post_root_id, neuropil, syn_count, nt_type).

    Streams in chunks. If ``keep_neuron_ids`` is given, keep rows where pre OR
    post is in the set; if ``keep_neuropils`` is given, keep matching neuropils.
    """
    neuropil_set = set(keep_neuropils) if keep_neuropils is not None else None
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        RAW_DIR / CONNECTIONS_FILE, compression="gzip", chunksize=CONNECTIONS_CHUNK_SIZE
    ):
        if neuropil_set is not None:
            chunk = chunk[chunk["neuropil"].isin(neuropil_set)]
        if keep_neuron_ids is not None:
            chunk = chunk[
                chunk["pre_root_id"].isin(keep_neuron_ids)
                | chunk["post_root_id"].isin(keep_neuron_ids)
            ]
        if len(chunk):
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(
            columns=["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"]
        )
    df = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %d connection rows (after streaming filter)", len(df))
    return df
