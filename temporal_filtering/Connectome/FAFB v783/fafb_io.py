"""Shared I/O for the FAFB build: paths and raw-CSV readers.

This is the one place that knows where the raw FAFB files live and how to read
them. ``build_network.py``, ``hex_grid.py`` and ``column_locator.py`` all import
from here (they never import each other), so the path constants and the three
CSV loaders are defined exactly once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Build directory (outputs) and the raw downloaded CSVs (download/).
DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "download"

VISUAL_NEURON_TYPES_FILE = "visual_neuron_types.csv.gz"
COLUMN_ASSIGNMENT_FILE = "column_assignment.csv.gz"
CONNECTIONS_FILE = "connections_princeton.csv.gz"

# Rows read per chunk when scanning the (large) connections file.
CONNECTIONS_CHUNK_SIZE = 500_000


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
