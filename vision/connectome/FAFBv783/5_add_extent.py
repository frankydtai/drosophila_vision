"""Crop a built network.json to a central hex disc (extent).

Reads ``4_built_networks/<run>/network.json`` and writes a sibling
``<run>_extent<N>/network.json`` (keeps column-positioned nodes with hex
radius <= extent, and edges between them). Run after ``4_build_network.py``.

    .venv/bin/python "connectome/FAFBv783/5_add_extent.py" 2
    .venv/bin/python "connectome/FAFBv783/5_add_extent.py" 1,3,5,10
    .venv/bin/python "connectome/FAFBv783/5_add_extent.py" 1,3,5,10 --run right_min_neuron1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Set

import path
from import_bootstrap import parse_comma_list
from path import DEFAULT_NETWORK_RUN
from build_hex import inside_mask
from build_network import _write_summary

logger = logging.getLogger(__name__)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}. Run 4_build_network.py first."
        )
    return path


def resolve_run_dir(spec: str) -> Path:
    """Resolve a ``4_built_networks`` run folder from a name or path."""
    return path.resolve_network_json(spec).parent


def add_extent(run_dir: Path, crop_extent: int) -> Path:
    """Crop a built network.json to the central hex disc of ``crop_extent``.

    Keeps only column-positioned nodes whose hex distance from the centre is
    ``<= crop_extent`` (and the edges between them), writing a sibling run folder
    ``<run_dir.name>_extent<crop_extent>/network.json``. Node ``column_id`` is
    the stable FAFB identity and is preserved as-is on crop.

    Args:
        run_dir: An existing run folder containing network.json.
        crop_extent: Hex-disc radius to keep around the centre (2 -> 19 columns).
    """
    if crop_extent < 0:
        raise ValueError(f"crop_extent must be >= 0, got {crop_extent}")

    src = _require(run_dir / "network.json")
    payload = json.load(open(src))
    nodes = payload["nodes"]
    edges = payload["edges"]

    kept_nodes = [
        n for n in nodes
        if n.get("u") is not None and bool(inside_mask(n["u"], n["v"], crop_extent))
    ]
    kept_ids: Set[int] = {n["id"] for n in kept_nodes}
    kept_edges = [
        e for e in edges if e["src"] in kept_ids and e["tar"] in kept_ids
    ]

    n_with_col = sum(1 for n in kept_nodes if n.get("u") is not None)
    src_meta = payload.get("metadata", {})
    metadata: Dict[str, object] = {
        "side": src_meta.get("side"),
        "min_neuron_count": src_meta.get("min_neuron_count"),
        "extent": crop_extent,
        "cropped_from": run_dir.name,
        "sign_mode": src_meta.get("sign_mode"),
        "nt_to_sign": src_meta.get("nt_to_sign"),
        "forced_negative_pre_cells": src_meta.get("forced_negative_pre_cells"),
        "n_nodes": len(kept_nodes),
        "n_nodes_with_column": n_with_col,
        "n_edges": len(kept_edges),
        "n_input_nodes": int(sum(bool(n.get("input")) for n in kept_nodes)),
        "n_cells": int(len({n["name"] for n in kept_nodes})),
    }

    out_dir = run_dir.parent / f"{run_dir.name}_extent{crop_extent}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "network.json"
    with open(out_path, "w") as fh:
        json.dump({"metadata": metadata, "nodes": kept_nodes, "edges": kept_edges}, fh)
    logger.info(
        "Cropped %s -> %s (extent=%d): %d nodes, %d edges, %d cells",
        run_dir.name, out_path, crop_extent,
        len(kept_nodes), len(kept_edges), metadata["n_cells"],
    )
    _write_summary(out_dir, metadata)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop a built FAFB network.json to central hex disc(s)."
    )
    parser.add_argument(
        "extents",
        metavar="EXTENT[,EXTENT...]",
        help="Comma-separated hex-disc radii (e.g. 1,3,5,10). Each must be >= 0.",
    )
    parser.add_argument(
        "--run", default=DEFAULT_NETWORK_RUN,
        help=f"Source run under 4_built_networks/ (default: {DEFAULT_NETWORK_RUN}).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    tokens = parse_comma_list(args.extents)
    if not tokens:
        raise SystemExit("extents must not be empty")
    try:
        extents = [int(t) for t in tokens]
    except ValueError as e:
        raise SystemExit(f"invalid extent in {args.extents!r}") from e
    if any(e < 0 for e in extents):
        raise SystemExit(f"each extent must be >= 0, got {extents}")

    run_dir = resolve_run_dir(args.run)
    for extent in extents:
        out = add_extent(run_dir, extent)
        meta = json.load(open(out))["metadata"]
        print(f"\n=== add_extent ({run_dir.name}, extent={extent}) ===")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print(f"  output: {out}")


if __name__ == "__main__":
    main()
