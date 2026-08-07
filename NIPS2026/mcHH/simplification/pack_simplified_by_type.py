"""
Pack simplified SWC files into grouped binary archives for optic-lobe loading.

Design goals:
- one binary file per abundant neuron type
- rare neuron types merged by alphabetical prefix
- compact sequential binary payload + sidecar manifest/index json
- resume-friendly, easy to inspect, easy to load later

Binary format (little-endian):
    magic[4]           = b'NPKG'
    version[u32]       = 1
    meta_len[u32]      = length of UTF-8 JSON metadata block
    meta_json[bytes]   = file-level metadata
    repeated neuron records:
        root_id[u64]
        type_name_len[u16]
        type_name[bytes]
        node_count[u32]
        node records[node_count], each packed as:
            id[i32], swc_type[i32], x[f32], y[f32], z[f32], r[f32], parent[i32]

A package-level manifest.json records:
- grouping strategy
- which neuron types are merged
- per-file counts and paths
- per-neuron offsets for fast loading
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

MAGIC = b"NPKG"
VERSION = 1
NODE_STRUCT = struct.Struct("<ii4fi")
RECORD_HEADER = struct.Struct("<QHI")  # root_id, type_name_len, node_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack simplified SWCs by neuron type")
    parser.add_argument(
        "--simplified-dir",
        type=str,
        default="/Users/lengyuner/Desktop/NIPS2026/mcHH/data/simplified_swc",
        help="Directory containing simplified .swc files and _summary.csv",
    )
    parser.add_argument(
        "--vnt-path",
        type=str,
        default="/Users/lengyuner/Desktop/data/flywire/Jun2025/visual_neuron_types.csv.gz",
        help="visual_neuron_types.csv.gz path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/lengyuner/Desktop/NIPS2026/mcHH/data/optic_lobe_type_packages_v1",
        help="Output directory for binary packages",
    )
    parser.add_argument(
        "--min-type-size",
        type=int,
        default=32,
        help="Types with >= this many neurons get their own package",
    )
    parser.add_argument(
        "--rare-prefix-min-size",
        type=int,
        default=64,
        help="Rare types sharing the same alpha prefix are grouped together if the prefix bucket reaches this size",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print one progress line every N neurons",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional neuron limit for dry runs",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def type_prefix(neuron_type: str) -> str:
    m = re.match(r"([A-Za-z]+)", neuron_type)
    return m.group(1) if m else "misc"


def read_summary_ids(summary_path: Path) -> List[int]:
    ids: List[int] = []
    with summary_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["root_id"]))
    return ids


def read_vnt_types(vnt_path: Path, allowed_ids: set[int]) -> Dict[int, str]:
    id_to_type: Dict[int, str] = {}
    with gzip.open(vnt_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = int(row["root_id"])
            if rid in allowed_ids:
                id_to_type[rid] = row["type"]
    return id_to_type


def build_groups(id_to_type: Dict[int, str], min_type_size: int, rare_prefix_min_size: int) -> Tuple[Dict[str, List[int]], Dict[str, dict]]:
    counts = Counter(id_to_type.values())
    groups: Dict[str, List[int]] = defaultdict(list)
    group_meta: Dict[str, dict] = {}

    rare_by_prefix: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for rid, neuron_type in id_to_type.items():
        if counts[neuron_type] >= min_type_size:
            group_name = f"type__{neuron_type}"
            groups[group_name].append(rid)
        else:
            rare_by_prefix[type_prefix(neuron_type)].append((rid, neuron_type))

    for neuron_type, count in counts.items():
        if count >= min_type_size:
            group_name = f"type__{neuron_type}"
            group_meta[group_name] = {
                "mode": "single_type",
                "types": [neuron_type],
                "count": count,
            }

    misc_bucket: List[Tuple[int, str]] = []
    for prefix, items in sorted(rare_by_prefix.items()):
        type_list = sorted({t for _, t in items})
        if len(items) >= rare_prefix_min_size:
            group_name = f"rare_prefix__{prefix}"
            groups[group_name].extend([rid for rid, _ in items])
            group_meta[group_name] = {
                "mode": "rare_prefix",
                "prefix": prefix,
                "types": type_list,
                "count": len(items),
            }
        else:
            misc_bucket.extend(items)

    if misc_bucket:
        misc_by_prefix: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for rid, neuron_type in misc_bucket:
            misc_by_prefix[type_prefix(neuron_type)].append((rid, neuron_type))
        for prefix, items in sorted(misc_by_prefix.items()):
            group_name = f"rare_mixed__{prefix}"
            groups[group_name].extend([rid for rid, _ in items])
            group_meta[group_name] = {
                "mode": "rare_mixed_prefix",
                "prefix": prefix,
                "types": sorted({t for _, t in items}),
                "count": len(items),
            }

    for group_name in groups:
        groups[group_name].sort()

    return dict(groups), group_meta


def parse_swc(path: Path) -> List[Tuple[int, int, float, float, float, float, int]]:
    nodes = []
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 7:
                continue
            nodes.append(
                (
                    int(p[0]),
                    int(p[1]),
                    float(p[2]),
                    float(p[3]),
                    float(p[4]),
                    float(p[5]),
                    int(p[6]),
                )
            )
    return nodes


def write_package(
    group_name: str,
    root_ids: List[int],
    id_to_type: Dict[int, str],
    simplified_dir: Path,
    output_dir: Path,
) -> dict:
    bin_name = sanitize_name(group_name) + ".bin"
    index_name = sanitize_name(group_name) + ".index.json"
    bin_path = output_dir / bin_name
    index_path = output_dir / index_name

    meta = {
        "group_name": group_name,
        "format": "NPKG",
        "version": VERSION,
        "neuron_count": len(root_ids),
        "node_record_struct": "<ii4fi",
    }
    meta_blob = json.dumps(meta, ensure_ascii=False).encode("utf-8")

    index_records = []
    with bin_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(meta_blob)))
        f.write(meta_blob)

        for rid in root_ids:
            swc_path = simplified_dir / f"{rid}.swc"
            neuron_type = id_to_type[rid]
            type_blob = neuron_type.encode("utf-8")
            nodes = parse_swc(swc_path)
            offset = f.tell()
            f.write(RECORD_HEADER.pack(int(rid), len(type_blob), len(nodes)))
            f.write(type_blob)
            for node in nodes:
                f.write(NODE_STRUCT.pack(*node))
            size = f.tell() - offset
            index_records.append(
                {
                    "root_id": int(rid),
                    "type": neuron_type,
                    "offset": offset,
                    "size": size,
                    "node_count": len(nodes),
                    "source_swc": swc_path.name,
                }
            )

    index_doc = {
        "group_name": group_name,
        "bin_file": bin_name,
        "record_count": len(index_records),
        "records": index_records,
    }
    index_path.write_text(json.dumps(index_doc, ensure_ascii=False, indent=2))

    return {
        "group_name": group_name,
        "bin_file": bin_name,
        "index_file": index_name,
        "neuron_count": len(root_ids),
        "types": sorted({id_to_type[rid] for rid in root_ids}),
    }


def main() -> None:
    args = parse_args()

    simplified_dir = Path(args.simplified_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = simplified_dir / "_summary.csv"
    vnt_path = Path(args.vnt_path)

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary csv: {summary_path}")

    root_ids = read_summary_ids(summary_path)
    if args.limit is not None:
        root_ids = root_ids[: args.limit]
    allowed_ids = set(root_ids)
    id_to_type = read_vnt_types(vnt_path, allowed_ids)

    missing_type = sorted(allowed_ids - set(id_to_type.keys()))
    if missing_type:
        raise RuntimeError(f"Missing type annotations for {len(missing_type)} neurons")

    missing_swc = [rid for rid in root_ids if not (simplified_dir / f"{rid}.swc").exists()]
    if missing_swc:
        raise RuntimeError(f"Missing simplified SWC for {len(missing_swc)} neurons")

    groups, group_meta = build_groups(
        id_to_type=id_to_type,
        min_type_size=args.min_type_size,
        rare_prefix_min_size=args.rare_prefix_min_size,
    )

    print(f"Simplified neurons: {len(root_ids):,}")
    print(f"Distinct types: {len(set(id_to_type.values()))}")
    print(f"Output dir: {output_dir}")
    print(f"Groups to write: {len(groups)}")
    print(f"Grouping: type >= {args.min_type_size} kept standalone; rare prefixes merged if >= {args.rare_prefix_min_size}")

    manifest_packages = []
    processed = 0
    total = sum(len(v) for v in groups.values())

    for i, group_name in enumerate(sorted(groups.keys()), start=1):
        root_batch = groups[group_name]
        pkg = write_package(
            group_name=group_name,
            root_ids=root_batch,
            id_to_type=id_to_type,
            simplified_dir=simplified_dir,
            output_dir=output_dir,
        )
        pkg.update(group_meta[group_name])
        manifest_packages.append(pkg)
        processed += len(root_batch)
        if processed % max(args.progress_every, 1) == 0 or i == len(groups):
            print(f"[{processed:6,d}/{total:,}] wrote group {i:4d}/{len(groups)}: {group_name} ({len(root_batch)} neurons)")

    manifest = {
        "format": "optic_lobe_type_packages_v1",
        "binary_magic": MAGIC.decode("ascii"),
        "version": VERSION,
        "source_dir": str(simplified_dir),
        "source_summary": str(summary_path),
        "total_neurons": len(root_ids),
        "total_groups": len(manifest_packages),
        "min_type_size": args.min_type_size,
        "rare_prefix_min_size": args.rare_prefix_min_size,
        "packages": manifest_packages,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    # lightweight README for later loaders
    readme = output_dir / "README.md"
    readme.write_text(
        "# Optic Lobe Type Packages\n\n"
        "This directory contains grouped binary packages built from simplified SWC files.\n\n"
        "## Format\n\n"
        "Each `.bin` file starts with magic `NPKG`, a version number, a JSON metadata blob, and then repeated neuron records.\n"
        "Use the matching `.index.json` file for random access.\n\n"
        "## Grouping rule\n\n"
        f"- neuron types with >= {args.min_type_size} neurons are stored as standalone `type__<name>.bin`\n"
        f"- rarer neuron types are merged by alphabetical prefix when the prefix bucket has >= {args.rare_prefix_min_size} neurons\n"
        "- remaining rare types are stored in `rare_mixed__<prefix>.bin`\n\n"
        "See `manifest.json` for the exact mapping from package to neuron types.\n",
        encoding="utf-8",
    )

    print("Done.")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
