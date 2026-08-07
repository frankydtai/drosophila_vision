"""Load grouped binary morphology packages generated from simplified SWCs."""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

MAGIC = b"NPKG"
VERSION = 1
NODE_DTYPE = np.dtype(
    [
        ("id", "<i4"),
        ("type", "<i4"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("r", "<f4"),
        ("parent", "<i4"),
    ]
)
RECORD_HEADER = struct.Struct("<QHI")  # root_id, type_name_len, node_count


@dataclass
class PackedNeuronMorphology:
    """One neuron morphology read from a packed binary package."""

    root_id: int
    neuron_type: str
    nodes: np.ndarray
    package: str

    @property
    def node_count(self) -> int:
        return int(len(self.nodes))

    def to_swc_array(self) -> np.ndarray:
        """Return numeric SWC array with columns id,type,x,y,z,r,parent."""
        arr = np.empty((len(self.nodes), 7), dtype=np.float32)
        arr[:, 0] = self.nodes["id"]
        arr[:, 1] = self.nodes["type"]
        arr[:, 2] = self.nodes["x"]
        arr[:, 3] = self.nodes["y"]
        arr[:, 4] = self.nodes["z"]
        arr[:, 5] = self.nodes["r"]
        arr[:, 6] = self.nodes["parent"]
        return arr


class PackedMorphologyLoader:
    """Reader for ``optic_lobe_type_packages_v1`` binary morphology packages.

    The loader keeps only manifests and per-neuron index entries in memory.
    Morphology node arrays are loaded lazily from ``.bin`` files.
    """

    def __init__(self, package_dir: Union[str, Path], preload_index: bool = True):
        self.package_dir = Path(package_dir)
        manifest_path = self.package_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        with manifest_path.open() as f:
            self.manifest = json.load(f)
        self.packages = self.manifest.get("packages", [])
        self._index_by_root: Dict[int, dict] = {}
        self._records_by_package: Dict[str, List[dict]] = {}
        if preload_index:
            self.load_index(show_progress=False)

    @property
    def total_neurons(self) -> int:
        return int(self.manifest.get("total_neurons", 0))

    @property
    def total_groups(self) -> int:
        return int(self.manifest.get("total_groups", len(self.packages)))

    def load_index(self, show_progress: bool = True) -> None:
        iterator = self.packages
        pbar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator, desc="Index packages", unit="pkg")
            except Exception:
                pbar = None
        for pkg in iterator:
            index_path = self.package_dir / pkg["index_file"]
            with index_path.open() as f:
                index_doc = json.load(f)
            records = index_doc.get("records", [])
            self._records_by_package[pkg["bin_file"]] = records
            for rec in records:
                rec2 = dict(rec)
                rec2["bin_file"] = pkg["bin_file"]
                rec2["group_name"] = pkg["group_name"]
                self._index_by_root[int(rec2["root_id"])] = rec2

    def type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if not self._index_by_root:
            self.load_index(show_progress=False)
        for rec in self._index_by_root.values():
            t = rec["type"]
            counts[t] = counts.get(t, 0) + 1
        return counts

    def package_summary(self) -> List[dict]:
        return list(self.packages)

    def get_index_record(self, root_id: int) -> dict:
        if not self._index_by_root:
            self.load_index(show_progress=False)
        return self._index_by_root[int(root_id)]

    def load_neuron(self, root_id: int) -> PackedNeuronMorphology:
        rec = self.get_index_record(root_id)
        bin_path = self.package_dir / rec["bin_file"]
        with bin_path.open("rb") as f:
            f.seek(int(rec["offset"]))
            rid, type_len, node_count = RECORD_HEADER.unpack(f.read(RECORD_HEADER.size))
            neuron_type = f.read(type_len).decode("utf-8")
            payload = f.read(node_count * NODE_DTYPE.itemsize)
        nodes = np.frombuffer(payload, dtype=NODE_DTYPE).copy()
        return PackedNeuronMorphology(
            root_id=int(rid), neuron_type=neuron_type, nodes=nodes, package=rec["bin_file"]
        )

    def iter_neurons(
        self,
        packages: Optional[Iterable[str]] = None,
        types: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        show_progress: bool = True,
        progress_every: int = 100,
        report_memory: bool = True,
    ) -> Iterator[PackedNeuronMorphology]:
        """Iterate neurons lazily, with progress and memory reporting."""
        if not self._records_by_package:
            self.load_index(show_progress=False)

        package_filter = set(packages) if packages is not None else None
        type_filter = set(types) if types is not None else None

        records = []
        for bin_file, recs in self._records_by_package.items():
            if package_filter is not None and bin_file not in package_filter:
                continue
            for rec in recs:
                if type_filter is not None and rec["type"] not in type_filter:
                    continue
                records.append(rec)
        if limit is not None:
            records = records[:limit]

        total = len(records)
        t0 = time.time()
        loaded_nodes = 0

        iterator = records
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(records, desc="Load morphologies", unit="neuron")
            except Exception:
                pass

        for i, rec in enumerate(iterator, start=1):
            morph = self.load_neuron(int(rec["root_id"]))
            loaded_nodes += morph.node_count
            if show_progress and (i % max(progress_every, 1) == 0 or i == total):
                msg = f"loaded {i:,}/{total:,} neurons, nodes={loaded_nodes:,}"
                if report_memory:
                    rss = _get_rss_mb()
                    if rss is not None:
                        msg += f", RSS={rss:.1f} MB"
                msg += f", elapsed={time.time() - t0:.1f}s"
                print(msg)
            yield morph

    def load_all(
        self,
        types: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        show_progress: bool = True,
        progress_every: int = 100,
    ) -> List[PackedNeuronMorphology]:
        return list(
            self.iter_neurons(
                types=types,
                limit=limit,
                show_progress=show_progress,
                progress_every=progress_every,
            )
        )


def _get_rss_mb() -> Optional[float]:
    """Return current process RSS in MB, if psutil or resource is available."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except Exception:
        pass
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KB.
        if rss > 10**8:
            return rss / (1024**2)
        return rss / 1024
    except Exception:
        return None


def estimate_package_memory(package_dir: Union[str, Path]) -> dict:
    loader = PackedMorphologyLoader(package_dir, preload_index=True)
    total_nodes = 0
    for rec in loader._index_by_root.values():
        total_nodes += int(rec["node_count"])
    return {
        "total_neurons": loader.total_neurons,
        "total_nodes": total_nodes,
        "node_dtype_bytes": NODE_DTYPE.itemsize,
        "estimated_node_payload_mb": total_nodes * NODE_DTYPE.itemsize / (1024**2),
    }


__all__ = [
    "PackedMorphologyLoader",
    "PackedNeuronMorphology",
    "estimate_package_memory",
    "NODE_DTYPE",
]
