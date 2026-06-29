# -*- coding: utf-8 -*-
"""Load a connectome ``network.json`` into a :class:`ScatterConn` + indices.

The JSON contract (see ``Connectome/FAFB v783/.../network.json``):

    metadata: {side, extent, nt_to_sign, forced_negative_pre_types, ...}
    nodes:    [{id, name, u, v, hex_index, input, output}, ...]
    edges:    [{src, tar, sign, n_syn, source_type, target_type, du, dv}, ...]

``sign`` already encodes ``nt_to_sign`` and the ``forced_negative_pre_types``
override, so the per-edge synaptic weight is simply ``sign * n_syn``.

Units are the nodes in file order; ``node_type[i]`` is the index of
``nodes[i]['name']`` in the sorted type vocabulary. This mirrors the Borst path
where unit ``i``'s type is ``i % nofcells`` and lets the schema broadcast a
``(n_types,)`` parameter to ``(n_units,)`` via ``param[node_type]``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from connectivity import ScatterConn

# Default synaptic scale (matches FiveCol exc_synweight == inh_synweight == 0.001).
DEFAULT_SYNWEIGHT = 0.001


@dataclass
class Connectome:
    """A loaded connectome: edge-list backend plus per-node geometry / indices."""

    conn: ScatterConn
    n_units: int
    node_type: torch.Tensor          # (N,) long, index into type_names
    type_names: List[str]            # type vocabulary (len = n_types)
    u: np.ndarray                    # (N,) axial u
    v: np.ndarray                    # (N,) axial v
    hex_index: np.ndarray            # (N,) column hex_index (or -1)
    is_input: np.ndarray             # (N,) bool photoreceptor / stimulus node
    node_ids: List[int]              # (N,) original node ids in unit order
    id_to_unit: Dict[int, int]       # node id -> unit index
    device: str = "cpu"
    meta: dict = field(default_factory=dict)

    @property
    def n_types(self) -> int:
        return len(self.type_names)

    @property
    def center_units(self) -> np.ndarray:
        """Units in the centre column (u == 0 and v == 0)."""
        return np.where((self.u == 0) & (self.v == 0))[0]

    def units_at(self, u: int, v: int) -> np.ndarray:
        """All unit indices sitting on column (u, v)."""
        return np.where((self.u == u) & (self.v == v))[0]

    def input_units_at(self, u: int, v: int) -> np.ndarray:
        """Stimulus (photoreceptor) unit indices on column (u, v)."""
        return np.where((self.u == u) & (self.v == v) & self.is_input)[0]

    def build_signal(
        self,
        maxtime: int = 200,
        baseline: float = 20.0,
        amp: float = 40.0,
        t_on: int = 50,
        center_uv=(0, 0),
    ) -> torch.Tensor:
        """(maxtime, n_units) stimulus current injected into one column's inputs.

        Mirrors the Borst ``signal``: baseline before ``t_on`` then ``amp`` after,
        applied only to the stimulus (photoreceptor) units of ``center_uv``.
        """
        sig = torch.zeros((maxtime, self.n_units), dtype=torch.float64, device=self.device)
        units = self.input_units_at(int(center_uv[0]), int(center_uv[1]))
        if len(units):
            idx = torch.as_tensor(units, dtype=torch.long, device=self.device)
            sig[:t_on, idx] = baseline
            sig[t_on:, idx] = amp
        return sig


def load_connectome(
    path,
    device: Optional[str] = None,
    exc_synweight: float = DEFAULT_SYNWEIGHT,
    inh_synweight: float = DEFAULT_SYNWEIGHT,
) -> Connectome:
    """Read ``network.json`` and return a :class:`Connectome`."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(path)
    with open(path) as f:
        doc = json.load(f)

    nodes = doc["nodes"]
    edges = doc["edges"]
    meta = doc.get("metadata", {})

    n_units = len(nodes)
    node_ids = [int(n["id"]) for n in nodes]
    id_to_unit = {nid: i for i, nid in enumerate(node_ids)}

    type_names = sorted({n["name"] for n in nodes})
    type_to_idx = {t: i for i, t in enumerate(type_names)}
    node_type = np.array([type_to_idx[n["name"]] for n in nodes], dtype=np.int64)

    u = np.array([n.get("u", 0) if n.get("u") is not None else 0 for n in nodes], dtype=np.int64)
    v = np.array([n.get("v", 0) if n.get("v") is not None else 0 for n in nodes], dtype=np.int64)
    hex_index = np.array(
        [n["hex_index"] if n.get("hex_index") is not None else -1 for n in nodes],
        dtype=np.int64,
    )
    is_input = np.array([bool(n.get("input", False)) for n in nodes], dtype=bool)

    # edge list -> unit indices + signed weight (sign * n_syn).
    src_idx = np.empty(len(edges), dtype=np.int64)
    tar_idx = np.empty(len(edges), dtype=np.int64)
    base_w = np.empty(len(edges), dtype=np.float64)
    for k, e in enumerate(edges):
        src_idx[k] = id_to_unit[int(e["src"])]
        tar_idx[k] = id_to_unit[int(e["tar"])]
        base_w[k] = float(e["sign"]) * float(e["n_syn"])

    conn = ScatterConn(
        src_idx=src_idx,
        tar_idx=tar_idx,
        base_w=base_w,
        n_units=n_units,
        node_type=node_type,
        exc_scale=exc_synweight,
        inh_scale=inh_synweight,
        device=device,
    )

    return Connectome(
        conn=conn,
        n_units=n_units,
        node_type=torch.as_tensor(node_type, dtype=torch.long, device=device),
        type_names=type_names,
        u=u,
        v=v,
        hex_index=hex_index,
        is_input=is_input,
        node_ids=node_ids,
        id_to_unit=id_to_unit,
        device=device,
        meta=meta,
    )


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else (
        "../Connectome/FAFB v783/right_min_neuron50_extent2_col/network.json"
    )
    c = load_connectome(p, device="cpu")
    print(f"loaded {p}")
    print(f"n_units={c.n_units}  n_types={c.n_types}  n_edges={len(c.conn.src_idx)}")
    print(f"center units (u=v=0): {c.center_units.tolist()}")
    print(f"input units total: {int(c.is_input.sum())}")
    x = torch.ones(c.n_units, dtype=torch.float64)
    ge, gi = c.conn.exc_inh_drive(x)
    print(f"exc_inh_drive ok: g_exc.sum={float(ge.sum()):.4f} g_inh.sum={float(gi.sum()):.4f}")
    xb = torch.ones((7, c.n_units), dtype=torch.float64)
    geb, _ = c.conn.exc_inh_drive(xb)
    print(f"batched (7,N) ok: shape={tuple(geb.shape)}")
    sig = c.build_signal()
    print(f"signal shape={tuple(sig.shape)}  nonzero cols={int((sig.abs().sum(0)>0).sum())}")
