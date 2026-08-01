# -*- coding: utf-8 -*-
"""Load a ``network.json`` into a :class:`ScatterConn` + indices.

The JSON contract (see ``connectome/FAFBv783/.../network.json``):

    metadata: {side, extent, nt_to_sign, forced_negative_pre_cells, ...}
    nodes:    [{id, name, u, v, column_id, input, output}, ...]
    edges:    [{src, tar, syn_sign, n_syn, source_cell, target_cell, du, dv}, ...]

``syn_sign`` already encodes ``nt_to_sign`` and the ``forced_negative_pre_cells``
override. ``--syn-mode per_cell`` uses ``edge_weight = syn_sign * n_syn``;
``--syn-mode per_edge`` uses ``edge_weight = syn_sign`` (ignore ``n_syn``).

Nodes follow ``network.json`` file order; ``node_cell[i]`` is the index of
``nodes[i]['name']`` in the family-ordered cell vocabulary
(:data:`CELL_FAMILY_ROWS`). This broadcasts per-cell params to nodes via
``param[node_cell]`` (shape ``(n_cells,)`` → ``(n_nodes,)``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import import_bootstrap  # noqa: F401
from .connectivity import ScatterConn
from neuron.schema import normalize_syn_mode

# Photoreceptor drive currents (pA) are injected by the caller
# (``training.defaults.I_*``); this module has no numeric bindings.

# Canonical cell order (photoreceptor → lamina → medulla families).
CELL_FAMILY_ROWS: List[List[str]] = [
    ['R1-6', 'R7', 'R8'],
    ['L1', 'L2', 'L3', 'L4', 'L5'],
    ['Mi1', 'Mi4', 'Mi9'],
    ['T1', 'T2', 'T2a', 'T3'],
    ['T4a', 'T4b', 'T4c', 'T4d'],
    ['T5a', 'T5b', 'T5c', 'T5d'],
    ['Tm1', 'Tm2', 'Tm20', 'Tm21', 'Tm3', 'Tm4', 'Tm9'],
    ['C2', 'C3'],
]


def cell_family_rows(present: Sequence[str]) -> List[List[str]]:
    """Present cells arranged into family rows; skip empty rows; append leftovers."""
    present_names = [str(t) for t in present]
    present_set = set(present_names)
    rows: List[List[str]] = []
    used: set = set()
    for row in CELL_FAMILY_ROWS:
        filtered = [str(t) for t in row if str(t) in present_set]
        if filtered:
            rows.append(filtered)
            used.update(filtered)
    for name in present_names:
        if name not in used:
            rows.append([name])
    return rows


def cell_names_in_family_order(present: Sequence[str]) -> List[str]:
    """Flat cell order from :func:`cell_family_rows`."""
    return [n for row in cell_family_rows(present) for n in row]


@dataclass
class Network:
    """A loaded network: edge-list backend plus per-node geometry / indices."""

    conn: ScatterConn
    n_nodes: int
    node_cell: torch.Tensor          # (N,) long, index into cell_names
    cell_names: List[str]            # cell vocabulary (len = n_cells)
    u: np.ndarray                    # (N,) axial u
    v: np.ndarray                    # (N,) axial v
    column_id: np.ndarray            # (N,) FAFB column_id (or -1)
    is_input: np.ndarray             # (N,) bool photoreceptor / stimulus node
    node_ids: List[int]              # (N,) original node ids in node order
    id_to_node: Dict[int, int]       # node id -> node index
    device: str = "cpu"
    meta: dict = field(default_factory=dict)
    source_json: Optional[Path] = None

    @property
    def n_cells(self) -> int:
        return len(self.cell_names)

    @property
    def center_nodes(self) -> np.ndarray:
        """Nodes in the center hex (u == 0 and v == 0)."""
        return np.where((self.u == 0) & (self.v == 0))[0]

    def nodes_at(self, u: int, v: int) -> np.ndarray:
        """All node indices sitting on hex (u, v)."""
        return np.where((self.u == u) & (self.v == v))[0]

    def input_nodes_at(self, u: int, v: int) -> np.ndarray:
        """Stimulus (photoreceptor) node indices on hex (u, v)."""
        return np.where((self.u == u) & (self.v == v) & self.is_input)[0]

    def build_i_sti(
        self,
        n_t: int,
        *,
        i_baseline: float,
        i_bright: float,
        t_onset: int,
        center_uv=(0, 0),
    ) -> torch.Tensor:
        """(n_t, n_nodes) injected PR current for one hex's inputs."""
        i_sti = torch.zeros((n_t, self.n_nodes), dtype=torch.float64, device=self.device)
        nodes = self.input_nodes_at(int(center_uv[0]), int(center_uv[1]))
        if len(nodes):
            idx = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
            i_sti[:t_onset, idx] = i_baseline
            i_sti[t_onset:, idx] = i_bright
        return i_sti


def node_cell_names(C: Network) -> np.ndarray:
    """(n_nodes,) array of each node's cell NAME."""
    return np.asarray(C.cell_names)[C.node_cell.detach().cpu().numpy()]


def col2sti(C: Network, u: int, v: int) -> np.ndarray:
    """Stimulus (photoreceptor / input) node indices on hex (u, v)."""
    return C.input_nodes_at(int(u), int(v))


def col2gt(
    C: Network,
    u: int,
    v: int,
    gt_type: str,
    names: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Node indices of cell ``gt_type`` on hex (u, v)."""
    if names is None:
        names = node_cell_names(C)
    return np.where(
        (C.u == int(u)) & (C.v == int(v)) & (names == gt_type),
    )[0]


def present_gt_cells(
    gt_cells: Optional[Sequence[str]],
    default_pool: Sequence[str],
    available: Sequence[str],
    *,
    context: str,
) -> List[str]:
    """Intersect requested (or default) gt cells with those present in the network."""
    pool = tuple(gt_cells) if gt_cells is not None else tuple(default_pool)
    present = [st for st in pool if st in set(available)]
    if not present:
        raise ValueError(
            f"{context} has no gt cells (requested {list(pool)!r})",
        )
    return present


def gt_cells_from_opts(opts) -> Optional[Tuple[str, ...]]:
    """``opts['gt_cells']`` as a tuple, or ``None`` if unset."""
    rs = (opts or {}).get("gt_cells")
    if rs is None:
        return None
    return tuple(str(s) for s in rs)


def normalize_gt_cells(gt_cells: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Serialize ``gt_cells`` for stimulus opts, or ``None`` if unset."""
    if gt_cells is None:
        return None
    return [str(s) for s in gt_cells]


def read_network_json(path) -> Tuple[List[dict], List[dict], List[str], dict]:
    """Load ``network.json`` → ``(nodes, edges, family-ordered cell_names, metadata)``."""
    path = Path(path)
    with open(path) as f:
        doc = json.load(f)
    nodes = doc.get("nodes")
    edges = doc.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError(f"invalid network.json (need nodes/edges lists): {path}")
    present = sorted(
        {n["name"] for n in nodes if isinstance(n.get("name"), str)}
    )
    cell_names = cell_names_in_family_order(present)
    meta = doc.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    return nodes, edges, cell_names, meta


def load_network(
    path,
    device: Optional[str] = None,
    *,
    syn_scale_exc: float,
    syn_scale_inh: float,
    syn_mode: str,
    dtype: torch.dtype,
) -> Network:
    """Read ``network.json`` and return a :class:`Network``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(path)
    nodes, edges, cell_names, meta = read_network_json(path)
    mode = normalize_syn_mode(syn_mode)

    n_nodes = len(nodes)
    node_ids = [int(n["id"]) for n in nodes]
    id_to_node = {nid: i for i, nid in enumerate(node_ids)}

    cell_to_idx = {t: i for i, t in enumerate(cell_names)}
    node_cell = np.array([cell_to_idx[n["name"]] for n in nodes], dtype=np.int64)

    u = np.array([n.get("u", 0) if n.get("u") is not None else 0 for n in nodes], dtype=np.int64)
    v = np.array([n.get("v", 0) if n.get("v") is not None else 0 for n in nodes], dtype=np.int64)
    column_id = np.array(
        [n["column_id"] if n.get("column_id") is not None else -1 for n in nodes],
        dtype=np.int64,
    )
    is_input = np.array([bool(n.get("input", False)) for n in nodes], dtype=bool)

    # edge list -> node indices + signed edge weight.
    src_idx = np.empty(len(edges), dtype=np.int64)
    tar_idx = np.empty(len(edges), dtype=np.int64)
    edge_weight = np.empty(len(edges), dtype=np.float64)
    for k, e in enumerate(edges):
        src_idx[k] = id_to_node[int(e["src"])]
        tar_idx[k] = id_to_node[int(e["tar"])]
        syn_sign = float(e["syn_sign"])
        if mode == "per_edge":
            edge_weight[k] = syn_sign
        else:
            edge_weight[k] = syn_sign * float(e["n_syn"])

    conn = ScatterConn(
        src_idx=src_idx,
        tar_idx=tar_idx,
        edge_weight=edge_weight,
        n_nodes=n_nodes,
        node_cell=node_cell,
        syn_scale_exc=syn_scale_exc,
        syn_scale_inh=syn_scale_inh,
        device=device,
        dtype=dtype,
    )

    return Network(
        conn=conn,
        n_nodes=n_nodes,
        node_cell=torch.as_tensor(node_cell, dtype=torch.long, device=device),
        cell_names=cell_names,
        u=u,
        v=v,
        column_id=column_id,
        is_input=is_input,
        node_ids=node_ids,
        id_to_node=id_to_node,
        device=device,
        meta=meta,
        source_json=path.resolve(),
    )


if __name__ == "__main__":
    import sys

    from path import BUILT_NETWORKS_DIR
    from training.defaults import SYN_SCALE_EXC, SYN_SCALE_INH, SYN_MODE
    from training.readout_pack import SIM_DTYPE

    p = sys.argv[1] if len(sys.argv) > 1 else str(
        BUILT_NETWORKS_DIR / "right_min_neuron1_extent2" / "network.json"
    )
    c = load_network(
        p, device="cpu",
        syn_scale_exc=SYN_SCALE_EXC, syn_scale_inh=SYN_SCALE_INH,
        syn_mode=SYN_MODE, dtype=SIM_DTYPE,
    )
    print(f"loaded {p}")
    print(f"n_nodes={c.n_nodes}  n_cells={c.n_cells}  n_edges={len(c.conn.src_idx)}")
    print(f"center nodes (u=v=0): {c.center_nodes.tolist()}")
    print(f"input nodes total: {int(c.is_input.sum())}")
    x = torch.ones(c.n_nodes, dtype=torch.float64)
    syn_strength = torch.ones(c.conn.n_pairs, dtype=torch.float64, device=c.device)
    ge, gi = c.conn.exc_inh_drive(x, syn_strength)
    print(f"exc_inh_drive ok: g_exc.sum={float(ge.sum()):.4f} g_inh.sum={float(gi.sum()):.4f} "
          f"n_pairs={c.conn.n_pairs}")
    xb = torch.ones((7, c.n_nodes), dtype=torch.float64)
    geb, _ = c.conn.exc_inh_drive(xb, syn_strength)
    print(f"batched (7,N) ok: shape={tuple(geb.shape)}")
    i_sti = c.build_i_sti(n_t=10, i_baseline=I_BASELINE, i_bright=I_BRIGHT, t_onset=5)
    print(f"i_sti shape={tuple(i_sti.shape)}  nonzero cols={int((i_sti.abs().sum(0)>0).sum())}")
