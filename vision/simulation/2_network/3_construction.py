# -*- coding: utf-8 -*-
"""Load a ``network.json`` into a :class:`ScatterConn` + idxs.

The JSON contract (see ``connectome/FAFBv783/.../network.json``):

    metadata: {side, radius, sign_from_nt, forced_negative_pre_cells, ...}
    nodes:    [{id, name, u, v, column_id, sti, output}, ...]
    edges:    [{src, tar, syn_sign, n_syn, source_cell, target_cell, du, dv}, ...]

``syn_sign`` already encodes ``sign_from_nt`` and the ``forced_negative_pre_cells``
rule. ``--syn-mode per_cell`` uses ``edge_weights = syn_sign * n_syn``;
``--syn-mode per_edge`` uses ``edge_weights = syn_sign`` (ignore ``n_syn``).

Nodes follow ``network.json`` file order; ``node_cells[i]`` is the index of
``nodes[i]['name']`` in the order-ordered cell vocabulary
(:data:`CELL_ROWS`). This broadcasts per-cell params to nodes via
``param[node_cells]`` (shape ``(n_cells,)`` → ``(n_nodes,)``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import import_bootstrap  # noqa: F401
from .connectivity import ScatterConn
# Photoreceptor sti currents (pA) are injected by the caller
# (``const_default.I_*``); this module has no numeric bindings.

# Canonical cell order for figure layout / param broadcast (cell rows).
# Leftovers (not listed) are appended alphabetically, five per row.
CELL_ROWS: list[list[str]] = [
    ['R1-6', 'R7', 'R8'],
    ['L1', 'L2', 'L3', 'L4', 'L5'],
    ['Mi1', 'Tm3', 'Mi4', 'Mi9'],
    ['T4a', 'T4b', 'T4c', 'T4d'],
    ['Tm1', 'Tm2', 'Tm4', 'Tm9'],
    ['T5a', 'T5b', 'T5c', 'T5d'],
    ['T1', 'T2', 'T2a', 'T3'],
    ['C2', 'C3', 'Lawf1', 'Lawf2'],
]

_LEFTOVER_ROW_LEN = 5


def cell_rows(active: Sequence[str]) -> list[list[str]]:
    """Active cells into rows; leftovers alphabetical, ``_LEFTOVER_ROW_LEN`` per row."""
    active_cells = [str(t) for t in active]
    active_set = set(active_cells)
    rows: list[list[str]] = []
    used: set[str] = set()
    for row in CELL_ROWS:
        filtered = [t for t in row if t in active_set]
        if filtered:
            rows.append(filtered)
            used.update(filtered)
    leftover = sorted(cell for cell in active_cells if cell not in used)
    for start in range(0, len(leftover), _LEFTOVER_ROW_LEN):
        rows.append(leftover[start:start + _LEFTOVER_ROW_LEN])
    return rows


def cells_in_order(active: Sequence[str]) -> list[str]:
    """Flat cell order from :func:`cell_rows`."""
    return [cell for row in cell_rows(active) for cell in row]


@dataclass
class Network:
    """A loaded network: edge-list backend plus per-node geometry / idxs."""

    conn: ScatterConn
    n_nodes: int
    node_cells: torch.Tensor          # (N,) long, index into cells
    cells: list[str]            # cell vocabulary (len = n_cells)
    us: np.ndarray                    # (N,) axial u
    vs: np.ndarray                    # (N,) axial v
    column_ids: np.ndarray            # (N,) FAFB column_ids (or -1)
    is_sti: np.ndarray             # (N,) bool sti (photoreceptor) node
    node_ids: list[int]              # (N,) original node ids in node order
    node_from_id: dict[int, int]   # node id -> node
    device: str = "cpu"
    meta: dict = field(default_factory=dict)
    source_json: Path | None = None

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def sti_nodes(self) -> np.ndarray:
        """Nodes with ``is_sti``."""
        return np.where(self.is_sti)[0]

    def sti_nodes_at_uv(self, u: int, v: int) -> np.ndarray:
        """Sti nodes on hex (u, v)."""
        return np.where((self.us == u) & (self.vs == v) & self.is_sti)[0]


def node_cells(connectome: Network) -> np.ndarray:
    """(n_nodes,) array of each node's cell NAME."""
    return np.asarray(connectome.cells)[connectome.node_cells.detach().cpu().numpy()]


def hex2gt(
    connectome: Network,
    u: int,
    v: int,
    gt_type: str,
    node_cell: np.ndarray | None = None,
) -> np.ndarray:
    """Nodes of cell ``gt_type`` on hex (u, v)."""
    if node_cell is None:
        node_cell = node_cells(connectome)
    return np.where(
        (connectome.us == int(u)) & (connectome.vs == int(v)) & (node_cell == gt_type),
    )[0]


def standardize_cost_radius(cost_radius=None):
    """``None`` or ``-1`` → unrestricted (all hexes); else non-negative int."""
    if cost_radius is None:
        return None
    v = int(cost_radius)
    if v == -1:
        return None
    return v


def cost_radius_mask(u, v, cost_radius=None) -> bool:
    """True when axial ``(u, v)`` has ``hex_radius <= cost_radius`` (``None`` = all hexes)."""
    cost_radius = standardize_cost_radius(cost_radius)
    if cost_radius is None:
        return True
    import build_hex
    return bool(build_hex.radius_mask(int(u), int(v), int(cost_radius)))


def active_gt_cells(
    gt_cells: Sequence[str] | None,
    fallback_gt_cells: Sequence[str],
    available: Sequence[str],
    *,
    context: str,
) -> list[str]:
    """Intersect requested (or fallback) gt cells with those active in the network."""
    keep = tuple(gt_cells) if gt_cells is not None else tuple(fallback_gt_cells)
    avail = set(available)
    active = [st for st in keep if st in avail]
    if not active:
        raise ValueError(
            f"{context} has no gt cells (requested {list(keep)!r})",
        )
    return active


def gt_cells_from_opts(opts) -> tuple[str, ...] | None:
    """``opts['gt_cells']`` as a tuple, or ``None`` if unset."""
    gt_cells = (opts or {}).get("gt_cells")
    if gt_cells is None:
        return None
    return tuple(str(cell) for cell in gt_cells)


def load_network_json(path) -> tuple[list[dict], list[dict], list[str], dict]:
    """Load ``network.json`` → ``(nodes, edges, order-ordered cells, metadata)``."""
    path = Path(path)
    with open(path) as f:
        doc = json.load(f)
    nodes = doc.get("nodes")
    edges = doc.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError(f"invalid network.json (need nodes/edges lists): {path}")
    active = sorted(
        {n["name"] for n in nodes if isinstance(n.get("name"), str)}
    )
    cells = cells_in_order(active)
    meta = doc.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    return nodes, edges, cells, meta


def load_network(
    path,
    device: str | None = None,
    *,
    a_syn_exc: float,
    a_syn_inh: float,
    syn_mode: str,
    dtype: torch.dtype,
) -> Network:
    """Read ``network.json`` and return a :class:`Network``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(path)
    nodes, edges, cells, meta = load_network_json(path)
    mode = syn_mode

    n_nodes = len(nodes)
    node_ids = [int(n["id"]) for n in nodes]
    node_from_id = {node_id: node for node, node_id in enumerate(node_ids)}

    cell_idx = dict(zip(cells, range(len(cells))))
    node_cells = np.array([cell_idx[n["name"]] for n in nodes], dtype=np.int64)

    us = np.array(
        [0 if n.get("u") is None else int(n["u"]) for n in nodes], dtype=np.int64,
    )
    vs = np.array(
        [0 if n.get("v") is None else int(n["v"]) for n in nodes], dtype=np.int64,
    )
    column_ids = np.array(
        [-1 if n.get("column_id") is None else int(n["column_id"]) for n in nodes],
        dtype=np.int64,
    )
    is_sti = np.array([bool(n.get("sti", False)) for n in nodes], dtype=bool)

    # edge list -> nodes + signed edge weight.
    source_idxs = np.empty(len(edges), dtype=np.int64)
    target_idxs = np.empty(len(edges), dtype=np.int64)
    edge_weights = np.empty(len(edges), dtype=np.float64)
    for k, e in enumerate(edges):
        source_idxs[k] = node_from_id[int(e["src"])]
        target_idxs[k] = node_from_id[int(e["tar"])]
        syn_sign = float(e["syn_sign"])
        edge_weights[k] = (
            syn_sign if mode == "per_edge" else syn_sign * float(e["n_syn"])
        )

    conn = ScatterConn(
        source_idxs=source_idxs,
        target_idxs=target_idxs,
        edge_weights=edge_weights,
        n_nodes=n_nodes,
        node_cells=node_cells,
        a_syn_exc=a_syn_exc,
        a_syn_inh=a_syn_inh,
        device=device,
        dtype=dtype,
    )

    return Network(
        conn=conn,
        n_nodes=n_nodes,
        node_cells=torch.as_tensor(node_cells, dtype=torch.long, device=device),
        cells=cells,
        us=us,
        vs=vs,
        column_ids=column_ids,
        is_sti=is_sti,
        node_ids=node_ids,
        node_from_id=node_from_id,
        device=device,
        meta=meta,
        source_json=path.resolve(),
    )
