# -*- coding: utf-8 -*-
"""Load a ``network.json`` into a :class:`ScatterConn` + indices.

The JSON contract (see ``connectome/FAFBv783/.../network.json``):

    metadata: {side, radius, sign_from_nt, forced_negative_pre_cells, ...}
    nodes:    [{id, name, u, v, column_id, sti, output}, ...]
    edges:    [{src, tar, syn_sign, n_syn, source_cell, target_cell, du, dv}, ...]

``syn_sign`` already encodes ``sign_from_nt`` and the ``forced_negative_pre_cells``
override. ``--syn-mode per_cell`` uses ``edge_weights = syn_sign * n_syn``;
``--syn-mode per_edge`` uses ``edge_weights = syn_sign`` (ignore ``n_syn``).

Nodes follow ``network.json`` file order; ``node_cells[i]`` is the index of
``nodes[i]['name']`` in the order-ordered cell vocabulary
(:data:`CELL_PLOT_ROWS`). This broadcasts per-cell params to nodes via
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
from neuron.schema import normalize_syn_mode

# Photoreceptor drive currents (pA) are injected by the caller
# (``default_params.I_*``); this module has no numeric bindings.

# Canonical cell order for plot / param broadcast (plot rows).
# Leftovers (not listed) are appended alphabetically, five per plot row.
CELL_PLOT_ROWS: list[list[str]] = [
    ['R1-6', 'R7', 'R8'],
    ['L1', 'L2', 'L3', 'L4', 'L5'],
    ['Mi1', 'Tm3', 'Mi4', 'Mi9'],
    ['T4a', 'T4b', 'T4c', 'T4d'],
    ['Tm1', 'Tm2', 'Tm4', 'Tm9'],
    ['T5a', 'T5b', 'T5c', 'T5d'],
    ['T1', 'T2', 'T2a', 'T3'],
    ['C2', 'C3', 'Lawf1', 'Lawf2'],
]

_LEFTOVER_PLOT_ROW_LEN = 5


def cell_plot_rows(present: Sequence[str]) -> list[list[str]]:
    """Present cells into plot rows; leftovers alphabetical, ``_LEFTOVER_PLOT_ROW_LEN`` per row."""
    present_names = [str(t) for t in present]
    present_set = set(present_names)
    plot_rows: list[list[str]] = []
    used: set[str] = set()
    for plot_row in CELL_PLOT_ROWS:
        filtered = [t for t in plot_row if t in present_set]
        if filtered:
            plot_rows.append(filtered)
            used.update(filtered)
    leftover = sorted(name for name in present_names if name not in used)
    for i in range(0, len(leftover), _LEFTOVER_PLOT_ROW_LEN):
        plot_rows.append(leftover[i : i + _LEFTOVER_PLOT_ROW_LEN])
    return plot_rows


def cells_in_order(present: Sequence[str]) -> list[str]:
    """Flat cell order from :func:`cell_plot_rows`."""
    return [n for plot_row in cell_plot_rows(present) for n in plot_row]


@dataclass
class Network:
    """A loaded network: edge-list backend plus per-node geometry / indices."""

    conn: ScatterConn
    n_nodes: int
    node_cells: torch.Tensor          # (N,) long, index into cells
    cells: list[str]            # cell vocabulary (len = n_cells)
    us: np.ndarray                    # (N,) axial u
    vs: np.ndarray                    # (N,) axial v
    column_ids: np.ndarray            # (N,) FAFB column_ids (or -1)
    is_sti: np.ndarray             # (N,) bool sti (photoreceptor) node
    node_ids: list[int]              # (N,) original node ids in node order
    node_from_id: dict[int, int]       # node id -> node index
    device: str = "cpu"
    meta: dict = field(default_factory=dict)
    source_json: Path | None = None

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def sti_node_indices(self) -> np.ndarray:
        """Node indices with ``is_sti``."""
        return np.where(self.is_sti)[0]

    def sti_nodes_at(self, u: int, v: int) -> np.ndarray:
        """Sti node indices on hex (u, v)."""
        return np.where((self.us == u) & (self.vs == v) & self.is_sti)[0]


def node_cell_names(connectome: Network) -> np.ndarray:
    """(n_nodes,) array of each node's cell NAME."""
    return np.asarray(connectome.cells)[connectome.node_cells.detach().cpu().numpy()]


def hex2gt(
    connectome: Network,
    u: int,
    v: int,
    gt_type: str,
    names: np.ndarray | None = None,
) -> np.ndarray:
    """Node indices of cell ``gt_type`` on hex (u, v)."""
    if names is None:
        names = node_cell_names(connectome)
    return np.where(
        (connectome.us == int(u)) & (connectome.vs == int(v)) & (names == gt_type),
    )[0]


def normalize_cost_radius(cost_radius=None):
    """``None`` or ``-1`` → unrestricted (all hexes); else non-negative int."""
    if cost_radius is None:
        return None
    v = int(cost_radius)
    if v == -1:
        return None
    return v


def hex_in_cost_radius(u, v, cost_radius=None) -> bool:
    """True when axial ``(u, v)`` lies in the cost hex disc (``None`` = all hexes)."""
    cost_radius = normalize_cost_radius(cost_radius)
    if cost_radius is None:
        return True
    import build_hex
    return bool(build_hex.inside_mask(int(u), int(v), int(cost_radius)))


def present_gt_cells(
    gt_cells: Sequence[str] | None,
    default_pool: Sequence[str],
    available: Sequence[str],
    *,
    context: str,
) -> list[str]:
    """Intersect requested (or default) gt cells with those present in the network."""
    pool = tuple(gt_cells) if gt_cells is not None else tuple(default_pool)
    avail = set(available)
    present = [st for st in pool if st in avail]
    if not present:
        raise ValueError(
            f"{context} has no gt cells (requested {list(pool)!r})",
        )
    return present


def gt_cells_from_opts(opts) -> tuple[str, ...] | None:
    """``opts['gt_cells']`` as a tuple, or ``None`` if unset."""
    rs = (opts or {}).get("gt_cells")
    if rs is None:
        return None
    return tuple(str(s) for s in rs)


def normalize_gt_cells(gt_cells: Sequence[str] | None) -> list[str] | None:
    """Serialize ``gt_cells`` for sti opts, or ``None`` if unset."""
    if gt_cells is None:
        return None
    return [str(s) for s in gt_cells]


def load_network_json(path) -> tuple[list[dict], list[dict], list[str], dict]:
    """Load ``network.json`` → ``(nodes, edges, order-ordered cells, metadata)``."""
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
    cells = cells_in_order(present)
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
    mode = normalize_syn_mode(syn_mode)

    n_nodes = len(nodes)
    node_ids = [int(n["id"]) for n in nodes]
    node_from_id = {nid: i for i, nid in enumerate(node_ids)}

    idx_from_cell = {t: i for i, t in enumerate(cells)}
    node_cells = np.array([idx_from_cell[n["name"]] for n in nodes], dtype=np.int64)

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

    # edge list -> node indices + signed edge weight.
    source_indices = np.empty(len(edges), dtype=np.int64)
    target_indices = np.empty(len(edges), dtype=np.int64)
    edge_weights = np.empty(len(edges), dtype=np.float64)
    for k, e in enumerate(edges):
        source_indices[k] = node_from_id[int(e["src"])]
        target_indices[k] = node_from_id[int(e["tar"])]
        syn_sign = float(e["syn_sign"])
        edge_weights[k] = (
            syn_sign if mode == "per_edge" else syn_sign * float(e["n_syn"])
        )

    conn = ScatterConn(
        source_indices=source_indices,
        target_indices=target_indices,
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
