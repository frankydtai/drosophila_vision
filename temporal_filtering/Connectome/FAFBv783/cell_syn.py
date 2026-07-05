#!/usr/bin/env python3
"""
For the connectome ``network.json`` files under this folder (built by
``build_network.py`` and stored in e.g. ``left_min_neuron1/network.json``),
tabulate the synaptic partners of one or more *cell types*.

By default (incoming / ``pre``) each CELL_TYPE is treated as the postsynaptic
``target_type`` and broken down by presynaptic ``source_type``. With ``--post``
(outgoing) each CELL_TYPE is treated as the presynaptic ``source_type`` and broken
down by postsynaptic ``target_type``.

A CELL_TYPE token may be a cell type (e.g. ``Mi1``); a *family* when prefixed with
``&`` (e.g. ``&Centrifugal``) which aggregates over all its member types; or a single
neuron when prefixed with ``@`` (e.g. ``@720575940622041087``) selected by FlyWire
root id. The breakdown column still shows individual ``source_type``/``target_type``
unless ``--family`` is given.

Optionally restrict to CELL_TYPE *instances* at a single location: axial ``(u, v)``
(``--at-uv U V``) or pixel ``(x, y)`` (``--at-xy X Y``) for FAFB ``network.json``,
or the central hex disc ``--extent N`` (0 = centre column, 1 = 7 columns, 2 = 19, …;
uses ``column_mapper.inside_mask``). ``--extent`` is FAFB-only and shows mean
``pre_d_xy``/``post_d_xy`` only.
With ``--borst`` (5-column ``multi_colM`` model) use ``--at-xy`` only — Borst column
centres sit on ``y=0`` at ``x`` in ``{-10,-5,0,5,10}`` (resolved via
``column_mapper.borst_col_at_xy``); ``--at-uv``, ``--extent``, and ``--family`` are invalid
(no ``type_counts_abc.csv`` on the Borst path). Borst output never shows axial
``(u,v)``; with ``--at-xy`` only ``pre_d_xy``/``post_d_xy``. Without ``--at-uv``/``--at-xy``,
only ``pre_d_xy``/``post_d_xy`` is shown (mean partner pixel delta). The reference is
the mean ``(x,y)`` of all queried *self* instances; each partner row shows **one**
mean ``(dx,dy)`` over that partner's instances. With ``--at-uv`` only ``pre_d_uv``/
``post_d_uv`` is added (hex reference). With ``--at-xy`` the reference is the
``--at`` centre; if ``n_neuron`` ≤ 5, distinct partner ``(dx,dy)`` pairs are listed,
otherwise the mean delta is shown.

Per (cell_type, partner_type): sum ``n_syn`` where ``sign > 0`` vs ``sign < 0``,
then express each as a percentage of **all** ``n_syn`` for that cell type. An
``n_neuron`` column is always shown. The TOTAL row omits the coord columns.

The ``network.json`` schema is ``{"metadata", "nodes", "edges"}`` where each node is
``{"id", "name", "u", "v", "column_id", "input", "output"}`` and each edge is
``{"src", "tar", "sign", "n_syn", "source_type", "target_type", "du", "dv"}``.

Example::

    python3 "cell_syn.py"
    python3 "cell_syn.py" L1,L2,L3,L4,L5
    python3 "cell_syn.py" T4a,T4b,T4c,T4d
    python3 "cell_syn.py" Mi1 --post
    python3 "cell_syn.py" &Centrifugal
    python3 "cell_syn.py" &Centrifugal --family
    python3 "cell_syn.py" Mi1 --family
    python3 "cell_syn.py" @720575940622041087
    python3 "cell_syn.py" Mi1 --network right_min_neuron1
    python3 "cell_syn.py" L1 --network /abs/path/to/some_folder
    python3 "cell_syn.py" Mi1 --post --at-uv 0 0
    python3 "cell_syn.py" Mi1 --post --at-xy 0 1
    python3 "cell_syn.py" Mi1 --extent 0
    python3 "cell_syn.py" Mi1 --extent 2
    python3 "cell_syn.py" Mi1 --borst
    python3 "cell_syn.py" Mi1 --borst --post
    python3 "cell_syn.py" Mi1 --borst --at-xy 0 0
    python3 "cell_syn.py" Mi1 --borst --at-xy -5 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, Union

_UvCoord = Tuple[Union[int, float], Union[int, float]]

from column_mapper import (
    DEFAULT_KERNEL_SIZE,
    borst_col_at_xy,
    borst_column_centers,
    hex_to_pixel,
    inside_mask,
    pixel_to_hex,
)
from connectome_io import (
    BORST_CTYPE_NPY,
    BORST_MULTI_COL_M,
    SIMULATION_CODE_DIR,
    resolve_network_json,
    type_counts_abc_path,
)

_DEFAULT_NETWORK = "right_min_neuron1"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


_MAX_PARTNER_LIST = 5


def _parse_cell_types(spec: str) -> List[str]:
    """Parse one comma-separated CELL_TYPE string (e.g. ``T4a,T4b,T4c``)."""
    return [part.strip() for part in spec.split(",") if part.strip()]


def _hex_disc_column_count(extent: int) -> int:
    """Hex cells in a disc of axial radius ``extent`` (0 -> 1, 1 -> 7, 2 -> 19, …)."""
    return 3 * extent * (extent + 1) + 1


def _load_type_to_family(json_path: Path) -> Dict[str, str]:
    """Map cell ``type`` -> ``family`` from ``type_counts_abc.csv`` next to network.json."""
    csv_path = type_counts_abc_path(json_path)
    out: Dict[str, str] = {}
    if not csv_path.is_file():
        logger.warning("No type_counts_abc.csv next to %s; family names won't resolve", json_path)
        return out
    import csv

    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            t = row.get("type")
            fam = row.get("family")
            if t:
                out[t] = fam if fam else t
    return out


def _import_medulla_library():
    """Lazy import of SimulationCode ``Medulla_Library`` (Borst path only)."""
    import os

    sim = SIMULATION_CODE_DIR
    sim_s = str(sim)
    if sim_s not in sys.path:
        sys.path.insert(0, sim_s)
    prev = os.getcwd()
    try:
        os.chdir(sim)
        import Medulla_Library as ml  # noqa: WPS433

        return ml
    finally:
        os.chdir(prev)


def _load_borst_graph() -> Tuple[List[dict], List[dict]]:
    """Build synthetic ``nodes``/``edges`` from patched ``multi_colM.npy``."""
    import numpy as np

    if not BORST_MULTI_COL_M.is_file():
        raise FileNotFoundError(f"Borst matrix not found: {BORST_MULTI_COL_M}")

    ml = _import_medulla_library()
    matrix = ml.apply_borst_connectivity_patches(np.load(BORST_MULTI_COL_M).copy())
    if not BORST_CTYPE_NPY.is_file():
        raise FileNotFoundError(f"Borst ctype not found: {BORST_CTYPE_NPY}")
    ctype = np.load(BORST_CTYPE_NPY, allow_pickle=True)
    centers = borst_column_centers()
    col_center = {c.col: c for c in centers}

    nodes: List[dict] = []
    for col in range(ml.nofcols):
        c = col_center[col]
        for type_idx in range(ml.nofcells):
            unit_id = col * ml.nofcells + type_idx
            nodes.append(
                {
                    "id": unit_id,
                    "name": str(ctype[type_idx]),
                    "u": c.u,
                    "v": c.v,
                    "x": c.x,
                    "y": c.y,
                    "column_id": col,
                    "input": False,
                    "output": False,
                }
            )

    edges: List[dict] = []
    n_units = ml.nofcols * ml.nofcells
    for post in range(n_units):
        post_col = post // ml.nofcells
        post_center = col_center[post_col]
        post_name = str(ctype[post % ml.nofcells])
        for pre in range(n_units):
            weight = float(matrix[post, pre])
            if weight == 0.0:
                continue
            pre_col = pre // ml.nofcells
            pre_center = col_center[pre_col]
            edges.append(
                {
                    "src": pre,
                    "tar": post,
                    "sign": 1.0 if weight > 0.0 else -1.0,
                    "n_syn": abs(weight),
                    "source_type": str(ctype[pre % ml.nofcells]),
                    "target_type": post_name,
                    "du": pre_center.u - post_center.u,
                    "dv": pre_center.v - post_center.v,
                }
            )

    return nodes, edges


def _resolve_query_labels(
    tokens: List[str], type_to_family: Dict[str, str]
) -> Tuple[List[str], Dict[str, Set[str]], Dict[int, Set[str]]]:
    """Resolve queried tokens to (ordered labels, self_type -> labels, self_id -> labels).

    Token prefixes:
      - ``&Family`` aggregates over every member type of that family.
      - ``@<root_id>`` selects a single neuron by FlyWire root id.
      - anything else is a literal cell type.
    The label shown in the output is the token as typed (e.g. ``&Centrifugal``,
    ``@720575940622041087``).
    """
    family_to_types: DefaultDict[str, List[str]] = defaultdict(list)
    for t, fam in type_to_family.items():
        family_to_types[fam].append(t)
    labels: List[str] = list(dict.fromkeys(tokens))
    self_type_to_labels: DefaultDict[str, Set[str]] = defaultdict(set)
    self_id_to_labels: DefaultDict[int, Set[str]] = defaultdict(set)
    for tok in labels:
        if tok.startswith("&"):
            fam = tok[1:]
            members = family_to_types.get(fam, [])
            if not members:
                logger.warning("Family %r not found in type_counts_abc.csv", fam)
            for t in members:
                self_type_to_labels[t].add(tok)
        elif tok.startswith("@"):
            try:
                self_id_to_labels[int(tok[1:])].add(tok)
            except ValueError:
                logger.warning("Invalid root id token %r (expected @<int>)", tok)
        else:
            self_type_to_labels[tok].add(tok)
    return labels, dict(self_type_to_labels), dict(self_id_to_labels)


def _format_scalar_for_table(z: float) -> str:
    if abs(z - round(z)) < 1e-9:
        return str(int(round(z)))
    return f"{z:g}"


def _format_mean_scalar_for_table(z: float) -> str:
    return f"{z:.2f}"


def _node_centers(
    n: dict,
    *,
    float_coords: bool,
    xy_kernel_size: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Parse one node's ``(u,v)`` and pixel ``(x,y)`` centres."""
    try:
        u = float(n["u"]) if float_coords else float(int(n["u"]))
        v = float(n["v"]) if float_coords else float(int(n["v"]))
        if "x" in n and "y" in n:
            xy = (float(n["x"]), float(n["y"]))
        else:
            x, y = hex_to_pixel(u, v, kernel_size=xy_kernel_size)
            xy = (float(x), float(y))
        return (u, v), xy
    except (KeyError, TypeError, ValueError):
        return None


def _format_delta_uv(
    uvs: Set[_UvCoord],
    origin: Tuple[float, float],
    *,
    mean: bool,
) -> str:
    """Format partner ``(du,dv)`` as one mean pair or a sorted distinct list."""
    if not uvs:
        return ""
    ou, ov = origin
    if mean:
        n = float(len(uvs))
        du = sum(float(u) - ou for u, v in uvs) / n
        dv = sum(float(v) - ov for u, v in uvs) / n
        return (
            f"({_format_mean_scalar_for_table(du)},{_format_mean_scalar_for_table(dv)})"
        )
    return ";".join(
        f"({_format_scalar_for_table(float(u) - ou)},{_format_scalar_for_table(float(v) - ov)})"
        for u, v in sorted(uvs)
    )


def _format_delta_xy(
    uvs: Set[_UvCoord],
    origin: Tuple[float, float],
    *,
    mean: bool,
    kernel_size: float = 1.0,
) -> str:
    """Format partner ``(dx,dy)`` in pixel space as mean or sorted distinct list."""
    if not uvs:
        return ""
    ox, oy = origin
    if mean:
        n = float(len(uvs))
        dx = dy = 0.0
        for u, v in uvs:
            x, y = hex_to_pixel(u, v, kernel_size=kernel_size)
            dx += float(x) - ox
            dy += float(y) - oy
        dx /= n
        dy /= n
        return (
            f"({_format_mean_scalar_for_table(dx)},{_format_mean_scalar_for_table(dy)})"
        )
    parts: List[str] = []
    for u, v in sorted(uvs):
        x, y = hex_to_pixel(u, v, kernel_size=kernel_size)
        parts.append(
            f"({_format_scalar_for_table(float(x) - ox)},"
            f"{_format_scalar_for_table(float(y) - oy)})"
        )
    return ";".join(parts)


def _self_node_origin(
    label: str,
    nodes: List[dict],
    *,
    float_coords: bool,
    xy_kernel_size: float,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """``@root_id`` label -> that node's ``(u,v)`` and ``(x,y)`` centres."""
    if not label.startswith("@"):
        return None, None
    try:
        nid = int(label[1:])
    except ValueError:
        return None, None
    for n in nodes:
        try:
            if int(n["id"]) != nid:
                continue
            centers = _node_centers(n, float_coords=float_coords, xy_kernel_size=xy_kernel_size)
            if centers is None:
                continue
            return centers
        except (KeyError, TypeError, ValueError):
            continue
    return None, None


def _mean_self_origin(
    label: str,
    nodes: List[dict],
    self_type_to_labels: Dict[str, Set[str]],
    ids_at_hex: Optional[Dict[str, Set[int]]],
    *,
    float_coords: bool,
    xy_kernel_size: float,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Mean ``(u,v)`` and ``(x,y)`` over all *self* instances for ``label``."""
    self_types = {t for t, labs in self_type_to_labels.items() if label in labs}
    if not self_types:
        return None, None
    uvs: List[Tuple[float, float]] = []
    xys: List[Tuple[float, float]] = []
    for n in nodes:
        name = n.get("name")
        if not isinstance(name, str) or name not in self_types:
            continue
        try:
            nid = int(n["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if ids_at_hex is not None:
            allowed = ids_at_hex.get(name, set())
            if nid not in allowed:
                continue
        centers = _node_centers(n, float_coords=float_coords, xy_kernel_size=xy_kernel_size)
        if centers is None:
            continue
        uv, xy = centers
        uvs.append(uv)
        xys.append(xy)
    if not uvs:
        return None, None
    n = float(len(uvs))
    mean_uv = (sum(p[0] for p in uvs) / n, sum(p[1] for p in uvs) / n)
    mean_xy = (sum(p[0] for p in xys) / n, sum(p[1] for p in xys) / n)
    return mean_uv, mean_xy


def _label_origins(
    label: str,
    nodes: List[dict],
    self_type_to_labels: Dict[str, Set[str]],
    ids_at_hex: Optional[Dict[str, Set[int]]],
    at_ref_uv: Optional[Tuple[float, float]],
    at_ref_xy: Optional[Tuple[float, float]],
    *,
    float_coords: bool,
    xy_kernel_size: float,
    need_uv: bool,
    need_xy: bool,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Reference ``(u,v)``/``(x,y)`` for partner deltas: ``--at`` centre or self mean."""
    origin_uv = at_ref_uv if need_uv else None
    origin_xy = at_ref_xy if need_xy else None
    if label.startswith("@"):
        self_uv, self_xy = _self_node_origin(
            label, nodes, float_coords=float_coords, xy_kernel_size=xy_kernel_size
        )
        if origin_uv is None:
            origin_uv = self_uv
        if origin_xy is None:
            origin_xy = self_xy
    elif origin_uv is None or origin_xy is None:
        mean_uv, mean_xy = _mean_self_origin(
            label,
            nodes,
            self_type_to_labels,
            ids_at_hex,
            float_coords=float_coords,
            xy_kernel_size=xy_kernel_size,
        )
        if origin_uv is None:
            origin_uv = mean_uv
        if origin_xy is None:
            origin_xy = mean_xy
    return origin_uv, origin_xy


def _xy_to_uv(x: float, y: float) -> Tuple[int, int]:
    """Inverse of ``column_mapper.hex_to_pixel(kernel_size=1)`` for hex centers."""
    return pixel_to_hex(x, y, kernel_size=1.0)


def _node_id_to_uv(nodes: List[dict], *, float_coords: bool = False) -> Dict[int, _UvCoord]:
    """Unit id -> hex (u, v) from network nodes."""
    m: Dict[int, _UvCoord] = {}
    for n in nodes:
        try:
            centers = _node_centers(n, float_coords=float_coords, xy_kernel_size=1.0)
            if centers is None:
                continue
            uv, _ = centers
            m[int(n["id"])] = uv
        except (KeyError, TypeError, ValueError):
            continue
    return m


def _instance_ids_at_hex(
    nodes: List[dict], u: int, v: int
) -> Dict[str, Set[int]]:
    """Map cell type (node ``name``) -> FlyWire root ids at hex (u, v)."""
    out: Dict[str, Set[int]] = {}
    for n in nodes:
        try:
            nu, nv = int(n["u"]), int(n["v"])
        except (KeyError, TypeError, ValueError):
            continue
        if nu != u or nv != v:
            continue
        name = n.get("name")
        if not isinstance(name, str):
            continue
        try:
            nid = int(n["id"])
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault(name, set()).add(nid)
    return out


def _instance_ids_in_disc(nodes: List[dict], extent: int) -> Dict[str, Set[int]]:
    """Map cell type -> root ids inside the central hex disc of radius ``extent``."""
    out: Dict[str, Set[int]] = {}
    for n in nodes:
        try:
            u, v = int(n["u"]), int(n["v"])
        except (KeyError, TypeError, ValueError):
            continue
        if not bool(inside_mask(u, v, extent)):
            continue
        name = n.get("name")
        if not isinstance(name, str):
            continue
        try:
            nid = int(n["id"])
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault(name, set()).add(nid)
    return out


def _instance_ids_at_col(nodes: List[dict], col: int) -> Dict[str, Set[int]]:
    """Map cell type -> unit ids in one Borst column (``column_id`` match)."""
    out: Dict[str, Set[int]] = {}
    for n in nodes:
        try:
            ncol = int(n["column_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if ncol != col:
            continue
        name = n.get("name")
        if not isinstance(name, str):
            continue
        try:
            nid = int(n["id"])
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault(name, set()).add(nid)
    return out


def _edge_sign(e: dict) -> float:
    """Signed weight for an edge from its ``sign`` field (±1)."""
    try:
        return float(e.get("sign", 0))
    except (TypeError, ValueError):
        return 0.0


def _accumulate_all(
    edges: List[dict],
    labels: List[str],
    self_type_to_labels: Dict[str, Set[str]],
    id_to_uv: Dict[int, _UvCoord],
    ids_at_hex: Optional[Dict[str, Set[int]]] = None,
    direction: str = "pre",
    type_to_family: Optional[Dict[str, str]] = None,
    self_id_to_labels: Optional[Dict[int, Set[str]]] = None,
) -> Dict[
    str,
    Tuple[
        DefaultDict[str, Dict[str, float]],
        float,
        Dict[str, int],
        Dict[str, Set[_UvCoord]],
        int,
    ],
]:
    """One pass over edges: per queried label, (per partner type syn+/syn-, total n_syn).

    ``labels`` is the ordered list of queried tokens (a cell type, a family entered as
    ``&Family``, or a single neuron entered as ``@<root_id>``). ``self_type_to_labels``
    maps each *self* cell type to its label(s); ``self_id_to_labels`` maps a *self* root
    id to its label(s). A family label aggregates over all its member types.

    ``direction="pre"`` (default): query each label as the **postsynaptic** side
    (``target_type``) and break down by presynaptic ``source_type`` (incoming).
    ``direction="post"``: query each label as the **presynaptic** side (``source_type``)
    and break down by postsynaptic ``target_type`` (outgoing).

    If ``ids_at_hex`` is set, only edges whose *self* instance id (``tar`` for ``pre``,
    ``src`` for ``post``) sits at the chosen hex are counted. The third return value
    maps partner type -> count of **distinct** *partner* instance ids. The fourth maps
    partner type -> distinct partner ``(u,v)`` centres.
    """
    if direction == "post":
        self_type_field, partner_type_field = "source_type", "target_type"
        self_id_field, partner_id_field = "src", "tar"
    else:
        self_type_field, partner_type_field = "target_type", "source_type"
        self_id_field, partner_id_field = "tar", "src"

    by_cell: Dict[str, DefaultDict[str, Dict[str, float]]] = {
        p: defaultdict(lambda: {"syn+": 0.0, "syn-": 0.0}) for p in labels
    }
    totals: Dict[str, float] = {p: 0.0 for p in labels}
    # Always count distinct partner neurons per partner type (-> n_neuron column).
    partner_ids: Dict[str, DefaultDict[str, Set[int]]] = {
        p: defaultdict(set) for p in labels
    }
    partner_uv: Dict[str, DefaultDict[str, Set[_UvCoord]]] = {
        p: defaultdict(set) for p in labels
    }
    self_ids: Dict[str, Set[int]] = {p: set() for p in labels}
    for e in edges:
        stype = e.get(self_type_field)
        self_id_raw = e.get(self_id_field)
        try:
            self_id_int: Optional[int] = int(self_id_raw)
        except (TypeError, ValueError):
            self_id_int = None

        cell_labels: Set[str] = set()
        type_labels = self_type_to_labels.get(stype)
        if type_labels:
            cell_labels |= type_labels
        if self_id_to_labels and self_id_int is not None:
            id_labels = self_id_to_labels.get(self_id_int)
            if id_labels:
                cell_labels |= id_labels
        if not cell_labels:
            continue
        if ids_at_hex is not None:
            allowed = ids_at_hex.get(stype, set())
            if not allowed or self_id_int is None or self_id_int not in allowed:
                continue
        pt = e.get(partner_type_field) or "?"
        if type_to_family is not None:
            pt = type_to_family.get(pt, pt)
        a = _edge_sign(e)
        ns = float(e.get("n_syn", 0))
        partner = e.get(partner_id_field)
        for cell in cell_labels:
            totals[cell] += ns
            if self_id_int is not None:
                self_ids[cell].add(self_id_int)
            if a > 0:
                by_cell[cell][pt]["syn+"] += ns
            elif a < 0:
                by_cell[cell][pt]["syn-"] += ns
            if partner is not None:
                try:
                    pid = int(partner)
                    partner_ids[cell][pt].add(pid)
                    uv = id_to_uv.get(pid)
                    if uv is not None:
                        partner_uv[cell][pt].add(uv)
                except (TypeError, ValueError):
                    pass
    out: Dict[
        str,
        Tuple[
            DefaultDict[str, Dict[str, float]],
            float,
            Dict[str, int],
            Dict[str, Set[_UvCoord]],
            int,
            int,
        ],
    ] = {}
    for p in labels:
        npartner_map = {pt: len(ids) for pt, ids in partner_ids[p].items()}
        row_sets = {pt: set(uvs) for pt, uvs in partner_uv[p].items()}
        n_self = len(self_ids[p])
        out[p] = (by_cell[p], totals[p], npartner_map, row_sets, n_self)
    return out


def print_table(
    cell_type: str,
    by_partner: DefaultDict[str, Dict[str, float]],
    total_syn: float,
    n_partner_by_type: Dict[str, int],
    partner_uv_by_type: Dict[str, Set[_UvCoord]],
    hex_note: str = "",
    direction: str = "pre",
    use_family: bool = False,
    min_pct: float = 0.0,
    xy_kernel_size: float = 1.0,
    show_uv: bool = True,
    show_xy: bool = True,
    origin_uv: Optional[Tuple[float, float]] = None,
    origin_xy: Optional[Tuple[float, float]] = None,
    mean_partner_delta: bool = False,
    n_self: int = 0,
) -> None:
    partner_dim = "family" if use_family else "type"
    self_dim = "id" if cell_type.startswith("@") else "type"
    if direction == "post":
        side = "post"
        self_field, partner_field = f"source_{self_dim}", f"target_{partner_dim}"
        flow_word = "out of"
    else:
        side = "pre"
        self_field, partner_field = f"target_{self_dim}", f"source_{partner_dim}"
        flow_word = "onto"
    n_label = "n_neuron"
    uv_label = f"{side}_d_uv"
    xy_label = f"{side}_d_xy"

    header = [partner_field, "% n_syn+", "% n_syn-", n_label]
    if show_uv:
        header.append(uv_label)
    if show_xy:
        header.append(xy_label)

    rows: List[List[str]] = []
    sum_p = sum_m = 0.0
    if total_syn <= 0:
        logger.warning("No n_syn for %s=%s", self_field, cell_type)
    else:
        for pt in sorted(by_partner):
            d = by_partner[pt]
            pp = 100.0 * d["syn+"] / total_syn
            pm = 100.0 * d["syn-"] / total_syn
            sum_p += pp
            sum_m += pm
            if pp + pm <= min_pct:
                continue
            row = [pt, f"{pp:.4f}", f"{pm:.4f}"]
            npv = int(n_partner_by_type.get(pt, 0))
            row.append(str(npv))
            uvs = partner_uv_by_type.get(pt, set())
            use_mean_delta = mean_partner_delta or npv > _MAX_PARTNER_LIST
            if show_uv:
                if origin_uv is None:
                    row.append("")
                else:
                    row.append(_format_delta_uv(uvs, origin_uv, mean=use_mean_delta))
            if show_xy:
                if origin_xy is None:
                    row.append("")
                else:
                    row.append(
                        _format_delta_xy(
                            uvs,
                            origin_xy,
                            mean=use_mean_delta,
                            kernel_size=xy_kernel_size,
                        )
                    )
            rows.append(row)

    total_row = ["TOTAL", f"{sum_p:.4f}", f"{sum_m:.4f}"]
    total_n = sum(int(n_partner_by_type.get(pt, 0)) for pt in by_partner)
    total_row.append(str(total_n))
    total_row += [""] * (int(show_uv) + int(show_xy))

    all_rows = [header] + rows + [total_row]
    n_cols = len(header)
    widths = [max(len(r[c]) for r in all_rows) for c in range(n_cols)]

    def _fmt(row: List[str]) -> str:
        cells = [row[0].ljust(widths[0])]
        cells += [row[c].rjust(widths[c]) for c in range(1, n_cols)]
        return "  ".join(cells).rstrip()

    n_count_label = "n_source" if direction == "post" else "n_target"
    title = (
        f"{self_field} = {cell_type}  |  {n_count_label} = {n_self}  |  "
        f"all n_syn {flow_word} {cell_type}{hex_note} = {total_syn:.1f}"
    )
    sep = "=" * max(60, len(title))
    print(sep)
    print(title)
    print(sep)
    print(_fmt(header))
    for row in rows:
        print(_fmt(row))
    print(_fmt(total_row))
    print()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synapse mix between a cell type and its synaptic partners from a connectome network.json."
    )
    parser.add_argument(
        "cell_types",
        nargs="?",
        default="L1",
        metavar="CELL_TYPE[,CELL_TYPE...]",
        help=(
            "Comma-separated cell types to query (e.g. T4a,T4b,T4c or Mi1). "
            "Prefix with & for a family "
            "(e.g. &Centrifugal) to aggregate its member types, or @ for a single "
            "neuron by root id (e.g. @720575940622041087). Default: L1 if omitted"
        ),
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help=(
            "Find outgoing (postsynaptic) connections: treat CELL_TYPE as presynaptic "
            "source_type and break down by target_type. Default is incoming "
            "(presynaptic) connections onto CELL_TYPE."
        ),
    )
    parser.add_argument(
        "--family",
        action="store_true",
        help=(
            "Break down partners by source_family/target_family instead of "
            "source_type/target_type (family from type_counts_abc.csv)."
        ),
    )
    parser.add_argument(
        "--min",
        type=float,
        default=0.0,
        metavar="PCT",
        help=(
            "Only list partner rows whose combined %% n_syn+ + %% n_syn- exceeds PCT "
            "(e.g. --min 5 lists only partners >5%%). TOTAL still reflects all partners."
        ),
    )
    parser.add_argument(
        "--borst",
        action="store_true",
        help=(
            "Use Borst 5-column multi_colM connectivity (ignores --network). "
            "Restrict to one column with --at-xy (Borst centres on y=0); "
            "--at-uv, --extent, and --family are not valid."
        ),
    )
    parser.add_argument(
        "--network",
        default=_DEFAULT_NETWORK,
        help=(
            "Network folder (e.g. right_min_neuron1, resolved next to this script) or a "
            "direct path to a folder / network.json. Default: right_min_neuron1"
        ),
    )
    at_group = parser.add_mutually_exclusive_group()
    at_group.add_argument(
        "--at-uv",
        nargs=2,
        type=int,
        metavar=("U", "V"),
        default=None,
        help=(
            "Only count edges whose CELL_TYPE *instance* sits at hex (u,v). "
            "Omit to aggregate over all instances of each cell type (default). "
            "When set: extra column pre_d_uv/post_d_uv for partner instances "
            "(n_neuron is always shown). If n_neuron ≤ 5, distinct deltas are "
            "listed; if n_neuron > 5, the mean delta is shown. "
            "TOTAL row omits the coord column."
        ),
    )
    at_group.add_argument(
        "--at-xy",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help=(
            "Only count edges whose CELL_TYPE *instance* sits at pixel (x,y). FAFB: "
            "converted via column_mapper.pixel_to_hex (integer hex centre). "
            "When set: extra column pre_d_xy/post_d_xy (partner pixel minus --at-xy). "
            "If n_neuron ≤ 5, distinct deltas are listed; if n_neuron > 5, mean delta. "
            "With --borst: resolved via column_mapper.borst_col_at_xy "
            "(centres at x in {-10,-5,0,5,10}, y=0)."
        ),
    )
    at_group.add_argument(
        "--extent",
        type=int,
        metavar="N",
        default=None,
        help=(
            "FAFB only: restrict to CELL_TYPE instances in the central hex disc of "
            "radius N (0 = centre column, 1 = 7 columns, 2 = 19, …; "
            "column_mapper.inside_mask). Shows mean pre_d_xy/post_d_xy only. "
            "Incompatible with --borst."
        ),
    )
    args = parser.parse_args(argv)

    if args.borst and args.at_uv is not None:
        logger.error(
            "--at-uv is invalid with --borst; use --at-xy (Borst column centres on y=0)"
        )
        return 1

    if args.borst and args.extent is not None:
        logger.error("--extent is invalid with --borst; use --network on FAFB network.json")
        return 1

    if args.extent is not None and args.extent < 0:
        logger.error("--extent must be >= 0")
        return 1

    if args.borst and args.family:
        logger.error(
            "--family is invalid with --borst; family names come from type_counts_abc.csv "
            "next to FAFB network.json, not multi_colM"
        )
        return 1

    direction = "post" if args.post else "pre"
    borst_prefix = " | Borst 5-col multi_colM" if args.borst else ""

    if args.borst:
        if args.network != _DEFAULT_NETWORK:
            logger.info("--borst: ignoring --network %s", args.network)
        try:
            nodes, edges = _load_borst_graph()
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("Loaded Borst graph from %s (%d nodes, %d edges)", BORST_MULTI_COL_M, len(nodes), len(edges))
        type_to_family_all: Dict[str, str] = {}
    else:
        json_path = resolve_network_json(args.network)

        if not json_path.is_file():
            logger.error("JSON not found: %s", json_path)
            return 1

        logger.info("Loading %s", json_path)
        with json_path.open() as f:
            spec = json.load(f)
        edges = spec.get("edges")
        if not isinstance(edges, list):
            logger.error("Invalid JSON: missing edges list")
            return 1

        nodes = spec.get("nodes")
        if not isinstance(nodes, list):
            logger.error("Invalid JSON: missing nodes list")
            return 1

        type_to_family_all = _load_type_to_family(json_path)

    ids_at_hex: Optional[Dict[str, Set[int]]] = None
    hex_note = borst_prefix
    xy_kernel_size = DEFAULT_KERNEL_SIZE if args.borst else 1.0
    at_ref_uv: Optional[Tuple[float, float]] = None
    at_ref_xy: Optional[Tuple[float, float]] = None

    if args.borst and args.at_xy is not None:
        try:
            bx, by = float(args.at_xy[0]), float(args.at_xy[1])
            col = borst_col_at_xy(bx, by)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        center = borst_column_centers()[col]
        ids_at_hex = _instance_ids_at_col(nodes, col)
        at_ref_xy = (float(center.x), float(center.y))
        hex_note += (
            f" at Borst col={col} (k={center.k}) "
            f"(x,y)=({_format_scalar_for_table(center.x)},"
            f"{_format_scalar_for_table(center.y)})"
        )
        logger.info(
            "Restricting to Borst column %d; %d cell types have ≥1 unit there",
            col,
            sum(1 for s in ids_at_hex.values() if s),
        )
    elif not args.borst:
        if args.extent is not None:
            ids_at_hex = _instance_ids_in_disc(nodes, args.extent)
            n_hex = _hex_disc_column_count(args.extent)
            hex_note += f" extent={args.extent} ({n_hex} hex cols)"
            logger.info(
                "Restricting to central hex disc extent=%d (%d hex columns); "
                "%d cell types have ≥1 node there",
                args.extent,
                n_hex,
                sum(1 for s in ids_at_hex.values() if s),
            )
        else:
            at_uv: Optional[Tuple[int, int]] = None
            if args.at_uv is not None:
                at_uv = (int(args.at_uv[0]), int(args.at_uv[1]))
            elif args.at_xy is not None:
                try:
                    at_uv = _xy_to_uv(args.at_xy[0], args.at_xy[1])
                except ValueError as exc:
                    logger.error("%s", exc)
                    return 1
            if at_uv is not None:
                hu, hv = at_uv
                ids_at_hex = _instance_ids_at_hex(nodes, hu, hv)
                if args.at_uv is not None:
                    at_ref_uv = (float(hu), float(hv))
                    hex_note += f" at hex (u,v)=({hu},{hv})"
                else:
                    at_ref_xy = (float(args.at_xy[0]), float(args.at_xy[1]))
                    hex_note += (
                        f" at (x,y)=({_format_scalar_for_table(at_ref_xy[0])},"
                        f"{_format_scalar_for_table(at_ref_xy[1])})"
                    )
                logger.info(
                    "Restricting to instances at (u,v)=(%s,%s); %d cell types have ≥1 node there",
                    hu,
                    hv,
                    sum(1 for s in ids_at_hex.values() if s),
                )

    if args.at_uv is not None:
        show_partner_uv, show_partner_xy = True, False
    else:
        show_partner_uv, show_partner_xy = False, True
    mean_partner_delta = args.at_uv is None and args.at_xy is None

    partner_type_to_family = type_to_family_all if args.family else None

    cell_types = _parse_cell_types(args.cell_types)
    labels, self_type_to_labels, self_id_to_labels = _resolve_query_labels(
        cell_types, type_to_family_all
    )
    if args.borst:
        if self_id_to_labels:
            logger.warning("@root_id queries are not supported with --borst; skipping those tokens")
            labels = [lab for lab in labels if not lab.startswith("@")]
            self_id_to_labels = {}
        if any(tok.startswith("&") for tok in cell_types):
            logger.warning(
                "&Family tokens are not supported with --borst; "
                "use a literal cell type from multi_colM ctype"
            )

    # Partner delta coords: always collected; reference is --at centre or mean self location.
    id_to_uv = _node_id_to_uv(nodes, float_coords=args.borst)
    acc = _accumulate_all(
        edges,
        labels,
        self_type_to_labels,
        id_to_uv,
        ids_at_hex=ids_at_hex,
        direction=direction,
        type_to_family=partner_type_to_family,
        self_id_to_labels=self_id_to_labels,
    )
    for label in labels:
        by_partner, total_syn, n_partner_by_type, partner_uv_by_type, n_self = acc[label]
        label_origin_uv, label_origin_xy = _label_origins(
            label,
            nodes,
            self_type_to_labels,
            ids_at_hex,
            at_ref_uv,
            at_ref_xy,
            float_coords=args.borst,
            xy_kernel_size=xy_kernel_size,
            need_uv=show_partner_uv,
            need_xy=show_partner_xy,
        )
        print_table(
            label,
            by_partner,
            total_syn,
            n_partner_by_type,
            partner_uv_by_type,
            hex_note=hex_note,
            direction=direction,
            use_family=args.family,
            min_pct=args.min,
            xy_kernel_size=xy_kernel_size,
            show_uv=show_partner_uv,
            show_xy=show_partner_xy,
            origin_uv=label_origin_uv,
            origin_xy=label_origin_xy,
            mean_partner_delta=mean_partner_delta,
            n_self=n_self,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
