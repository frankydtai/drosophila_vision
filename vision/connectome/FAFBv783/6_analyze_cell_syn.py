"""
For the connectome ``network.json`` files under this folder (built by
``4_build_network.py`` and stored in e.g. ``left_min_neuron1/network.json``),
tabulate the synaptic partners of one or more *cells*.

By default (incoming / ``pre``) each CELL is treated as the postsynaptic
``target_cell`` and broken down by presynaptic ``source_cell``. With ``--post``
(outgoing) each CELL is treated as the presynaptic ``source_cell`` and broken
down by postsynaptic ``target_cell``.

A CELL token may be a cell (e.g. ``Mi1``); a *family* when prefixed with
``:`` (e.g. ``:Centrifugal``) which sums over all its cells; or a single
neuron when prefixed with ``@`` (e.g. ``@720575940622041087``) selected by FlyWire
root id. The breakdown column still shows individual ``source_cell``/``target_cell``
unless ``--family`` is given.

Optionally restrict to CELL ids by location: axial ``(u, v)`` with
``--u`` and/or ``--v`` (one axis for every column on that line, or both for a single
column); hex-step ``(x, y)`` with ``--x`` and/or ``--y``; or the central hex disc
``--radius N`` (0 = centre column, 1 = 7 columns, 2 = 19, …; uses
``build_hex.radius_mask``); or a single hex shell ``--shell N`` (0 = centre
column, 1 = 6 columns, 2 = 12, …; uses ``build_hex.hex_radius``). Both are
FAFB-only and show mean ``pre_d_xy``/``post_d_xy`` only.

Per (cell, partner_cell): sum ``n_syn`` where ``sign > 0`` vs ``sign < 0``,
then express each as a percentage of **all** ``n_syn`` for that cell. An
``n_neuron`` column is always shown. The SUM row omits the u/v/x/y columns.

The ``network.json`` schema is ``{"metadata", "nodes", "edges"}`` where each node is
``{"id", "name", "u", "v", "column_id", "sti", "output"}`` and each edge is
``{"src", "tar", "syn_sign", "n_syn", "source_cell", "target_cell", "du", "dv"}``.

Example::

    python3 "analyze_cell_syn.py"
    python3 "analyze_cell_syn.py" L1,L2,L3,L4,L5
    python3 "analyze_cell_syn.py" T4a,T4b,T4c,T4d
    python3 "analyze_cell_syn.py" Mi1 --post
    python3 "analyze_cell_syn.py" :Centrifugal
    python3 "analyze_cell_syn.py" :Centrifugal --family
    python3 "analyze_cell_syn.py" Mi1 --family
    python3 "analyze_cell_syn.py" @720575940622041087
    python3 "analyze_cell_syn.py" Mi1 --network right_min_neuron1
    python3 "analyze_cell_syn.py" L1 --network /abs/path/to/some_folder
    python3 "analyze_cell_syn.py" Mi1 --post --u 0 --v 0
    python3 "analyze_cell_syn.py" Mi1 --post --u 0
    python3 "analyze_cell_syn.py" Mi1 --post --x 0 --y 1
    python3 "analyze_cell_syn.py" Mi1 --x 0
    python3 "analyze_cell_syn.py" Mi1 --radius 0
    python3 "analyze_cell_syn.py" Mi1 --radius 2
    python3 "analyze_cell_syn.py" Mi1 --shell 0
    python3 "analyze_cell_syn.py" Mi1 --shell 2
    python3 "analyze_cell_syn.py" Mi1 --u 0 --v 0 --p-xy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from build_hex import (
    hex_radius,
    radius_mask,
    xy_from_uv,
    uv_from_xy,
)
from path import (
    resolve_network_json,
    resolve_cell_counts_abc_path,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_MAX_PARTNER_LIST = 5


def family_from_cell_csv(json_path: Path) -> Dict[str, str]:
    """Map cell -> ``family`` from ``cell_counts_abc.csv`` for this network."""
    csv_path = resolve_cell_counts_abc_path(json_path)
    family_by_cell: Dict[str, str] = {}
    if not csv_path.is_file():
        logger.warning("No cell_counts_abc.csv at %s; family names won't resolve", csv_path)
        return family_by_cell
    import csv

    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            t = row.get("cell")
            fam = row.get("family")
            if t:
                family_by_cell[t] = fam if fam else t
    return family_by_cell


def resolve_query_labels(
    tokens: List[str], family_from_cell: Dict[str, str]
) -> Tuple[List[str], Dict[str, Set[str]], Dict[int, Set[str]]]:
    """Resolve queried tokens to (ordered labels, self_cell -> labels, self_id -> labels).

    Token prefixes:
      - ``:Family`` sums over every cell of that family.
      - ``@<id>`` selects a single neuron by FAFB id (CSV field ``root_id``).
      - anything else is a literal cell.
    The label shown in the output is the token as typed (e.g. ``:Centrifugal``,
    ``@720575940622041087``).
    """
    cells_from_family: DefaultDict[str, List[str]] = defaultdict(list)
    for t, fam in family_from_cell.items():
        cells_from_family[fam].append(t)
    labels: List[str] = list(dict.fromkeys(tokens))
    labels_from_self_cell: DefaultDict[str, Set[str]] = defaultdict(set)
    labels_from_self_id: DefaultDict[int, Set[str]] = defaultdict(set)
    for label in labels:
        if label.startswith(":"):
            fam = label[1:]
            fam_cells = cells_from_family.get(fam, [])
            if not fam_cells:
                logger.warning("Family %r not found in cell_counts_abc.csv", fam)
            for t in fam_cells:
                labels_from_self_cell[t].add(label)
        elif label.startswith("@"):
            try:
                labels_from_self_id[int(label[1:])].add(label)
            except ValueError:
                logger.warning("Invalid root id token %r (expected @<int>)", label)
        else:
            labels_from_self_cell[label].add(label)
    return labels, dict(labels_from_self_cell), dict(labels_from_self_id)


def _node_uv_xy(
    n: dict,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Parse one FAFB node's ``(u,v)`` and hex-step ``(x,y)``."""
    try:
        u = float(int(n["u"]))
        v = float(int(n["v"]))
        x, y = xy_from_uv(u, v)
        xy = (float(x), float(y))
        return (u, v), xy
    except (KeyError, TypeError, ValueError):
        return None


def _self_node_origin(
    label: str,
    nodes: List[dict],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """``@<id>`` label -> FAFB ``(u,v)``/``(x,y)``."""
    if not label.startswith("@"):
        return None, None
    try:
        node_id = int(label[1:])
    except ValueError:
        return None, None
    for n in nodes:
        try:
            if int(n["id"]) != node_id:
                continue
            node_uv_xy = _node_uv_xy(n)
            if node_uv_xy is None:
                continue
            return node_uv_xy
        except (KeyError, TypeError, ValueError):
            continue
    return None, None


def _mean_self_origin(
    label: str,
    nodes: List[dict],
    labels_from_self_cell: Dict[str, Set[str]],
    ids_by_cell: Optional[Dict[str, Set[int]]],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Mean self location: FAFB ``(u,v)``/``(x,y)``."""
    self_cells = {t for t, labs in labels_from_self_cell.items() if label in labs}
    if not self_cells:
        return None, None
    uvs: List[Tuple[float, float]] = []
    for n in nodes:
        name = n.get("name")
        if not isinstance(name, str) or name not in self_cells:
            continue
        try:
            node_id = int(n["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if ids_by_cell is not None:
            allowed = ids_by_cell.get(name, set())
            if node_id not in allowed:
                continue
        node_uv_xy = _node_uv_xy(n)
        if node_uv_xy is None:
            continue
        uvs.append(node_uv_xy[0])
    if not uvs:
        return None, None
    n = float(len(uvs))
    mean_uv = (sum(u for u, _v in uvs) / n, sum(v for _u, v in uvs) / n)
    mean_xy = (
        sum(xy_from_uv(u, v)[0] for u, v in uvs) / n,
        sum(xy_from_uv(u, v)[1] for u, v in uvs) / n,
    )
    return mean_uv, mean_xy


def _label_origins(
    label: str,
    nodes: List[dict],
    labels_from_self_cell: Dict[str, Set[str]],
    ids_by_cell: Optional[Dict[str, Set[int]]],
    at_ref_uv: Optional[Tuple[float, float]],
    at_ref_xy: Optional[Tuple[float, float]],
    *,
    need_uv: bool,
    need_xy: bool,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Reference ``(u,v)``/``(x,y)`` for partner deltas."""
    origin_uv = at_ref_uv if need_uv else None
    origin_xy = at_ref_xy if need_xy else None
    if label.startswith("@"):
        self_uv, self_xy = _self_node_origin(label, nodes)
        if origin_uv is None:
            origin_uv = self_uv
        if origin_xy is None:
            origin_xy = self_xy
    elif origin_uv is None or origin_xy is None:
        mean_uv, mean_xy = _mean_self_origin(
            label,
            nodes,
            labels_from_self_cell,
            ids_by_cell,
        )
        if origin_uv is None:
            origin_uv = mean_uv
        if origin_xy is None:
            origin_xy = mean_xy
    return origin_uv, origin_xy


def uv_from_node_id(nodes: List[dict]) -> Dict[int, Tuple[float, float]]:
    """Unit id -> hex (u, v) from network nodes."""
    m: Dict[int, Tuple[float, float]] = {}
    for n in nodes:
        try:
            node_uv_xy = _node_uv_xy(n)
            if node_uv_xy is None:
                continue
            uv, _ = node_uv_xy
            m[int(n["id"])] = uv
        except (KeyError, TypeError, ValueError):
            continue
    return m


def build_ids_by_cell(
    nodes: List[dict],
    *,
    at_u: Optional[int] = None,
    at_v: Optional[int] = None,
    at_x: Optional[float] = None,
    at_y: Optional[float] = None,
    radius: Optional[int] = None,
    shell: Optional[int] = None,
    tol: float = 1e-6,
) -> Dict[str, Set[int]]:
    """Map cell -> root ids under one spatial filter (FAFB).

    Exactly one mode: ``radius``, ``shell``, ``at_u``/``at_v``, or ``at_x``/``at_y``.
    """
    has_radius = radius is not None
    has_shell = shell is not None
    has_uv = at_u is not None or at_v is not None
    has_xy = at_x is not None or at_y is not None
    if has_radius + has_shell + has_uv + has_xy != 1:
        raise ValueError(
            "build_ids_by_cell requires exactly one of radius, shell, at_u/at_v, at_x/at_y"
        )
    ids_by_cell: Dict[str, Set[int]] = {}
    for n in nodes:
        name = n.get("name")
        if not isinstance(name, str):
            continue
        try:
            node_id = int(n["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if has_xy:
            node_uv_xy = _node_uv_xy(n)
            if node_uv_xy is None:
                continue
            _, (x, y) = node_uv_xy
            if at_x is not None and abs(float(x) - float(at_x)) > tol:
                continue
            if at_y is not None and abs(float(y) - float(at_y)) > tol:
                continue
        else:
            try:
                u, v = int(n["u"]), int(n["v"])
            except (KeyError, TypeError, ValueError):
                continue
            if has_radius and not bool(radius_mask(u, v, int(radius))):
                continue
            if has_shell and hex_radius(u, v) != int(shell):
                continue
            if has_uv:
                if at_u is not None and u != at_u:
                    continue
                if at_v is not None and v != at_v:
                    continue
        ids_by_cell.setdefault(name, set()).add(node_id)
    return ids_by_cell



def _syn_sign(edge: dict) -> float:
    """Signed weight for an edge from its ``syn_sign`` field (±1)."""
    try:
        return float(edge.get("syn_sign", 0))
    except (TypeError, ValueError):
        return 0.0


def partner_syn_by_label(
    edges: List[dict],
    labels: List[str],
    labels_from_self_cell: Dict[str, Set[str]],
    uv_from_id: Dict[int, Tuple[float, float]],
    *,
    xy_from_id: Optional[Dict[int, Tuple[float, float]]] = None,
    ids_by_cell: Optional[Dict[str, Set[int]]] = None,
    direction: str = "pre",
    family_from_cell: Optional[Dict[str, str]] = None,
    labels_from_self_id: Optional[Dict[int, Set[str]]] = None,
) -> Dict[
    str,
    Tuple[
        DefaultDict[str, Dict[str, float]],
        float,
        Dict[str, int],
        Dict[str, Set[Tuple[float, float]]],
        Dict[str, Set[Tuple[float, float]]],
        int,
    ],
]:
    """One pass over edges: per queried label, (per partner type syn+/syn-, n_syn sum).

    ``labels`` is the ordered list of queried tokens (a cell, a family entered as
    ``:Family``, or a single neuron entered as ``@<id>``). ``labels_from_self_cell``
    maps each *self* cell to its label(s); ``labels_from_self_id`` maps a *self* root
    id to its label(s). A family label sums over all its cells.

    ``direction="pre"`` (default): query each label as the **postsynaptic** side
    (``target_cell``) and break down by presynaptic ``source_cell`` (incoming).
    ``direction="post"``: query each label as the **presynaptic** side (``source_cell``)
    and break down by postsynaptic ``target_cell`` (outgoing).

    If ``ids_by_cell`` is set, only edges whose *self* id (``tar`` for ``pre``,
    ``src`` for ``post``) is in that cell→ids bag are counted. The third return value
    maps partner type -> count of **distinct** *partner* ids. The fourth maps
    partner type -> distinct partner ``(u,v)`` centres.
    """
    if direction == "post":
        self_cell_field, partner_cell_field = "source_cell", "target_cell"
        self_id_field, partner_id_field = "src", "tar"
    else:
        self_cell_field, partner_cell_field = "target_cell", "source_cell"
        self_id_field, partner_id_field = "tar", "src"

    by_cell: Dict[str, DefaultDict[str, Dict[str, float]]] = {
        p: defaultdict(lambda: {"syn+": 0.0, "syn-": 0.0}) for p in labels
    }
    sums: Dict[str, float] = {p: 0.0 for p in labels}
    # Always count distinct partner neurons per partner type (-> n_neuron column).
    partner_ids: Dict[str, DefaultDict[str, Set[int]]] = {
        p: defaultdict(set) for p in labels
    }
    partner_uv: Dict[str, DefaultDict[str, Set[Tuple[float, float]]]] = {
        p: defaultdict(set) for p in labels
    }
    partner_xy: Dict[str, DefaultDict[str, Set[Tuple[float, float]]]] = {
        p: defaultdict(set) for p in labels
    }
    self_ids: Dict[str, Set[int]] = {p: set() for p in labels}
    for edge in edges:
        stype = edge.get(self_cell_field)
        try:
            self_id_int: Optional[int] = int(edge.get(self_id_field))
        except (TypeError, ValueError):
            self_id_int = None

        cell_labels: Set[str] = set()
        type_labels = labels_from_self_cell.get(stype)
        if type_labels:
            cell_labels |= type_labels
        if labels_from_self_id and self_id_int is not None:
            id_labels = labels_from_self_id.get(self_id_int)
            if id_labels:
                cell_labels |= id_labels
        if not cell_labels:
            continue
        if ids_by_cell is not None:
            allowed = ids_by_cell.get(stype, set())
            if not allowed or self_id_int is None or self_id_int not in allowed:
                continue
        pt = edge.get(partner_cell_field) or "?"
        if family_from_cell is not None:
            pt = family_from_cell.get(pt, pt)
        a = _syn_sign(edge)
        ns = float(edge.get("n_syn", 0))
        partner = edge.get(partner_id_field)
        for cell in cell_labels:
            sums[cell] += ns
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
                    uv = uv_from_id.get(pid)
                    if uv is not None:
                        partner_uv[cell][pt].add(uv)
                    if xy_from_id is not None:
                        xy = xy_from_id.get(pid)
                        if xy is not None:
                            partner_xy[cell][pt].add(xy)
                except (TypeError, ValueError):
                    pass
    return {
        p: (
            by_cell[p],
            sums[p],
            {pt: len(ids) for pt, ids in partner_ids[p].items()},
            {pt: set(uvs) for pt, uvs in partner_uv[p].items()},
            {pt: set(partner_xy[p][pt]) for pt in partner_xy[p]},
            len(self_ids[p]),
        )
        for p in labels
    }


def query_partner_syn(
    nodes: List[dict],
    edges: List[dict],
    cells: List[str],
    *,
    direction: str = "pre",
    ids_by_cell: Optional[Dict[str, Set[int]]] = None,
    family_from_cell: Optional[Dict[str, str]] = None,
) -> Dict[
    str,
    Tuple[
        DefaultDict[str, Dict[str, float]],
        float,
        Dict[str, int],
        Dict[str, Set[Tuple[float, float]]],
        Dict[str, Set[Tuple[float, float]]],
        int,
    ],
]:
    """Resolve ``cells`` and return ``partner_syn_by_label`` stats (no print)."""
    fam = family_from_cell if family_from_cell is not None else {}
    labels, labels_from_self_cell, labels_from_self_id = resolve_query_labels(
        list(cells), fam
    )
    return partner_syn_by_label(
        edges,
        labels,
        labels_from_self_cell,
        uv_from_node_id(nodes),
        ids_by_cell=ids_by_cell,
        direction=direction,
        labels_from_self_id=labels_from_self_id,
    )


def print_table(
    cell: str,
    by_partner: DefaultDict[str, Dict[str, float]],
    n_syn_sum: float,
    n_partner_by_type: Dict[str, int],
    partner_uv_by_type: Dict[str, Set[Tuple[float, float]]],
    partner_xy_by_type: Optional[Dict[str, Set[Tuple[float, float]]]] = None,
    hex_note: str = "",
    direction: str = "pre",
    use_family: bool = False,
    min_percent: float = 0.0,
    show_uv: bool = True,
    show_d_xy: bool = True,
    show_xy: bool = False,
    origin_uv: Optional[Tuple[float, float]] = None,
    origin_xy: Optional[Tuple[float, float]] = None,
    mean_partner_delta: bool = False,
    n_self: int = 0,
    syn_strength_by_partner: Optional[Dict[str, str]] = None,
    after_title: Optional[str] = None,
) -> None:
    partner_dim = "family" if use_family else "cell"
    self_dim = "id" if cell.startswith("@") else "cell"
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
    d_xy_label = f"{side}_d_xy"
    xy_label = f"{side}_xy"
    show_syn_strength = syn_strength_by_partner is not None

    header = [partner_field]
    if show_syn_strength:
        header.append("syn_strength")
    header += ["% n_syn+", "% n_syn-", n_label]
    if show_uv:
        header.append(uv_label)
    if show_d_xy:
        header.append(d_xy_label)
    if show_xy:
        header.append(xy_label)

    rows: List[List[str]] = []
    percent_positive_sum = percent_negative_sum = 0.0
    if n_syn_sum <= 0:
        logger.warning("No n_syn for %s=%s", self_field, cell)
    else:
        for pt in sorted(by_partner):
            d = by_partner[pt]
            percent_positive = 100.0 * d["syn+"] / n_syn_sum
            percent_negative = 100.0 * d["syn-"] / n_syn_sum
            percent_positive_sum += percent_positive
            percent_negative_sum += percent_negative
            if percent_positive + percent_negative <= min_percent:
                continue
            row = [pt]
            if show_syn_strength:
                row.append(syn_strength_by_partner.get(pt, "-"))
            row += [f"{percent_positive:.4f}", f"{percent_negative:.4f}"]
            npv = int(n_partner_by_type.get(pt, 0))
            row.append(str(npv))
            uvs = partner_uv_by_type.get(pt, set())
            use_mean_delta = mean_partner_delta or npv > _MAX_PARTNER_LIST
            if show_uv:
                if origin_uv is None:
                    row.append("")
                elif use_mean_delta:
                    ou, ov = origin_uv
                    n = 0
                    du_sum = 0.0
                    dv_sum = 0.0
                    for u, v in uvs:
                        n += 1
                        du_sum += float(u) - ou
                        dv_sum += float(v) - ov
                    row.append(
                        f"({du_sum / n:.2f},{dv_sum / n:.2f})" if n else ""
                    )
                elif not uvs:
                    row.append("")
                else:
                    ou, ov = origin_uv
                    row.append(";".join(
                        f"({float(u) - ou:g},{float(v) - ov:g})"
                        for u, v in sorted(uvs)
                    ))
            if show_d_xy:
                if origin_xy is None:
                    row.append("")
                elif use_mean_delta:
                    ox, oy = origin_xy
                    partner_xy = (partner_xy_by_type or {}).get(pt, set())
                    n = 0
                    dx_sum = 0.0
                    dy_sum = 0.0
                    if partner_xy:
                        for x, y in partner_xy:
                            n += 1
                            dx_sum += float(x) - ox
                            dy_sum += float(y) - oy
                    else:
                        for u, v in uvs:
                            n += 1
                            hx, hy = xy_from_uv(u, v)
                            dx_sum += float(hx) - ox
                            dy_sum += float(hy) - oy
                    row.append(
                        f"({dx_sum / n:.2f},{dy_sum / n:.2f})" if n else ""
                    )
                else:
                    partner_xy = (partner_xy_by_type or {}).get(pt, set())
                    ox, oy = origin_xy
                    if partner_xy:
                        row.append(";".join(
                            f"({x - ox:g},{y - oy:g})"
                            for x, y in sorted(partner_xy)
                        ))
                    elif not uvs:
                        row.append("")
                    else:
                        row.append(";".join(
                            f"({float(xy_from_uv(u, v)[0]) - ox:g},"
                            f"{float(xy_from_uv(u, v)[1]) - oy:g})"
                            for u, v in sorted(uvs)
                        ))
            if show_xy:
                partner_xy = (partner_xy_by_type or {}).get(pt, set())
                if use_mean_delta:
                    n = 0
                    dx_sum = 0.0
                    dy_sum = 0.0
                    if partner_xy:
                        for x, y in partner_xy:
                            n += 1
                            dx_sum += float(x)
                            dy_sum += float(y)
                    else:
                        for u, v in uvs:
                            n += 1
                            hx, hy = xy_from_uv(u, v)
                            dx_sum += float(hx)
                            dy_sum += float(hy)
                    row.append(
                        f"({dx_sum / n:.2f},{dy_sum / n:.2f})" if n else ""
                    )
                elif partner_xy:
                    row.append(";".join(
                        f"({float(x):g},{float(y):g})"
                        for x, y in sorted(partner_xy)
                    ))
                elif not uvs:
                    row.append("")
                else:
                    row.append(";".join(
                        f"({float(xy_from_uv(u, v)[0]):g},"
                        f"{float(xy_from_uv(u, v)[1]):g})"
                        for u, v in sorted(uvs)
                    ))
            rows.append(row)

    sum_row = ["SUM"]
    if show_syn_strength:
        sum_row.append("")
    sum_row += [f"{percent_positive_sum:.4f}", f"{percent_negative_sum:.4f}"]
    n_partner_sum = sum(int(n_partner_by_type.get(pt, 0)) for pt in by_partner)
    sum_row.append(str(n_partner_sum))
    sum_row += [""] * (int(show_uv) + int(show_d_xy) + int(show_xy))

    table_rows = [header] + rows + [sum_row]
    n_field = len(header)
    ws = [max(len(row[field_idx]) for row in table_rows) for field_idx in range(n_field)]

    def _fmt(row: List[str]) -> str:
        cells = [row[0].ljust(ws[0])]
        cells += [row[field_idx].rjust(ws[field_idx]) for field_idx in range(1, n_field)]
        return "  ".join(cells).rstrip()

    n_count_label = "n_source" if direction == "post" else "n_target"
    title = (
        f"{self_field} = {cell}  |  {n_count_label} = {n_self}  |  "
        f"all n_syn {flow_word} {cell}{hex_note} = {n_syn_sum:.1f}"
    )
    sep = "=" * max(60, len(title))
    print(sep)
    print(title)
    print(sep)
    if after_title:
        print(after_title)
    print(_fmt(header))
    for row in rows:
        print(_fmt(row))
    print(_fmt(sum_row))
    print()




def cli_xy_filter(
    x: Optional[float],
    y: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Parse optional ``--x`` / ``--y`` (either or both)."""
    at_x = float(x) if x is not None else None
    at_y = float(y) if y is not None else None
    return at_x, at_y


def resolve_xy_ids(
    nodes: List[dict],
    at_x: Optional[float],
    at_y: Optional[float],
) -> Tuple[
    Optional[Dict[str, Set[int]]],
    str,
    Optional[Tuple[float, float]],
    bool,
]:
    """FAFB ``--x``/``--y`` → ``(ids_by_cell, hex_note, at_ref_xy, single_xy_column)``.

    Raises ``ValueError`` when the filter matches no ids or ``uv_from_xy`` fails.
    With neither ``at_x`` nor ``at_y`` set, returns ``(None, "", None, False)``.
    """
    has_xy = at_x is not None or at_y is not None
    if not has_xy:
        return None, "", None, False
    single_xy = at_x is not None and at_y is not None
    if single_xy:
        hu, hv = uv_from_xy(at_x, at_y)
        ids_by_cell = build_ids_by_cell(nodes, at_u=hu, at_v=hv)
        at_ref_xy = (float(at_x), float(at_y))
        hex_note = (
            f" at (x,y)=({at_ref_xy[0]:g},"
            f"{at_ref_xy[1]:g})"
        )
        logger.info(
            "Restricting to ids at (x,y)=(%s,%s) (u,v)=(%s,%s); "
            "%d cells have ≥1 node there",
            f"{at_ref_xy[0]:g}",
            f"{at_ref_xy[1]:g}",
            hu,
            hv,
            sum(1 for ids in ids_by_cell.values() if ids),
        )
        return ids_by_cell, hex_note, at_ref_xy, True

    ids_by_cell = build_ids_by_cell(nodes, at_x=at_x, at_y=at_y)
    if not any(ids_by_cell.values()):
        raise ValueError(f"no ids match --x={at_x!r} --y={at_y!r}")
    parts = []
    if at_x is not None:
        parts.append(f"x={at_x:g}")
    if at_y is not None:
        parts.append(f"y={at_y:g}")
    hex_note = " at " + ", ".join(parts)
    logger.info(
        "Restricting to ids on %s; %d cells have ≥1 node there",
        ", ".join(parts),
        sum(1 for ids in ids_by_cell.values() if ids),
    )
    return ids_by_cell, hex_note, None, False


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synapse mix between a cell and its synaptic partners from a connectome network.json."
    )
    parser.add_argument(
        "cells",
        nargs="*",
        default=["L1"],
        metavar="CELL",
        help=(
            "Cells to query (e.g. T4a T4b T4c or Mi1). "
            "Prefix with : for a family "
            "(e.g. :Centrifugal) to sum its cells, or @ for a single "
            "neuron by root id (e.g. @720575940622041087). Default: L1 if omitted"
        ),
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help=(
            "Find outgoing (postsynaptic) connections: treat CELL as presynaptic "
            "source_cell and break down by target_cell. Default is incoming "
            "(presynaptic) connections onto CELL."
        ),
    )
    parser.add_argument(
        "--family",
        action="store_true",
        help=(
            "Break down partners by source_family/target_family instead of "
            "source_cell/target_cell (family from cell_counts_abc.csv)."
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
        "--u",
        type=int,
        default=None,
        metavar="U",
        help="FAFB only: axial u; with --v selects one column, alone selects a u line",
    )
    parser.add_argument(
        "--v",
        type=int,
        default=None,
        metavar="V",
        help="FAFB only: axial v; with --u selects one column, alone selects a v line",
    )
    parser.add_argument(
        "--x",
        type=float,
        default=None,
        metavar="X",
        help="FAFB hex-step x",
    )
    parser.add_argument(
        "--y",
        type=float,
        default=None,
        metavar="Y",
        help="FAFB hex-step y",
    )
    parser.add_argument(
        "--network",
        default="right_min_neuron1_r10",
        help=(
            "Network folder (e.g. right_min_neuron1_r10, resolved next to this script) or a "
            "direct path to a folder / network.json."
        ),
    )
    parser.add_argument(
        "--radius",
        type=int,
        metavar="N",
        default=None,
        help=(
            "FAFB only: restrict to CELL ids in the central hex disc of "
            "radius N (0 = centre column, 1 = 7 columns, 2 = 19, …; "
            "build_hex.radius_mask). Shows mean pre_d_xy/post_d_xy only. "
            "Incompatible with --shell, --u/--v, and --x/--y."
        ),
    )
    parser.add_argument(
        "--shell",
        type=int,
        metavar="N",
        default=None,
        help=(
            "FAFB only: restrict to CELL ids on hex shell N "
            "(0 = centre column, 1 = 6 columns, 2 = 12, …; "
            "build_hex.hex_radius). Shows mean pre_d_xy/post_d_xy only. "
            "Incompatible with --radius, --u/--v, and --x/--y."
        ),
    )
    parser.add_argument(
        "--p-xy",
        action="store_true",
        help=(
            "Append pre_xy/post_xy as the last column: absolute partner (x,y) "
            "in addition to pre_d_xy/pre_d_uv."
        ),
    )
    args = parser.parse_args(argv)
    at_u, at_v = args.u, args.v
    has_uv_filter = at_u is not None or at_v is not None
    single_uv_hex = at_u is not None and at_v is not None
    at_x, at_y = cli_xy_filter(args.x, args.y)
    has_xy_filter = at_x is not None or at_y is not None
    single_xy_column = at_x is not None and at_y is not None

    if has_uv_filter and has_xy_filter:
        logger.error("--u/--v cannot be used with --x/--y")
        return 1

    if args.radius is not None and has_uv_filter:
        logger.error("--radius cannot be used with --u/--v")
        return 1

    if args.radius is not None and has_xy_filter:
        logger.error("--radius cannot be used with --x/--y")
        return 1

    if args.shell is not None and has_uv_filter:
        logger.error("--shell cannot be used with --u/--v")
        return 1

    if args.shell is not None and has_xy_filter:
        logger.error("--shell cannot be used with --x/--y")
        return 1

    if args.radius is not None and args.shell is not None:
        logger.error("--radius cannot be used with --shell")
        return 1

    if args.radius is not None and args.radius < 0:
        logger.error("--radius must be >= 0")
        return 1

    if args.shell is not None and args.shell < 0:
        logger.error("--shell must be >= 0")
        return 1

    direction = "post" if args.post else "pre"

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

    family_from_cell_all = family_from_cell_csv(json_path)

    ids_by_cell: Optional[Dict[str, Set[int]]] = None
    hex_note = ""
    at_ref_uv: Optional[Tuple[float, float]] = None
    at_ref_xy: Optional[Tuple[float, float]] = None

    if args.radius is not None:
        ids_by_cell = build_ids_by_cell(nodes, radius=args.radius)
        n_hex = 3 * args.radius * (args.radius + 1) + 1
        hex_note += f" radius={args.radius} ({n_hex} hexes)"
        logger.info(
            "Restricting to central hex disc radius=%d (%d hexes); "
            "%d cells have ≥1 node there",
            args.radius,
            n_hex,
            sum(1 for ids in ids_by_cell.values() if ids),
        )
    elif args.shell is not None:
        ids_by_cell = build_ids_by_cell(nodes, shell=args.shell)
        n_hex = 1 if args.shell == 0 else 6 * args.shell
        hex_note += f" shell={args.shell} ({n_hex} hexes)"
        logger.info(
            "Restricting to hex shell=%d (%d hexes); "
            "%d cells have ≥1 node there",
            args.shell,
            n_hex,
            sum(1 for ids in ids_by_cell.values() if ids),
        )
    elif has_uv_filter:
        if single_uv_hex:
            hu, hv = at_u, at_v
            ids_by_cell = build_ids_by_cell(nodes, at_u=hu, at_v=hv)
            at_ref_uv = (float(hu), float(hv))
            hx, hy = (float(v) for v in xy_from_uv(hu, hv))
            at_ref_xy = (hx, hy)
            hex_note += (
                f" at hex (u,v)=({hu},{hv}) "
                f"(x,y)=({hx:g},"
                f"{hy:g})"
            )
            logger.info(
                "Restricting to ids at (u,v)=(%s,%s) "
                "(x,y)=(%s,%s); %d cells have ≥1 node there",
                hu,
                hv,
                f"{hx:g}",
                f"{hy:g}",
                sum(1 for ids in ids_by_cell.values() if ids),
            )
        else:
            ids_by_cell = build_ids_by_cell(nodes, at_u=at_u, at_v=at_v)
            if not any(ids_by_cell.values()):
                logger.error("no ids match --u=%r --v=%r", at_u, at_v)
                return 1
            parts = []
            if at_u is not None:
                parts.append(f"u={at_u}")
            if at_v is not None:
                parts.append(f"v={at_v}")
            hex_note += " at " + ", ".join(parts)
            logger.info(
                "Restricting to ids on %s; %d cells have ≥1 node there",
                ", ".join(parts),
                sum(1 for ids in ids_by_cell.values() if ids),
            )
    elif has_xy_filter:
        try:
            ids_by_cell, xy_note, at_ref_xy, single_xy_column = (
                resolve_xy_ids(nodes, at_x, at_y)
            )
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        hex_note += xy_note

    if has_uv_filter:
        show_partner_uv, show_partner_d_xy = True, False
    else:
        show_partner_uv, show_partner_d_xy = False, True
    mean_partner_delta = not single_uv_hex and not single_xy_column

    family_from_partner_cell = family_from_cell_all if args.family else None

    cells = list(args.cells)
    labels, labels_from_self_cell, labels_from_self_id = resolve_query_labels(
        cells, family_from_cell_all
    )

    # Partner delta (x,y): always collected; reference is --at uv or mean self location.
    uv_from_id = uv_from_node_id(nodes)
    xy_from_id = None
    for label, (
        by_partner,
        n_syn_sum,
        n_partner_by_type,
        partner_uv_by_type,
        partner_xy_by_type,
        n_self,
    ) in partner_syn_by_label(
        edges,
        labels,
        labels_from_self_cell,
        uv_from_id,
        xy_from_id=xy_from_id,
        ids_by_cell=ids_by_cell,
        direction=direction,
        family_from_cell=family_from_partner_cell,
        labels_from_self_id=labels_from_self_id,
    ).items():
        label_origin_uv, label_origin_xy = _label_origins(
            label,
            nodes,
            labels_from_self_cell,
            ids_by_cell,
            at_ref_uv,
            at_ref_xy,
            need_uv=show_partner_uv,
            need_xy=show_partner_d_xy,
        )
        print_table(
            label,
            by_partner,
            n_syn_sum,
            n_partner_by_type,
            partner_uv_by_type,
            partner_xy_by_type=None,
            hex_note=hex_note,
            direction=direction,
            use_family=args.family,
            min_percent=args.min,
            show_uv=show_partner_uv,
            show_d_xy=show_partner_d_xy,
            show_xy=args.p_xy,
            origin_uv=label_origin_uv,
            origin_xy=label_origin_xy,
            mean_partner_delta=mean_partner_delta,
            n_self=n_self,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
