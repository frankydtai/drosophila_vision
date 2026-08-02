"""
For the connectome ``network.json`` files under this folder (built by
``4_build_network.py`` and stored in e.g. ``left_min_neuron1/network.json``),
tabulate the synaptic partners of one or more *cells*.

By default (incoming / ``pre``) each CELL is treated as the postsynaptic
``target_cell`` and broken down by presynaptic ``source_cell``. With ``--post``
(outgoing) each CELL is treated as the presynaptic ``source_cell`` and broken
down by postsynaptic ``target_cell``.

A CELL token may be a cell (e.g. ``Mi1``); a *family* when prefixed with
``:`` (e.g. ``:Centrifugal``) which aggregates over all its member cells; or a single
neuron when prefixed with ``@`` (e.g. ``@720575940622041087``) selected by FlyWire
root id. The breakdown column still shows individual ``source_cell``/``target_cell``
unless ``--family`` is given.

Optionally restrict to CELL *instances* by location: axial ``(u, v)`` with
``--u`` and/or ``--v`` (one axis for every column on that line, or both for a single
column); hex-step ``(x, y)`` with ``--x`` and/or ``--y``; or the central hex disc
``--extent N`` (0 = centre column, 1 = 7 columns, 2 = 19, …; uses
``build_hex.inside_mask``); or a single hex shell ``--shell N`` (0 = centre
column, 1 = 6 columns, 2 = 12, …; uses ``build_hex.hex_radius``). Both are
FAFB-only and show mean ``pre_d_xy``/``post_d_xy`` only.

Per (cell, partner_cell): sum ``n_syn`` where ``sign > 0`` vs ``sign < 0``,
then express each as a percentage of **all** ``n_syn`` for that cell. An
``n_neuron`` column is always shown. The TOTAL row omits the coord columns.

The ``network.json`` schema is ``{"metadata", "nodes", "edges"}`` where each node is
``{"id", "name", "u", "v", "column_id", "input", "output"}`` and each edge is
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
    python3 "analyze_cell_syn.py" Mi1 --extent 0
    python3 "analyze_cell_syn.py" Mi1 --extent 2
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
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, Union

_UvCoord = Tuple[Union[int, float], Union[int, float]]

from build_hex import (
    hex_radius,
    inside_mask,
    uv_to_xy,
    xy_to_uv,
)
from import_bootstrap import parse_comma_list
from path import (
    DEFAULT_NETWORK_RUN,
    resolve_network_json,
    resolve_cell_counts_abc_path,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_MAX_PARTNER_LIST = 5


def _hex_disc_count(extent: int) -> int:
    """Hex cells in a disc of axial radius ``extent`` (0 -> 1, 1 -> 7, 2 -> 19, …)."""
    return 3 * extent * (extent + 1) + 1


def _shell_hex_count(shell: int) -> int:
    """Hex cells on shell ``shell`` (0 -> 1, 1 -> 6, 2 -> 12, …)."""
    return 1 if shell == 0 else 6 * shell


def _load_cell_to_family(json_path: Path) -> Dict[str, str]:
    """Map cell -> ``family`` from ``cell_counts_abc.csv`` for this network."""
    csv_path = resolve_cell_counts_abc_path(json_path)
    out: Dict[str, str] = {}
    if not csv_path.is_file():
        logger.warning("No cell_counts_abc.csv at %s; family names won't resolve", csv_path)
        return out
    import csv

    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            t = row.get("cell")
            fam = row.get("family")
            if t:
                out[t] = fam if fam else t
    return out


def resolve_query_labels(
    tokens: List[str], cell_to_family: Dict[str, str]
) -> Tuple[List[str], Dict[str, Set[str]], Dict[int, Set[str]]]:
    """Resolve queried tokens to (ordered labels, self_cell -> labels, self_id -> labels).

    Token prefixes:
      - ``:Family`` aggregates over every member cell of that family.
      - ``@<root_id>`` selects a single neuron by FlyWire root id.
      - anything else is a literal cell.
    The label shown in the output is the token as typed (e.g. ``:Centrifugal``,
    ``@720575940622041087``).
    """
    family_to_cells: DefaultDict[str, List[str]] = defaultdict(list)
    for t, fam in cell_to_family.items():
        family_to_cells[fam].append(t)
    labels: List[str] = list(dict.fromkeys(tokens))
    self_cell_to_labels: DefaultDict[str, Set[str]] = defaultdict(set)
    self_id_to_labels: DefaultDict[int, Set[str]] = defaultdict(set)
    for tok in labels:
        if tok.startswith(":"):
            fam = tok[1:]
            members = family_to_cells.get(fam, [])
            if not members:
                logger.warning("Family %r not found in cell_counts_abc.csv", fam)
            for t in members:
                self_cell_to_labels[t].add(tok)
        elif tok.startswith("@"):
            try:
                self_id_to_labels[int(tok[1:])].add(tok)
            except ValueError:
                logger.warning("Invalid root id token %r (expected @<int>)", tok)
        else:
            self_cell_to_labels[tok].add(tok)
    return labels, dict(self_cell_to_labels), dict(self_id_to_labels)


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
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Parse one FAFB node's ``(u,v)`` and hex-step ``(x,y)`` centres."""
    try:
        u = float(n["u"]) if float_coords else float(int(n["u"]))
        v = float(n["v"]) if float_coords else float(int(n["v"]))
        x, y = uv_to_xy(u, v)
        xy = (float(x), float(y))
        return (u, v), xy
    except (KeyError, TypeError, ValueError):
        return None


def _format_pairs(
    pairs: Iterable[Tuple[float, float]],
    *,
    mean: bool,
) -> str:
    items = list(pairs)
    if not items:
        return ""
    if mean:
        n = float(len(items))
        dx = sum(p[0] for p in items) / n
        dy = sum(p[1] for p in items) / n
        return (
            f"({_format_mean_scalar_for_table(dx)},{_format_mean_scalar_for_table(dy)})"
        )
    return ";".join(
        f"({_format_scalar_for_table(p[0])},{_format_scalar_for_table(p[1])})"
        for p in sorted(items)
    )


def _format_partner_uv(
    uvs: Set[_UvCoord],
    origin: Optional[Tuple[float, float]] = None,
    *,
    mean: bool,
) -> str:
    """Format partner ``(u,v)`` or ``(du,dv)`` when ``origin`` is set."""
    if origin is None:
        pairs = ((float(u), float(v)) for u, v in uvs)
    else:
        ou, ov = origin
        pairs = ((float(u) - ou, float(v) - ov) for u, v in uvs)
    return _format_pairs(pairs, mean=mean)


def _format_partner_xy(
    uvs: Set[_UvCoord],
    coords: Set[Tuple[float, float]],
    origin: Optional[Tuple[float, float]] = None,
    *,
    mean: bool,
) -> str:
    """Format partner ``(x,y)`` or ``(dx,dy)``; prefer explicit ``coords``."""
    if coords:
        if origin is None:
            pairs = ((float(x), float(y)) for x, y in coords)
        else:
            ox, oy = origin
            pairs = ((x - ox, y - oy) for x, y in coords)
    elif origin is None:
        pairs = (
            (float(uv_to_xy(u, v)[0]), float(uv_to_xy(u, v)[1])) for u, v in uvs
        )
    else:
        ox, oy = origin
        pairs = (
            (
                float(uv_to_xy(u, v)[0]) - ox,
                float(uv_to_xy(u, v)[1]) - oy,
            )
            for u, v in uvs
        )
    return _format_pairs(pairs, mean=mean)


def _self_node_origin(
    label: str,
    nodes: List[dict],
    *,
    float_coords: bool
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """``@root_id`` label -> FAFB ``(u,v)``/``(x,y)``."""
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
            centers = _node_centers(n, float_coords=float_coords)
            if centers is None:
                continue
            return centers
        except (KeyError, TypeError, ValueError):
            continue
    return None, None


def _mean_self_origin(
    label: str,
    nodes: List[dict],
    self_cell_to_labels: Dict[str, Set[str]],
    ids_at_hex: Optional[Dict[str, Set[int]]],
    *,
    float_coords: bool
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Mean self centre: FAFB ``(u,v)``/``(x,y)``."""
    self_cells = {t for t, labs in self_cell_to_labels.items() if label in labs}
    if not self_cells:
        return None, None
    uvs: List[Tuple[float, float]] = []
    xys: List[Tuple[float, float]] = []
    for n in nodes:
        name = n.get("name")
        if not isinstance(name, str) or name not in self_cells:
            continue
        try:
            nid = int(n["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if ids_at_hex is not None:
            allowed = ids_at_hex.get(name, set())
            if nid not in allowed:
                continue
        centers = _node_centers(n, float_coords=float_coords)
        if centers is None:
            continue
        uv, xy = centers
        uvs.append(uv)
        xys.append(xy)
    if not xys:
        return None, None
    n = float(len(uvs))
    mean_uv = (sum(p[0] for p in uvs) / n, sum(p[1] for p in uvs) / n)
    mean_xy = (sum(p[0] for p in xys) / n, sum(p[1] for p in xys) / n)
    return mean_uv, mean_xy


def _label_origins(
    label: str,
    nodes: List[dict],
    self_cell_to_labels: Dict[str, Set[str]],
    ids_at_hex: Optional[Dict[str, Set[int]]],
    at_ref_uv: Optional[Tuple[float, float]],
    at_ref_xy: Optional[Tuple[float, float]],
    *,
    float_coords: bool,
    need_uv: bool,
    need_xy: bool,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Reference ``(u,v)``/``(x,y)`` for partner deltas."""
    origin_uv = at_ref_uv if need_uv else None
    origin_xy = at_ref_xy if need_xy else None
    if label.startswith("@"):
        self_uv, self_xy = _self_node_origin(
            label, nodes, float_coords=float_coords,
        )
        if origin_uv is None:
            origin_uv = self_uv
        if origin_xy is None:
            origin_xy = self_xy
    elif origin_uv is None or origin_xy is None:
        mean_uv, mean_xy = _mean_self_origin(
            label,
            nodes,
            self_cell_to_labels,
            ids_at_hex,
            float_coords=float_coords,
           
        )
        if origin_uv is None:
            origin_uv = mean_uv
        if origin_xy is None:
            origin_xy = mean_xy
    return origin_uv, origin_xy




def node_id_to_uv(nodes: List[dict], *, float_coords: bool = False) -> Dict[int, _UvCoord]:
    """Unit id -> hex (u, v) from network nodes."""
    m: Dict[int, _UvCoord] = {}
    for n in nodes:
        try:
            centers = _node_centers(n, float_coords=float_coords)
            if centers is None:
                continue
            uv, _ = centers
            m[int(n["id"])] = uv
        except (KeyError, TypeError, ValueError):
            continue
    return m


def _instance_ids_on_uv_line(
    nodes: List[dict],
    *,
    at_u: Optional[int] = None,
    at_v: Optional[int] = None,
) -> Dict[str, Set[int]]:
    """Map cell -> root ids on a hex ``u`` and/or ``v`` line (FAFB)."""
    out: Dict[str, Set[int]] = {}
    for n in nodes:
        try:
            u, v = int(n["u"]), int(n["v"])
        except (KeyError, TypeError, ValueError):
            continue
        if at_u is not None and u != at_u:
            continue
        if at_v is not None and v != at_v:
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


def instance_ids_at_hex(
    nodes: List[dict], u: int, v: int
) -> Dict[str, Set[int]]:
    """Map cell (node ``name``) -> FlyWire root ids at hex (u, v)."""
    return _instance_ids_on_uv_line(nodes, at_u=u, at_v=v)


def _instance_ids_in_disc(nodes: List[dict], extent: int) -> Dict[str, Set[int]]:
    """Map cell -> root ids inside the central hex disc of radius ``extent``."""
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


def _instance_ids_on_shell(nodes: List[dict], shell: int) -> Dict[str, Set[int]]:
    """Map cell -> root ids on hex shell ``shell`` (exact distance from origin)."""
    out: Dict[str, Set[int]] = {}
    for n in nodes:
        try:
            u, v = int(n["u"]), int(n["v"])
        except (KeyError, TypeError, ValueError):
            continue
        if hex_radius(u, v) != shell:
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



def _syn_sign(e: dict) -> float:
    """Signed weight for an edge from its ``syn_sign`` field (±1)."""
    try:
        return float(e.get("syn_sign", 0))
    except (TypeError, ValueError):
        return 0.0


def accumulate_all(
    edges: List[dict],
    labels: List[str],
    self_cell_to_labels: Dict[str, Set[str]],
    id_to_uv: Dict[int, _UvCoord],
    *,
    id_to_xy: Optional[Dict[int, Tuple[float, float]]] = None,
    ids_at_hex: Optional[Dict[str, Set[int]]] = None,
    direction: str = "pre",
    cell_to_family: Optional[Dict[str, str]] = None,
    self_id_to_labels: Optional[Dict[int, Set[str]]] = None,
) -> Dict[
    str,
    Tuple[
        DefaultDict[str, Dict[str, float]],
        float,
        Dict[str, int],
        Dict[str, Set[_UvCoord]],
        Dict[str, Set[Tuple[float, float]]],
        int,
    ],
]:
    """One pass over edges: per queried label, (per partner type syn+/syn-, total n_syn).

    ``labels`` is the ordered list of queried tokens (a cell, a family entered as
    ``:Family``, or a single neuron entered as ``@<root_id>``). ``self_cell_to_labels``
    maps each *self* cell to its label(s); ``self_id_to_labels`` maps a *self* root
    id to its label(s). A family label aggregates over all its member cells.

    ``direction="pre"`` (default): query each label as the **postsynaptic** side
    (``target_cell``) and break down by presynaptic ``source_cell`` (incoming).
    ``direction="post"``: query each label as the **presynaptic** side (``source_cell``)
    and break down by postsynaptic ``target_cell`` (outgoing).

    If ``ids_at_hex`` is set, only edges whose *self* instance id (``tar`` for ``pre``,
    ``src`` for ``post``) sits at the chosen hex are counted. The third return value
    maps partner type -> count of **distinct** *partner* instance ids. The fourth maps
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
    totals: Dict[str, float] = {p: 0.0 for p in labels}
    # Always count distinct partner neurons per partner type (-> n_neuron column).
    partner_ids: Dict[str, DefaultDict[str, Set[int]]] = {
        p: defaultdict(set) for p in labels
    }
    partner_uv: Dict[str, DefaultDict[str, Set[_UvCoord]]] = {
        p: defaultdict(set) for p in labels
    }
    partner_xy: Dict[str, DefaultDict[str, Set[Tuple[float, float]]]] = {
        p: defaultdict(set) for p in labels
    }
    self_ids: Dict[str, Set[int]] = {p: set() for p in labels}
    for e in edges:
        stype = e.get(self_cell_field)
        self_id_raw = e.get(self_id_field)
        try:
            self_id_int: Optional[int] = int(self_id_raw)
        except (TypeError, ValueError):
            self_id_int = None

        cell_labels: Set[str] = set()
        type_labels = self_cell_to_labels.get(stype)
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
        pt = e.get(partner_cell_field) or "?"
        if cell_to_family is not None:
            pt = cell_to_family.get(pt, pt)
        a = _syn_sign(e)
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
                    if id_to_xy is not None:
                        xy = id_to_xy.get(pid)
                        if xy is not None:
                            partner_xy[cell][pt].add(xy)
                except (TypeError, ValueError):
                    pass
    out: Dict[
        str,
        Tuple[
            DefaultDict[str, Dict[str, float]],
            float,
            Dict[str, int],
            Dict[str, Set[_UvCoord]],
            Dict[str, Set[Tuple[float, float]]],
            int,
        ],
    ] = {}
    for p in labels:
        npartner_map = {pt: len(ids) for pt, ids in partner_ids[p].items()}
        row_sets = {pt: set(uvs) for pt, uvs in partner_uv[p].items()}
        row_xy_sets = {pt: set(coords) for pt, coords in partner_xy[p].items()}
        n_self = len(self_ids[p])
        out[p] = (by_cell[p], totals[p], npartner_map, row_sets, row_xy_sets, n_self)
    return out


def query_partner_syn(
    nodes: List[dict],
    edges: List[dict],
    cells: List[str],
    *,
    direction: str = "pre",
    ids_at_hex: Optional[Dict[str, Set[int]]] = None,
    cell_to_family: Optional[Dict[str, str]] = None,
) -> Dict[
    str,
    Tuple[
        DefaultDict[str, Dict[str, float]],
        float,
        Dict[str, int],
        Dict[str, Set[_UvCoord]],
        Dict[str, Set[Tuple[float, float]]],
        int,
    ],
]:
    """Resolve ``cells`` and return ``accumulate_all`` partner syn stats (no print)."""
    fam = cell_to_family if cell_to_family is not None else {}
    labels, self_cell_to_labels, self_id_to_labels = resolve_query_labels(
        list(cells), fam
    )
    return accumulate_all(
        edges,
        labels,
        self_cell_to_labels,
        node_id_to_uv(nodes, float_coords=False),
        ids_at_hex=ids_at_hex,
        direction=direction,
        self_id_to_labels=self_id_to_labels,
    )


def print_table(
    cell: str,
    by_partner: DefaultDict[str, Dict[str, float]],
    total_syn: float,
    n_partner_by_type: Dict[str, int],
    partner_uv_by_type: Dict[str, Set[_UvCoord]],
    partner_xy_by_type: Optional[Dict[str, Set[Tuple[float, float]]]] = None,
    hex_note: str = "",
    direction: str = "pre",
    use_family: bool = False,
    min_pct: float = 0.0,
    show_uv: bool = True,
    show_d_xy: bool = True,
    show_xy: bool = False,
    origin_uv: Optional[Tuple[float, float]] = None,
    origin_xy: Optional[Tuple[float, float]] = None,
    mean_partner_delta: bool = False,
    n_self: int = 0,
    alpha_by_partner: Optional[Dict[str, str]] = None,
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
    show_alpha = alpha_by_partner is not None

    header = [partner_field]
    if show_alpha:
        header.append("alpha")
    header += ["% n_syn+", "% n_syn-", n_label]
    if show_uv:
        header.append(uv_label)
    if show_d_xy:
        header.append(d_xy_label)
    if show_xy:
        header.append(xy_label)

    rows: List[List[str]] = []
    sum_p = sum_m = 0.0
    if total_syn <= 0:
        logger.warning("No n_syn for %s=%s", self_field, cell)
    else:
        for pt in sorted(by_partner):
            d = by_partner[pt]
            pp = 100.0 * d["syn+"] / total_syn
            pm = 100.0 * d["syn-"] / total_syn
            sum_p += pp
            sum_m += pm
            if pp + pm <= min_pct:
                continue
            row = [pt]
            if show_alpha:
                row.append(alpha_by_partner.get(pt, "-"))
            row += [f"{pp:.4f}", f"{pm:.4f}"]
            npv = int(n_partner_by_type.get(pt, 0))
            row.append(str(npv))
            uvs = partner_uv_by_type.get(pt, set())
            coords = (partner_xy_by_type or {}).get(pt, set())
            use_mean_delta = mean_partner_delta or npv > _MAX_PARTNER_LIST
            if show_uv:
                if origin_uv is None:
                    row.append("")
                else:
                    row.append(
                        _format_partner_uv(uvs, origin_uv, mean=use_mean_delta)
                    )
            if show_d_xy:
                if origin_xy is None:
                    row.append("")
                else:
                    row.append(
                        _format_partner_xy(
                            uvs, coords, origin_xy, mean=use_mean_delta,
                        )
                    )
            if show_xy:
                row.append(
                    _format_partner_xy(uvs, coords, None, mean=use_mean_delta)
                )
            rows.append(row)

    total_row = ["TOTAL"]
    if show_alpha:
        total_row.append("")
    total_row += [f"{sum_p:.4f}", f"{sum_m:.4f}"]
    total_n = sum(int(n_partner_by_type.get(pt, 0)) for pt in by_partner)
    total_row.append(str(total_n))
    total_row += [""] * (int(show_uv) + int(show_d_xy) + int(show_xy))

    all_rows = [header] + rows + [total_row]
    n_hexes = len(header)
    widths = [max(len(r[c]) for r in all_rows) for c in range(n_hexes)]

    def _fmt(row: List[str]) -> str:
        cells = [row[0].ljust(widths[0])]
        cells += [row[c].rjust(widths[c]) for c in range(1, n_hexes)]
        return "  ".join(cells).rstrip()

    n_count_label = "n_source" if direction == "post" else "n_target"
    title = (
        f"{self_field} = {cell}  |  {n_count_label} = {n_self}  |  "
        f"all n_syn {flow_word} {cell}{hex_note} = {total_syn:.1f}"
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
    print(_fmt(total_row))
    print()


def _coord_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def instance_ids_on_xy_line(
    nodes: List[dict],
    *,
    at_x: Optional[float] = None,
    at_y: Optional[float] = None,
    tol: float = 1e-6,
) -> Dict[str, Set[int]]:
    """Map cell -> root ids on a hex-step ``x`` and/or ``y`` line (FAFB)."""
    out: Dict[str, Set[int]] = {}
    for n in nodes:
        centers = _node_centers(n, float_coords=False)
        if centers is None:
            continue
        _uv, (x, y) = centers
        if at_x is not None and not _coord_close(x, at_x, tol=tol):
            continue
        if at_y is not None and not _coord_close(y, at_y, tol=tol):
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





def cli_xy_filter(
    x: Optional[float],
    y: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Parse optional ``--x`` / ``--y`` (either or both)."""
    at_x = float(x) if x is not None else None
    at_y = float(y) if y is not None else None
    return at_x, at_y


def resolve_xy_instance_ids(
    nodes: List[dict],
    at_x: Optional[float],
    at_y: Optional[float],
) -> Tuple[
    Optional[Dict[str, Set[int]]],
    str,
    Optional[Tuple[float, float]],
    bool,
]:
    """FAFB ``--x``/``--y`` → ``(ids_at_hex, hex_note, at_ref_xy, single_xy_column)``.

    Raises ``ValueError`` when the filter matches no instances or ``xy_to_uv`` fails.
    With neither coordinate set, returns ``(None, "", None, False)``.
    """
    has_xy = at_x is not None or at_y is not None
    if not has_xy:
        return None, "", None, False
    single_xy = at_x is not None and at_y is not None
    if single_xy:
        hu, hv = xy_to_uv(at_x, at_y)
        ids_at_hex = instance_ids_at_hex(nodes, hu, hv)
        at_ref_xy = (float(at_x), float(at_y))
        hex_note = (
            f" at (x,y)=({_format_scalar_for_table(at_ref_xy[0])},"
            f"{_format_scalar_for_table(at_ref_xy[1])})"
        )
        logger.info(
            "Restricting to instances at (x,y)=(%s,%s) (u,v)=(%s,%s); "
            "%d cells have ≥1 node there",
            _format_scalar_for_table(at_ref_xy[0]),
            _format_scalar_for_table(at_ref_xy[1]),
            hu,
            hv,
            sum(1 for s in ids_at_hex.values() if s),
        )
        return ids_at_hex, hex_note, at_ref_xy, True

    ids_at_hex = instance_ids_on_xy_line(nodes, at_x=at_x, at_y=at_y)
    if not any(ids_at_hex.values()):
        raise ValueError(f"no instances match --x={at_x!r} --y={at_y!r}")
    parts = []
    if at_x is not None:
        parts.append(f"x={_format_scalar_for_table(at_x)}")
    if at_y is not None:
        parts.append(f"y={_format_scalar_for_table(at_y)}")
    hex_note = " at " + ", ".join(parts)
    logger.info(
        "Restricting to instances on %s; %d cells have ≥1 node there",
        ", ".join(parts),
        sum(1 for s in ids_at_hex.values() if s),
    )
    return ids_at_hex, hex_note, None, False


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synapse mix between a cell and its synaptic partners from a connectome network.json."
    )
    parser.add_argument(
        "cells",
        nargs="?",
        default="L1",
        metavar="CELL[,CELL...]",
        help=(
            "Comma-separated cells to query (e.g. T4a,T4b,T4c or Mi1). "
            "Prefix with : for a family "
            "(e.g. :Centrifugal) to aggregate its member cells, or @ for a single "
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
        default="right_min_neuron1_extent10",
        help=(
            "Network folder (e.g. right_min_neuron1_extent10, resolved next to this script) or a "
            "direct path to a folder / network.json."
        ),
    )
    parser.add_argument(
        "--extent",
        type=int,
        metavar="N",
        default=None,
        help=(
            "FAFB only: restrict to CELL instances in the central hex disc of "
            "radius N (0 = centre column, 1 = 7 columns, 2 = 19, …; "
            "build_hex.inside_mask). Shows mean pre_d_xy/post_d_xy only. "
            "Incompatible with --shell, --u/--v, and --x/--y."
        ),
    )
    parser.add_argument(
        "--shell",
        type=int,
        metavar="N",
        default=None,
        help=(
            "FAFB only: restrict to CELL instances on hex shell N "
            "(0 = centre column, 1 = 6 columns, 2 = 12, …; "
            "build_hex.hex_radius). Shows mean pre_d_xy/post_d_xy only. "
            "Incompatible with --extent, --u/--v, and --x/--y."
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

    if args.extent is not None and has_uv_filter:
        logger.error("--extent cannot be used with --u/--v")
        return 1

    if args.extent is not None and has_xy_filter:
        logger.error("--extent cannot be used with --x/--y")
        return 1

    if args.shell is not None and has_uv_filter:
        logger.error("--shell cannot be used with --u/--v")
        return 1

    if args.shell is not None and has_xy_filter:
        logger.error("--shell cannot be used with --x/--y")
        return 1

    if args.extent is not None and args.shell is not None:
        logger.error("--extent cannot be used with --shell")
        return 1

    if args.extent is not None and args.extent < 0:
        logger.error("--extent must be >= 0")
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

    cell_to_family_all = _load_cell_to_family(json_path)

    ids_at_hex: Optional[Dict[str, Set[int]]] = None
    hex_note = ""
    at_ref_uv: Optional[Tuple[float, float]] = None
    at_ref_xy: Optional[Tuple[float, float]] = None

    if args.extent is not None:
        ids_at_hex = _instance_ids_in_disc(nodes, args.extent)
        n_hex = _hex_disc_count(args.extent)
        hex_note += f" extent={args.extent} ({n_hex} hex cols)"
        logger.info(
            "Restricting to central hex disc extent=%d (%d hexes); "
            "%d cells have ≥1 node there",
            args.extent,
            n_hex,
            sum(1 for s in ids_at_hex.values() if s),
        )
    elif args.shell is not None:
        ids_at_hex = _instance_ids_on_shell(nodes, args.shell)
        n_hex = _shell_hex_count(args.shell)
        hex_note += f" shell={args.shell} ({n_hex} hex cols)"
        logger.info(
            "Restricting to hex shell=%d (%d hexes); "
            "%d cells have ≥1 node there",
            args.shell,
            n_hex,
            sum(1 for s in ids_at_hex.values() if s),
        )
    elif has_uv_filter:
        if single_uv_hex:
            hu, hv = at_u, at_v
            ids_at_hex = instance_ids_at_hex(nodes, hu, hv)
            at_ref_uv = (float(hu), float(hv))
            hx, hy = (float(v) for v in uv_to_xy(hu, hv))
            at_ref_xy = (hx, hy)
            hex_note += (
                f" at hex (u,v)=({hu},{hv}) "
                f"(x,y)=({_format_scalar_for_table(hx)},"
                f"{_format_scalar_for_table(hy)})"
            )
            logger.info(
                "Restricting to instances at (u,v)=(%s,%s) "
                "(x,y)=(%s,%s); %d cells have ≥1 node there",
                hu,
                hv,
                _format_scalar_for_table(hx),
                _format_scalar_for_table(hy),
                sum(1 for s in ids_at_hex.values() if s),
            )
        else:
            ids_at_hex = _instance_ids_on_uv_line(nodes, at_u=at_u, at_v=at_v)
            if not any(ids_at_hex.values()):
                logger.error("no instances match --u=%r --v=%r", at_u, at_v)
                return 1
            parts = []
            if at_u is not None:
                parts.append(f"u={at_u}")
            if at_v is not None:
                parts.append(f"v={at_v}")
            hex_note += " at " + ", ".join(parts)
            logger.info(
                "Restricting to instances on %s; %d cells have ≥1 node there",
                ", ".join(parts),
                sum(1 for s in ids_at_hex.values() if s),
            )
    elif has_xy_filter:
        try:
            ids_at_hex, xy_note, at_ref_xy, single_xy_column = (
                resolve_xy_instance_ids(nodes, at_x, at_y)
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

    partner_cell_to_family = cell_to_family_all if args.family else None

    cells = parse_comma_list(args.cells)
    labels, self_cell_to_labels, self_id_to_labels = resolve_query_labels(
        cells, cell_to_family_all
    )

    # Partner delta coords: always collected; reference is --at centre or mean self location.
    id_to_uv = node_id_to_uv(nodes, float_coords=False)
    id_to_xy = None
    acc = accumulate_all(
        edges,
        labels,
        self_cell_to_labels,
        id_to_uv,
        id_to_xy=id_to_xy,
        ids_at_hex=ids_at_hex,
        direction=direction,
        cell_to_family=partner_cell_to_family,
        self_id_to_labels=self_id_to_labels,
    )
    for label in labels:
        by_partner, total_syn, n_partner_by_type, partner_uv_by_type, partner_xy_by_type, n_self = acc[label]
        label_origin_uv, label_origin_xy = _label_origins(
            label,
            nodes,
            self_cell_to_labels,
            ids_at_hex,
            at_ref_uv,
            at_ref_xy,
            float_coords=False,
            need_uv=show_partner_uv,
            need_xy=show_partner_d_xy,
        )
        print_table(
            label,
            by_partner,
            total_syn,
            n_partner_by_type,
            partner_uv_by_type,
            partner_xy_by_type=None,
            hex_note=hex_note,
            direction=direction,
            use_family=args.family,
            min_pct=args.min,
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
