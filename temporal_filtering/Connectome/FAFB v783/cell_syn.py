#!/usr/bin/env python3
"""
For the connectome ``network.json`` files under this folder (built by
``build_network.py`` and stored in e.g. ``left_min_neuron1/network.json``),
tabulate the synaptic partners of one or more *cell types*.

By default (incoming / ``pre``) each CELL_TYPE is treated as the postsynaptic
``target_type`` and broken down by presynaptic ``source_type``. With ``--post``
(outgoing) each CELL_TYPE is treated as the presynaptic ``source_type`` and broken
down by postsynaptic ``target_type``.

A CELL_TYPE token may be a cell type (e.g. ``Mi1``) or, when prefixed with ``@``, a
*family* (e.g. ``@Centrifugal``) which aggregates over all its member types. The
breakdown column still shows individual ``source_type``/``target_type`` unless
``--family`` is given.

Optionally restrict to CELL_TYPE *instances* at a single hex ``(u, v)``
(``--at-hex U V``).

Per (cell_type, partner_type): sum ``n_syn`` where ``sign > 0`` vs ``sign < 0``,
then express each as a percentage of **all** ``n_syn`` for that cell type.
With ``--at-hex``, extra columns ``n_pre``/``n_post``, ``pre_uv``/``post_uv``
(hex; truncated when count>5), and ``pre_xy``/``post_xy`` (``x=v``, ``y=u+v/2``)
for the *partner* instances. The TOTAL row omits the uv/xy columns.

The ``network.json`` schema is ``{"metadata", "nodes", "edges"}`` where each node is
``{"id", "name", "u", "v", "hex_index", "input", "output"}`` and each edge is
``{"src", "tar", "sign", "n_syn", "source_type", "target_type", "du", "dv"}``.

Example::

    python3 "cell_syn.py"
    python3 "cell_syn.py" L1 L2 L3 L4 L5
    python3 "cell_syn.py" Mi1 --post
    python3 "cell_syn.py" @Centrifugal
    python3 "cell_syn.py" @Centrifugal --family
    python3 "cell_syn.py" Mi1 --family
    python3 "cell_syn.py" Mi1 --dir right_min_neuron1
    python3 "cell_syn.py" L1 --dir /abs/path/to/some_folder
    python3 "cell_syn.py" L1 -q
    python3 "cell_syn.py" Mi1 --post --at-hex 0 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_DIR = "left_min_neuron1"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


_MAX_PRE_UV_COORDS = 5


def _resolve_network_json(folder: str) -> Path:
    """Resolve a network folder/path to its ``network.json``.

    ``folder`` may be a folder name (e.g. ``left_min_neuron1``) resolved relative to
    this script's directory, an absolute/relative folder path, or a direct path to a
    ``network.json``.
    """
    p = Path(folder)
    if not p.is_absolute():
        cand = _BASE_DIR / folder
        p = cand if cand.exists() else p
    if p.is_dir():
        return (p / "network.json").resolve()
    return p.resolve()


def _load_type_to_family(json_path: Path) -> Dict[str, str]:
    """Map cell ``type`` -> ``family`` from ``type_counts_abc.csv`` next to network.json."""
    csv_path = json_path.parent / "type_counts_abc.csv"
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


def _resolve_query_labels(
    tokens: List[str], type_to_family: Dict[str, str]
) -> Tuple[List[str], Dict[str, Set[str]]]:
    """Resolve queried tokens to (ordered labels, self_type -> labels).

    A token prefixed with ``@`` is a *family*; it expands to every member type. Any other
    token is taken as a literal cell type. The label shown in the output is the token as
    typed (e.g. ``@Centrifugal`` for a family).
    """
    family_to_types: DefaultDict[str, List[str]] = defaultdict(list)
    for t, fam in type_to_family.items():
        family_to_types[fam].append(t)
    labels: List[str] = list(dict.fromkeys(tokens))
    self_type_to_labels: DefaultDict[str, Set[str]] = defaultdict(set)
    for tok in labels:
        if tok.startswith("@"):
            fam = tok[1:]
            members = family_to_types.get(fam, [])
            if not members:
                logger.warning("Family %r not found in type_counts_abc.csv", fam)
            for t in members:
                self_type_to_labels[t].add(tok)
        else:
            self_type_to_labels[tok].add(tok)
    return labels, self_type_to_labels


def _truncated_uv_pairs(
    uvs: Set[Tuple[int, int]], n_pre: int, max_coords: int = _MAX_PRE_UV_COORDS
) -> List[Tuple[int, int]]:
    """Sorted distinct (u,v), truncated to ``max_coords`` pairs when ``n_pre`` > ``max_coords``."""
    if not uvs:
        return []
    ordered = sorted(uvs)
    if n_pre > max_coords:
        ordered = ordered[:max_coords]
    return ordered


def _format_uv_coords_trunc(
    uvs: Set[Tuple[int, int]], n_pre: int, max_coords: int = _MAX_PRE_UV_COORDS
) -> str:
    """Semicolon-separated ``(u,v)``; if ``n_pre`` > ``max_coords``, only first ``max_coords`` pairs."""
    return ";".join(f"({u},{v})" for u, v in _truncated_uv_pairs(uvs, n_pre, max_coords))


def _format_scalar_for_table(z: float) -> str:
    if abs(z - round(z)) < 1e-9:
        return str(int(round(z)))
    return f"{z:g}"


def _format_pre_xy_from_uvs(
    uvs: Set[Tuple[int, int]],
    n_pre: int,
    max_coords: int = _MAX_PRE_UV_COORDS,
) -> str:
    """Same (u,v) order/truncation as ``pre_uv``; ``x=v``, ``y=u+v/2``."""
    parts: List[str] = []
    for u, v in _truncated_uv_pairs(uvs, n_pre, max_coords):
        x = float(v)
        y = float(u) + float(v) / 2.0
        parts.append(f"({_format_scalar_for_table(x)},{_format_scalar_for_table(y)})")
    return ";".join(parts)


def _node_id_to_uv(nodes: List[dict]) -> Dict[int, Tuple[int, int]]:
    """FlyWire root id -> hex (u, v) from network nodes."""
    m: Dict[int, Tuple[int, int]] = {}
    for n in nodes:
        try:
            m[int(n["id"])] = (int(n["u"]), int(n["v"]))
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
    ids_at_hex: Optional[Dict[str, Set[int]]] = None,
    id_to_uv: Optional[Dict[int, Tuple[int, int]]] = None,
    direction: str = "pre",
    type_to_family: Optional[Dict[str, str]] = None,
) -> Dict[
    str,
    Tuple[
        DefaultDict[str, Dict[str, float]],
        float,
        Optional[Dict[str, int]],
        Optional[Tuple[Dict[str, Set[Tuple[int, int]]], Set[Tuple[int, int]]]],
    ],
]:
    """One pass over edges: per queried label, (per partner type syn+/syn-, total n_syn).

    ``labels`` is the ordered list of queried tokens (a cell type, or a family entered as
    ``@Family``); ``self_type_to_labels`` maps each *self* cell type to the label(s) that
    cover it. A family label therefore aggregates over all its member types.

    ``direction="pre"`` (default): query each label as the **postsynaptic** side
    (``target_type``) and break down by presynaptic ``source_type`` (incoming).
    ``direction="post"``: query each label as the **presynaptic** side (``source_type``)
    and break down by postsynaptic ``target_type`` (outgoing).

    If ``ids_at_hex`` is set, only edges whose *self* instance id (``tar`` for ``pre``,
    ``src`` for ``post``) sits at the chosen hex are counted. In that case the third
    return value maps partner type -> count of **distinct** *partner* instance ids. If
    ``id_to_uv`` is also passed, the fourth value is ``(coords_by_partner, union)``.
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
    partner_ids: Optional[Dict[str, DefaultDict[str, Set[int]]]] = None
    partner_uv: Optional[Dict[str, DefaultDict[str, Set[Tuple[int, int]]]]] = None
    if ids_at_hex is not None:
        partner_ids = {p: defaultdict(set) for p in labels}
        if id_to_uv is not None:
            partner_uv = {p: defaultdict(set) for p in labels}
    for e in edges:
        stype = e.get(self_type_field)
        cell_labels = self_type_to_labels.get(stype)
        if not cell_labels:
            continue
        if ids_at_hex is not None:
            allowed = ids_at_hex.get(stype, set())
            if not allowed:
                continue
            self_id = e.get(self_id_field)
            if self_id is None:
                continue
            try:
                if int(self_id) not in allowed:
                    continue
            except (TypeError, ValueError):
                continue
        pt = e.get(partner_type_field) or "?"
        if type_to_family is not None:
            pt = type_to_family.get(pt, pt)
        a = _edge_sign(e)
        ns = float(e.get("n_syn", 0))
        partner = e.get(partner_id_field) if partner_ids is not None else None
        for cell in cell_labels:
            totals[cell] += ns
            if a > 0:
                by_cell[cell][pt]["syn+"] += ns
            elif a < 0:
                by_cell[cell][pt]["syn-"] += ns
            if partner_ids is not None and partner is not None:
                try:
                    pid = int(partner)
                    partner_ids[cell][pt].add(pid)
                    if partner_uv is not None and id_to_uv is not None:
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
            Optional[Dict[str, int]],
            Optional[Tuple[Dict[str, Set[Tuple[int, int]]], Set[Tuple[int, int]]]],
        ],
    ] = {}
    for p in labels:
        npartner_map: Optional[Dict[str, int]] = None
        coord_block: Optional[Tuple[Dict[str, Set[Tuple[int, int]]], Set[Tuple[int, int]]]] = None
        if partner_ids is not None:
            npartner_map = {pt: len(ids) for pt, ids in partner_ids[p].items()}
        if partner_uv is not None:
            union_uv: Set[Tuple[int, int]] = set()
            for uvs in partner_uv[p].values():
                union_uv |= uvs
            row_sets = {pt: set(uvs) for pt, uvs in partner_uv[p].items()}
            coord_block = (row_sets, union_uv)
        out[p] = (by_cell[p], totals[p], npartner_map, coord_block)
    return out


def print_table(
    cell_type: str,
    by_partner: DefaultDict[str, Dict[str, float]],
    total_syn: float,
    hex_note: str = "",
    n_partner_by_type: Optional[Dict[str, int]] = None,
    partner_coords_block: Optional[Tuple[Dict[str, Set[Tuple[int, int]]], Set[Tuple[int, int]]]] = None,
    direction: str = "pre",
    use_family: bool = False,
) -> None:
    partner_dim = "family" if use_family else "type"
    if direction == "post":
        self_field, partner_field = "source_type", f"target_{partner_dim}"
        flow_word = "out of"
        n_label, uv_label, xy_label = "n_post", "post_uv", "post_xy"
    else:
        self_field, partner_field = "target_type", f"source_{partner_dim}"
        flow_word = "onto"
        n_label, uv_label, xy_label = "n_pre", "pre_uv", "pre_xy"

    show_n = n_partner_by_type is not None
    show_coords = show_n and partner_coords_block is not None

    header = [partner_field, "% n_syn+", "% n_syn-"]
    if show_n:
        header.append(n_label)
    if show_coords:
        header += [uv_label, xy_label]

    rows: List[List[str]] = []
    sum_p = sum_m = 0.0
    rows_uv_sets, _ = partner_coords_block if partner_coords_block is not None else ({}, set())
    if total_syn <= 0:
        logger.warning("No n_syn for %s=%s", self_field, cell_type)
    else:
        for pt in sorted(by_partner):
            d = by_partner[pt]
            pp = 100.0 * d["syn+"] / total_syn
            pm = 100.0 * d["syn-"] / total_syn
            sum_p += pp
            sum_m += pm
            row = [pt, f"{pp:.4f}", f"{pm:.4f}"]
            if show_n:
                npv = int(n_partner_by_type.get(pt, 0))
                row.append(str(npv))
            if show_coords:
                npv = int(n_partner_by_type.get(pt, 0))
                uvs = rows_uv_sets.get(pt, set())
                row.append(_format_uv_coords_trunc(uvs, npv))
                row.append(_format_pre_xy_from_uvs(uvs, npv))
            rows.append(row)

    total_row = ["TOTAL", f"{sum_p:.4f}", f"{sum_m:.4f}"]
    if show_n:
        total_n = sum(int(n_partner_by_type.get(pt, 0)) for pt in by_partner)
        total_row.append(str(total_n))
    if show_coords:
        total_row += ["", ""]

    all_rows = [header] + rows + [total_row]
    n_cols = len(header)
    widths = [max(len(r[c]) for r in all_rows) for c in range(n_cols)]

    def _fmt(row: List[str]) -> str:
        cells = [row[0].ljust(widths[0])]
        cells += [row[c].rjust(widths[c]) for c in range(1, n_cols)]
        return "  ".join(cells).rstrip()

    title = f"{self_field} = {cell_type}  |  all n_syn {flow_word} {cell_type}{hex_note} = {total_syn:.1f}"
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
        nargs="*",
        default=["L1"],
        metavar="CELL_TYPE",
        help=(
            "Cell type name(s) to query, e.g. Mi1. Prefix with @ for a family "
            "(e.g. @Centrifugal) to aggregate its member types. Default: L1 if omitted"
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
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress INFO log (e.g. JSON path)",
    )
    parser.add_argument(
        "--dir",
        "--json",
        dest="folder",
        default=_DEFAULT_DIR,
        help=(
            "Network folder (e.g. left_min_neuron1, resolved next to this script) or a "
            "direct path to a folder / network.json. Default: left_min_neuron1"
        ),
    )
    parser.add_argument(
        "--at-hex",
        nargs=2,
        type=int,
        metavar=("U", "V"),
        default=None,
        help=(
            "Only count edges whose CELL_TYPE *instance* sits at hex (u,v). "
            "Omit to aggregate over all instances of each cell type (default). "
            "When set: columns n_pre/n_post, pre_uv/post_uv (hex, ≤5 pairs if >5), "
            "pre_xy/post_xy (x=v, y=u+v/2) for the partner instances. "
            "TOTAL row omits the uv/xy columns."
        ),
    )
    args = parser.parse_args(argv)
    if args.quiet:
        logger.setLevel(logging.WARNING)

    direction = "post" if args.post else "pre"

    json_path = _resolve_network_json(args.folder)

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

    ids_at_hex: Optional[Dict[str, Set[int]]] = None
    hex_note = ""
    if args.at_hex is not None:
        hu, hv = int(args.at_hex[0]), int(args.at_hex[1])
        ids_at_hex = _instance_ids_at_hex(nodes, hu, hv)
        xh = float(hv)
        yh = float(hu) + float(hv) / 2.0
        hex_note = (
            f" at hex (u,v)=({hu},{hv}) "
            f"(x,y)=({_format_scalar_for_table(xh)},{_format_scalar_for_table(yh)})"
        )
        logger.info(
            "Restricting to instances at (u,v)=(%s,%s); %d cell types have ≥1 node there",
            hu,
            hv,
            sum(1 for s in ids_at_hex.values() if s),
        )

    type_to_family_all = _load_type_to_family(json_path)
    partner_type_to_family = type_to_family_all if args.family else None

    id_to_uv = _node_id_to_uv(nodes) if args.at_hex is not None else None
    labels, self_type_to_labels = _resolve_query_labels(
        list(args.cell_types), type_to_family_all
    )
    acc = _accumulate_all(
        edges,
        labels,
        self_type_to_labels,
        ids_at_hex=ids_at_hex,
        id_to_uv=id_to_uv,
        direction=direction,
        type_to_family=partner_type_to_family,
    )
    for label in labels:
        by_partner, total_syn, n_partner_by_type, partner_coords_block = acc[label]
        print_table(
            label,
            by_partner,
            total_syn,
            hex_note=hex_note,
            n_partner_by_type=n_partner_by_type if args.at_hex is not None else None,
            partner_coords_block=partner_coords_block if args.at_hex is not None else None,
            direction=direction,
            use_family=args.family,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
