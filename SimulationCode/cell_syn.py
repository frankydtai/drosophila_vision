"""
Tabulate synaptic partners of one or more cell types from the FiveCol / Srini
``multi_colM`` under ``Circuits/`` (5 columns × 65 types = 325 neurons).

By default (incoming / ``pre``) each CELL_TYPE is the postsynaptic type and is
broken down by presynaptic partner type. With ``--post`` (outgoing) each
CELL_TYPE is the presynaptic type and is broken down by postsynaptic partner.

Matrix convention matches MedSim: row = post, column = pre; value = signed
synapse count (``>0`` excitatory, ``<0`` inhibitory). Each type has one neuron
per column → ``n_target`` / ``n_source`` = 5 (or fewer with ``--x``).
Partner column ``pre_d_x`` / ``post_d_x`` is the synapse-weighted mean of
``partner_x - self_x``.

Column index ``x`` runs ``-2,-1,0,1,2`` (left → right; ``0`` = centre,
MedSim ``CENTER_COL``).

Example::

    python cell_syn.py
    python cell_syn.py L1,L2,L3,L4,L5
    python cell_syn.py Mi1 --post
    python cell_syn.py L1 --min 1
    python cell_syn.py L3 --x=0
    python cell_syn.py L3 --x=-2,0,2
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_CIRCUITS = Path(__file__).resolve().parent / "Circuits"
_NOFCOLS = 5
_VALID_X = (-2, -1, 0, 1, 2)


def _parse_comma_list(token: str) -> List[str]:
    return [p.strip() for p in str(token).split(",") if p.strip()]


def parse_x_list(token: str) -> List[int]:
    """Parse comma-separated column ``x`` in ``{-2,-1,0,1,2}`` (deduped, sorted)."""
    out: List[int] = []
    seen: Set[int] = set()
    for part in _parse_comma_list(token):
        try:
            x = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid --x value {part!r} (need integer)") from exc
        if x not in _VALID_X:
            raise ValueError(
                f"invalid --x={x}; allowed values are {list(_VALID_X)}"
            )
        if x not in seen:
            seen.add(x)
            out.append(x)
    out.sort()
    return out


def x_to_col(x: int) -> int:
    """Map column ``x`` (-2..2) to multi_colM block index 0..4."""
    return int(x) + 2


def load_connM(
    circuits_dir: Path = _CIRCUITS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load single-column ``ctype`` and five-column ``multi_colM``."""
    ctype = np.load(circuits_dir / "ctype.npy", allow_pickle=True)
    ctype = np.asarray(ctype, dtype=str)
    connM = np.asarray(
        np.load(circuits_dir / "multi_colM.npy"), dtype=np.float64
    )
    n = len(ctype)
    if connM.shape != (n * _NOFCOLS, n * _NOFCOLS):
        raise ValueError(
            f"multi_colM shape {connM.shape} != "
            f"({n * _NOFCOLS}, {n * _NOFCOLS})"
        )
    return connM, ctype


def _instance_ids(
    type_index: int,
    nofcells: int,
    *,
    xs: Optional[Sequence[int]] = None,
) -> List[int]:
    """Neuron indices for ``type_index`` in selected columns (all if ``xs`` is None)."""
    if xs is None:
        cols = range(_NOFCOLS)
    else:
        cols = [x_to_col(x) for x in xs]
    return [type_index + c * nofcells for c in cols]


def _type_name(neuron_index: int, ctype: Sequence[str]) -> str:
    return str(ctype[neuron_index % len(ctype)])


def neuron_x(neuron_index: int, nofcells: int) -> int:
    """Column ``x`` in ``{-2..2}`` for a ``multi_colM`` neuron index."""
    return int(neuron_index) // int(nofcells) - 2


def _format_mean_dx(z: float) -> str:
    return f"{z:.2f}"


def resolve_query_labels(
    tokens: List[str], ctype: Sequence[str]
) -> Tuple[List[str], Dict[str, Set[str]]]:
    """Resolve queried type tokens to (ordered labels, self_type -> labels).

    Rejects ``@root_id`` and ``:family`` tokens. Unknown types raise ``ValueError``.
    """
    known = set(ctype)
    labels: List[str] = []
    self_type_to_labels: Dict[str, Set[str]] = defaultdict(set)
    for tok in tokens:
        if tok.startswith("@") or tok.startswith(":"):
            raise ValueError(
                f"unsupported query token {tok!r} "
                "(no @root_id or :family in SimulationCode/cell_syn.py)"
            )
        if tok not in known:
            raise ValueError(f"unknown cell type {tok!r}")
        if tok not in labels:
            labels.append(tok)
        self_type_to_labels[tok].add(tok)
    return labels, dict(self_type_to_labels)


def accumulate_all(
    connM: np.ndarray,
    ctype: Sequence[str],
    labels: List[str],
    *,
    direction: str = "pre",
    xs: Optional[Sequence[int]] = None,
) -> Dict[
    str,
    Tuple[
        DefaultDict[str, Dict[str, float]],
        float,
        Dict[str, int],
        Dict[str, float],
        int,
    ],
]:
    """Per queried label: (syn+/syn-, total |n_syn|, n_partner, mean d_x, n_self).

    Aggregates over column instances of each type in ``multi_colM`` (all five, or
    only columns in ``xs`` if given). ``direction="pre"``: label is postsynaptic
    (rows); partners are presynaptic columns. ``direction="post"``: label is
    presynaptic (columns); partners are postsynaptic rows. ``n_partner`` counts
    distinct partner neuron indices. Mean ``d_x`` is synapse-weighted
    ``partner_x - self_x``.
    """
    nofcells = len(ctype)
    name_to_idx = {str(n): i for i, n in enumerate(ctype)}
    by_cell: Dict[str, DefaultDict[str, Dict[str, float]]] = {
        p: defaultdict(lambda: {"syn+": 0.0, "syn-": 0.0}) for p in labels
    }
    totals: Dict[str, float] = {p: 0.0 for p in labels}
    partner_ids: Dict[str, DefaultDict[str, Set[int]]] = {
        p: defaultdict(set) for p in labels
    }
    dx_sum: Dict[str, DefaultDict[str, float]] = {
        p: defaultdict(float) for p in labels
    }
    dx_w: Dict[str, DefaultDict[str, float]] = {
        p: defaultdict(float) for p in labels
    }
    n_self: Dict[str, int] = {p: 0 for p in labels}

    for label in labels:
        self_ids = _instance_ids(name_to_idx[label], nofcells, xs=xs)
        n_self[label] = len(self_ids)
        for sid in self_ids:
            sx = neuron_x(sid, nofcells)
            if direction == "post":
                weights = connM[:, sid]
            else:
                weights = connM[sid, :]
            for pid, w in enumerate(weights):
                w = float(w)
                if w == 0.0:
                    continue
                pt = _type_name(pid, ctype)
                ns = abs(w)
                totals[label] += ns
                if w > 0:
                    by_cell[label][pt]["syn+"] += ns
                else:
                    by_cell[label][pt]["syn-"] += ns
                partner_ids[label][pt].add(pid)
                dx_sum[label][pt] += (neuron_x(pid, nofcells) - sx) * ns
                dx_w[label][pt] += ns

    out: Dict[
        str,
        Tuple[
            DefaultDict[str, Dict[str, float]],
            float,
            Dict[str, int],
            Dict[str, float],
            int,
        ],
    ] = {}
    for p in labels:
        mean_dx = {
            pt: dx_sum[p][pt] / dx_w[p][pt]
            for pt in dx_w[p]
            if dx_w[p][pt] > 0
        }
        out[p] = (
            by_cell[p],
            totals[p],
            {pt: len(ids) for pt, ids in partner_ids[p].items()},
            mean_dx,
            n_self[p],
        )
    return out


def query_partner_syn(
    connM: np.ndarray,
    ctype: Sequence[str],
    cell_types: List[str],
    *,
    direction: str = "pre",
    xs: Optional[Sequence[int]] = None,
) -> Dict[
    str,
    Tuple[
        DefaultDict[str, Dict[str, float]],
        float,
        Dict[str, int],
        Dict[str, float],
        int,
    ],
]:
    """Resolve ``cell_types`` and return ``accumulate_all`` partner syn stats."""
    labels, _ = resolve_query_labels(list(cell_types), ctype)
    return accumulate_all(connM, ctype, labels, direction=direction, xs=xs)


def print_table(
    cell_type: str,
    by_partner: DefaultDict[str, Dict[str, float]],
    total_syn: float,
    n_partner_by_type: Dict[str, int],
    partner_dx_by_type: Dict[str, float],
    *,
    direction: str = "pre",
    min_pct: float = 0.0,
    n_self: int = 0,
    hex_note: str = "",
) -> None:
    """Print partner mix table including ``pre_d_x`` / ``post_d_x``."""
    if direction == "post":
        self_field, partner_field = "source_type", "target_type"
        flow_word = "out of"
        n_count_label = "n_source"
        dx_label = "post_d_x"
    else:
        self_field, partner_field = "target_type", "source_type"
        flow_word = "onto"
        n_count_label = "n_target"
        dx_label = "pre_d_x"

    header = [partner_field, "% n_syn+", "% n_syn-", "n_neuron", dx_label]
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
            dx = partner_dx_by_type.get(pt)
            rows.append(
                [
                    pt,
                    f"{pp:.4f}",
                    f"{pm:.4f}",
                    str(int(n_partner_by_type.get(pt, 0))),
                    "" if dx is None else _format_mean_dx(dx),
                ]
            )

    total_n = sum(int(n_partner_by_type.get(pt, 0)) for pt in by_partner)
    total_row = ["TOTAL", f"{sum_p:.4f}", f"{sum_m:.4f}", str(total_n), ""]
    all_rows = [header] + rows + [total_row]
    n_cols = len(header)
    widths = [max(len(r[c]) for r in all_rows) for c in range(n_cols)]

    def _fmt(row: List[str]) -> str:
        cells = [row[0].ljust(widths[0])]
        cells += [row[c].rjust(widths[c]) for c in range(1, n_cols)]
        return "  ".join(cells).rstrip()

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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Synapse mix between a cell type and its synaptic partners from "
            "SimulationCode Circuits multi_colM (5 columns)."
        )
    )
    parser.add_argument(
        "cell_types",
        nargs="?",
        default="L1",
        metavar="CELL_TYPE[,CELL_TYPE...]",
        help="Comma-separated cell types to query (e.g. L1,Mi1). Default: L1",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help=(
            "Outgoing connections: CELL_TYPE as presynaptic source_type, "
            "break down by target_type. Default is incoming onto CELL_TYPE."
        ),
    )
    parser.add_argument(
        "--min",
        type=float,
        default=0.0,
        metavar="PCT",
        help=(
            "Only list partner rows whose combined %% n_syn+ + %% n_syn- "
            "exceeds PCT. TOTAL still reflects all partners."
        ),
    )
    parser.add_argument(
        "--x",
        default=None,
        metavar="X[,X...]",
        help=(
            "Restrict to CELL_TYPE instances in column x "
            f"({','.join(str(v) for v in _VALID_X)}; 0 = centre). "
            "Comma-separated allowed (e.g. --x=-2,0,2). "
            "Use --x=-2 form for negatives. Default: all five columns."
        ),
    )
    args = parser.parse_args(argv)
    direction = "post" if args.post else "pre"

    circuits = _CIRCUITS
    if not (circuits / "multi_colM.npy").is_file():
        logger.error("multi_colM.npy not found under %s", circuits)
        return 1

    logger.info("Loading %s/multi_colM.npy (%d columns)", circuits, _NOFCOLS)
    try:
        connM, ctype = load_connM(circuits)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    xs: Optional[List[int]] = None
    hex_note = ""
    if args.x is not None:
        try:
            xs = parse_x_list(args.x)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        if not xs:
            logger.error("--x is empty")
            return 1
        hex_note = " at x=" + ",".join(str(v) for v in xs)
        logger.info(
            "Restricting to column x=%s (%d of %d columns)",
            ",".join(str(v) for v in xs),
            len(xs),
            _NOFCOLS,
        )

    try:
        cell_types = _parse_comma_list(args.cell_types)
        labels, _ = resolve_query_labels(cell_types, ctype)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    acc = accumulate_all(connM, ctype, labels, direction=direction, xs=xs)
    for label in labels:
        by_partner, total_syn, n_partner_by_type, partner_dx, n_self = acc[label]
        print_table(
            label,
            by_partner,
            total_syn,
            n_partner_by_type,
            partner_dx,
            direction=direction,
            min_pct=args.min,
            n_self=n_self,
            hex_note=hex_note,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
