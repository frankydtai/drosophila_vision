"""Query trained syn_strength (alpha) joined to connectome partner % n_syn.

Reads ``best_param.npy`` + ``train_opts.json`` only (no training session rebuild).
Partner % comes from ``cell_syn``; alpha / gains from the saved parameter vector.

Examples
--------
  # GUI Run on this file, or:
  ../.venv/bin/python analysis/syn_strength.py T4a --post
  ../.venv/bin/python -m analysis.syn_strength Mi1 --x 0 --y 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import FiveCol_MedSim_Pytorch as fc
import network_bootstrap  # noqa: F401  # FAFB on sys.path
import cell_syn
import plot_trained
import train as train_mod
from connectome_io import parse_comma_list
from network.connectivity import build_type_pair_index
from network.construction import read_network_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RUN_NAME = """
27252028-train-nofsteps-1000-lrs-0.1-shift-extent-1-cost-extent-9
""".strip()
DEFAULT_RUN_PATH = "conductance/" + DEFAULT_RUN_NAME


def _alpha_by_partner(
    *,
    cell: str,
    partner: str,
    direction: str,
    name_to_i: dict[str, int],
    key_to_i: dict[tuple[int, int], int],
    syn_strength: np.ndarray,
) -> str:
    """Format alpha for one partner row; ``-`` if type pair absent from connectome."""
    if cell not in name_to_i or partner not in name_to_i:
        return "-"
    if direction == "post":
        pair = (name_to_i[cell], name_to_i[partner])
    else:
        pair = (name_to_i[partner], name_to_i[cell])
    pi = key_to_i.get(pair)
    if pi is None:
        return "-"
    return f"{float(syn_strength[pi]):.6g}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Trained syn_strength (alpha) per partner type, with cell_syn "
            "% n_syn+/- from the run's network.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "cell_types",
        nargs="?",
        default="L1",
        metavar="CELL_TYPE[,CELL_TYPE...]",
        help="comma-separated cell types (plain names only). Default: L1",
    )
    ap.add_argument(
        "--run",
        default=DEFAULT_RUN_PATH,
        help="run folder under PARAMETER_DIR or absolute path (default: %(default)s)",
    )
    ap.add_argument(
        "--post",
        action="store_true",
        help=(
            "outgoing: treat CELL_TYPE as source_type, break down by target_type. "
            "Default is incoming onto CELL_TYPE."
        ),
    )
    ap.add_argument(
        "--x",
        type=float,
        default=None,
        metavar="X",
        help="hex-step x (FAFB); with --y selects one column, alone a line",
    )
    ap.add_argument(
        "--y",
        type=float,
        default=None,
        metavar="Y",
        help="hex-step y (FAFB); with --x selects one column, alone a line",
    )
    args = ap.parse_args(argv)

    tokens = parse_comma_list(args.cell_types)
    for tok in tokens:
        if tok.startswith(":") or tok.startswith("@"):
            raise SystemExit(
                f"plain cell-type names only (got {tok!r}); "
                "use cell_syn.py for :family / @root_id"
            )

    outdir = plot_trained.resolve_run_dir(args.run)
    opts = plot_trained.load_train_opts(outdir)
    if not opts:
        raise SystemExit(f"missing train_opts.json under {outdir}")
    if opts.get("model", "conductance") != "conductance":
        raise SystemExit(
            f"syn_strength requires conductance model, got {opts.get('model')!r}"
        )
    network_json = opts.get("network_json")
    if not network_json:
        raise SystemExit("train_opts.json missing network_json")

    try:
        z = train_mod.load_best_param(outdir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        nodes, edges, type_names, _meta = read_network_json(network_json)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    name_to_i = {n: i for i, n in enumerate(type_names)}
    n_types = len(type_names)
    for tok in tokens:
        if tok not in name_to_i:
            raise SystemExit(f"unknown cell type {tok!r}; known e.g. {type_names[:8]}...")

    src_t = np.array([name_to_i[e["source_type"]] for e in edges], dtype=np.int64)
    tar_t = np.array([name_to_i[e["target_type"]] for e in edges], dtype=np.int64)
    _pair_idx, n_pairs, pair_keys = build_type_pair_index(src_t, tar_t, n_types)
    key_to_i = {k: i for i, k in enumerate(pair_keys)}

    try:
        slices = fc.unpack_conductance_z(z, type_names, n_pairs, opts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if "syn_strength" not in slices:
        raise SystemExit("schema missing syn_strength")
    syn_strength = slices["syn_strength"]
    for key in ("in_gain", "out_gain", "out_scale"):
        if key not in slices:
            raise SystemExit(f"schema missing {key}")

    at_x, at_y = cell_syn.cli_xy_filter(args.x, args.y)
    hex_note = ""
    ids_at_hex = None
    if at_x is not None or at_y is not None:
        try:
            ids_at_hex, hex_note, _ref_xy, _single = cell_syn.resolve_xy_instance_ids(
                nodes, at_x, at_y
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    direction = "post" if args.post else "pre"
    acc = cell_syn.query_partner_syn(
        nodes, edges, tokens, direction=direction, ids_at_hex=ids_at_hex,
    )

    print(f"outdir={outdir}")
    print(f"n_pairs={n_pairs}  best_param={z.shape[0]} params")
    for cell in tokens:
        if cell not in acc:
            logger.warning("no accumulate result for %s", cell)
            continue
        ti = name_to_i[cell]
        by_partner, total_syn, n_partner, partner_uv, partner_xy, n_self = acc[cell]
        alpha_map = {
            pt: _alpha_by_partner(
                cell=cell,
                partner=pt,
                direction=direction,
                name_to_i=name_to_i,
                key_to_i=key_to_i,
                syn_strength=syn_strength,
            )
            for pt in by_partner
        }
        cell_syn.print_table(
            cell,
            by_partner,
            total_syn,
            n_partner,
            partner_uv,
            partner_xy_by_type=partner_xy,
            hex_note=hex_note,
            direction=direction,
            show_uv=False,
            show_d_xy=False,
            n_self=int(n_self),
            alpha_by_partner=alpha_map,
            after_title=(
                f"in_gain={fc.raw_segment_at_type(slices['in_gain'], ti, n_types):g}, "
                f"out_gain={fc.raw_segment_at_type(slices['out_gain'], ti, n_types):g}, "
                f"scale={fc.raw_segment_at_type(slices['out_scale'], ti, n_types):g}"
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
