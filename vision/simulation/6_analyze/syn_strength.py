"""Query trained syn_strength_cell joined to connectome partner % n_syn.

Reads ``best_param.npz`` + ``train_opts.json`` only (no train session rebuild).
Partner % comes from ``analyze_cell_syn``; syn_strength_cell / a_* from the named npz.

Examples
--------
  ../.venv/bin/python analyze/syn_strength.py T4a --post
  ../.venv/bin/python analyze/syn_strength.py Mi1 --x 0 --y 1
"""

from __future__ import annotations

from default_params import (
    RUN_PATH,
)

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401
import network.path  # noqa: F401  # FAFB on sys.path
import analyze_cell_syn
import train
import figure.plot as plot_trained
import train.implementation as train_mod
from import_bootstrap import parse_comma_list
from network.connectivity import build_cell_pair_indices
from network.construction import load_network_json
from default_params import RUN_PATH


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Trained syn_strength_cell per partner type, with analyze_cell_syn "
            "% n_syn+/- from the run's network.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "cells",
        nargs="?",
        default="L1",
        metavar="CELL[,CELL...]",
        help="comma-separated cells (plain names only). Default: L1",
    )
    ap.add_argument(
        "--run",
        default=RUN_PATH,
        help="run folder under PARAMETER_DIR or absolute path (default: %(default)s)",
    )
    ap.add_argument(
        "--post",
        action="store_true",
        help=(
            "outgoing: treat CELL as source_cell, break down by target_cell. "
            "Default is incoming onto CELL."
        ),
    )
    ap.add_argument(
        "--x",
        type=float,
        default=None,
        metavar="X",
        help="hex-step x (FAFB); with --y selects one hex, alone a line",
    )
    ap.add_argument(
        "--y",
        type=float,
        default=None,
        metavar="Y",
        help="hex-step y (FAFB); with --x selects one hex, alone a line",
    )
    args = ap.parse_args(argv)

    tokens = parse_comma_list(args.cells)
    for tok in tokens:
        if tok.startswith(":") or tok.startswith("@"):
            raise SystemExit(
                f"plain cell names only (got {tok!r}); "
                "use analyze_cell_syn.py for :family / @root_id"
            )

    outdir = plot_trained.resolve_run_dir(args.run)
    opts = plot_trained.load_train_opts(outdir)
    if not opts:
        raise SystemExit(f"missing train_opts.json under {outdir}")
    if opts.get("model", "borst") not in ("borst", "hp_lp"):
        raise SystemExit(
            f"syn_strength_cell requires borst/hp_lp model, got {opts.get('model')!r}"
        )
    network_json = opts.get("network_json")
    if not network_json:
        raise SystemExit("train_opts.json missing network_json")

    try:
        named, cells_npz, pair_names = train_mod.load_best_param_named(outdir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        nodes, edges, cells, _meta = load_network_json(network_json)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if list(cells) != list(cells_npz):
        raise SystemExit(
            f"cells mismatch: network.json vs best_param.npz "
            f"({len(cells)} vs {len(cells_npz)})"
        )

    i_from_name = {n: i for i, n in enumerate(cells)}
    n_cells = len(cells)
    for tok in tokens:
        if tok not in i_from_name:
            raise SystemExit(f"unknown cell {tok!r}; known e.g. {cells[:8]}...")

    src_t = np.array([i_from_name[e["source_cell"]] for e in edges], dtype=np.int64)
    tar_t = np.array([i_from_name[e["target_cell"]] for e in edges], dtype=np.int64)
    _, n_pairs, pair_keys = build_cell_pair_indices(src_t, tar_t, n_cells)
    i_from_key = {k: i for i, k in enumerate(pair_keys)}
    if pair_names is not None:
        expected = [f"{cells[s]}{train.PAIR_SEP}{cells[t]}" for s, t in pair_keys]
        if list(pair_names) != expected:
            raise SystemExit("pair_names in best_param.npz do not match network.json edges")

    for key in ("syn_strength_cell", "a_in", "a_out", "a_gt", "bias_gt"):
        if key not in named:
            raise SystemExit(f"best_param.npz missing {key}")
    syn_strength_cell = np.asarray(named["syn_strength_cell"], dtype=np.float64).reshape(-1)
    if syn_strength_cell.shape[0] != n_pairs:
        raise SystemExit(
            f"syn_strength_cell length {syn_strength_cell.shape[0]} != n_pairs {n_pairs}"
        )

    at_x, at_y = analyze_cell_syn.cli_xy_filter(args.x, args.y)
    hex_note = ""
    ids_at_hex = None
    if at_x is not None or at_y is not None:
        try:
            ids_at_hex, hex_note, _ref_xy, _single = analyze_cell_syn.resolve_xy_instance_ids(
                nodes, at_x, at_y
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    direction = "post" if args.post else "pre"
    partner_syn_by_cell = analyze_cell_syn.query_partner_syn(
        nodes, edges, tokens, direction=direction, ids_at_hex=ids_at_hex,
    )

    print(f"n_pairs={n_pairs}  best_param.npz syn_strength_cell={syn_strength_cell.shape[0]}")
    for cell in tokens:
        if cell not in partner_syn_by_cell:
            print(f"warning: no partner_syn for {cell}", flush=True)
            continue
        ti = i_from_name[cell]
        by_partner, n_syn_sum, n_partner, partner_uv, partner_xy, n_self = partner_syn_by_cell[cell]
        alpha_map = {}
        for partner in by_partner:
            if partner not in i_from_name:
                alpha_map[partner] = "-"
                continue
            pair = (
                (i_from_name[cell], i_from_name[partner]) if direction == "post"
                else (i_from_name[partner], i_from_name[cell])
            )
            pi = i_from_key.get(pair)
            alpha_map[partner] = (
                "-" if pi is None else f"{float(syn_strength_cell[pi]):.6g}"
            )
        analyze_cell_syn.print_table(
            cell,
            by_partner,
            n_syn_sum,
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
                f"a_in={float(named['a_in'][ti]):g}, "
                f"a_out={float(named['a_out'][ti]):g}, "
                f"a_gt={float(named['a_gt'][ti]):g}, "
                f"bias_gt={float(named['bias_gt'][ti]):g}"
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
