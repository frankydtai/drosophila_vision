#!/usr/bin/env python3
"""Plot legacy ``with_Ih`` conductance fits (138-param Borst, ``out_scale`` fixed).

``FiveCol_Parameter/with_Ih/best_parameter.npy`` uses the historical layout
(65 inp + 65 out + 5 Ih + 3 scalars). Match training with
``--mode out_scale=fixed`` (``nparams=138``) — no padding.

Delegates to :func:`plot_trained.plot_param_set` / :func:`run.restore_training_context`
(same Borst-vs-network rule as ``run_training``: network only when the context
run folder has network sidecars).

  - ``model_data_tile.png``, ``model_all_tile.png`` — Borst 5-column (**65** cells)
  - ``model_data_bar.png``, ``model_all_bar.png`` (``--bar``) — connectome moving-bar
    via ``--network-ref`` context (32 types; legacy z name-mapped)

Usage:
    ../.venv/bin/python test/plot_with_Ih_legacy.py
    ../.venv/bin/python test/plot_with_Ih_legacy.py \\
        --params FiveCol_Parameter/with_Ih/best_parameter.npy \\
        --outdir FiveCol_Parameter/with_Ih
    ../.venv/bin/python test/plot_with_Ih_legacy.py --bar \\
        --network-ref FiveCol_Parameter/conductance/run_20260703_031746
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import Medulla_Library as ml
import FiveCol_MedSim_Pytorch as fc
import run
from plot_trained import plot_param_set

DEFAULT_PARAMS = os.path.join("FiveCol_Parameter", "with_Ih", "best_parameter.npy")
DEFAULT_OUTDIR = os.path.join("FiveCol_Parameter", "with_Ih")
LEGACY_NPARAMS = ml.legacy_conductance_z_slices()["n_params"]
LEGACY_PARAM_MODES = {"out_scale": "fixed"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=DEFAULT_PARAMS,
                    help=f"legacy conductance z vector ({LEGACY_NPARAMS},) .npy")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="directory for PNG output")
    ap.add_argument("--bar", action="store_true",
                    help="also write model_*_bar.png (connectome moving-bar target)")
    ap.add_argument("--network-ref", default=None,
                    help="run dir with network sidecars (required when --bar)")
    return ap.parse_args()


def load_legacy_z(path: str) -> np.ndarray:
    z = np.load(path)
    if z.ndim != 1:
        raise SystemExit(f"expected 1d param vector, got shape {z.shape} from {path!r}")
    if z.shape[0] != LEGACY_NPARAMS:
        raise SystemExit(
            f"legacy with_Ih params must have {LEGACY_NPARAMS} values, got {z.shape[0]} "
            f"from {path!r}; use plot_trained.py for full-schema runs"
        )
    return z


def map_legacy_to_network_z(z138: np.ndarray, net_type_names: list[str]) -> np.ndarray:
    """Name-map legacy 65-type inp/out gains onto a connectome type vocabulary."""
    legacy_ct = np.load("Circuits/ctype.npy", allow_pickle=True)
    leg_idx = {str(n): i for i, n in enumerate(legacy_ct)}
    slices = ml.legacy_conductance_z_slices()
    inp = np.array([float(z138[slices["inp_gain"].start + leg_idx[t]]) if t in leg_idx else 0.5
                    for t in net_type_names])
    out = np.array([float(z138[slices["out_gain"].start + leg_idx[t]]) if t in leg_idx else 0.5
                    for t in net_type_names])
    ih_sc = z138[slices["Ih_gmax"].start:slices["n_params"]]
    return np.concatenate([inp, out, ih_sc])


def main():
    args = parse_args()
    z138 = load_legacy_z(args.params)
    os.makedirs(args.outdir, exist_ok=True)

    _, tile_cost = plot_param_set(
        z138[None], args.outdir, model_type="conductance", model_all=True,
        param_modes=LEGACY_PARAM_MODES,
    )
    print(f"params: {args.params}  ({LEGACY_NPARAMS} legacy, out_scale=fixed)")
    print(f"tile:   Borst (no network sidecars in {args.outdir})  cost={tile_cost:.4f}%")

    if args.bar:
        if not args.network_ref:
            raise SystemExit("--bar requires --network-ref <run_dir with network sidecars>")
        if not run.has_network_run(args.network_ref):
            raise SystemExit(f"no network sidecars in {args.network_ref!r}")
        run.restore_training_context(
            args.network_ref, "conductance", param_modes=LEGACY_PARAM_MODES,
        )
        z_net = map_legacy_to_network_z(z138, list(fc.NETWORK.type_names))
        n = fc.schema_nparams(fc.CONDUCTANCE_SCHEMA)
        if z_net.shape[0] != n:
            raise SystemExit(f"mapped network z length {z_net.shape[0]} != schema {n}")
        _, bar_cost = plot_param_set(
            z_net[None], args.outdir, model_type="conductance", model_all=True,
            context_dir=args.network_ref, param_modes=LEGACY_PARAM_MODES,
            plot_targets=["moving_bar"],
        )
        print(f"bar:    network context from {args.network_ref}  cost={bar_cost:.4f}%  "
              f"({len(fc.NETWORK.type_names)} types)")


if __name__ == "__main__":
    main()
