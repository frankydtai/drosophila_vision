#!/usr/bin/env python3
"""Plot legacy ``with_Ih`` conductance fits (138-param Borst, ``out_scale`` fixed)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import Medulla_Library as ml
import FiveCol_MedSim_Pytorch as fc
import run
from plot_trained import (
    load_session,
    load_train_opts,
    plot_model_all_moving_bar,
    plot_model_data_moving_bar,
    plot_param_set,
)

DEFAULT_PARAMS = os.path.join("FiveCol_Parameter", "with_Ih", "best_parameter.npy")
DEFAULT_OUTDIR = os.path.join("FiveCol_Parameter", "with_Ih")
LEGACY_NPARAMS = ml.legacy_conductance_z_slices()["n_params"]
LEGACY_PARAM_MODES = {"out_scale": "fixed"}
LEGACY_PARAM_FIXES = {"out_scale": 1.0}
LEGACY_TILE_STIMULUS_OPTS = {
    "target": "tile",
    "mode": "borst",
    "i_baseline": 0.0,
    "i_bright": float(ml.I_BRIGHT),
    "t_on": ml.T_ON,
    "maxtime": ml.IMPULSE_MAXTIME,
    "deltat_ms": 10.0,
}
LEGACY_BAR_STIMULUS_OPTS = {
    "target": "moving_bar",
    "mode": "borst",
    "i_baseline_bright_bar": 0.0,
    "i_baseline_dark_bar": float(ml.I_BRIGHT),
    "i_bright_bar": float(ml.I_BRIGHT),
    "i_dark_bar": float(ml.I_DARK),
    "t_on": ml.T_ON,
    "deltat_ms": 10.0,
    "center_column": False,
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=DEFAULT_PARAMS,
                    help=f"legacy conductance z vector ({LEGACY_NPARAMS},) .npy")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="directory for PNG output")
    ap.add_argument("--bar", action="store_true",
                    help="also write model_*_bar.png (Borst moving-bar by default)")
    ap.add_argument("--network-ref", default=None,
                    help="optional run dir with network sidecars for connectome moving-bar")
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
    legacy_ct = np.load("Circuits/ctype.npy", allow_pickle=True)
    leg_idx = {str(n): i for i, n in enumerate(legacy_ct)}
    slices = ml.legacy_conductance_z_slices()
    inp = np.array([float(z138[slices["inp_gain"].start + leg_idx[t]]) if t in leg_idx else 0.5
                    for t in net_type_names])
    out = np.array([float(z138[slices["out_gain"].start + leg_idx[t]]) if t in leg_idx else 0.5
                    for t in net_type_names])
    ih_sc = z138[slices["Ih_gmax"].start:slices["n_params"]]
    return np.concatenate([inp, out, ih_sc])


def borst_bar_session(outdir):
    train = load_train_opts(outdir)
    opts = (train or {}).get("bar_stimulus_opts") or LEGACY_BAR_STIMULUS_OPTS
    session = fc.open_session(
        fc.make_train_opts(
            backend="borst", target_list=["moving_bar"],
            bar_stimulus_opts=opts,
            moving_bar_center_column=bool(opts.get("center_column", False)),
        ),
        "conductance",
    )
    return run.apply_param_modes(session, LEGACY_PARAM_MODES, LEGACY_PARAM_FIXES)


def borst_tile_session():
    session = fc.open_session(
        fc.make_train_opts(
            backend="borst", target_list=["tile"],
            tile_stimulus_opts=LEGACY_TILE_STIMULUS_OPTS,
        ),
        "conductance",
    )
    return run.apply_param_modes(session, LEGACY_PARAM_MODES, LEGACY_PARAM_FIXES)


def main():
    args = parse_args()
    z138 = load_legacy_z(args.params)
    os.makedirs(args.outdir, exist_ok=True)

    if load_train_opts(args.outdir) is None:
        tile_session = borst_tile_session()
    else:
        tile_session = None

    _, tile_cost = plot_param_set(
        z138[None], args.outdir, model_type="conductance", model_all=True,
        param_modes=LEGACY_PARAM_MODES, param_fixes=LEGACY_PARAM_FIXES,
        session=tile_session,
    )
    print(f"params: {args.params}  ({LEGACY_NPARAMS} legacy, out_scale=fixed)")
    print(f"tile:   Borst  cost={tile_cost:.4f}%")

    if args.bar:
        if args.network_ref:
            ref_opts = load_train_opts(args.network_ref)
            if ref_opts is None or ref_opts.get("backend") != "network":
                raise SystemExit(f"no train_opts.json (network) in {args.network_ref!r}")
            net_session = load_session(
                args.network_ref, "conductance",
                param_modes=LEGACY_PARAM_MODES, param_fixes=LEGACY_PARAM_FIXES,
            )
            z_net = map_legacy_to_network_z(z138, list(net_session.backend.network.type_names))
            n = fc.schema_nparams(list(net_session.schema))
            if z_net.shape[0] != n:
                raise SystemExit(f"mapped network z length {z_net.shape[0]} != schema {n}")
            _, bar_cost = plot_param_set(
                z_net[None], args.outdir, model_type="conductance", model_all=True,
                context_dir=args.network_ref, param_modes=LEGACY_PARAM_MODES,
                param_fixes=LEGACY_PARAM_FIXES,
                plot_targets=["moving_bar"],
            )
            print(f"bar:    network context from {args.network_ref}  cost={bar_cost:.4f}%  "
                  f"({len(net_session.backend.network.type_names)} types)")
        else:
            bar_session = borst_bar_session(args.outdir)
            z_t = torch.tensor(z138, dtype=torch.float64, device=bar_session.device)
            bar_cost = float(fc.calc_cost(z_t, bar_session).item())
            suffix = f"trained, cost {bar_cost:.2f}% of data power"
            plot_model_data_moving_bar(
                bar_session, z_t, os.path.join(args.outdir, "model_data_bar.png"),
                title=f"Moving-bar model-data ({suffix})",
            )
            plot_model_all_moving_bar(
                bar_session, z_t, os.path.join(args.outdir, "model_all_bar.png"),
                title=f"Moving-bar model-all ({suffix})",
            )
            print(f"bar:    Borst moving-bar  cost={bar_cost:.4f}%")


if __name__ == "__main__":
    main()
