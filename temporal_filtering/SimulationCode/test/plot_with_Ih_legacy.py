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
import train
from plot_trained import (
    load_session,
    load_train_opts,
    plot_moving_bar_all,
    plot_moving_bar_data,
    plot_param_set,
)

DEFAULT_PARAMS = os.path.join("FiveCol_Parameter", "with_Ih", "best_parameter.npy")
DEFAULT_OUTDIR = os.path.join("FiveCol_Parameter", "with_Ih")


def _legacy_conductance_z_slices():
    """138-parameter conductance z layout (this test script only)."""
    ih_start = 2 * ml.nofcells
    n_lamina_ih = ml.LAMINA_SLICE.stop - ml.LAMINA_SLICE.start
    return {
        "inp_gain": slice(0, ml.nofcells),
        "out_gain": slice(ml.nofcells, 2 * ml.nofcells),
        "Ih_gmax": slice(ih_start, ih_start + n_lamina_ih),
        "Ih_midv": ih_start + n_lamina_ih,
        "Ih_slope": ih_start + n_lamina_ih + 1,
        "tau_midv": ih_start + n_lamina_ih + 2,
        "n_params": 2 * ml.nofcells + n_lamina_ih + 3,
        "n_selp_correlation": ih_start + n_lamina_ih,
    }


LEGACY_NPARAMS = _legacy_conductance_z_slices()["n_params"]
LEGACY_PARAM_MODES = {"out_scale": "fixed"}
LEGACY_PARAM_FIXES = {"out_scale": 1.0}
LEGACY_SPOT_STIMULUS_OPTS = {
    "target": "spot_bright",
    "mode": "borst",
    "i_baseline": 0.0,
    "i_bright": float(ml.I_BRIGHT),
    "t_on": ml.T_ON,
    "maxtime": ml.IMPULSE_MAXTIME,
    "deltat_ms": 10.0,
}
LEGACY_BAR_BRIGHT_OPTS = {
    "target": "moving_bar_bright",
    "mode": "borst",
    "i_baseline_bright_bar": 0.0,
    "i_bright_bar": float(ml.I_BRIGHT),
    "t_on": ml.T_ON,
    "deltat_ms": 10.0,
    "center_column": False,
}
LEGACY_BAR_DARK_OPTS = {
    "target": "moving_bar_dark",
    "mode": "borst",
    "i_baseline_dark_bar": float(ml.I_BRIGHT),
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
    slices = _legacy_conductance_z_slices()
    inp = np.array([float(z138[slices["inp_gain"].start + leg_idx[t]]) if t in leg_idx else 0.5
                    for t in net_type_names])
    out = np.array([float(z138[slices["out_gain"].start + leg_idx[t]]) if t in leg_idx else 0.5
                    for t in net_type_names])
    ih_sc = z138[slices["Ih_gmax"].start:slices["n_params"]]
    return np.concatenate([inp, out, ih_sc])


def _legacy_bar_opts_from_sidecar(train):
    bright = dict(LEGACY_BAR_BRIGHT_OPTS)
    dark = dict(LEGACY_BAR_DARK_OPTS)
    if not train:
        return bright, dark
    old = train.get("bar_stimulus_opts")
    if old:
        for k, v in old.items():
            if k in bright or k.startswith("i_baseline_bright") or k == "i_bright_bar":
                bright[k.replace("moving_bar", "moving_bar_bright") if k == "target" else k] = v
            if k in dark or k.startswith("i_baseline_dark") or k == "i_dark_bar":
                dark[k.replace("moving_bar", "moving_bar_dark") if k == "target" else k] = v
        if "center_column" in old:
            bright["center_column"] = dark["center_column"] = bool(old["center_column"])
    bright.update(train.get("moving_bar_bright_stimulus_opts") or {})
    dark.update(train.get("moving_bar_dark_stimulus_opts") or {})
    return bright, dark


def borst_bar_sessions(outdir):
    train = load_train_opts(outdir)
    bright_opts, dark_opts = _legacy_bar_opts_from_sidecar(train)
    center = bool(bright_opts.get("center_column", False))
    base = fc.make_train_opts(
        backend="borst",
        target_list=["moving_bar_bright", "moving_bar_dark"],
        moving_bar_bright_stimulus_opts=bright_opts,
        moving_bar_dark_stimulus_opts=dark_opts,
        moving_bar_center_column=center,
    )
    session = fc.open_session(base, "conductance")
    session = train.apply_param_modes(session, LEGACY_PARAM_MODES, LEGACY_PARAM_FIXES)
    s_bright = fc.open_session({**base, "target_list": ["moving_bar_bright"], "packs": None},
                               "conductance", model_backend=session.backend)
    s_dark = fc.open_session({**base, "target_list": ["moving_bar_dark"], "packs": None},
                             "conductance", model_backend=session.backend)
    return s_bright, s_dark


def borst_spot_session():
    session = fc.open_session(
        fc.make_train_opts(
            backend="borst", target_list=["spot_bright"],
            spot_bright_stimulus_opts=LEGACY_SPOT_STIMULUS_OPTS,
        ),
        "conductance",
    )
    return train.apply_param_modes(session, LEGACY_PARAM_MODES, LEGACY_PARAM_FIXES)


def main():
    args = parse_args()
    z138 = load_legacy_z(args.params)
    os.makedirs(args.outdir, exist_ok=True)

    if load_train_opts(args.outdir) is None:
        spot_session = borst_spot_session()
    else:
        spot_session = None

    _, spot_cost = plot_param_set(
        z138[None], args.outdir, model="conductance", model_all=True,
        param_modes=LEGACY_PARAM_MODES, param_fixes=LEGACY_PARAM_FIXES,
        session=spot_session,
    )
    print(f"params: {args.params}  ({LEGACY_NPARAMS} legacy, out_scale=fixed)")
    print(f"spot_bright: Borst  cost={spot_cost:.4f}%")

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
                z_net[None], args.outdir, model="conductance", model_all=True,
                context_dir=args.network_ref, param_modes=LEGACY_PARAM_MODES,
                param_fixes=LEGACY_PARAM_FIXES,
                plot_targets=["moving_bar_bright", "moving_bar_dark"],
            )
            print(f"bar:    network context from {args.network_ref}  cost={bar_cost:.4f}%  "
                  f"({len(net_session.backend.network.type_names)} types)")
        else:
            s_bright, s_dark = borst_bar_sessions(args.outdir)
            z_t = torch.tensor(z138, dtype=torch.float64, device=s_bright.device)
            bar_cost = float(
                fc.calc_cost(z_t, fc.open_session({
                    **s_bright.train_opts,
                    "target_list": ["moving_bar_bright", "moving_bar_dark"],
                    "packs": None,
                    "backend": "borst",
                }, "conductance", model_backend=s_bright.backend))
            )
            suffix = f"trained, cost {bar_cost:.2f}% of data power"
            plot_moving_bar_data(
                s_bright, z_t, os.path.join(args.outdir, "model_data_bar.png"),
                session_2=s_dark,
                title=f"Moving-bar model-data ({suffix})",
            )
            plot_moving_bar_all(
                s_bright, z_t, os.path.join(args.outdir, "model_all_bar.png"),
                session_2=s_dark,
                title=f"Moving-bar model-all ({suffix})",
            )
            print(f"bar:    Borst moving-bar  cost={bar_cost:.4f}%")


if __name__ == "__main__":
    main()
