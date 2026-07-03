#!/usr/bin/env python
"""Unified training driver for the FiveCol medulla model.

`local` and `gpu` run the EXACT same code path (do_many_runs + save_training_outputs + make_plots);
they differ only in whether CUDA is disabled. Output folders are named by
model_type only (not by local/gpu). All results of a run land in one folder:

``training_config.PARAMETER_DIR`` (default ``SimulationCode/FiveCol_Parameter``):

    <model_type>/run_<id>/

where <id> is the SLURM job id (under SLURM) or a timestamp otherwise.

    # short LOCAL CPU smoke test (CUDA disabled)
    python run.py local --model_type adaptive --nofsteps 30 --lrs 0.1

    # full GPU training
    python run.py gpu --model_type conductance --nofruns 20 --nofsteps 10000 \
                      --lrs 0.1 0.01 0.001

    # moving-bar (``--network`` = folder under built_network/)
    python run.py local --target moving_bar --network right_min_neuron1_extent2 \\
                      --nofsteps 5 --lrs 0.1 --sequential

Import-safe: importing this module does NOT parse argv or touch CUDA, so test
scripts can `import run` and reuse run_training / save_training_outputs / etc.
"""
import argparse
import json
import os
import sys
import time

# When executed as a script, run from this file's own directory so `fc` finds
# Circuits/ regardless of where it was launched (no need to cd first). Done
# before importing fc (Borst paths resolve relative to SimulationCode/). NOT done on
# `import run`, so importers keep control of cwd / CUDA_VISIBLE_DEVICES.
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # CLI `local` mode must disable CUDA *before* the model library is imported.
    if len(sys.argv) > 1 and sys.argv[1] == "local":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

import network_bootstrap  # noqa: F401 — connectome_io on sys.path
from connectome_io import DEFAULT_NETWORK_RUN, NETWORK_DIR, resolve_network_json
from FiveCol_MedSim_Pytorch import do_many_runs
import FiveCol_MedSim_Pytorch as fc
from plot_trained import plot_param_set, run_dir


def make_plots(fname, outdir, session, result=None):
    """Cost curve + model-vs-data + all-cell-types."""
    if result is not None:
        plot_param_set(
            result.all_params, outdir, session=session,
            final_costs=result.final_costs,
            cost_curve=result.cost_curve,
            best_i=result.best_i,
            save_artifacts=False,
        )
        return
    params = np.load(os.path.join(outdir, fname))
    final_costs, cost_curve = load_stored_costs(outdir, fname, np.atleast_2d(params).shape[0])
    plot_param_set(
        params, outdir, session=session,
        final_costs=final_costs, cost_curve=cost_curve,
        save_artifacts=False,
    )


def ctype_labels(session):
    if session.backend.network is not None:
        return np.asarray(session.backend.network.type_names)
    path = os.path.join(os.path.dirname(os.path.abspath(fc.__file__)), "Circuits", "ctype.npy")
    return np.load(path, allow_pickle=True)


def decompose_params(z_t, session):
    """Return (per_cell_cols, global_scalars) for one parameter vector."""
    n = session.backend.n_types
    schema = list(session.schema)
    p = fc.assign_params(z_t, schema, session.backend)
    cols, glob = {}, {}
    for seg in schema:
        name, v = seg["name"], p[seg["name"]]
        if seg["kind"] == "scalar":
            glob[name] = float(v.item() if torch.is_tensor(v) else v)
        elif seg["kind"] == "output":
            cols[name] = v.detach().cpu().numpy()
        else:
            cols[name] = v[:n].detach().cpu().numpy()
    if session.model_type == "adaptive":
        glob["gate_pivot"] = float(fc.GATE_PIVOT)
    return cols, glob


def write_param_table(z_t, session, table_path, extra_cols=None):
    cols, glob = decompose_params(z_t, session)
    if extra_cols:
        cols.update(extra_cols)
    ctype = ctype_labels(session)
    cell_names = list(cols.keys())
    glob_names = list(glob.keys())
    n = session.backend.n_types
    with open(table_path, "w") as f:
        f.write("idx,ctype," + ",".join(cell_names + glob_names) + "\n")
        for i in range(n):
            row = ["%.6f" % cols[nm][i] for nm in cell_names] + ["%.6f" % glob[nm] for nm in glob_names]
            f.write("%d,%s," % (i, ctype[i]) + ",".join(row) + "\n")
    return table_path


def _artifact_stem(fname):
    return fname.replace('.npy', '')


def _final_costs_path(outdir, fname):
    return os.path.join(outdir, _artifact_stem(fname) + '_final_costs.npy')


def _cost_curve_path(outdir, fname):
    return os.path.join(outdir, _artifact_stem(fname) + '_costs.npy')


def final_costs_for_params(all_params, session, final_costs=None):
    """Per-run final costs; recompute only when not supplied."""
    all_params = np.atleast_2d(all_params)
    if final_costs is not None:
        return np.asarray(final_costs, dtype=np.float64), int(np.argmin(final_costs))
    costs = np.array([
        fc.calc_cost(
            torch.tensor(all_params[i], dtype=torch.float64, device=session.device),
            session,
        ).item()
        for i in range(all_params.shape[0])
    ])
    return costs, int(np.argmin(costs))


def write_best_artifacts(outdir, fname, session, all_params, best_i, final_costs):
    """Write ``best_param.npy`` and ``*_table.csv`` for one best index."""
    all_params = np.atleast_2d(all_params)
    best = all_params[best_i]
    np.save(os.path.join(outdir, 'best_param.npy'), best)
    table_path = os.path.join(outdir, _artifact_stem(fname) + '_table.csv')
    z_best = torch.tensor(best, dtype=torch.float64, device=session.device)
    write_param_table(z_best, session, table_path)
    print("wrote table: %s (best run #%d, cost=%.4f)" % (
        table_path, best_i, final_costs[best_i]))
    return best


def load_stored_costs(outdir, fname, n_runs):
    """Load ``*_final_costs.npy`` and step ``*_costs.npy`` when present."""
    final_costs = None
    cost_curve = None
    fp = _final_costs_path(outdir, fname)
    if os.path.isfile(fp):
        final_costs = np.load(fp)
    cp = _cost_curve_path(outdir, fname)
    if os.path.isfile(cp):
        arr = np.load(cp)
        if arr.ndim == 1 and arr.shape[0] == n_runs and final_costs is None:
            final_costs = arr
        else:
            cost_curve = arr
    return final_costs, cost_curve


def save_training_outputs(fname, outdir, session, result):
    """Write the full run artifact set (convention §5)."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'model_type.txt'), 'w') as f:
        f.write(session.model_type)
    if session.train_opts is not None:
        with open(os.path.join(outdir, fc.TRAIN_OPTS_FILE), 'w') as f:
            json.dump(session.train_opts, f, indent=2)
            f.write('\n')
    np.save(os.path.join(outdir, fname), result.all_params)
    np.save(_cost_curve_path(outdir, fname), result.cost_curve)
    np.save(_final_costs_path(outdir, fname), result.final_costs)
    write_best_artifacts(
        outdir, fname, session, result.all_params, result.best_i, result.final_costs,
    )


def save_param_tables(fname, outdir, session):
    """Regenerate ``*_table.csv`` and ``best_param.npy`` from saved ``fname`` on disk."""
    all_params = np.load(os.path.join(outdir, fname))
    final_costs, _ = load_stored_costs(outdir, fname, np.atleast_2d(all_params).shape[0])
    final_costs, best_i = final_costs_for_params(all_params, session, final_costs=final_costs)
    write_best_artifacts(outdir, fname, session, all_params, best_i, final_costs)


def apply_param_modes(session, param_modes=None, param_fixes=None):
    schema = fc.apply_modes(list(session.schema), param_modes, param_fixes)
    session = session.with_schema(schema)
    summary = ", ".join(f"{s['name']}:{fc.seg_mode(s)}({fc.seg_ntrain(s)})" for s in schema)
    print("param modes -> " + summary)
    return session


def apply_per_type_schema(session):
    schema = fc.expand_schema_per_type(list(session.schema), session.backend.n_types)
    return session.with_schema(schema)


def resolve_network(network):
    return str(resolve_network_json(network))


def build_session(
    model_type,
    *,
    network=None,
    multi_column=False,
    share_edges=False,
    sequential=None,
    target="tile",
    target_list=None,
    loss_weights=None,
    moving_bar_center_column=False,
    tile_center_column=False,
    per_type=False,
    bar_stimulus_opts=None,
    tile_stimulus_opts=None,
    i_baseline=None,
    i_bright=None,
    param_modes=None,
    param_fixes=None,
    pack_overrides=None,
    model_backend=None,
    schema=None,
):
    """Create a :class:`TrainSession` from run options."""
    tl = list(target_list) if target_list else [target]
    dev = fc.active_device()
    if network:
        network = resolve_network(network)
        mb = model_backend or fc.load_network_backend(network, dev=dev)
        opts = fc.make_train_opts(
            backend="network",
            target_list=tl,
            network=mb.network,
            network_json=network,
            multi_column=multi_column,
            share_edges=share_edges,
            sequential=sequential,
            moving_bar_center_column=moving_bar_center_column,
            tile_center_column=tile_center_column,
            loss_weights=loss_weights,
            pack_overrides=pack_overrides,
            i_baseline=i_baseline,
            i_bright=i_bright,
            dev=dev,
        )
        model_backend = mb
    else:
        opts = fc.make_train_opts(
            backend="borst",
            target_list=tl,
            loss_weights=loss_weights,
            pack_overrides=pack_overrides,
            moving_bar_center_column=moving_bar_center_column,
            bar_stimulus_opts=bar_stimulus_opts,
            tile_stimulus_opts=tile_stimulus_opts,
            i_baseline=i_baseline,
            i_bright=i_bright,
            sequential=sequential,
            per_type=per_type,
        )
    session = fc.open_session(opts, model_type, schema=schema, model_backend=model_backend)
    if per_type:
        session = apply_per_type_schema(session)
        if session.train_opts is not None:
            session.train_opts["per_type"] = True
        print(f"per_type schema -> nparams={fc.schema_nparams(list(session.schema))}")
    if param_modes or param_fixes:
        session = apply_param_modes(session, param_modes, param_fixes)
    return session


def run_training(model_type, nofruns, nofsteps, lrs, fname=None, outdir=None,
                 param_modes=None, param_fixes=None,
                 network=None, multi_column=False, share_edges=False,
                 sequential=None, target="tile",
                 target_list=None, loss_weights=None,
                 moving_bar_center_column=False, tile_center_column=False,
                 per_type=False, bar_stimulus_opts=None,
                 tile_stimulus_opts=None,
                 i_baseline=None, i_bright=None,
                 pack_overrides=None, model_backend=None, schema=None):
    """Full training pipeline (do_many_runs + save_training_outputs + plots). Returns (fname, outdir, session)."""
    session = build_session(
        model_type,
        network=network,
        multi_column=multi_column,
        share_edges=share_edges,
        sequential=sequential,
        target=target,
        target_list=target_list,
        loss_weights=loss_weights,
        moving_bar_center_column=moving_bar_center_column,
        tile_center_column=tile_center_column,
        per_type=per_type,
        bar_stimulus_opts=bar_stimulus_opts,
        tile_stimulus_opts=tile_stimulus_opts,
        i_baseline=i_baseline,
        i_bright=i_bright,
        param_modes=param_modes,
        param_fixes=param_fixes,
        pack_overrides=pack_overrides,
        model_backend=model_backend,
        schema=schema,
    )
    suffix = "" if model_type == "conductance" else f"_{model_type}"
    fname = fname or f"training{suffix or '_with_Ih'}.npy"
    outdir = outdir or run_dir(model_type)

    print(f"device={session.device}, model_type={model_type}, nofruns={nofruns}, nofsteps={nofsteps}, "
          f"lrs={lrs}, nparams={fc.schema_nparams(list(session.schema))}, fname={fname}, outdir={outdir}")
    t0 = time.time()
    result = do_many_runs(session, nofruns, nofsteps, lrs=lrs)
    print(f"done in {(time.time() - t0) / 3600:.2f} hours")

    save_training_outputs(fname, outdir, session, result)
    make_plots(fname, outdir, session, result=result)
    return fname, outdir, session


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p, nofruns, nofsteps):
        p.add_argument("--model_type", default="conductance",
                       choices=["conductance", "adaptive"])
        p.add_argument("--nofruns", type=int, default=nofruns)
        p.add_argument("--nofsteps", type=int, default=nofsteps)
        p.add_argument("--lrs", type=float, nargs="+", default=[0.1, 0.01, 0.001],
                       help="learning-rate stages; each runs for --nofsteps steps")
        p.add_argument("--fname", default=None,
                       help="params filename (default derived from --model_type)")
        p.add_argument("--outdir", default=None,
                       help="output dir (default derived from --model_type)")
        p.add_argument("--mode", nargs="+", default=[], metavar="NAME=MODE",
                       help="per-param mode override, e.g. --mode out_scale=shared "
                            "inp_gain=fixed (MODE in individual|shared|fixed)")
        p.add_argument("--fix", nargs="+", default=[], metavar="NAME=VALUE",
                       help="hold a param fixed at VALUE (implies fixed mode), "
                            "e.g. --fix Ih_midv=-50 out_scale=1.0")
        p.add_argument("--per_type", action="store_true",
                       help="train Ih (and adaptive lamina) params per cell type "
                            "instead of shared lamina/scalar values")
        p.add_argument("--network", default=None, metavar="RUN",
                       help=f"built_network run folder name (under {NETWORK_DIR}), "
                            f"e.g. {DEFAULT_NETWORK_RUN}; "
                            f"moving_bar default if omitted")
        p.add_argument(
            "--target",
            default="tile",
            help="target name(s): 'tile' or 'moving_bar', or comma-separated "
                 "multi-target list, e.g. moving_bar,tile",
        )
        p.add_argument(
            "--loss_weight",
            nargs="+",
            default=[],
            metavar="NAME=VALUE",
            help="per-target loss weights, e.g. moving_bar=1 tile=0.5",
        )
        p.add_argument("--shift", action="store_true",
                       help="tile: use 7 shifts (centre + 6 neighbours)")
        p.add_argument("--share_edges", action="store_true",
                       help="full-graph tiling: 43 edge-sharing tiles (default 31 disjoint)")
        p.add_argument(
            "--center_only",
            default="",
            help="comma-separated targets that use centre-column-only cost; "
                 "choices: tile,moving_bar (e.g. --center_only tile,moving_bar)",
        )
        p.add_argument("--i_baseline", type=float, default=None,
                       help="tile: PR baseline (pA) before t_on")
        p.add_argument("--i_bright", type=float, default=None,
                       help="tile: PR current (pA) from t_on")
        p.add_argument("--i_baseline_bar", type=float, default=None,
                       help="moving_bar: shared PR baseline (pA) before bar sweep")
        p.add_argument("--i_bright_bar", type=float, default=None,
                       help="moving_bar: PR current (pA) under bright bar")
        p.add_argument("--i_dark_bar", type=float, default=None,
                       help="moving_bar: PR current (pA) under dark bar")
        p.add_argument("--i_baseline_bright_bar", type=float, default=None,
                       help="moving_bar: bright-field baseline (pA); overrides --i_baseline_bar")
        p.add_argument("--i_baseline_dark_bar", type=float, default=None,
                       help="moving_bar: dark-field baseline (pA); overrides --i_baseline_bar")

    add_common(sub.add_parser("local", help="short local CPU run (CUDA disabled)"), 1, 100)
    add_common(sub.add_parser("gpu", help="full training run"), 20, 10000)
    add_common(sub.add_parser(
        "auto",
        help="auto pick CPU/GPU (CPU uses sequential cost by default)",
    ), 1, 10000)
    return parser.parse_args()


def parse_kv(tokens, cast=str):
    out = {}
    for tok in tokens:
        if "=" not in tok:
            raise SystemExit(f"expected NAME=VALUE, got {tok!r}")
        name, val = tok.split("=", 1)
        out[name] = cast(val)
    return out


def main():
    args = parse_args()
    param_modes = parse_kv(args.mode)
    param_fixes = parse_kv(args.fix, float)
    loss_weights = parse_kv(getattr(args, "loss_weight", []) or [], float)
    target_raw = str(args.target).strip()
    target_list = [t.strip() for t in target_raw.split(",") if t.strip()]
    target_single = target_list[0] if len(target_list) == 1 else "tile"
    if target_single not in ("tile", "moving_bar"):
        raise SystemExit(f"unknown --target {target_single!r} (expected tile|moving_bar)")
    if len(target_list) > 1:
        bad = [t for t in target_list if t not in ("tile", "moving_bar")]
        if bad:
            raise SystemExit(f"unknown target(s) in --target: {bad} (expected tile|moving_bar)")
    center_only = [t.strip() for t in str(args.center_only).split(",") if t.strip()]
    bad_center = [t for t in center_only if t not in ("tile", "moving_bar")]
    if bad_center:
        raise SystemExit(f"unknown target(s) in --center_only: {bad_center} (expected tile|moving_bar)")
    moving_bar_center_column = "moving_bar" in center_only
    tile_center_column = "tile" in center_only
    bar_stimulus_opts = fc.bar_stimulus_opts_from_cli(
        i_baseline_bar=args.i_baseline_bar,
        i_bright_bar=args.i_bright_bar,
        i_dark_bar=args.i_dark_bar,
        i_baseline_bright_bar=args.i_baseline_bright_bar,
        i_baseline_dark_bar=args.i_baseline_dark_bar,
    )
    tile_stimulus_opts = fc.tile_stimulus_opts_from_cli(
        i_baseline=args.i_baseline,
        i_bright=args.i_bright,
    )
    outdir = run_dir(args.model_type, parent=args.outdir)
    run_training(args.model_type, args.nofruns, args.nofsteps, args.lrs,
                 fname=args.fname, outdir=outdir,
                 param_modes=param_modes, param_fixes=param_fixes,
                 network=args.network, multi_column=args.shift,
                 share_edges=args.share_edges,
                 target=target_single,
                 target_list=(target_list if len(target_list) > 1 else None), loss_weights=loss_weights,
                 moving_bar_center_column=moving_bar_center_column,
                 tile_center_column=tile_center_column,
                 per_type=args.per_type,
                 bar_stimulus_opts=bar_stimulus_opts,
                 tile_stimulus_opts=tile_stimulus_opts,
                 i_baseline=args.i_baseline,
                 i_bright=args.i_bright)


if __name__ == "__main__":
    main()
