#!/usr/bin/env python
"""Unified training driver for the FiveCol medulla model.

All results of a run land in one folder under
``training_config.PARAMETER_DIR`` (default ``SimulationCode/FiveCol_Parameter``):

    <model_type>/<run_name>/

NumPy artifacts (``.npy`` / ``.npz``) live in ``<run_name>/np/``; PNGs, CSV, and
JSON sidecars stay in ``<run_name>/``.

where <run_name> encodes the CLI, e.g.
``26758480-run-nofsteps-50-target-moving_bar,tile-network-right_min_neuron1_extent2-shift``
(job id under SLURM, else a timestamp prefix).

    # short smoke test
    python run.py --model_type adaptive --nofsteps 30 --lrs 0.1

    # full training
    python run.py --model_type conductance --nofruns 1 --nofsteps 10000 \\
                  --lrs 0.1,0.01,0.001

    # per-target PR currents (comma-separated TARGET=VALUE)
    python run.py --target tile,moving_bar --i_baseline tile=20,moving_bar=22 \\
                  --i_bright tile_bright=45 --i_dark tile_dark=0,moving_bar_dark=2

    # moving-bar (``--network`` = folder under built_network/)
    python run.py --target moving_bar --network right_min_neuron1_extent2 \\
                  --nofsteps 5 --lrs 0.1

Import-safe: importing this module does NOT parse argv or touch CUDA, so test
scripts can `import run` and reuse run_training / save_training_outputs / etc.
"""
import argparse
import json
import os
import time

# When executed as a script, run from this file's own directory so `fc` finds
# Circuits/ regardless of where it was launched (no need to cd first). Done
# before importing fc (Borst paths resolve relative to SimulationCode/). NOT done on
# `import run`, so importers keep control of cwd / CUDA_VISIBLE_DEVICES.
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import network_bootstrap  # noqa: F401 — connectome_io on sys.path
from connectome_io import DEFAULT_NETWORK_RUN, NETWORK_DIR, resolve_network_json
from FiveCol_MedSim_Pytorch import do_many_runs
import FiveCol_MedSim_Pytorch as fc
from plot_trained import command_run_name, plot_param_set, run_dir


def make_plots(fname, outdir, session, result=None, *,
               ref_cubes=None, ref_cubes_off=None, mvd_group_list=None):
    """Cost curve + model-vs-data + all-cell-types."""
    plot_kw = dict(ref_cubes=ref_cubes, ref_cubes_off=ref_cubes_off,
                   mvd_group_list=mvd_group_list)
    if result is not None:
        plot_param_set(
            result.all_params, outdir, session=session,
            final_costs=result.final_costs,
            cost_curve=result.cost_curve,
            costs_by_target=result.cost_curves_by_target,
            best_i=result.best_i,
            save_artifacts=False,
            **plot_kw,
        )
        return
    params = np.load(params_path(outdir, fname))
    final_costs, cost_curve, costs_by_target, _ = load_stored_costs(
        outdir, fname, np.atleast_2d(params).shape[0],
    )
    plot_param_set(
        params, outdir, session=session,
        final_costs=final_costs, cost_curve=cost_curve,
        costs_by_target=costs_by_target,
        save_artifacts=False,
        **plot_kw,
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


NP_SUBDIR = 'np'


def np_dir(outdir):
    return os.path.join(outdir, NP_SUBDIR)


def params_path(outdir, fname):
    return os.path.join(np_dir(outdir), fname)


def best_param_path(outdir):
    return os.path.join(np_dir(outdir), 'best_param.npy')


def _artifact_stem(fname):
    return fname.replace('.npy', '')


def _final_costs_path(outdir, fname):
    return os.path.join(np_dir(outdir), _artifact_stem(fname) + '_final_costs.npy')


def _cost_curve_path(outdir, fname):
    return os.path.join(np_dir(outdir), _artifact_stem(fname) + '_costs.npy')


def _costs_by_target_path(outdir, fname):
    return os.path.join(np_dir(outdir), _artifact_stem(fname) + '_costs_by_target.npz')


def _final_costs_by_target_path(outdir, fname):
    return os.path.join(np_dir(outdir), _artifact_stem(fname) + '_final_costs_by_target.npz')


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
    os.makedirs(np_dir(outdir), exist_ok=True)
    np.save(best_param_path(outdir), best)
    table_path = os.path.join(outdir, _artifact_stem(fname) + '_table.csv')
    z_best = torch.tensor(best, dtype=torch.float64, device=session.device)
    write_param_table(z_best, session, table_path)
    print("wrote table: %s (best run #%d, cost=%.4f)" % (
        table_path, best_i, final_costs[best_i]))
    return best


def load_stored_costs(outdir, fname, n_runs):
    """Load ``*_final_costs.npy``, step ``*_costs.npy``, and per-target npz when present."""
    final_costs = None
    cost_curve = None
    costs_by_target = None
    final_costs_by_target = None
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
    cbt = _costs_by_target_path(outdir, fname)
    if os.path.isfile(cbt):
        with np.load(cbt) as d:
            costs_by_target = {k: np.asarray(d[k]) for k in d.files}
    fbt = _final_costs_by_target_path(outdir, fname)
    if os.path.isfile(fbt):
        with np.load(fbt) as d:
            final_costs_by_target = {k: np.asarray(d[k]) for k in d.files}
    return final_costs, cost_curve, costs_by_target, final_costs_by_target


def save_training_outputs(fname, outdir, session, result):
    """Write the full run artifact set (convention §5)."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'model_type.txt'), 'w') as f:
        f.write(session.model_type)
    if session.train_opts is not None:
        with open(os.path.join(outdir, fc.TRAIN_OPTS_FILE), 'w') as f:
            json.dump(session.train_opts, f, indent=2)
            f.write('\n')
    os.makedirs(np_dir(outdir), exist_ok=True)
    np.save(params_path(outdir, fname), result.all_params)
    np.save(_cost_curve_path(outdir, fname), result.cost_curve)
    np.save(_final_costs_path(outdir, fname), result.final_costs)
    if result.cost_curves_by_target:
        np.savez(_costs_by_target_path(outdir, fname), **result.cost_curves_by_target)
    if result.final_costs_by_target:
        np.savez(_final_costs_by_target_path(outdir, fname), **result.final_costs_by_target)
    write_best_artifacts(
        outdir, fname, session, result.all_params, result.best_i, result.final_costs,
    )


def save_param_tables(fname, outdir, session):
    """Regenerate ``*_table.csv`` and ``best_param.npy`` from saved ``fname`` on disk."""
    all_params = np.load(params_path(outdir, fname))
    final_costs, _, _, _ = load_stored_costs(outdir, fname, np.atleast_2d(all_params).shape[0])
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
    sequential=None,
    target_list=None,
    loss_weights=None,
    center_only_targets=None,
    multi_shift_targets=None,
    share_edges_targets=None,
    i_cli=None,
    per_type=False,
    moving_bar_bright_stimulus_opts=None,
    moving_bar_dark_stimulus_opts=None,
    tile_bright_stimulus_opts=None,
    tile_dark_stimulus_opts=None,
    param_modes=None,
    param_fixes=None,
    ih_off=fc.IH_OFF_DEFAULT,
    pack_overrides=None,
    model_backend=None,
    schema=None,
):
    """Create a :class:`TrainSession` from run options."""
    tl = list(target_list) if target_list is not None else ["tile_bright"]
    dev = fc.active_device()
    mkw = dict(
        target_list=tl,
        loss_weights=loss_weights,
        pack_overrides=pack_overrides,
        sequential=sequential,
        center_only_targets=center_only_targets,
        multi_shift_targets=multi_shift_targets,
        share_edges_targets=share_edges_targets,
        i_cli=i_cli,
        tile_bright_stimulus_opts=tile_bright_stimulus_opts,
        tile_dark_stimulus_opts=tile_dark_stimulus_opts,
        moving_bar_bright_stimulus_opts=moving_bar_bright_stimulus_opts,
        moving_bar_dark_stimulus_opts=moving_bar_dark_stimulus_opts,
    )
    if network:
        network = resolve_network(network)
        mb = model_backend or fc.load_network_backend(network, dev=dev)
        opts = fc.make_train_opts(
            backend="network",
            network=mb.network,
            network_json=network,
            dev=dev,
            ih_off=ih_off,
            **mkw,
        )
        model_backend = mb
    else:
        opts = fc.make_train_opts(backend="borst", per_type=per_type, ih_off=ih_off, **mkw)
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
                 ih_off=fc.IH_OFF_DEFAULT,
                 network=None, sequential=None,
                 target_list=None, loss_weights=None,
                 center_only_targets=None, multi_shift_targets=None,
                 share_edges_targets=None, i_cli=None,
                 per_type=False, moving_bar_bright_stimulus_opts=None,
                 moving_bar_dark_stimulus_opts=None,
                 tile_bright_stimulus_opts=None,
                 tile_dark_stimulus_opts=None,
                 pack_overrides=None, model_backend=None, schema=None,
                 plot_ref_cubes=None, plot_ref_cubes_off=None,
                 plot_mvd_group_list=None):
    """Full training pipeline (do_many_runs + save_training_outputs + plots). Returns (fname, outdir, session)."""
    session = build_session(
        model_type,
        network=network,
        sequential=sequential,
        target_list=target_list,
        loss_weights=loss_weights,
        center_only_targets=center_only_targets,
        multi_shift_targets=multi_shift_targets,
        share_edges_targets=share_edges_targets,
        i_cli=i_cli,
        per_type=per_type,
        moving_bar_bright_stimulus_opts=moving_bar_bright_stimulus_opts,
        moving_bar_dark_stimulus_opts=moving_bar_dark_stimulus_opts,
        tile_bright_stimulus_opts=tile_bright_stimulus_opts,
        tile_dark_stimulus_opts=tile_dark_stimulus_opts,
        param_modes=param_modes,
        param_fixes=param_fixes,
        ih_off=ih_off,
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
    make_plots(
        fname, outdir, session, result=result,
        ref_cubes=plot_ref_cubes, ref_cubes_off=plot_ref_cubes_off,
        mvd_group_list=plot_mvd_group_list,
    )
    return fname, outdir, session


def add_training_arguments(parser):
    """Register run.py training CLI flags on *parser*."""
    parser.add_argument("--model_type", default="conductance",
                        choices=["conductance", "adaptive"])
    parser.add_argument("--nofruns", type=int, default=1)
    parser.add_argument("--nofsteps", type=int, default=50)
    parser.add_argument("--lrs", default="0.1",
                        help="comma-separated learning-rate stages; each runs for --nofsteps steps")
    parser.add_argument("--fname", default=None,
                        help="params filename (default derived from --model_type)")
    parser.add_argument("--outdir", default=None,
                        help="output dir (default derived from --model_type)")
    parser.add_argument("--mode", default="", metavar="NAME=MODE,...",
                        help="per-param mode override, e.g. out_scale=shared,inp_gain=fixed "
                             "(MODE in individual|shared|fixed)")
    parser.add_argument("--fix", default="", metavar="NAME=VALUE,...",
                        help="hold a param fixed at VALUE (implies fixed mode), "
                             "e.g. Ih_midv=-50,out_scale=1.0")
    parser.add_argument("--ih_off", default=fc.IH_OFF_DEFAULT,
                        choices=list(fc.IH_OFF_MODES),
                        help="conductance ON/OFF Ih coupling: split (train Ih_gmax_off+OFF "
                             "scalars; default), shared (OFF uses ON Ih_gmax+shape), "
                             "off (disable OFF channel)")
    parser.add_argument("--per_type", action="store_true",
                        help="train Ih (and adaptive lamina) params per cell type "
                             "instead of shared lamina/scalar values")
    parser.add_argument("--network", default=None, metavar="RUN",
                        help=f"connectome backend: built_network run folder under {NETWORK_DIR} "
                             f"(e.g. {DEFAULT_NETWORK_RUN}); "
                             f"default Borst 5-column simulator if omitted")
    parser.add_argument(
        "--target",
        default="tile,moving_bar",
        help="target name(s): tile (=tile_bright+tile_dark), moving_bar (=bright+dark), "
             "or explicit names / comma-separated list, e.g. tile,moving_bar",
    )
    parser.add_argument(
        "--loss_weight",
        default="",
        metavar="TARGET=VALUE,...",
        help="per-part loss weights, e.g. tile=1,PD=1.5,ND=1.0 "
             "(aliases: tile, moving_bar, moving_bar_bright/dark, PD/ND)",
    )
    parser.add_argument(
        "--shift",
        action="store_true",
        help="enable 7 sub-tile shifts for tile targets in --target",
    )
    parser.add_argument(
        "--share_edges",
        default="",
        help="comma-separated tile targets for edge-sharing tiling "
             "(choices: tile,tile_bright,tile_dark)",
    )
    parser.add_argument(
        "--center_only",
        default="",
        help="comma-separated targets that use centre-column-only cost; "
             "choices: tile,tile_bright,tile_dark,moving_bar,moving_bar_bright,moving_bar_dark",
    )
    parser.add_argument(
        "--i_baseline",
        default="",
        metavar="TARGET=VALUE,...",
        help="per-target PR baseline (pA); aliases: tile, moving_bar",
    )
    parser.add_argument(
        "--i_bright",
        default="",
        metavar="TARGET=VALUE,...",
        help="bright peak/step current (pA); targets: tile_bright, moving_bar_bright "
             "(aliases tile, moving_bar)",
    )
    parser.add_argument(
        "--i_dark",
        default="",
        metavar="TARGET=VALUE,...",
        help="dark peak/step current (pA); targets: tile_dark, moving_bar_dark "
             "(aliases tile, moving_bar)",
    )


def make_training_argparser(description):
    """Argparse parser with the full run.py training CLI."""
    common = argparse.ArgumentParser(add_help=False)
    add_training_arguments(common)
    return argparse.ArgumentParser(
        description=description,
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def parse_comma_list(text):
    """Split a comma-separated token list (empty string → ``[]``)."""
    return [t.strip() for t in str(text or "").split(",") if t.strip()]


def parse_comma_kv(text, cast=str):
    """Parse comma-separated ``NAME=VALUE`` pairs."""
    out = {}
    for tok in parse_comma_list(text):
        if "=" not in tok:
            raise ValueError(f"expected NAME=VALUE, got {tok!r}")
        name, val = tok.split("=", 1)
        out[name.strip()] = cast(val.strip())
    return out


def parse_comma_floats(text):
    """Parse comma-separated floats (e.g. learning rates)."""
    return [float(x) for x in parse_comma_list(text)]


def parse_target_list(text):
    """Parse comma-separated training targets (with alias expansion)."""
    return fc._normalize_target_list(parse_comma_list(text))


def parse_target_names(text):
    """Parse comma-separated target names without alias expansion."""
    return parse_comma_list(text)


def training_kwargs_from_args(
    args,
    *,
    script_stem="run",
):
    """Parse a training CLI namespace into kwargs for :func:`run_training`."""
    param_modes = parse_comma_kv(args.mode)
    param_fixes = parse_comma_kv(args.fix, float)
    loss_weights = fc.expand_loss_weights(parse_comma_kv(args.loss_weight, float))
    target_list = parse_target_list(args.target)
    center_only_targets = fc.expand_target_aliases(parse_target_names(args.center_only))
    multi_shift_targets = (
        [t for t in target_list if t in fc.TILE_TARGETS] if args.shift else []
    )
    share_edges_targets = fc.expand_target_aliases(parse_target_names(args.share_edges))
    i_cli = fc.build_i_cli_by_target({
        "i_baseline": parse_comma_kv(args.i_baseline, float),
        "i_bright": parse_comma_kv(args.i_bright, float),
        "i_dark": parse_comma_kv(args.i_dark, float),
    })
    lrs = parse_comma_floats(args.lrs)
    if not lrs:
        raise ValueError("--lrs must list at least one learning rate")
    bad_center = [t for t in center_only_targets if t not in fc.VALID_TARGETS]
    if bad_center:
        raise ValueError(
            f"unknown target(s) in --center_only: {bad_center} "
            f"(expected {'|'.join(fc.CLI_TARGET_NAMES)})",
        )
    bad_share = [t for t in share_edges_targets if t not in fc.TILE_TARGETS]
    if bad_share:
        raise ValueError(
            f"unknown target(s) in --share_edges: {bad_share} "
            f"(expected {'|'.join(fc.TILE_TARGETS + ('tile',))})",
        )
    run_name = command_run_name(script_stem)
    outdir = run_dir(args.model_type, parent=args.outdir, name=run_name)
    return dict(
        model_type=args.model_type,
        nofruns=args.nofruns,
        nofsteps=args.nofsteps,
        lrs=lrs,
        fname=args.fname,
        outdir=outdir,
        param_modes=param_modes,
        param_fixes=param_fixes,
        network=args.network,
        target_list=target_list,
        loss_weights=loss_weights,
        center_only_targets=center_only_targets,
        multi_shift_targets=multi_shift_targets,
        share_edges_targets=share_edges_targets,
        i_cli=i_cli,
        per_type=args.per_type,
        ih_off=args.ih_off,
    )


def main():
    parser = make_training_argparser(__doc__)
    args = parser.parse_args()
    try:
        kw = training_kwargs_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_training(**kw)


if __name__ == "__main__":
    main()
