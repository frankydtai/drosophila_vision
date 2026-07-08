#!/usr/bin/env python
"""Unified training driver for the FiveCol medulla model.

All results of a run land in one folder under
``training_config.PARAMETER_DIR`` (default ``SimulationCode/FiveCol_Parameter``):

    <model>/<run_name>/

Run artifacts (``.npy`` / ``.npz``, ``train_opts.json``) live in
``<run_name>/data/``; PNGs and ``*_table.csv`` stay in ``<run_name>/``.

where <run_name> encodes the CLI, e.g.
``26758480-run-nofsteps-50-target-moving_bar,spot-network-right_min_neuron1_extent2-shift``
(job id under SLURM, else a timestamp prefix).

    # short smoke test
    python train.py --model adaptive --nofsteps 30 --lrs 0.1

    # full training
    python train.py --model conductance --nofruns 1 --nofsteps 10000 \\
                  --lrs 0.1,0.01,0.001

    # per-target PR currents (comma-separated TARGET=VALUE)
    python train.py --target spot,moving_bar --i-baseline spot=20,moving_bar=22 \\
                  --i-bright spot_bright=45 --i-dark spot_dark=0,moving_bar_dark=2

    # moving-bar (``--network`` = folder under built_network/)
    python train.py --target moving_bar --network right_min_neuron1_extent2 \\
                  --nofsteps 5 --lrs 0.1

    # warm-start from a previous run (params only; all other settings from this CLI)
    python train.py --init-from run_26693975 \\
                  --network right_min_neuron1_extent2 --target moving_bar_bright \\
                  --nofsteps 10000 --lrs 0.1,0.01,0.001

Import-safe: importing this module does NOT parse argv or touch CUDA, so test
scripts can `import train` and reuse run_training / save_training_outputs / etc.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import network_bootstrap  # noqa: F401 — connectome_io on sys.path
from connectome_io import DEFAULT_NETWORK_RUN, NETWORK_DIR, resolve_network_json
from FiveCol_MedSim_Pytorch import do_many_runs
import FiveCol_MedSim_Pytorch as fc
from plot_trained import (
    add_plot_arguments,
    command_run_name,
    plot_kwargs_from_args,
    plot_param_set,
    resolve_run_dir,
    run_dir,
)
from training_config import BORST_CTYPE_NPY, run_data_dir


def make_plots(fname, outdir, session, result=None, *,
               ref_cubes=None, ref_cubes_off=None, mvd_group_list=None,
               plot_right_only=True, at_x=None, at_y=None, plot_vm=False):
    """Cost curve + model-vs-data + all-cell-types."""
    plot_kw = dict(
        ref_cubes=ref_cubes, ref_cubes_off=ref_cubes_off,
        mvd_group_list=mvd_group_list, plot_right_only=plot_right_only,
        at_x=at_x, at_y=at_y,
        plot_vm=plot_vm,
    )
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
    return np.load(BORST_CTYPE_NPY, allow_pickle=True)


def decompose_params(z_t, session):
    """Return (per_cell_cols, global_scalars) for one parameter vector."""
    n = session.backend.n_types
    schema = list(session.schema)
    p = fc.assign_params(z_t, schema, session.backend)
    cols, glob = {}, {}
    for seg in schema:
        name, v = seg["name"], p[seg["name"]]
        mode = fc.seg_mode(seg)
        if seg["kind"] == "output":
            arr = v.detach().cpu().numpy()
        else:
            arr = v[:n].detach().cpu().numpy()
        if mode in ("shared", "fixed"):
            glob[name] = float(arr.mean())
        else:
            cols[name] = arr
    if session.model == "adaptive":
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


def data_dir(outdir):
    return run_data_dir(outdir)


def params_path(outdir, fname):
    return os.path.join(data_dir(outdir), fname)


def best_param_path(outdir):
    return os.path.join(data_dir(outdir), 'best_param.npy')


def best_i_path(outdir):
    return os.path.join(data_dir(outdir), 'best_i.txt')


def write_best_i(outdir, best_i):
    os.makedirs(data_dir(outdir), exist_ok=True)
    with open(best_i_path(outdir), 'w') as f:
        f.write(f'{int(best_i)}\n')


def load_best_i(outdir):
    fp = best_i_path(outdir)
    if os.path.isfile(fp):
        return int(Path(fp).read_text().strip())
    return None


def _artifact_stem(fname):
    return fname.replace('.npy', '')


def _final_costs_path(outdir, fname):
    return os.path.join(data_dir(outdir), _artifact_stem(fname) + '_final_costs.npy')


def _cost_curve_path(outdir, fname):
    return os.path.join(data_dir(outdir), _artifact_stem(fname) + '_costs.npy')


def _costs_by_target_path(outdir, fname):
    return os.path.join(data_dir(outdir), _artifact_stem(fname) + '_costs_by_target.npz')


def _final_costs_by_target_path(outdir, fname):
    return os.path.join(data_dir(outdir), _artifact_stem(fname) + '_final_costs_by_target.npz')


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
    os.makedirs(data_dir(outdir), exist_ok=True)
    np.save(best_param_path(outdir), best)
    write_best_i(outdir, best_i)
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


def load_init_z(init_from, session):
    """Load a parameter vector from a prior run folder (no train_opts replay)."""
    try:
        outdir = resolve_run_dir(init_from)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc
    n_expected = fc.schema_nparams(list(session.schema))

    best_fp = best_param_path(outdir)
    if os.path.isfile(best_fp):
        z = np.load(best_fp)
        source = best_fp
    else:
        tables = sorted(Path(outdir).glob('training*_table.csv'))
        if len(tables) != 1:
            raise ValueError(
                f'expected exactly one training*_table.csv in {outdir!r}, found {len(tables)}',
            )
        fname = tables[0].name.replace('_table.csv', '') + '.npy'
        params_fp = params_path(outdir, fname)
        if not os.path.isfile(params_fp):
            raise ValueError(f'missing training params: {params_fp!r}')
        params = np.atleast_2d(np.load(params_fp))
        final_costs, _, _, _ = load_stored_costs(outdir, fname, params.shape[0])
        if final_costs is not None:
            z = params[int(np.argmin(final_costs))]
        else:
            valid = params[np.any(params != 0, axis=1)]
            if len(valid) == 0:
                raise ValueError(f'no trained parameter sets in {params_fp!r}')
            z = valid[0]
        source = params_fp

    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.shape != (n_expected,):
        raise ValueError(
            f'init params length {z.shape[0]} != session nparams {n_expected} '
            f'(from {source!r}); align schema CLI: --network, --ih-off, --ih-group, --mode',
        )
    print(f'init-from {outdir!r} -> {source!r} ({z.shape[0]} params)')
    return torch.tensor(z, dtype=session.sim_dtype, device=session.device)


def save_training_outputs(fname, outdir, session, result):
    """Write the full run artifact set (convention §5)."""
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(data_dir(outdir), exist_ok=True)
    if session.train_opts is not None:
        with open(os.path.join(data_dir(outdir), fc.TRAIN_OPTS_FILE), 'w') as f:
            json.dump(session.train_opts, f, indent=2)
            f.write('\n')
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


def apply_param_modes(session, param_modes=None):
    modes = fc.expand_mode_dict(param_modes, list(session.schema))
    schema = fc.apply_modes(list(session.schema), modes)
    session = session.with_schema(schema)
    summary = ", ".join(f"{s['name']}:{fc.seg_mode(s)}({fc.seg_ntrain(s)})" for s in schema)
    print("param modes -> " + summary)
    return session


def resolve_network(network):
    return str(resolve_network_json(network))


def build_session(
    model,
    *,
    network=None,
    sequential=False,
    target_list=None,
    cost_weights=None,
    cost_extent_by_target=None,
    shift_extent=0,
    spot_cost_radius_weight=None,
    i_cli=None,
    ih_group_names=None,
    moving_bar_bright_stimulus_opts=None,
    moving_bar_dark_stimulus_opts=None,
    spot_bright_stimulus_opts=None,
    spot_dark_stimulus_opts=None,
    param_modes=None,
    ih_off=fc.IH_OFF_DEFAULT,
    fp32=False,
    pack_overrides=None,
    model_backend=None,
    schema=None,
):
    """Create a :class:`TrainSession` from run options."""
    tl = list(target_list) if target_list is not None else ["spot_bright"]
    dev = fc.active_device()
    mkw = dict(
        target_list=tl,
        cost_weights=cost_weights,
        pack_overrides=pack_overrides,
        sequential=sequential,
        cost_extent_by_target=cost_extent_by_target,
        shift_extent=shift_extent,
        spot_cost_radius_weight=spot_cost_radius_weight,
        i_cli=i_cli,
        spot_bright_stimulus_opts=spot_bright_stimulus_opts,
        spot_dark_stimulus_opts=spot_dark_stimulus_opts,
        moving_bar_bright_stimulus_opts=moving_bar_bright_stimulus_opts,
        moving_bar_dark_stimulus_opts=moving_bar_dark_stimulus_opts,
    )
    if network:
        network = resolve_network(network)
        opts = fc.make_train_opts(
            backend="network",
            network_json=network,
            dev=dev,
            ih_off=ih_off,
            ih_group_names=ih_group_names,
            fp32=fp32,
            **mkw,
        )
    else:
        opts = fc.make_train_opts(
            backend="borst", ih_off=ih_off, ih_group_names=ih_group_names, fp32=fp32, **mkw,
        )
    session = fc.open_session(opts, model, schema=schema, model_backend=model_backend)
    if param_modes:
        session = apply_param_modes(session, param_modes)
    return session


def run_training(model, nofruns, nofsteps, lrs, fname=None, outdir=None,
                 param_modes=None,
                 ih_off=fc.IH_OFF_DEFAULT,
                 network=None, sequential=False,
                 target_list=None, cost_weights=None,
                 cost_extent_by_target=None, shift_extent=0,
                 spot_cost_radius_weight=None,
                 i_cli=None,
                 ih_group_names=None,
                 moving_bar_bright_stimulus_opts=None,
                 moving_bar_dark_stimulus_opts=None,
                 spot_bright_stimulus_opts=None,
                 spot_dark_stimulus_opts=None,
                 pack_overrides=None, model_backend=None, schema=None,
                 fp32=False,
                 plot_ref_cubes=None, plot_ref_cubes_off=None,
                 plot_mvd_group_list=None, plot_right_only=True,
                 at_x=None, at_y=None,
                 plot_vm=False,
                 init_from=None):
    """Full training pipeline (do_many_runs + save_training_outputs + plots). Returns (fname, outdir, session)."""
    session = build_session(
        model,
        network=network,
        sequential=sequential,
        target_list=target_list,
        cost_weights=cost_weights,
        cost_extent_by_target=cost_extent_by_target,
        shift_extent=shift_extent,
        spot_cost_radius_weight=spot_cost_radius_weight,
        i_cli=i_cli,
        ih_group_names=ih_group_names,
        moving_bar_bright_stimulus_opts=moving_bar_bright_stimulus_opts,
        moving_bar_dark_stimulus_opts=moving_bar_dark_stimulus_opts,
        spot_bright_stimulus_opts=spot_bright_stimulus_opts,
        spot_dark_stimulus_opts=spot_dark_stimulus_opts,
        param_modes=param_modes,
        ih_off=ih_off,
        pack_overrides=pack_overrides,
        model_backend=model_backend,
        schema=schema,
        fp32=fp32,
    )
    suffix = "" if model == "conductance" else f"_{model}"
    fname = fname or f"training{suffix or '_with_Ih'}.npy"
    outdir = outdir or run_dir(model)

    print(f"device={session.device}, model={model}, nofruns={nofruns}, nofsteps={nofsteps}, "
          f"lrs={lrs}, nparams={fc.schema_nparams(list(session.schema))}, fname={fname}, outdir={outdir}")
    z_init = load_init_z(init_from, session) if init_from else None
    t0 = time.time()
    result = do_many_runs(session, nofruns, nofsteps, lrs=lrs, z_init=z_init)
    print(f"done in {(time.time() - t0) / 3600:.2f} hours")

    save_training_outputs(fname, outdir, session, result)
    make_plots(
        fname, outdir, session, result=result,
        ref_cubes=plot_ref_cubes, ref_cubes_off=plot_ref_cubes_off,
        mvd_group_list=plot_mvd_group_list,
        plot_right_only=plot_right_only, at_x=at_x, at_y=at_y,
        plot_vm=plot_vm,
    )
    return fname, outdir, session


def add_training_arguments(parser):
    """Register train.py training CLI flags on *parser*."""
    parser.add_argument("--model", default="conductance",
                        choices=["conductance", "adaptive"])
    parser.add_argument("--nofruns", type=int, default=1)
    parser.add_argument("--nofsteps", type=int, default=50)
    parser.add_argument("--lrs", default="0.1",
                        help="comma-separated learning-rate stages; each runs for --nofsteps steps")
    parser.add_argument("--fname", default=None,
                        help="params filename (default derived from --model)")
    parser.add_argument("--outdir", default=None,
                        help="output dir (default derived from --model)")
    parser.add_argument("--init-from", default=None, metavar="RUN",
                        help="prior run folder NAME only (no model/ prefix); "
                             "resolved under FiveCol_Parameter/<model>/NAME unless an absolute path is given; "
                             "load params as z init only (settings come from this CLI, not train_opts.json)")
    parser.add_argument("--mode", default="", metavar="NAME=MODE,...",
                        help="per-param mode override, e.g. out_scale=shared,ih=indi "
                             "(MODE in indi|shared|fixed; ih expands to 6 shape params)")
    parser.add_argument("--ih-group", default=",".join(fc.DEFAULT_IH_GROUP_NAMES),
                        help="comma-separated cell types receiving Ih_gmax/Ih_gmax_off "
                             "(one trainable slot per name when mode=indi); "
                             "or 'all' for every cell type on the backend")
    parser.add_argument("--ih-off", default=fc.IH_OFF_DEFAULT,
                        choices=list(fc.IH_OFF_MODES),
                        help="OFF-channel Ih: on (train Ih_gmax_off+OFF shape; default), "
                             "mirrored (OFF copies ON; shared/indi from --mode), "
                             "off (disable OFF channel)")
    parser.add_argument("--fp32", action="store_true",
                        help="run simulation in float32 (default float64)")
    parser.add_argument("--sequential", action="store_true",
                        help="one stimulus batch per forward (default: batched forward)")
    parser.add_argument("--network", default=None, metavar="RUN",
                        help=f"connectome backend: built_network run folder under {NETWORK_DIR} "
                             f"(e.g. {DEFAULT_NETWORK_RUN}); "
                             f"default Borst 5-column simulator if omitted")
    parser.add_argument(
        "--target",
        default="spot,moving_bar",
        help="target name(s): spot (=spot_bright+spot_dark), moving_bar (=bright+dark), "
             "or explicit names / comma-separated list, e.g. spot,moving_bar",
    )
    parser.add_argument(
        "--cost-weight",
        default="",
        metavar="TARGET=VALUE,...",
        help="per-part cost weights, e.g. spot=1,PD=1.5,ND=1.0 "
             "(aliases: spot, moving_bar, moving_bar_bright/dark, PD/ND)",
    )
    parser.add_argument(
        "--shift-extent",
        type=int,
        default=0,
        help="spot sub-shift hex-disc radius for spot targets in --target "
             "(n_shifts=1+3k(k+1); 0->1, 1->7, 2->19, 3->37, ...)",
    )
    parser.add_argument(
        "--spot-cost-r-w",
        default="",
        metavar="R=W,...",
        help="spot cost weights by Euclidean r from stim column (r=w); "
             "default 0=1,1=1/6,2=1/6; omitted radii → 0 (excluded); "
             "e.g. 0=1,1=1/6,2=1/12",
    )
    parser.add_argument(
        "--cost-extent",
        default="",
        metavar="N|TARGET=N,...",
        help="network cost hex-disc radius (default -1 = all columns with --network): "
             "bare N for all --target, or per-target e.g. moving_bar_bright=0 "
             "(aliases: spot, moving_bar); -1 = all columns; requires --network",
    )
    parser.add_argument(
        "--i-baseline",
        default="",
        metavar="TARGET=VALUE,...",
        help="per-target PR baseline (pA); aliases: spot, moving_bar",
    )
    parser.add_argument(
        "--i-bright",
        default="",
        metavar="TARGET=VALUE,...",
        help="bright peak/step current (pA); targets: spot_bright, moving_bar_bright "
             "(aliases spot, moving_bar)",
    )
    parser.add_argument(
        "--i-dark",
        default="",
        metavar="TARGET=VALUE,...",
        help="dark peak/step current (pA); targets: spot_dark, moving_bar_dark "
             "(aliases spot, moving_bar)",
    )
    add_plot_arguments(parser)


def make_training_argparser(description):
    """Argparse parser with the full train.py training CLI."""
    common = argparse.ArgumentParser(add_help=False)
    add_training_arguments(common)
    return argparse.ArgumentParser(
        description=description,
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def parse_bool(text):
    """Parse CLI boolean (true/false, 1/0, yes/no)."""
    v = str(text).lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError(f"expected true|false, got {text!r}")


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


def parse_cost_extent(text):
    """Parse ``--cost-extent``: optional bare ``N`` plus ``TARGET=N`` pairs."""
    default = None
    by_target = {}
    for tok in parse_comma_list(text):
        if "=" in tok:
            name, val = tok.split("=", 1)
            by_target[name.strip()] = int(val.strip())
        else:
            if default is not None:
                raise ValueError("only one bare extent allowed in --cost-extent")
            default = int(tok)
    return default, by_target


def parse_spot_cost_r_w(text):
    """Parse ``--spot-cost-r-w`` (r=w); empty → default ``0=1,1=1/6,2=1/6``."""
    from network.spot_target import (
        expand_spot_cost_r_w_dict,
        parse_spot_cost_radius_weight_value,
    )

    if not str(text or "").strip():
        return None
    return expand_spot_cost_r_w_dict(
        parse_comma_kv(text, cast=parse_spot_cost_radius_weight_value),
    )


def training_kwargs_from_args(
    args,
    *,
    script_stem="train",
):
    """Parse a training CLI namespace into kwargs for :func:`run_training`."""
    init_from = args.init_from
    if init_from:
        p = Path(str(init_from)).expanduser()
        if not p.is_absolute():
            # HARD STOP: no backward-compat. --init-from now takes a run folder name only.
            if "/" in str(init_from) or "\\" in str(init_from):
                raise ValueError(
                    "--init-from must be a run folder name only (no path); "
                    "the model subfolder is inferred from --model (default: conductance). "
                    "Use an absolute path to reference runs outside FiveCol_Parameter.",
                )
            init_from = f"{args.model}/{init_from}"
    param_modes = parse_comma_kv(args.mode)
    cost_weights = fc.expand_cost_weight_dict(parse_comma_kv(args.cost_weight, float))
    target_list = parse_target_list(args.target)
    default_extent, extent_kv = parse_cost_extent(args.cost_extent)
    if (
        default_extent is None
        and not extent_kv
        and args.network is not None
    ):
        default_extent = -1
    cost_extent_by_target = fc.resolve_cost_extent_by_target(
        target_list, default_extent, extent_kv,
    )
    if cost_extent_by_target and args.network is None:
        raise ValueError("--cost-extent requires --network")
    if default_extent is not None and default_extent != -1 and default_extent < 0:
        raise ValueError("--cost-extent must be -1 or >= 0")
    if any(v != -1 and v < 0 for v in extent_kv.values()):
        raise ValueError("--cost-extent must be -1 or >= 0")
    if args.shift_extent < 0:
        raise ValueError("--shift-extent must be >= 0")
    spot_cost_radius_weight = parse_spot_cost_r_w(args.spot_cost_r_w)
    i_cli = fc.build_i_cli_by_target({
        "i_baseline": parse_comma_kv(args.i_baseline, float),
        "i_bright": parse_comma_kv(args.i_bright, float),
        "i_dark": parse_comma_kv(args.i_dark, float),
    })
    lrs = parse_comma_floats(args.lrs)
    if not lrs:
        raise ValueError("--lrs must list at least one learning rate")
    run_name = command_run_name(script_stem)
    outdir = run_dir(args.model, parent=args.outdir, name=run_name)
    return dict(
        model=args.model,
        nofruns=args.nofruns,
        nofsteps=args.nofsteps,
        lrs=lrs,
        fname=args.fname,
        outdir=outdir,
        param_modes=param_modes,
        network=args.network,
        target_list=target_list,
        cost_weights=cost_weights,
        cost_extent_by_target=cost_extent_by_target,
        shift_extent=int(args.shift_extent),
        spot_cost_radius_weight=spot_cost_radius_weight,
        i_cli=i_cli,
        ih_group_names=parse_comma_list(args.ih_group),
        ih_off=args.ih_off,
        fp32=args.fp32,
        sequential=args.sequential,
        init_from=init_from,
        **plot_kwargs_from_args(args),
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
