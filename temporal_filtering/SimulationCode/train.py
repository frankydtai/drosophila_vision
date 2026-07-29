"""Unified training driver for the FiveCol medulla model.

All results of a run land in one folder under
``training_config.PARAMETER_DIR`` (default ``SimulationCode/0trained``):

    <model>/<run_name>/

Run artifacts (``.npy`` / ``.npz``, ``train_opts.json``) live in
``<run_name>/data/``; PNGs, ``param.csv``, and ``syn_strength.csv`` stay in
``<run_name>/``.

where <run_name> encodes the CLI, e.g.
``26758480-run-nofsteps-50-target-moving_bar,spot-network-right_min_neuron1_extent2-shift``
(job id under SLURM, else a timestamp prefix).

    # short smoke test
    python train.py --model hp_lp --nofsteps 30 --lrs 0.1

    # full training
    python train.py --model conductance --nofruns 1 --nofsteps 10000 \\
                  --lrs 0.1,0.01,0.001

    # per-target PR currents (comma-separated TARGET=VALUE)
    python train.py --target spot,moving_bar --i-baseline spot=20,moving_bar=22 \\
                  --i-bright spot_bright=45 --i-dark spot_dark=0,moving_bar_dark=2

    # moving-bar (``--network`` = folder under built_network/)
    python train.py --target moving_bar --network right_min_neuron1_extent2 \\
                  --nofsteps 5 --lrs 0.1

    # warm-start from a previous run (named best_param.npz; settings from this CLI)
    python train.py --from run_26693975 \\
                  --network right_min_neuron1_extent2 --target moving_bar_bright \\
                  --nofsteps 10000 --lrs 0.1,0.01,0.001

    # freeze all syn_strength; only L1,L2,L4,L5 train Ih_gmax
    python train.py --syn-strength frozen=all \\
                  --ih-gmax indi=L1,L2,L4,L5 fixed=all --ih-shape shared=all

    # same partition on every parameter (per-param flags override)
    python train.py --all-param indi=all --syn-strength frozen=all

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
from connectome_io import (
    DEFAULT_NETWORK_RUN,
    NETWORK_DIR,
    parse_comma_list,
    resolve_network_json,
)
from network.spot_target import DEFAULT_SHIFT_EXTENT
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
from param_defaults import DEFAULT_IH_GMAX_INDI_NAMES
from training_config import (
    PARAM_CSV,
    SYN_STRENGTH_CSV,
    run_data_dir,
)


DEFAULT_NOFSTEPS_CPU = 50
DEFAULT_NOFSTEPS_GPU = 200


def make_plots(fname, outdir, session, result=None, *,
               ref_cubes=None, ref_cubes_2=None,
               plot_right_only=True, at_x=None, at_y=None,
               align_at_x=None, align_at_y=None,
               plot_vm=False, show_pre=True):
    """Cost curve + model-vs-data + all-cell-types."""
    plot_kw = dict(
        ref_cubes=ref_cubes, ref_cubes_2=ref_cubes_2,
        plot_right_only=plot_right_only,
        at_x=at_x, at_y=at_y,
        align_at_x=align_at_x, align_at_y=align_at_y,
        plot_vm=plot_vm, show_pre=show_pre,
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
    if session.backend.network is None:
        raise ValueError("ctype_labels requires session.backend.network")
    return np.asarray(session.backend.network.type_names)



def decompose_params(z_t, session):
    """Return (per_cell_cols, global_scalars) for one parameter vector.

    Per-type columns come from full-width unit values (partition-aware).
    Shared-only segments also emit a global scalar (mean of shared units).
    """
    n = session.backend.n_types
    schema = list(session.schema)
    unit_vals = fc.z_to_unit_values(z_t, schema)
    cols, glob = {}, {}
    for seg in schema:
        name = seg["name"]
        if seg["kind"] == "edge_pair":
            continue
        arr = np.asarray(unit_vals[name], dtype=np.float64).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"{name}: unit width {arr.shape[0]} != n_types {n}")
        cols[name] = arr
        if seg.get("shared") and not seg.get("indi"):
            glob[name] = float(arr[list(seg["shared"])].mean()) if seg["shared"] else float(arr.mean())
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


def write_syn_strength_table(z_t, session, table_path):
    """Write edge-pair ``syn_strength`` as source×target matrix CSV.

    Rows = source types, columns = target types. Absent connectome pairs are blank.
    """
    schema = list(session.schema)
    seg = next((s for s in schema if s["name"] == "syn_strength"), None)
    if seg is None or seg["kind"] != "edge_pair":
        return None
    unit_vals = fc.z_to_unit_values(z_t, schema)
    arr = np.asarray(unit_vals["syn_strength"], dtype=np.float64).reshape(-1)
    names = [str(n) for n in ctype_labels(session)]
    keys = list(session.backend.conn.pair_keys)
    if arr.shape[0] != len(keys):
        raise ValueError(
            f"syn_strength length {arr.shape[0]} != n_pairs {len(keys)}"
        )
    mat = {(int(s), int(t)): float(v) for (s, t), v in zip(keys, arr)}
    n = len(names)
    with open(table_path, "w") as f:
        f.write("," + ",".join(names) + "\n")
        for i, src in enumerate(names):
            cells = [
                ("%.6f" % mat[(i, j)]) if (i, j) in mat else ""
                for j in range(n)
            ]
            f.write(src + "," + ",".join(cells) + "\n")
    return table_path


def data_dir(outdir):
    return run_data_dir(outdir)


def params_path(outdir, fname):
    return os.path.join(data_dir(outdir), fname)


def best_param_path(outdir):
    return os.path.join(data_dir(outdir), 'best_param.npz')


def save_best_param_named(outdir, z, session):
    """Write named full-width unit values to ``data/best_param.npz``."""
    schema = list(session.schema)
    named = fc.z_to_unit_values(z, schema)
    type_names = np.asarray(fc.type_unit_names(session.backend), dtype=object)
    payload = {k: np.asarray(v, dtype=np.float64) for k, v in named.items()}
    payload['type_names'] = type_names
    if any(s['kind'] == 'edge_pair' for s in schema):
        payload['pair_names'] = np.asarray(fc.pair_unit_names(session.backend), dtype=object)
    os.makedirs(data_dir(outdir), exist_ok=True)
    np.savez(best_param_path(outdir), **payload)


def load_best_param_named(outdir):
    """Load ``data/best_param.npz`` → (named dict, type_names, pair_names|None)."""
    fp = best_param_path(outdir)
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)
    with np.load(fp, allow_pickle=True) as d:
        type_names = [str(x) for x in d['type_names'].tolist()]
        pair_names = None
        if 'pair_names' in d.files:
            pair_names = [str(x) for x in d['pair_names'].tolist()]
        named = {
            k: np.asarray(d[k], dtype=np.float64)
            for k in d.files
            if k not in ('type_names', 'pair_names')
        }
    return named, type_names, pair_names


def load_best_param(outdir, session=None):
    """Load best params as 1-D z for *session* (remap from named npz)."""
    if session is None:
        raise TypeError("load_best_param requires session for named best_param.npz")
    named, type_names, pair_names = load_best_param_named(outdir)
    schema = fc.attach_param_carry(
        list(session.schema),
        fc.remap_named_unit_values(
            named, type_names, pair_names, list(session.schema), session.backend,
        ),
    )
    remapped = fc.remap_named_unit_values(
        named, type_names, pair_names, schema, session.backend,
    )
    z = fc.unit_values_to_z(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )
    return z.detach().cpu().numpy().astype(np.float64)


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
    """Write ``best_param.npz``, ``param.csv``, and ``syn_strength.csv`` for one best index."""
    all_params = np.atleast_2d(all_params)
    best = all_params[best_i]
    os.makedirs(data_dir(outdir), exist_ok=True)
    z_best = torch.tensor(best, dtype=session.sim_dtype, device=session.device)
    save_best_param_named(outdir, z_best, session)
    write_best_i(outdir, best_i)
    table_path = os.path.join(outdir, PARAM_CSV)
    write_param_table(z_best, session, table_path)
    print("wrote table: %s (best run #%d, cost=%.4f)" % (
        table_path, best_i, final_costs[best_i]))
    syn_path = write_syn_strength_table(
        z_best, session, os.path.join(outdir, SYN_STRENGTH_CSV),
    )
    if syn_path is not None:
        print("wrote table: %s" % syn_path)
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
    """Load named best params; return ``(session, z)`` with frozen carry attached."""
    try:
        outdir = resolve_run_dir(init_from)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc
    named, type_names, pair_names = load_best_param_named(outdir)
    schema = list(session.schema)
    remapped = fc.remap_named_unit_values(
        named, type_names, pair_names, schema, session.backend,
    )
    schema = fc.attach_param_carry(schema, remapped)
    session = session.with_schema(schema)
    z = fc.unit_values_to_z(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )
    print(
        f'from {outdir!r} -> {best_param_path(outdir)!r} '
        f'({fc.schema_nparams(schema)} trainable slots)'
    )
    return session, z


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
    """Regenerate ``param.csv`` / ``syn_strength.csv`` and ``best_param.npz`` from saved ``fname``."""
    all_params = np.load(params_path(outdir, fname))
    final_costs, _, _, _ = load_stored_costs(outdir, fname, np.atleast_2d(all_params).shape[0])
    final_costs, best_i = final_costs_for_params(all_params, session, final_costs=final_costs)
    write_best_artifacts(outdir, fname, session, all_params, best_i, final_costs)


def print_param_partitions(session):
    """Print one schema segment per line: indi/shared/fixed/frozen counts and ntrain."""
    schema = list(session.schema)
    if not schema:
        return
    print("param partitions:")
    w = max(len(s["name"]) for s in schema)
    for s in schema:
        print(
            f"  {s['name']:<{w}}  "
            f"indi={len(s.get('indi') or [])}/"
            f"shared={len(s.get('shared') or [])}/"
            f"fixed={len(s.get('fixed') or [])}/"
            f"frozen={len(s.get('frozen') or [])} "
            f"({fc.seg_ntrain(s)})"
        )


def apply_param_partitions(session, partitions_by_name):
    """Apply CLI/name partitions onto session schema and refresh train_opts record."""
    if not partitions_by_name:
        return session
    from dataclasses import replace
    backend = session.backend
    schema = fc.apply_partitions(
        list(session.schema),
        partitions_by_name,
        lambda seg: fc.unit_names_for_segment(seg, backend),
    )
    if session.model == 'conductance':
        ih_off = (session.train_opts or {}).get('ih_off', fc.IH_OFF_DEFAULT)
        schema = fc.apply_ih_off_mode(schema, ih_off)
    schema = fc.attach_param_carry(schema)
    opts = dict(session.train_opts or {})
    opts['param_partitions'] = fc.schema_partitions_record(
        schema, lambda seg: fc.unit_names_for_segment(seg, backend),
    )
    session = replace(session, schema=tuple(schema), train_opts=opts)
    print_param_partitions(session)
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
    shift_extent=DEFAULT_SHIFT_EXTENT,
    spot_extent=None,
    multi_spot=True,
    fully_inside=True,
    spot_cost_radius_weight=None,
    i_cli=None,
    moving_bar_bright_stimulus_opts=None,
    moving_bar_dark_stimulus_opts=None,
    spot_bright_stimulus_opts=None,
    spot_dark_stimulus_opts=None,
    param_partitions=None,
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
        spot_extent=spot_extent,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_weight=spot_cost_radius_weight,
        i_cli=i_cli,
        spot_bright_stimulus_opts=spot_bright_stimulus_opts,
        spot_dark_stimulus_opts=spot_dark_stimulus_opts,
        moving_bar_bright_stimulus_opts=moving_bar_bright_stimulus_opts,
        moving_bar_dark_stimulus_opts=moving_bar_dark_stimulus_opts,
    )
    if not network:
        raise ValueError("build_session requires network")
    network = resolve_network(network)
    opts = fc.make_train_opts(
        backend="network",
        network_json=network,
        dev=dev,
        ih_off=ih_off,
        param_partitions=param_partitions,
        fp32=fp32,
        **mkw,
    )
    return fc.open_session(opts, model, schema=schema, model_backend=model_backend)


def run_training(model, nofruns, nofsteps, lrs, fname=None, outdir=None,
                 param_partitions=None,
                 ih_off=fc.IH_OFF_DEFAULT,
                 network=None, sequential=False,
                 target_list=None, cost_weights=None,
                 cost_extent_by_target=None, shift_extent=DEFAULT_SHIFT_EXTENT,
                 spot_extent=None,
                 multi_spot=True,
                 fully_inside=True,
                 spot_cost_radius_weight=None,
                 i_cli=None,
                 moving_bar_bright_stimulus_opts=None,
                 moving_bar_dark_stimulus_opts=None,
                 spot_bright_stimulus_opts=None,
                 spot_dark_stimulus_opts=None,
                 pack_overrides=None, model_backend=None, schema=None,
                 fp32=False,
                 plot_ref_cubes=None, plot_ref_cubes_2=None,
                 plot_right_only=True,
                 at_x=None, at_y=None,
                 align_at_x=None, align_at_y=None,
                 plot_vm=False, show_pre=True,
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
        spot_extent=spot_extent,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_weight=spot_cost_radius_weight,
        i_cli=i_cli,
        moving_bar_bright_stimulus_opts=moving_bar_bright_stimulus_opts,
        moving_bar_dark_stimulus_opts=moving_bar_dark_stimulus_opts,
        spot_bright_stimulus_opts=spot_bright_stimulus_opts,
        spot_dark_stimulus_opts=spot_dark_stimulus_opts,
        param_partitions=param_partitions,
        ih_off=ih_off,
        pack_overrides=pack_overrides,
        model_backend=model_backend,
        schema=schema,
        fp32=fp32,
    )
    suffix = "" if model == "conductance" else f"_{model}"
    fname = fname or f"training{suffix or '_with_Ih'}.npy"
    outdir = outdir or run_dir(model)

    print_param_partitions(session)
    print(f"device={session.device}, model={model}, nofruns={nofruns}, nofsteps={nofsteps}, "
          f"lrs={lrs}, nparams={fc.schema_nparams(list(session.schema))}, fname={fname}, outdir={outdir}")
    z_init = None
    if init_from:
        session, z_init = load_init_z(init_from, session)
    t0 = time.time()
    result = do_many_runs(session, nofruns, nofsteps, lrs=lrs, z_init=z_init)
    print(f"done in {(time.time() - t0) / 3600:.2f} hours")

    save_training_outputs(fname, outdir, session, result)
    make_plots(
        fname, outdir, session, result=result,
        ref_cubes=plot_ref_cubes, ref_cubes_2=plot_ref_cubes_2,
        plot_right_only=plot_right_only, at_x=at_x, at_y=at_y,
        align_at_x=align_at_x, align_at_y=align_at_y,
        plot_vm=plot_vm, show_pre=show_pre,
    )
    return fname, outdir, session


def add_spot_layout_arguments(parser):
    """Spot centre tiling flags (``--multi-spot``, ``--fully-inside``)."""
    from network.spot_target import DEFAULT_FULLY_INSIDE, DEFAULT_MULTI_SPOT

    parser.add_argument(
        "--multi-spot",
        type=parse_bool,
        default=DEFAULT_MULTI_SPOT,
        metavar="BOOL",
        help="tile simultaneous spot centres on network connectome "
             f"(default: {str(DEFAULT_MULTI_SPOT).lower()}; false → centre (0,0) only)",
    )
    parser.add_argument(
        "--fully-inside",
        type=parse_bool,
        default=DEFAULT_FULLY_INSIDE,
        metavar="BOOL",
        help="with --multi-spot: keep only centres whose spot footprint lies inside "
             f"connectome extent (default: {str(DEFAULT_FULLY_INSIDE).lower()})",
    )


def add_training_arguments(parser):
    """Register train.py training CLI flags on *parser*."""
    parser.add_argument("--model", default="hp_lp",
                        choices=list(fc.KNOWN_MODELS))
    parser.add_argument("--nofruns", type=int, default=1)
    parser.add_argument(
        "--nofsteps",
        type=int,
        default=None,
        help=f"steps per learning-rate stage (default: {DEFAULT_NOFSTEPS_GPU} on GPU, "
             f"{DEFAULT_NOFSTEPS_CPU} on CPU)",
    )
    parser.add_argument("--lrs", default="0.1",
                        help="comma-separated learning-rate stages; each runs for --nofsteps steps")
    parser.add_argument("--fname", default=None,
                        help="params filename (default derived from --model)")
    parser.add_argument("--outdir", default=None,
                        help="output dir (default derived from --model)")
    parser.add_argument("--from", dest="init_from", default=None, metavar="RUN",
                        help="prior run folder NAME only (no model/ prefix); "
                             "resolved under 0trained/<model>/NAME unless an absolute path is given; "
                             "load named best_param.npz as z init only "
                             "(settings come from this CLI, not train_opts.json)")
    _ih_gmax_default = (
        "indi=" + ",".join(DEFAULT_IH_GMAX_INDI_NAMES) + " fixed=all"
    )
    _partition_help = (
        "indi=/shared=/fixed=/frozen= lists space-separated; 'all' in one bucket = remainder; "
        "types or Src:Tar pairs (syn-strength). Example: indi=L1,L2 frozen=all"
    )
    _partition_kwargs = dict(default=None, nargs='+', metavar="PART")
    parser.add_argument("--all-param", **_partition_kwargs,
                        help=f"apply partitions to every parameter segment "
                             f"({_partition_help}; overridden by --ih-shape and per-param flags)")
    parser.add_argument("--in-gain", **_partition_kwargs,
                        help=f"in_gain partitions ({_partition_help}; default fixed=all)")
    parser.add_argument("--out-gain", **_partition_kwargs,
                        help=f"out_gain partitions ({_partition_help}; default fixed=all)")
    parser.add_argument("--out-scale", **_partition_kwargs,
                        help=f"out_scale partitions ({_partition_help}; default indi=all)")
    parser.add_argument("--syn-strength", **_partition_kwargs,
                        help=f"syn_strength partitions ({_partition_help}; default indi=all)")
    parser.add_argument("--v-th", **_partition_kwargs,
                        help=f"v_th partitions ({_partition_help}; default indi=all)")
    parser.add_argument("--ih-gmax", **_partition_kwargs,
                        help=f"Ih_gmax partitions ({_partition_help}; default {_ih_gmax_default})")
    parser.add_argument("--ih-gmax-off", **_partition_kwargs,
                        help=f"Ih_gmax_off partitions ({_partition_help}; default {_ih_gmax_default})")
    parser.add_argument("--ih-shape", **_partition_kwargs,
                        help="batch partitions for Ih_midv/Ih_slope/tau_midv and OFF "
                             f"({_partition_help}; default shared=all)")
    parser.add_argument("--ih-midv", **_partition_kwargs,
                        help=f"Ih_midv partitions (overrides --ih-shape; {_partition_help})")
    parser.add_argument("--ih-slope", **_partition_kwargs,
                        help=f"Ih_slope partitions (overrides --ih-shape; {_partition_help})")
    parser.add_argument("--tau-midv", **_partition_kwargs,
                        help=f"tau_midv partitions (overrides --ih-shape; {_partition_help})")
    parser.add_argument("--ih-midv-off", **_partition_kwargs,
                        help=f"Ih_midv_off partitions (overrides --ih-shape; {_partition_help})")
    parser.add_argument("--ih-slope-off", **_partition_kwargs,
                        help=f"Ih_slope_off partitions (overrides --ih-shape; {_partition_help})")
    parser.add_argument("--tau-midv-off", **_partition_kwargs,
                        help=f"tau_midv_off partitions (overrides --ih-shape; {_partition_help})")
    parser.add_argument("--tau-lp", **_partition_kwargs,
                        help=f"hp_lp tau_lp partitions ({_partition_help}; default indi=all)")
    parser.add_argument("--bias", **_partition_kwargs,
                        help=f"hp_lp bias partitions ({_partition_help}; default indi=all)")
    parser.add_argument("--tau-hp", **_partition_kwargs,
                        help=f"hp_lp tau_hp partitions ({_partition_help}; default indi=all)")
    parser.add_argument("--hp-gain", **_partition_kwargs,
                        help=f"hp_lp hp_gain partitions ({_partition_help}; default {_ih_gmax_default})")
    parser.add_argument("--ih-off", default=fc.IH_OFF_DEFAULT,
                        choices=list(fc.IH_OFF_MODES),
                        help="OFF-channel Ih: on (train Ih_gmax_off+OFF shape; default), "
                             "mirrored (OFF copies ON), off (disable OFF channel)")
    parser.add_argument("--fp32", action="store_true",
                        help="run simulation in float32 (default float64 on CUDA; "
                             "forced on when CUDA is unavailable)")
    parser.add_argument("--sequential", action="store_true",
                        help="one stimulus batch per forward (default: batched forward)")
    parser.add_argument("--network", default=DEFAULT_NETWORK_RUN, metavar="RUN",
                        help=f"connectome backend: built_network run folder under {NETWORK_DIR} "
                             f"(default: {DEFAULT_NETWORK_RUN})")
    parser.add_argument(
        "--multi-bar",
        type=parse_bool,
        default=True,
        metavar="BOOL",
        help="network moving-bar: tile simultaneous lane-clipped bars "
             "(default true); false → whole-field single bar over the full network field",
    )
    parser.add_argument(
        "--target",
        default="spot_bright",
        help="target name(s): spot (=spot_bright+spot_dark), moving_bar (=bright+dark), "
             "or explicit names / comma-separated list, e.g. spot,moving_bar",
    )
    parser.add_argument(
        "--cost-weight",
        default="",
        metavar="NAME|NAME=VALUE,...",
        help="per-part cost weights. NAME=VALUE merges onto default 1; bare NAME "
             "(aliases: spot, moving_bar, moving_bar_bright/dark, PD/ND/DSI) zeros "
             "all parts for --target then sets those to 1. "
             "e.g. DSI (=DSI-only), DSI=1 (PD/ND stay 1), DSI,PD=0.2",
    )
    parser.add_argument(
        "--shift-extent",
        type=int,
        default=DEFAULT_SHIFT_EXTENT,
        help="spot sub-shift hex-disc radius for spot targets in --target "
             "(n_shifts=1+3k(k+1); 0->1, 1->7, 2->19, 3->37, ...)",
    )
    parser.add_argument(
        "--spot-extent",
        type=float,
        default=None,
        metavar="R",
        help="spot footprint / centre-tiling radius (0.5 multiples; default 1); "
             "extent=1 folds RecF(2) into r=1 target amp and defaults cost weights "
             "to 0=1,1=1/6; extent 1.5/2 keep RecF(r) and 0=1,1=1/6,2=1/6",
    )
    add_spot_layout_arguments(parser)
    parser.add_argument(
        "--spot-cost-r-w",
        default="",
        metavar="R|R=W,...",
        help="spot cost weights by Euclidean r from stim column. Same rules as "
             "--cost-weight: R=W merges onto extent defaults; bare R zeros all "
             "known radii then sets R=1. Empty → extent default "
             "(1→0=1,1=1/6; else 0=1,1=1/6,2=1/6). Keys: 0,1,2,sqrt3. "
             "Weights only (does not change RecF data)",
    )
    parser.add_argument(
        "--cost-extent",
        default="",
        metavar="N|TARGET=N,...",
        help="network cost hex-disc radius (moving-bar default: network extent - 1; "
             "network extent 0/-1 and spot default to all columns): bare N for all "
             "--target, or per-target e.g. moving_bar_bright=0 "
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
    parser.add_argument(
        "--t-on-ms",
        type=float,
        default=500.0,
        metavar="MS",
        help="spot stimulus onset time in ms (default %(default)s; "
             "maxtime auto-extends to t_on + RESPONSE_DURATION)",
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


def parse_cost_weight(text, target_list):
    """Parse ``--cost-weight``: bare alias exclusive, ``NAME=VALUE`` merge.

    Bare tokens (aliases or concrete part keys) zero every cost part for
    ``target_list``, then set those names to ``1``. Explicit ``NAME=VALUE``
    always applied last (merge onto defaults / exclusive map). Empty → ``{}``
    (runtime default weight 1).
    """
    tokens = parse_comma_list(text)
    bare: list[str] = []
    explicit: dict[str, float] = {}
    for tok in tokens:
        if "=" in tok:
            name, val = tok.split("=", 1)
            explicit[name.strip()] = float(val.strip())
        else:
            bare.append(tok.strip())
    weights: dict[str, float] = {}
    if bare:
        weights = {key: 0.0 for key in fc.session_cost_part_keys(target_list)}
        weights.update(fc.expand_cost_weight_dict({name: 1.0 for name in bare}))
    weights.update(fc.expand_cost_weight_dict(explicit))
    return weights


def parse_spot_cost_r_w(text, spot_extent):
    """Parse ``--spot-cost-r-w``; empty → ``None`` (extent default at resolve)."""
    from network.spot_target import parse_spot_cost_r_w_tokens

    return parse_spot_cost_r_w_tokens(text, spot_extent=spot_extent)


def _partition_cli_text(parts):
    """Join space-separated partition CLI tokens into one parse string."""
    if parts is None:
        return None
    return ' '.join(parts)


def _partition_cli_map(args):
    """Build ``{seg_name: {indi/shared/fixed/frozen: tokens}}`` from partition CLI flags.

    Precedence: ``--all-param`` → ``--ih-shape`` → per-param flags.
    Omitted segments keep schema defaults (not listed here).
    """
    texts = {}
    all_param = _partition_cli_text(getattr(args, "all_param", None))
    if all_param is not None:
        for name in fc.ALL_PARAM_NAMES:
            texts[name] = all_param
    shape_text = _partition_cli_text(getattr(args, "ih_shape", None))
    if shape_text is not None:
        for name in fc.IH_SHAPE_PARAM_NAMES:
            texts[name] = shape_text
    per_param = {
        "in_gain": _partition_cli_text(getattr(args, "in_gain", None)),
        "out_gain": _partition_cli_text(getattr(args, "out_gain", None)),
        "out_scale": _partition_cli_text(getattr(args, "out_scale", None)),
        "syn_strength": _partition_cli_text(getattr(args, "syn_strength", None)),
        "v_th": _partition_cli_text(getattr(args, "v_th", None)),
        "Ih_gmax": _partition_cli_text(getattr(args, "ih_gmax", None)),
        "Ih_gmax_off": _partition_cli_text(getattr(args, "ih_gmax_off", None)),
        "Ih_midv": _partition_cli_text(getattr(args, "ih_midv", None)),
        "Ih_slope": _partition_cli_text(getattr(args, "ih_slope", None)),
        "tau_midv": _partition_cli_text(getattr(args, "tau_midv", None)),
        "Ih_midv_off": _partition_cli_text(getattr(args, "ih_midv_off", None)),
        "Ih_slope_off": _partition_cli_text(getattr(args, "ih_slope_off", None)),
        "tau_midv_off": _partition_cli_text(getattr(args, "tau_midv_off", None)),
        "tau_lp": _partition_cli_text(getattr(args, "tau_lp", None)),
        "bias": _partition_cli_text(getattr(args, "bias", None)),
        "tau_hp": _partition_cli_text(getattr(args, "tau_hp", None)),
        "hp_gain": _partition_cli_text(getattr(args, "hp_gain", None)),
    }
    for name, text in per_param.items():
        if text is not None:
            texts[name] = text
    return {name: fc.parse_partition_text(text) for name, text in texts.items()}


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
            # HARD STOP: no backward-compat. --from takes a run folder name only.
            if "/" in str(init_from) or "\\" in str(init_from):
                raise ValueError(
                    "--from must be a run folder name only (no path); "
                    "the model subfolder is inferred from --model (default: hp_lp). "
                    "Use an absolute path to reference runs outside 0trained.",
                )
            init_from = f"{args.model}/{init_from}"
    param_partitions = _partition_cli_map(args) or None
    target_list = parse_target_list(args.target)
    cost_weights = parse_cost_weight(args.cost_weight, target_list)
    default_extent, extent_kv = parse_cost_extent(args.cost_extent)
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
    from network.spot_target import DEFAULT_SPOT_EXTENT, spot_extent_half_steps

    spot_extent = DEFAULT_SPOT_EXTENT if args.spot_extent is None else float(args.spot_extent)
    spot_extent_half_steps(spot_extent)
    spot_cost_radius_weight = parse_spot_cost_r_w(args.spot_cost_r_w, spot_extent)
    from training_config import DELTAT_MS, RESPONSE_DURATION_MS, ms_to_steps
    _t_on_step = ms_to_steps(args.t_on_ms)
    _maxtime_step = ms_to_steps(args.t_on_ms + RESPONSE_DURATION_MS)
    _timing = {"t_on": _t_on_step, "maxtime": _maxtime_step, "deltat_ms": DELTAT_MS}
    moving_bar_bright_stimulus_opts = {"multi_bar": bool(args.multi_bar), "t_on": _t_on_step, "deltat_ms": DELTAT_MS}
    moving_bar_dark_stimulus_opts = {"multi_bar": bool(args.multi_bar), "t_on": _t_on_step, "deltat_ms": DELTAT_MS}
    spot_bright_stimulus_opts = dict(_timing)
    spot_dark_stimulus_opts = dict(_timing)
    i_cli = fc.build_i_cli_by_target({
        "i_baseline": parse_comma_kv(args.i_baseline, float),
        "i_bright": parse_comma_kv(args.i_bright, float),
        "i_dark": parse_comma_kv(args.i_dark, float),
    })
    lrs = parse_comma_floats(args.lrs)
    if not lrs:
        raise ValueError("--lrs must list at least one learning rate")
    # CLI-only: float64 on CPU is too heavy; force fp32 when CUDA is absent.
    # Keep CUDA probe out of import / run_training defaults (import-safe).
    # Run folder name stays strict CLI (command_run_name); do not inject --fp32.
    cuda_available = torch.cuda.is_available()
    fp32 = bool(args.fp32) or not cuda_available
    nofsteps = args.nofsteps
    if nofsteps is None:
        nofsteps = DEFAULT_NOFSTEPS_GPU if cuda_available else DEFAULT_NOFSTEPS_CPU
    run_name = command_run_name(script_stem)
    outdir = run_dir(args.model, parent=args.outdir, name=run_name)
    return dict(
        model=args.model,
        nofruns=args.nofruns,
        nofsteps=nofsteps,
        lrs=lrs,
        fname=args.fname,
        outdir=outdir,
        param_partitions=param_partitions,
        network=args.network,
        target_list=target_list,
        cost_weights=cost_weights,
        cost_extent_by_target=cost_extent_by_target,
        shift_extent=int(args.shift_extent),
        spot_extent=spot_extent,
        multi_spot=args.multi_spot,
        fully_inside=args.fully_inside,
        spot_cost_radius_weight=spot_cost_radius_weight,
        moving_bar_bright_stimulus_opts=moving_bar_bright_stimulus_opts,
        moving_bar_dark_stimulus_opts=moving_bar_dark_stimulus_opts,
        spot_bright_stimulus_opts=spot_bright_stimulus_opts,
        spot_dark_stimulus_opts=spot_dark_stimulus_opts,
        i_cli=i_cli,
        ih_off=args.ih_off,
        fp32=fp32,
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
