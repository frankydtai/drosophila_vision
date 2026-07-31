"""Pure training driver (no plotting).

Orchestration that trains then plots lives in ``SimulationCode/6_run.py``:

    ../.venv/bin/python 6_run.py --model hp_lp --nofsteps 30 --lrs 0.1

All results of a run land under ``training.config.PARAMETER_DIR``
(``SimulationCode/0_runs/<model>/<run_name>/``). Artifacts (``.npy`` /
``.npz``, ``train_opts.json``) go in ``data/``; CSV tables in the run
folder. Checkpoint PNGs are written by ``6_run.py``, not here.

Import-safe: does NOT parse argv or touch CUDA.
"""
import argparse
import re
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

_SIMULATION_CODE = Path(__file__).resolve().parent.parent
if str(_SIMULATION_CODE) not in sys.path:
    sys.path.insert(0, str(_SIMULATION_CODE))

import import_bootstrap  # noqa: F401
import network.path  # noqa: F401 — connectome_io on sys.path
from connectome_io import (
    NETWORK_DIR,
    parse_comma_list,
    resolve_network_json,
)
from training.defaults import (
    CHECKPOINT_INTERVAL,
    DELTA_MS,
    FP,
    FULLY_INSIDE,
    IH_GMAX_INDI_NAMES,
    IH_OFF,
    LRS,
    MODEL,
    MULTI_BAR,
    MULTI_SPOT,
    NETWORK,
    NOFRUNS,
    NOFSTEPS_CPU,
    NOFSTEPS_GPU,
    PHYSICS,
    PRE_GRAD,
    PRE_MS,
    PULSE_MS,
    RESPONSE_MS,
    SEQUENTIAL,
    SHIFT_EXTENT,
    SPOT_COST_RADII,
    SPOT_COST_RADIUS_KEY_ALIASES,
    SPOT_COST_RADIUS_WEIGHT,
    SPOT_COST_RADIUS_WEIGHT_EXTENT1,
    SPOT_EXTENT,
    SYN_MODE,
    TARGET,
)
from task.spot.data import default_spot_cost_radius_weight
from training import do_many_runs
import training as fc
from training.config import (
    EDGE_WEIGHT_CSV,
    PARAM_CSV,
    SYN_STRENGTH_CSV,
    run_data_dir,
)


RUN_NAME_MAX = 255


def _slug(text):
    """Filesystem-safe token for a CLI flag value."""
    return re.sub(r'[^\w.,-]+', '-', str(text)).strip('-')


def _argv_cli_tokens(argv):
    """Drop the script path; yield long-option tokens from *argv*."""
    if argv and argv[0].endswith('.py'):
        argv = argv[1:]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ('-h', '--help'):
            i += 1
            continue
        if not tok.startswith('--'):
            i += 1
            continue
        key, sep, val = tok[2:].partition('=')
        if sep:
            yield _slug(key), _slug(val)
            i += 1
        elif i + 1 < len(argv) and not argv[i + 1].startswith('-'):
            yield _slug(key), _slug(argv[i + 1])
            i += 2
        else:
            yield _slug(key), None
            i += 1


def command_run_name(script_stem, argv=None):
    """Build a run folder name from flags on the command line (``sys.argv``)."""
    if argv is None:
        argv = sys.argv[1:]
    prefix = os.environ.get('SLURM_JOB_ID') or time.strftime('%m%d_%H%M%S')
    parts = [prefix, script_stem]
    for key, val in _argv_cli_tokens(argv):
        parts.append(key)
        if val is not None:
            parts.append(val)
    name = '-'.join(parts)
    if len(name) <= RUN_NAME_MAX:
        return name
    return name[:RUN_NAME_MAX].rstrip('-')


def run_dir(model, root=None, parent=None, name=None):
    """``<PARAMETER_DIR>/<model>/<name>`` (or under *parent*)."""
    from training.config import PARAMETER_DIR
    if parent is None:
        root = str(PARAMETER_DIR) if root is None else root
        parent = os.path.join(root, model)
    if name is None:
        job_id = os.environ.get('SLURM_JOB_ID')
        name = f'run_{job_id}' if job_id else time.strftime('run_%m%d_%H%M%S')
    return os.path.join(parent, name)


def resolve_run_dir(path):
    """Resolve a run folder under ``PARAMETER_DIR`` or an absolute path."""
    from pathlib import Path as _Path
    from training.config import PARAMETER_DIR
    pth = _Path(path).expanduser()
    outdir = pth.resolve() if pth.is_absolute() else (PARAMETER_DIR / pth).resolve()
    if not outdir.is_dir():
        raise SystemExit(f'run folder not found: {path!r} -> {outdir}')
    return str(outdir)




def checkpoint_step_tag(step):
    return f'{int(step):05d}'




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
        if seg["kind"] in ("edge_pair", "edge"):
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


def write_edge_weight_table(z_t, session, table_path):
    """Write per-edge ``edge_weight`` CSV (network edge order)."""
    schema = list(session.schema)
    seg = next((s for s in schema if s["name"] == "edge_weight"), None)
    if seg is None or seg["kind"] != "edge":
        return None
    unit_vals = fc.z_to_unit_values(z_t, schema)
    arr = np.asarray(unit_vals["edge_weight"], dtype=np.float64).reshape(-1)
    conn = session.backend.conn
    if arr.shape[0] != conn.n_edges:
        raise ValueError(
            f"edge_weight length {arr.shape[0]} != n_edges {conn.n_edges}"
        )
    names = [str(n) for n in ctype_labels(session)]
    src = conn.src_idx.detach().cpu().numpy()
    tar = conn.tar_idx.detach().cpu().numpy()
    node_type = conn.node_type.detach().cpu().numpy()
    sign = torch.sign(conn.w_signed).detach().cpu().numpy()
    with open(table_path, "w") as f:
        f.write("edge_idx,src_unit,tar_unit,source_type,target_type,sign,edge_weight\n")
        for i in range(conn.n_edges):
            si, ti = int(src[i]), int(tar[i])
            f.write(
                "%d,%d,%d,%s,%s,%.0f,%.6f\n"
                % (
                    i, si, ti,
                    names[int(node_type[si])], names[int(node_type[ti])],
                    float(sign[i]), float(arr[i]),
                )
            )
    return table_path


def write_syn_table(z_t, session, outdir_or_path, *, tag=None):
    """Write ``syn_strength.csv`` or ``edge_weight.csv`` for the active syn mode."""
    if tag is None:
        syn_path = write_syn_strength_table(
            z_t, session, os.path.join(outdir_or_path, SYN_STRENGTH_CSV),
        )
        edge_path = write_edge_weight_table(
            z_t, session, os.path.join(outdir_or_path, EDGE_WEIGHT_CSV),
        )
    else:
        syn_path = write_syn_strength_table(
            z_t, session, os.path.join(outdir_or_path, f"syn_strength_{tag}.csv"),
        )
        edge_path = write_edge_weight_table(
            z_t, session, os.path.join(outdir_or_path, f"edge_weight_{tag}.csv"),
        )
    return syn_path or edge_path


def data_dir(outdir):
    return run_data_dir(outdir)


def params_path(outdir, fname):
    return os.path.join(data_dir(outdir), fname)


def best_param_path(outdir):
    return os.path.join(data_dir(outdir), 'best_param.npz')


def save_param_named(outdir, z, session, filename):
    """Write named full-width unit values to ``data/<filename>``."""
    schema = list(session.schema)
    named = fc.z_to_unit_values(z, schema)
    type_names = np.asarray(fc.type_unit_names(session.backend), dtype=object)
    payload = {k: np.asarray(v, dtype=np.float64) for k, v in named.items()}
    payload['type_names'] = type_names
    if any(s['kind'] == 'edge_pair' for s in schema):
        payload['pair_names'] = np.asarray(fc.pair_unit_names(session.backend), dtype=object)
    os.makedirs(data_dir(outdir), exist_ok=True)
    np.savez(os.path.join(data_dir(outdir), filename), **payload)


def save_best_param_named(outdir, z, session):
    """Write named full-width unit values to ``data/best_param.npz``."""
    save_param_named(outdir, z, session, 'best_param.npz')


def checkpoint_param_filename(step, run_i=0, nofruns=1):
    suffix = '' if nofruns == 1 else f'_run{run_i}'
    return f'best_param_step_{checkpoint_step_tag(step)}{suffix}.npz'


def write_checkpoint_csv(outdir, step, z_best, session):
    tag = checkpoint_step_tag(step)
    csv_dir = os.path.join(outdir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    param_path = os.path.join(csv_dir, f'param_{tag}.csv')
    write_param_table(z_best, session, param_path)
    syn_path = write_syn_table(z_best, session, csv_dir, tag=tag)
    print(f'wrote checkpoint csv: {param_path}')
    if syn_path is not None:
        print(f'wrote checkpoint csv: {syn_path}')




def make_checkpoint_callback(outdir, session, *, run_i=0, nofruns=1, on_png=None):
    """Write interval-best npz/csv; optional *on_png* for plot layer (from ``6_run.py``)."""

    def on_interval_best(step, z_best, cost_best):
        name = checkpoint_param_filename(step, run_i=run_i, nofruns=nofruns)
        save_param_named(outdir, z_best, session, name)
        write_checkpoint_csv(outdir, step, z_best, session)
        if on_png is not None:
            on_png(outdir, step, z_best, cost_best, session)
        print(f'wrote checkpoint {name} (cost={cost_best:.4f})')
    return on_interval_best


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
    """Write ``best_param.npz``, ``param.csv``, and syn/edge CSV for one best index."""
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
    syn_path = write_syn_table(z_best, session, outdir)
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
    """Regenerate ``param.csv`` / syn or edge CSV and ``best_param.npz`` from saved ``fname``."""
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
    network=NETWORK,
    sequential=SEQUENTIAL,
    target_list=None,
    cost_weights=None,
    cost_extent_by_target=None,
    shift_extent=SHIFT_EXTENT,
    spot_extent=SPOT_EXTENT,
    multi_spot=MULTI_SPOT,
    fully_inside=FULLY_INSIDE,
    spot_cost_radius_weight=None,
    i_cli=None,
    moving_bar_bright_stimulus_opts=None,
    moving_bar_dark_stimulus_opts=None,
    spot_bright_stimulus_opts=None,
    spot_dark_stimulus_opts=None,
    param_partitions=None,
    syn_mode=SYN_MODE,
    ih_off=IH_OFF,
    fp=FP,
    pre_grad=PRE_GRAD,
    pack_overrides=None,
    model_backend=None,
    schema=None,
    physics=None,
):
    """Create a :class:`TrainSession` from run options."""
    tl = list(target_list) if target_list is not None else list(
        fc.normalize_target_list([TARGET])
    )
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
        syn_mode=syn_mode,
        fp=fp,
        pre_grad=pre_grad,
        physics=physics,
        **mkw,
    )
    return fc.open_session(opts, model, schema=schema, model_backend=model_backend)


def run_training(model, nofruns, nofsteps, lrs, fname=None, outdir=None,
                 param_partitions=None,
                 syn_mode=SYN_MODE,
                 ih_off=IH_OFF,
                 network=NETWORK, sequential=SEQUENTIAL,
                 target_list=None, cost_weights=None,
                 cost_extent_by_target=None, shift_extent=SHIFT_EXTENT,
                 spot_extent=SPOT_EXTENT,
                 multi_spot=MULTI_SPOT,
                 fully_inside=FULLY_INSIDE,
                 spot_cost_radius_weight=None,
                 i_cli=None,
                 moving_bar_bright_stimulus_opts=None,
                 moving_bar_dark_stimulus_opts=None,
                 spot_bright_stimulus_opts=None,
                 spot_dark_stimulus_opts=None,
                 pack_overrides=None, model_backend=None, schema=None,
                 fp=FP,
                 pre_grad=PRE_GRAD,
                 physics=None,
                 init_from=None,
                 checkpoint_interval=None,
                 make_checkpoint_callback=make_checkpoint_callback,
                 checkpoint_on_png=None):
    """Train + save artifacts (no plotting). Returns ``(fname, outdir, session, result)``.

    Plotting belongs in ``6_run.py``. Pass *checkpoint_on_png* from the run layer
    when ``--checkpoint-interval`` should also write PNGs.
    """
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
        syn_mode=syn_mode,
        ih_off=ih_off,
        pack_overrides=pack_overrides,
        model_backend=model_backend,
        schema=schema,
        fp=fp,
        pre_grad=pre_grad,
        physics=physics,
    )
    suffix = "" if model == "borst" else f"_{model}"
    fname = fname or f"training{suffix or '_with_Ih'}.npy"
    outdir = outdir or run_dir(model)

    print_param_partitions(session)
    syn_mode = (session.train_opts or {}).get("syn_mode", SYN_MODE)
    print(f"device={session.device}, model={model}, syn_mode={syn_mode}, "
          f"nofruns={nofruns}, nofsteps={nofsteps}, "
          f"lrs={lrs}, nparams={fc.schema_nparams(list(session.schema))}, fname={fname}, outdir={outdir}")
    if checkpoint_interval is not None:
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be a positive integer")
        print(f"checkpoint_interval={checkpoint_interval}")
    z_init = None
    if init_from:
        session, z_init = load_init_z(init_from, session)
    t0 = time.time()
    result = do_many_runs(
        session, nofruns, nofsteps, lrs=lrs, z_init=z_init,
        checkpoint_interval=checkpoint_interval,
        checkpoint_outdir=outdir if checkpoint_interval is not None else None,
        make_checkpoint_callback=make_checkpoint_callback,
        checkpoint_on_png=checkpoint_on_png if checkpoint_interval is not None else None,
    )
    print(f"done in {(time.time() - t0) / 3600:.2f} hours")

    save_training_outputs(fname, outdir, session, result)
    return fname, outdir, session, result


def add_spot_layout_arguments(parser):
    """Spot center tiling flags (``--multi-spot``, ``--fully-inside``)."""
    parser.add_argument(
        "--multi-spot",
        type=parse_bool,
        default=MULTI_SPOT,
        metavar="BOOL",
        help="tile simultaneous spot centers on network connectome "
             f"(default: {str(MULTI_SPOT).lower()}; false → center (0,0) only)",
    )
    parser.add_argument(
        "--fully-inside",
        type=parse_bool,
        default=FULLY_INSIDE,
        metavar="BOOL",
        help="with --multi-spot: keep only centers whose spot footprint lies inside "
             f"connectome extent (default: {str(FULLY_INSIDE).lower()})",
    )


def add_training_arguments(parser):
    """Register train.py training CLI flags on *parser*.

    Concrete omitted-flag values live in :mod:`training.defaults` and are
    wired here as ``default=CONST``. ``None`` only for omit-disabled flags.
    """
    parser.add_argument("--model", default=MODEL, choices=list(fc.KNOWN_MODELS))
    parser.add_argument(
        "--syn-mode",
        default=SYN_MODE,
        choices=list(fc.SYN_MODES),
        help="synaptic scale: type_pair (sign*n_syn + type→type syn_strength; default) "
             "or per_edge (sign only + per-edge edge_weight magnitude)",
    )
    parser.add_argument("--nofruns", type=int, default=NOFRUNS)
    parser.add_argument(
        "--nofsteps",
        type=int,
        default=None,
        help=f"steps per learning-rate stage (default: {NOFSTEPS_GPU} on GPU, "
             f"{NOFSTEPS_CPU} on CPU)",
    )
    parser.add_argument("--lrs", default=LRS,
                        help="comma-separated learning-rate stages; each runs for --nofsteps steps")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        metavar="N",
        help="every N global training steps, snap to the interval-best params and write "
             "data/best_param_step_XXXXX.npz, csv/param_XXXXX.csv, "
             "csv/syn_strength_XXXXX.csv or csv/edge_weight_XXXXX.csv, and png/*_XXXXX.png "
             f"(default: {CHECKPOINT_INTERVAL})",
    )
    parser.add_argument("--fname", default=None,
                        help="params filename (default derived from --model)")
    parser.add_argument("--outdir", default=None,
                        help="output dir (default derived from --model)")
    parser.add_argument("--from", dest="init_from", default=None, metavar="RUN",
                        help="prior run folder NAME only (no model/ prefix); "
                             "resolved under 0_runs/<model>/NAME unless an absolute path is given; "
                             "load named best_param.npz as z init only "
                             "(settings come from this CLI, not train_opts.json)")
    _ih_gmax_default = (
        "indi=" + ",".join(IH_GMAX_INDI_NAMES) + " fixed=all"
    )
    _partition_help = (
        "indi=/shared=/fixed=/frozen= lists space-separated; 'all' in one bucket = remainder; "
        "types or Src:Tar pairs (syn-strength); init=NAME:VAL,... overrides initial values. "
        "Example: indi=all init=L1,L2,L4,L5:200 all:10000"
    )
    _edge_weight_help = (
        "only indi=all / fixed=all / frozen=all "
        "(--syn-mode per_edge; no shared= / named edges)"
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
                        help=f"syn_strength partitions ({_partition_help}; default indi=all; "
                             f"--syn-mode type_pair only)")
    parser.add_argument("--edge-weight", **_partition_kwargs,
                        help=f"edge_weight partitions ({_edge_weight_help}; default indi=all)")
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
    parser.add_argument("--v-rest", **_partition_kwargs,
                        help=f"hp_lp v_rest partitions ({_partition_help}; default indi=all)")
    parser.add_argument("--tau-hp", **_partition_kwargs,
                        help=f"hp_lp tau_hp partitions ({_partition_help}; default indi=all)")
    parser.add_argument("--hp-gain", **_partition_kwargs,
                        help=f"hp_lp hp_gain partitions ({_partition_help}; default fixed=all)")
    parser.add_argument("--ih-off", default=IH_OFF,
                        choices=list(fc.IH_OFF_MODES),
                        help="OFF-channel Ih: on (train Ih_gmax_off+OFF shape; default), "
                             "mirrored (OFF copies ON), off (disable OFF channel)")
    parser.add_argument(
        "--fp",
        type=int,
        default=FP,
        choices=(16, 32, 64),
        metavar="N",
        help=f"simulation float width (default: {FP}); "
             "64 is forced to 32 when CUDA is unavailable",
    )
    parser.add_argument(
        "--pre-grad",
        type=parse_bool,
        default=PRE_GRAD,
        metavar="BOOL",
        help="include t < t_onset in BPTT / v_onset grads "
             f"(default: {str(PRE_GRAD).lower()}); "
             "false → no_grad pre + detach state/v/v_onset at onset",
    )
    parser.add_argument(
        "--sequential",
        type=parse_bool,
        default=SEQUENTIAL,
        metavar="BOOL",
        help=f"one stimulus batch per forward (default: {str(SEQUENTIAL).lower()})",
    )
    parser.add_argument("--network", default=NETWORK, metavar="RUN",
                        help=f"connectome backend: built_network run folder under {NETWORK_DIR} "
                             f"(default: {NETWORK})")
    parser.add_argument(
        "--multi-bar",
        type=parse_bool,
        default=MULTI_BAR,
        metavar="BOOL",
        help="network moving-bar: tile simultaneous lane-clipped bars "
             f"(default: {str(MULTI_BAR).lower()}); "
             "false → whole-field single bar over the full network field",
    )
    parser.add_argument(
        "--target",
        default=TARGET,
        help="target name(s): spot (=spot_bright+spot_dark), moving_bar (=bright+dark), "
             "or explicit names / comma-separated list, e.g. spot,moving_bar",
    )
    parser.add_argument(
        "--cost-weight",
        default=None,
        metavar="NAME|NAME=VALUE,...",
        help="per-part cost weights. NAME=VALUE merges onto default 1; bare NAME "
             "(aliases: spot, moving_bar, moving_bar_bright/dark, PD/ND/DSI) zeros "
             "all parts for --target then sets those to 1. "
             "e.g. DSI (=DSI-only), DSI=1 (PD/ND stay 1), DSI,PD=0.2",
    )
    parser.add_argument(
        "--shift-extent",
        type=int,
        default=SHIFT_EXTENT,
        help="spot sub-shift hex-disc radius for spot targets in --target "
             "(n_shifts=1+3k(k+1); 0->1, 1->7, 2->19, 3->37, ...)",
    )
    parser.add_argument(
        "--spot-extent",
        type=float,
        default=SPOT_EXTENT,
        metavar="R",
        help=f"spot footprint / center-tiling radius (0.5 multiples; default {SPOT_EXTENT}); "
             "extent=1 folds RecF(2) into r=1 target amp and defaults cost weights "
             "to 0=1,1=1/6; extent 1.5/2 keep RecF(r) and 0=1,1=1/6,2=1/6",
    )
    add_spot_layout_arguments(parser)
    parser.add_argument(
        "--spot-cost-r-w",
        default=None,
        metavar="R|R=W,...",
        help="spot cost weights by Euclidean r from stim column. Same rules as "
             "--cost-weight: R=W merges onto extent defaults; bare R zeros all "
             "known radii then sets R=1. Omit → extent default "
             "(1→0=1,1=1/6; else 0=1,1=1/6,2=1/6). Keys: 0,1,2,sqrt3. "
             "Weights only (does not change RecF data)",
    )
    parser.add_argument(
        "--cost-extent",
        default=None,
        metavar="N|TARGET=N,...",
        help="network cost hex-disc radius (moving-bar default: network extent - 1; "
             "network extent 0/-1 and spot default to all columns): bare N for all "
             "--target, or per-target e.g. moving_bar_bright=0 "
             "(aliases: spot, moving_bar); -1 = all columns; requires --network",
    )
    parser.add_argument(
        "--i-baseline",
        default=None,
        metavar="TARGET=VALUE,...",
        help="per-target PR baseline (pA); aliases: spot, moving_bar",
    )
    parser.add_argument(
        "--i-bright",
        default=None,
        metavar="TARGET=VALUE,...",
        help="bright peak/step current (pA); targets: spot_bright, moving_bar_bright "
             "(aliases spot, moving_bar)",
    )
    parser.add_argument(
        "--i-dark",
        default=None,
        metavar="TARGET=VALUE,...",
        help="dark peak/step current (pA); targets: spot_dark, moving_bar_dark "
             "(aliases spot, moving_bar)",
    )
    parser.add_argument(
        "--pre-ms",
        type=float,
        default=PRE_MS,
        metavar="MS",
        help=f"pre-stimulus baseline duration in ms (default: {PRE_MS}; "
             "t_onset = ms_to_t(pre_ms); "
             "n_t = ms_to_t(pre_ms)+ms_to_t(response_ms)+1)",
    )
    parser.add_argument(
        "--response-ms",
        type=float,
        default=RESPONSE_MS,
        metavar="MS",
        help=f"spot: post-onset response window in ms (default: {RESPONSE_MS}; "
             "n_t = ms_to_t(pre_ms)+ms_to_t(response_ms)+1)",
    )
    parser.add_argument(
        "--pulse-ms",
        type=float,
        default=PULSE_MS,
        metavar="MS",
        help=f"spot: bright/dark PR pulse duration in ms from onset "
             f"(default: {PULSE_MS})",
    )
    parser.add_argument(
        "--delta-ms",
        type=float,
        default=DELTA_MS,
        metavar="MS",
        help=f"simulation / stimulus time step in ms (default: {DELTA_MS}; "
             "sets Physics.delta_ms and all stimulus opts delta_ms)",
    )
    parser.add_argument(
        "--cost-interval-ms",
        type=float,
        default=None,
        metavar="MS",
        help="spot: train on post-onset times 0, interval, 2*interval, ... "
             "through response window; omit = every post-onset t (#4)",
    )


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
    return fc.normalize_target_list(parse_comma_list(text))


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
    from task.spot.data import parse_spot_cost_r_w_tokens

    return parse_spot_cost_r_w_tokens(
        text,
        default_weights=default_spot_cost_radius_weight(
            spot_extent,
            weights=SPOT_COST_RADIUS_WEIGHT,
            weights_extent1=SPOT_COST_RADIUS_WEIGHT_EXTENT1,
        ),
        spot_cost_radii=SPOT_COST_RADII,
        aliases=SPOT_COST_RADIUS_KEY_ALIASES,
    )


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
    syn_mode = fc.normalize_syn_mode(getattr(args, "syn_mode", SYN_MODE))
    syn_text = _partition_cli_text(getattr(args, "syn_strength", None))
    edge_text = _partition_cli_text(getattr(args, "edge_weight", None))
    if syn_mode == "per_edge" and syn_text is not None:
        raise ValueError("--syn-strength requires --syn-mode type_pair")
    if syn_mode == "type_pair" and edge_text is not None:
        raise ValueError("--edge-weight requires --syn-mode per_edge")
    texts = {}
    all_param = _partition_cli_text(getattr(args, "all_param", None))
    if all_param is not None:
        for name in fc.ALL_PARAM_NAMES:
            if name == "syn_strength" and syn_mode != "type_pair":
                continue
            if name == "edge_weight" and syn_mode != "per_edge":
                continue
            texts[name] = all_param
    shape_text = _partition_cli_text(getattr(args, "ih_shape", None))
    if shape_text is not None:
        for name in fc.IH_SHAPE_PARAM_NAMES:
            texts[name] = shape_text
    per_param = {
        "in_gain": _partition_cli_text(getattr(args, "in_gain", None)),
        "out_gain": _partition_cli_text(getattr(args, "out_gain", None)),
        "out_scale": _partition_cli_text(getattr(args, "out_scale", None)),
        "syn_strength": syn_text,
        "edge_weight": edge_text,
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
        "v_rest": _partition_cli_text(getattr(args, "v_rest", None)),
        "tau_hp": _partition_cli_text(getattr(args, "tau_hp", None)),
        "hp_gain": _partition_cli_text(getattr(args, "hp_gain", None)),
    }
    for name, text in per_param.items():
        if text is not None:
            texts[name] = text
    out = {name: fc.parse_partition_text(text) for name, text in texts.items()}
    if "edge_weight" in out:
        fc.validate_edge_weight_partition(out["edge_weight"])
    return out


def training_kwargs_from_args(
    args,
    *,
    script_stem="train",
):
    """Parse a training CLI namespace into kwargs for :func:`run_training`."""
    model = args.model
    init_from = args.init_from
    if init_from:
        p = Path(str(init_from)).expanduser()
        if not p.is_absolute():
            # HARD STOP: no backward-compat. --from takes a run folder name only.
            if "/" in str(init_from) or "\\" in str(init_from):
                raise ValueError(
                    "--from must be a run folder name only (no path); "
                    "the model subfolder is inferred from --model "
                    f"(default: {MODEL}). "
                    "Use an absolute path to reference runs outside 0_runs.",
                )
            init_from = f"{model}/{init_from}"
    param_partitions = _partition_cli_map(args) or None
    target_list = parse_target_list(args.target)
    cost_weights = parse_cost_weight(args.cost_weight, target_list)
    default_extent, extent_kv = parse_cost_extent(args.cost_extent)
    cost_extent_by_target = fc.resolve_cost_extent_by_target(
        target_list, default_extent, extent_kv,
    )
    if default_extent is not None and default_extent != -1 and default_extent < 0:
        raise ValueError("--cost-extent must be -1 or >= 0")
    if any(v != -1 and v < 0 for v in extent_kv.values()):
        raise ValueError("--cost-extent must be -1 or >= 0")
    from task.spot.input import spot_extent_half_steps

    shift_extent = int(args.shift_extent)
    if shift_extent < 0:
        raise ValueError("--shift-extent must be >= 0")
    spot_extent = float(args.spot_extent)
    spot_extent_half_steps(spot_extent)
    spot_cost_radius_weight = parse_spot_cost_r_w(args.spot_cost_r_w, spot_extent)
    multi_spot = bool(args.multi_spot)
    fully_inside = bool(args.fully_inside)
    pre_ms = float(args.pre_ms)
    response_ms = float(args.response_ms)
    delta_ms = float(args.delta_ms)
    if delta_ms <= 0:
        raise ValueError("--delta-ms must be > 0")
    physics = replace(PHYSICS, delta_ms=delta_ms)
    multi_bar = bool(args.multi_bar)
    _timing = {
        "pre_ms": pre_ms,
        "response_ms": response_ms,
        "delta_ms": delta_ms,
    }
    moving_bar_bright_stimulus_opts = {
        "multi_bar": multi_bar,
        "pre_ms": pre_ms,
        "delta_ms": delta_ms,
    }
    moving_bar_dark_stimulus_opts = {
        "multi_bar": multi_bar,
        "pre_ms": pre_ms,
        "delta_ms": delta_ms,
    }
    spot_bright_stimulus_opts = dict(_timing)
    spot_dark_stimulus_opts = dict(_timing)
    for _o in (spot_bright_stimulus_opts, spot_dark_stimulus_opts):
        _o["pulse_ms"] = float(args.pulse_ms)
    if args.cost_interval_ms is not None:
        if float(args.cost_interval_ms) <= 0:
            raise ValueError("--cost-interval-ms must be > 0")
        for _o in (spot_bright_stimulus_opts, spot_dark_stimulus_opts):
            _o["cost_interval_ms"] = float(args.cost_interval_ms)
    i_cli = fc.build_i_cli_by_target({
        "i_baseline": parse_comma_kv(args.i_baseline, float),
        "i_bright": parse_comma_kv(args.i_bright, float),
        "i_dark": parse_comma_kv(args.i_dark, float),
    })
    lrs = parse_comma_floats(args.lrs)
    if not lrs:
        raise ValueError("--lrs must list at least one learning rate")
    cuda_available = torch.cuda.is_available()
    fp = int(args.fp)
    if not cuda_available and fp == 64:
        fp = 32
    nofsteps = args.nofsteps
    if nofsteps is None:
        nofsteps = NOFSTEPS_GPU if cuda_available else NOFSTEPS_CPU
    run_name = command_run_name(script_stem)
    outdir = run_dir(model, parent=args.outdir, name=run_name)
    return dict(
        model=model,
        nofruns=int(args.nofruns),
        nofsteps=nofsteps,
        lrs=lrs,
        fname=args.fname,
        outdir=outdir,
        param_partitions=param_partitions,
        syn_mode=fc.normalize_syn_mode(args.syn_mode),
        network=args.network,
        target_list=target_list,
        cost_weights=cost_weights,
        cost_extent_by_target=cost_extent_by_target,
        shift_extent=shift_extent,
        spot_extent=spot_extent,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_weight=spot_cost_radius_weight,
        moving_bar_bright_stimulus_opts=moving_bar_bright_stimulus_opts,
        moving_bar_dark_stimulus_opts=moving_bar_dark_stimulus_opts,
        spot_bright_stimulus_opts=spot_bright_stimulus_opts,
        spot_dark_stimulus_opts=spot_dark_stimulus_opts,
        i_cli=i_cli,
        ih_off=args.ih_off,
        fp=fp,
        pre_grad=bool(args.pre_grad),
        sequential=bool(args.sequential),
        physics=physics,
        init_from=init_from,
        checkpoint_interval=args.checkpoint_interval,
    )



def main():
    raise SystemExit(
        "training.driver is the pure training library; use SimulationCode/6_run.py:\n"
        "  ../.venv/bin/python 6_run.py --help"
    )


if __name__ == "__main__":
    main()
