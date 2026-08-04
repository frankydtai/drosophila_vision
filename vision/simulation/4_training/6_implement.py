"""Pure training implement (no plotting).

Orchestration that trains then plots lives in ``simulation/run.py``:

    ../.venv/bin/python run.py --model hp_lp --nofsteps 30 --lrs 0.1

All results of a run land under ``training.config.PARAMETER_DIR``
(``simulation/0_runs/<model>/<run_name>/``). Artifacts (``.npy`` /
``.npz``, ``train_opts.json``) go in ``data/``; CSV tables in the run
folder. Checkpoint PNGs are written by ``run.py``, not here.

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
import network.path  # noqa: F401 — FAFB path on sys.path
from import_bootstrap import normalize_option_dashes, parse_bool, parse_comma_list
from path import (
    BUILT_NETWORKS_DIR,
    resolve_network_json,
)
from param_defaults import (
    CHECKPOINT_INTERVAL,
    COST_NORM,
    DELTA_MS,
    EULER,
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
    DELTA_MS,
    PARAM_BOXES,
    PRE_GRAD,
    MS_PRE,
    MS_POST,
    MS_PULSE,
    MS_RESPONSE,
    SEQUENTIAL,
    SHIFT_EXTENT,
    SPOT_COST_RADII,
    SPOT_COST_RADIUS_KEY_ALIASES,
    SPOT_COST_RADIUS_WEIGHT,
    SPOT_COST_RADIUS_WEIGHT_EXTENT1,
    SPOT_EXTENT,
    PRE_STEADY,
    PRE_STEADY_DAMP,
    PRE_STEADY_ITERS,
    SYN_MODE,
    TASK,
)
from task.spot.gt import (
    default_spot_cost_radius_weight,
    parse_spot_cost_r_w_tokens,
)
from training import do_many_runs
import training
from training.config import (
    COST_NORMS,
    PARAM_CSV,
    PRE_STEADY_BY_MODEL,
    SYN_STRENGTH_CELL_CSV,
    SYN_STRENGTH_EDGE_CSV,
    expand_cost_norm,
    expand_pre_steady_dict,
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
    argv = normalize_option_dashes(argv)
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




def cell_labels(session):
    if session.backend.network is None:
        raise ValueError("cell_labels requires session.backend.network")
    return np.asarray(session.backend.network.cell_names)



def decompose_params(z_t, session):
    """Return (per_cell_cols, global_scalars) for one parameter vector.

    Per-cell columns come from full-width node values (train-mode-aware).
    Shared-only segments also emit a global scalar (mean of shared nodes).
    """
    n = session.backend.n_cells
    schema = list(session.schema)
    node_vals = training.z_to_node_values(z_t, schema)
    cols, glob = {}, {}
    for seg in schema:
        name = seg["name"]
        if seg["kind"] in ("edge_pair", "edge"):
            continue
        arr = np.asarray(node_vals[name], dtype=np.float64).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"{name}: node width {arr.shape[0]} != n_cells {n}")
        cols[name] = arr
        if seg.get("shared") and not seg.get("indi"):
            glob[name] = float(arr[list(seg["shared"])].mean()) if seg["shared"] else float(arr.mean())
    return cols, glob


def write_param_table(z_t, session, table_path, extra_cols=None):
    cols, glob = decompose_params(z_t, session)
    if extra_cols:
        cols.update(extra_cols)
    cell_col = cell_labels(session)
    cell_names = list(cols.keys())
    glob_names = list(glob.keys())
    n = session.backend.n_cells
    with open(table_path, "w") as f:
        f.write("idx,cell," + ",".join(cell_names + glob_names) + "\n")
        for i in range(n):
            row = ["%.6f" % cols[nm][i] for nm in cell_names] + ["%.6f" % glob[nm] for nm in glob_names]
            f.write("%d,%s," % (i, cell_col[i]) + ",".join(row) + "\n")
    return table_path


def write_syn_strength_cell_table(z_t, session, table_path):
    """Write edge-pair ``syn_strength_cell`` as source×target matrix CSV.

    Rows = source types, columns = target types. Absent connectome pairs are blank.
    """
    schema = list(session.schema)
    seg = next((s for s in schema if s["name"] == "syn_strength_cell"), None)
    if seg is None or seg["kind"] != "edge_pair":
        return None
    node_vals = training.z_to_node_values(z_t, schema)
    arr = np.asarray(node_vals["syn_strength_cell"], dtype=np.float64).reshape(-1)
    names = [str(n) for n in cell_labels(session)]
    keys = list(session.backend.conn.pair_keys)
    if arr.shape[0] != len(keys):
        raise ValueError(
            f"syn_strength_cell length {arr.shape[0]} != n_pairs {len(keys)}"
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


def write_syn_strength_edge_table(z_t, session, table_path):
    """Write per-edge ``syn_strength_edge`` CSV (network edge order)."""
    schema = list(session.schema)
    seg = next((s for s in schema if s["name"] == "syn_strength_edge"), None)
    if seg is None or seg["kind"] != "edge":
        return None
    node_vals = training.z_to_node_values(z_t, schema)
    arr = np.asarray(node_vals["syn_strength_edge"], dtype=np.float64).reshape(-1)
    conn = session.backend.conn
    if arr.shape[0] != conn.n_edges:
        raise ValueError(
            f"syn_strength_edge length {arr.shape[0]} != n_edges {conn.n_edges}"
        )
    names = [str(n) for n in cell_labels(session)]
    src = conn.src_idx.detach().cpu().numpy()
    tar = conn.tar_idx.detach().cpu().numpy()
    node_cell = conn.node_cell.detach().cpu().numpy()
    syn_sign = torch.sign(conn.w_signed).detach().cpu().numpy()
    with open(table_path, "w") as f:
        f.write("edge_idx,src_node,tar_node,source_cell,target_cell,syn_sign,syn_strength_edge\n")
        for i in range(conn.n_edges):
            si, ti = int(src[i]), int(tar[i])
            f.write(
                "%d,%d,%d,%s,%s,%.0f,%.6f\n"
                % (
                    i, si, ti,
                    names[int(node_cell[si])], names[int(node_cell[ti])],
                    float(syn_sign[i]), float(arr[i]),
                )
            )
    return table_path


def write_syn_table(z_t, session, outdir_or_path, *, tag=None):
    """Write ``syn_strength_cell.csv`` or ``syn_strength_edge.csv`` for the active syn mode."""
    if tag is None:
        cell_path = write_syn_strength_cell_table(
            z_t, session, os.path.join(outdir_or_path, SYN_STRENGTH_CELL_CSV),
        )
        edge_path = write_syn_strength_edge_table(
            z_t, session, os.path.join(outdir_or_path, SYN_STRENGTH_EDGE_CSV),
        )
    else:
        cell_path = write_syn_strength_cell_table(
            z_t, session, os.path.join(outdir_or_path, f"syn_strength_cell_{tag}.csv"),
        )
        edge_path = write_syn_strength_edge_table(
            z_t, session, os.path.join(outdir_or_path, f"syn_strength_edge_{tag}.csv"),
        )
    return cell_path or edge_path


def data_dir(outdir):
    return run_data_dir(outdir)


def params_path(outdir, fname):
    return os.path.join(data_dir(outdir), fname)


def best_param_path(outdir):
    return os.path.join(data_dir(outdir), 'best_param.npz')


def save_param_named(outdir, z, session, filename):
    """Write named full-width node values to ``data/<filename>``."""
    schema = list(session.schema)
    named = training.z_to_node_values(z, schema)
    cell_names = np.asarray(training.cell_node_names(session.backend), dtype=object)
    payload = {k: np.asarray(v, dtype=np.float64) for k, v in named.items()}
    payload['cell_names'] = cell_names
    if any(s['kind'] == 'edge_pair' for s in schema):
        payload['pair_names'] = np.asarray(training.pair_node_names(session.backend), dtype=object)
    os.makedirs(data_dir(outdir), exist_ok=True)
    np.savez(os.path.join(data_dir(outdir), filename), **payload)


def save_best_param_named(outdir, z, session):
    """Write named full-width node values to ``data/best_param.npz``."""
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
    """Write interval-best npz/csv; optional *on_png* for plot layer (from ``run.py``)."""

    def on_interval_best(step, z_best, cost_best):
        name = checkpoint_param_filename(step, run_i=run_i, nofruns=nofruns)
        save_param_named(outdir, z_best, session, name)
        write_checkpoint_csv(outdir, step, z_best, session)
        if on_png is not None:
            on_png(outdir, step, z_best, cost_best, session)
        print(f'wrote checkpoint {name} (cost={cost_best:.4f})')
    return on_interval_best


def load_best_param_named(outdir):
    """Load ``data/best_param.npz`` → (named dict, cell_names, pair_names|None)."""
    fp = best_param_path(outdir)
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)
    with np.load(fp, allow_pickle=True) as d:
        cell_names = [str(x) for x in d['cell_names'].tolist()]
        pair_names = None
        if 'pair_names' in d.files:
            pair_names = [str(x) for x in d['pair_names'].tolist()]
        named = {
            k: np.asarray(d[k], dtype=np.float64)
            for k in d.files
            if k not in ('cell_names', 'pair_names')
        }
    return named, cell_names, pair_names


def load_best_param(outdir, session=None):
    """Load best params as 1-D z for *session* (remap from named npz)."""
    if session is None:
        raise TypeError("load_best_param requires session for named best_param.npz")
    named, cell_names, pair_names = load_best_param_named(outdir)
    schema = training.attach_param_carry(
        list(session.schema),
        training.remap_named_node_values(
            named, cell_names, pair_names, list(session.schema), session.backend,
        ),
    )
    remapped = training.remap_named_node_values(
        named, cell_names, pair_names, schema, session.backend,
    )
    z = training.node_values_to_z(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )
    return z.detach().cpu().numpy().astype(np.float64)


def _costs_path(outdir):
    """Per-run end weighted total costs."""
    return os.path.join(data_dir(outdir), 'costs.npy')


def _best_costs_path(outdir):
    """Best-run per-step weighted total cost curve."""
    return os.path.join(data_dir(outdir), 'best_costs.npy')


def _costs_by_part_path(outdir):
    """Per-run end per-part costs."""
    return os.path.join(data_dir(outdir), 'costs_by_part.npz')


def _best_costs_by_part_path(outdir):
    """Best-run per-step per-part cost curves."""
    return os.path.join(data_dir(outdir), 'best_costs_by_part.npz')


def final_costs_for_params(all_params, session, final_costs=None):
    """Per-run final costs; recompute only when not supplied."""
    all_params = np.atleast_2d(all_params)
    if final_costs is not None:
        return np.asarray(final_costs, dtype=np.float64)
    return np.array([
        training.calc_cost(
            torch.tensor(all_params[i], dtype=torch.float64, device=session.device),
            session,
        ).item()
        for i in range(all_params.shape[0])
    ])


def write_best_artifacts(outdir, fname, session, all_params, final_costs):
    """Write ``best_param.npz``, ``param.csv``, and syn/edge CSV for ``argmin(final_costs)``."""
    all_params = np.atleast_2d(all_params)
    final_costs = np.asarray(final_costs, dtype=np.float64)
    run_i = int(np.argmin(final_costs))
    best = all_params[run_i]
    os.makedirs(data_dir(outdir), exist_ok=True)
    z_best = torch.tensor(best, dtype=session.sim_dtype, device=session.device)
    save_best_param_named(outdir, z_best, session)
    table_path = os.path.join(outdir, PARAM_CSV)
    write_param_table(z_best, session, table_path)
    print("wrote table: %s (best run #%d, cost=%.4f)" % (
        table_path, run_i, final_costs[run_i]))
    syn_path = write_syn_table(z_best, session, outdir)
    if syn_path is not None:
        print("wrote table: %s" % syn_path)
    return best


def load_stored_costs(outdir):
    """Load ``costs.npy``, ``best_costs.npy``, and per-part npz when present."""
    final_costs = None
    cost_curve = None
    costs_by_part = None
    final_costs_by_part = None
    fp = _costs_path(outdir)
    if os.path.isfile(fp):
        final_costs = np.load(fp)
    cp = _best_costs_path(outdir)
    if os.path.isfile(cp):
        cost_curve = np.load(cp)
    cbt = _best_costs_by_part_path(outdir)
    if os.path.isfile(cbt):
        with np.load(cbt) as d:
            costs_by_part = {k: np.asarray(d[k]) for k in d.files}
    fbt = _costs_by_part_path(outdir)
    if os.path.isfile(fbt):
        with np.load(fbt) as d:
            final_costs_by_part = {k: np.asarray(d[k]) for k in d.files}
    return final_costs, cost_curve, costs_by_part, final_costs_by_part


def load_init_z(init_from, session):
    """Load named best params; return ``(session, z)`` with frozen carry attached."""
    try:
        outdir = resolve_run_dir(init_from)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc
    named, cell_names, pair_names = load_best_param_named(outdir)
    schema = list(session.schema)
    remapped = training.remap_named_node_values(
        named, cell_names, pair_names, schema, session.backend,
    )
    schema = training.attach_param_carry(schema, remapped)
    session = session.with_schema(schema)
    z = training.node_values_to_z(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )
    print(
        f'from {outdir!r} -> {best_param_path(outdir)!r} '
        f'({training.schema_nparams(schema)} trainable slots)'
    )
    return session, z


def save_training_outputs(fname, outdir, session, result):
    """Write the full run artifact set (convention §5)."""
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(data_dir(outdir), exist_ok=True)
    if session.train_opts is not None:
        with open(os.path.join(data_dir(outdir), training.TRAIN_OPTS_FILE), 'w') as f:
            json.dump(session.train_opts, f, indent=2)
            f.write('\n')
    np.save(params_path(outdir, fname), result.all_params)
    np.save(_best_costs_path(outdir), result.cost_curve)
    np.save(_costs_path(outdir), result.final_costs)
    if result.cost_curves_by_part:
        np.savez(_best_costs_by_part_path(outdir), **result.cost_curves_by_part)
    if result.final_costs_by_part:
        np.savez(_costs_by_part_path(outdir), **result.final_costs_by_part)
    write_best_artifacts(
        outdir, fname, session, result.all_params, result.final_costs,
    )


def save_param_tables(fname, outdir, session):
    """Regenerate ``param.csv`` / syn or edge CSV and ``best_param.npz`` from saved ``fname``."""
    all_params = np.load(params_path(outdir, fname))
    final_costs, _, _, _ = load_stored_costs(outdir)
    final_costs = final_costs_for_params(all_params, session, final_costs=final_costs)
    write_best_artifacts(outdir, fname, session, all_params, final_costs)


def print_train_modes(session):
    """Print one schema segment per line: indi/shared/fixed/frozen counts and ntrain."""
    schema = list(session.schema)
    if not schema:
        return
    print("train_modes:")
    w = max(len(s["name"]) for s in schema)
    for s in schema:
        print(
            f"  {s['name']:<{w}}  "
            f"indi={len(s.get('indi') or [])}/"
            f"shared={len(s.get('shared') or [])}/"
            f"fixed={len(s.get('fixed') or [])}/"
            f"frozen={len(s.get('frozen') or [])} "
            f"({training.seg_ntrain(s)})"
        )


def apply_session_train_modes(session, train_modes_by_name):
    """Apply CLI/name train_modes onto session schema and refresh train_opts record."""
    if not train_modes_by_name:
        return session
    from dataclasses import replace
    backend = session.backend
    schema = training.apply_train_modes(
        list(session.schema),
        train_modes_by_name,
        lambda seg: training.node_names_for_segment(seg, backend),
    )
    schema = training.attach_param_carry(schema)
    opts = dict(session.train_opts or {})
    opts['train_modes'] = training.schema_train_modes_record(
        schema, lambda seg: training.node_names_for_segment(seg, backend),
    )
    session = replace(session, schema=tuple(schema), train_opts=opts)
    print_train_modes(session)
    return session


def resolve_network(network):
    return str(resolve_network_json(network))


def build_session(
    model,
    *,
    network=NETWORK,
    sequential=SEQUENTIAL,
    tasks=None,
    cost_weights=None,
    cost_norm=COST_NORM,
    cost_extent_by_task=None,
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
    train_modes=None,
    syn_mode=SYN_MODE,
    ih_off=IH_OFF,
    euler=EULER,
    pre_steady=None,
    pre_steady_iters=PRE_STEADY_ITERS,
    pre_steady_damp=PRE_STEADY_DAMP,
    fp=FP,
    pre_grad=PRE_GRAD,
    pack_overrides=None,
    model_backend=None,
    schema=None,
):
    """Create a :class:`TrainSession` from run options."""
    tl = list(tasks) if tasks is not None else list(
        training.normalize_tasks([TASK])
    )
    dev = training.active_device()
    mkw = dict(
        tasks=tl,
        cost_weights=cost_weights,
        cost_norm=expand_cost_norm(cost_norm),
        pack_overrides=pack_overrides,
        sequential=sequential,
        cost_extent_by_task=cost_extent_by_task,
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
    opts = training.make_train_opts(
        backend="network",
        network_json=network,
        dev=dev,
        ih_off=ih_off,
        euler=euler,
        pre_steady=pre_steady,
        pre_steady_iters=pre_steady_iters,
        pre_steady_damp=pre_steady_damp,
        train_modes=train_modes,
        syn_mode=syn_mode,
        fp=fp,
        pre_grad=pre_grad,
        **mkw,
    )
    return training.open_session(opts, model, schema=schema, model_backend=model_backend)


def run_training(model, nofruns, nofsteps, lrs, fname=None, outdir=None,
                 train_modes=None,
                 syn_mode=SYN_MODE,
                 ih_off=IH_OFF,
                 euler=EULER,
                 pre_steady=None,
                 pre_steady_iters=PRE_STEADY_ITERS,
                 pre_steady_damp=PRE_STEADY_DAMP,
                 network=NETWORK, sequential=SEQUENTIAL,
                 tasks=None, cost_weights=None,
                 cost_norm=COST_NORM,
                 cost_extent_by_task=None, shift_extent=SHIFT_EXTENT,
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
                 init_from=None,
                 checkpoint_interval=None,
                 make_checkpoint_callback=make_checkpoint_callback,
                 checkpoint_on_png=None):
    """Train + save artifacts (no plotting). Returns ``(fname, outdir, session, result)``.

    Plotting belongs in ``run.py``. Pass *checkpoint_on_png* from the run layer
    when ``--checkpoint-interval`` should also write PNGs.
    """
    session = build_session(
        model,
        network=network,
        sequential=sequential,
        tasks=tasks,
        cost_weights=cost_weights,
        cost_norm=cost_norm,
        cost_extent_by_task=cost_extent_by_task,
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
        train_modes=train_modes,
        syn_mode=syn_mode,
        ih_off=ih_off,
        euler=euler,
        pre_steady=pre_steady,
        pre_steady_iters=pre_steady_iters,
        pre_steady_damp=pre_steady_damp,
        pack_overrides=pack_overrides,
        model_backend=model_backend,
        schema=schema,
        fp=fp,
        pre_grad=pre_grad,
    )
    suffix = "" if model == "borst" else f"_{model}"
    fname = fname or f"training{suffix or '_with_Ih'}.npy"
    outdir = outdir or run_dir(model)

    print_train_modes(session)
    syn_mode = (session.train_opts or {}).get("syn_mode", SYN_MODE)
    print(f"device={session.device}, model={model}, syn_mode={syn_mode}, euler={session.euler}, "
          f"pre_steady={session.pre_steady}, "
          f"nofruns={nofruns}, nofsteps={nofsteps}, "
          f"lrs={lrs}, nparams={training.schema_nparams(list(session.schema))}, fname={fname}, outdir={outdir}")
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


def add_multi_spot_arguments(parser):
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

    Concrete omitted-flag values live in :mod:`param_defaults` and are
    wired here as ``default=CONST``. ``None`` only for omit-disabled flags.
    """
    parser.add_argument("--model", default=MODEL, choices=list(training.KNOWN_MODELS))
    parser.add_argument(
        "--syn-mode",
        default=SYN_MODE,
        choices=list(training.SYN_MODES),
        help="synaptic scale: per_cell (syn_sign*n_syn + type→type syn_strength_cell; default) "
             "or per_edge (syn_sign only + per-edge syn_strength_edge magnitude)",
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
             "csv/syn_strength_cell_XXXXX.csv or csv/syn_strength_edge_XXXXX.csv, and png/*_XXXXX.png "
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

    def _box_train_mode_default(name):
        tm = PARAM_BOXES[name]["train_mode"]
        if tm == "indi_named":
            return _ih_gmax_default
        return f"{tm}=all"

    _train_mode_help = (
        "indi=/shared=/fixed=/frozen= lists space-separated; 'all' in one train_mode = remainder; "
        "types or Src:Tar pairs (syn-strength-cell); init.NAMES=VAL / all=VAL overrides initial values. "
        "Example: indi=all init.L1,L2,L4,L5=200 all=10000"
    )
    _syn_strength_edge_help = (
        "only indi=all / fixed=all / frozen=all "
        "(--syn-mode per_edge; no shared= / named edges)"
    )
    _train_mode_kwargs = dict(default=None, nargs='+', metavar="MODE")
    parser.add_argument("--all-param", **_train_mode_kwargs,
                        help=f"apply train_modes to every parameter segment "
                             f"({_train_mode_help}; overridden by --ih-shape and per-param flags)")
    parser.add_argument("--a-in", **_train_mode_kwargs,
                        help=f"a_in train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('a_in')})")
    parser.add_argument("--a-out", **_train_mode_kwargs,
                        help=f"a_out train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('a_out')})")
    parser.add_argument("--bias-out", **_train_mode_kwargs,
                        help=f"hp_lp bias_out train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('bias_out')})")
    parser.add_argument("--a-gt", **_train_mode_kwargs,
                        help=f"a_gt train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('a_gt')})")
    parser.add_argument("--bias-gt", **_train_mode_kwargs,
                        help=f"bias_gt train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('bias_gt')})")
    parser.add_argument("--syn-strength-cell", **_train_mode_kwargs,
                        help=f"syn_strength_cell train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('syn_strength_cell')}; "
                             f"--syn-mode per_cell only)")
    parser.add_argument("--syn-strength-edge", **_train_mode_kwargs,
                        help=f"syn_strength_edge train_modes ({_syn_strength_edge_help}; "
                             f"default {_box_train_mode_default('syn_strength_edge')})")
    parser.add_argument("--v-th", **_train_mode_kwargs,
                        help=f"v_th train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('v_th')})")
    parser.add_argument("--ih-gmax", **_train_mode_kwargs,
                        help=f"Ih_gmax train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('Ih_gmax')})")
    parser.add_argument("--ih-gmax-off", **_train_mode_kwargs,
                        help=f"Ih_gmax_off train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('Ih_gmax_off')})")
    parser.add_argument("--ih-shape", **_train_mode_kwargs,
                        help="batch train_modes for Ih_midv/Ih_slope/tau_midv and OFF "
                             f"({_train_mode_help}; default {_box_train_mode_default('Ih_midv')})")
    parser.add_argument("--ih-midv", **_train_mode_kwargs,
                        help=f"Ih_midv train_modes (overrides --ih-shape; {_train_mode_help})")
    parser.add_argument("--ih-slope", **_train_mode_kwargs,
                        help=f"Ih_slope train_modes (overrides --ih-shape; {_train_mode_help})")
    parser.add_argument("--tau-midv", **_train_mode_kwargs,
                        help=f"tau_midv train_modes (overrides --ih-shape; {_train_mode_help})")
    parser.add_argument("--ih-midv-off", **_train_mode_kwargs,
                        help=f"Ih_midv_off train_modes (overrides --ih-shape; {_train_mode_help})")
    parser.add_argument("--ih-slope-off", **_train_mode_kwargs,
                        help=f"Ih_slope_off train_modes (overrides --ih-shape; {_train_mode_help})")
    parser.add_argument("--tau-midv-off", **_train_mode_kwargs,
                        help=f"tau_midv_off train_modes (overrides --ih-shape; {_train_mode_help})")
    parser.add_argument("--tau-lp", **_train_mode_kwargs,
                        help=f"hp_lp tau_lp train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('tau_lp')})")
    parser.add_argument("--v-rest", **_train_mode_kwargs,
                        help=f"hp_lp v_rest train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('v_rest')})")
    parser.add_argument("--tau-hp", **_train_mode_kwargs,
                        help=f"hp_lp tau_hp train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('tau_hp')})")
    parser.add_argument("--a-slow", **_train_mode_kwargs,
                        help=f"hp_lp a_slow train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('a_slow')})")
    parser.add_argument("--ih-off", default=IH_OFF,
                        choices=list(training.IH_OFF_MODES),
                        help="OFF-channel Ih: on (train Ih_gmax_off+OFF shape; default), "
                             "mirrored (OFF copies ON), off (disable OFF channel)")
    parser.add_argument(
        "--euler",
        default=EULER,
        choices=list(training.EULER_CLI),
        help="membrane Euler: im=implicit (default), ex=explicit; "
             "Ih gates always explicit",
    )
    _ss_help = "; ".join(
        f"{m}=" + "|".join(modes)
        for m, modes in PRE_STEADY_BY_MODEL.items()
    )
    parser.add_argument(
        "--pre-steady",
        nargs="+",
        default=None,
        metavar="MODEL=MODE",
        help=(
            "t=0 membrane pre steady as MODEL=MODE tokens "
            f"({_ss_help}; defaults: "
            + " ".join(f"{m}={PRE_STEADY[m]}" for m in PRE_STEADY_BY_MODEL)
            + "); hp_lp solve uses fixed iters/damp from param_defaults"
        ),
    )
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
        help="include t < t_onset in BPTT "
             f"(default: {str(PRE_GRAD).lower()}); "
             "false → no_grad pre + detach state/v at onset",
    )
    parser.add_argument(
        "--sequential",
        type=parse_bool,
        default=SEQUENTIAL,
        metavar="BOOL",
        help=f"one stimulus batch per forward (default: {str(SEQUENTIAL).lower()})",
    )
    parser.add_argument("--network", default=NETWORK, metavar="RUN",
                        help=f"connectome backend: 4_built_networks run folder under {BUILT_NETWORKS_DIR} "
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
        "--task",
        default=TASK,
        help="task name(s): spot (=spot_bright+spot_dark), moving_bar (=bright+dark), "
             "or explicit names / comma-separated list, e.g. spot,moving_bar",
    )
    parser.add_argument(
        "--cost-weight",
        default=None,
        nargs="+",
        metavar="NAME|NAME=VALUE",
        help="per-part cost weights (space-separated tokens). NAME=VALUE merges "
             "onto default 1; bare NAME (aliases: spot, moving_bar, "
             "moving_bar_bright/dark, PD/ND/DSI) zeros all parts for --task then "
             "sets those to 1. e.g. DSI (=DSI-only), DSI=1 (PD/ND stay 1), "
             "DSI PD=0.2",
    )
    parser.add_argument(
        "--shift-extent",
        type=int,
        default=SHIFT_EXTENT,
        help="spot sub-shift hex-disc radius for spot tasks in --task "
             "(n_shifts=1+3k(k+1); 0->1, 1->7, 2->19, 3->37, ...)",
    )
    parser.add_argument(
        "--spot-extent",
        type=float,
        default=SPOT_EXTENT,
        metavar="R",
        help=f"spot footprint / center-tiling radius (0.5 multiples; default {SPOT_EXTENT}); "
             "extent=1 folds RecF(2) into r=1 gt amp and defaults cost weights "
             "to 0=1 1=1/6; extent 1.5/2 keep RecF(r) and 0=1 1=1/6 2=1/6",
    )
    add_multi_spot_arguments(parser)
    parser.add_argument(
        "--spot-cost-r-w",
        default=None,
        nargs="+",
        metavar="R|R=W",
        help="spot cost weights by Euclidean r from stim hex (space-separated). "
             "Same rules as --cost-weight: R=W merges onto extent defaults; bare R "
             "zeros all known radii then sets R=1. Omit → extent default "
             "(1→0=1 1=1/6; else 0=1 1=1/6 2=1/6). Keys: 0,1,2,sqrt3. "
             "Weights only (does not change RecF gt)",
    )
    parser.add_argument(
        "--cost-extent",
        default=None,
        nargs="+",
        metavar="N|TASK=N",
        help="network cost hex-disc radius (moving-bar default: network extent - 1; "
             "network extent 0/-1 and spot default to all hexes): bare N for all "
             "--task, or per-task space-separated e.g. moving_bar_bright=0 "
             "(aliases: spot, moving_bar); -1 = all hexes; requires --network",
    )
    parser.add_argument(
        "--gt",
        default=None,
        nargs="+",
        metavar="TASK=CELLS",
        help="final gt cell keep-set per task (space-separated TASK=CELLS; "
             "CELLS comma-separated). Aliases: spot, moving_bar; moving-bar "
             "cell aliases T4, T5. e.g. --gt moving_bar=T4 spot=L1,L2,L3,L4,L5",
    )
    parser.add_argument(
        "--i-baseline",
        default=None,
        nargs="+",
        metavar="TASK=VALUE",
        help="per-task PR baseline (pA; space-separated TASK=VALUE); "
             "aliases: spot, moving_bar",
    )
    parser.add_argument(
        "--i-bright",
        default=None,
        nargs="+",
        metavar="TASK=VALUE",
        help="bright peak/step current (pA; space-separated TASK=VALUE); "
             "tasks: spot_bright, moving_bar_bright (aliases spot, moving_bar)",
    )
    parser.add_argument(
        "--i-dark",
        default=None,
        nargs="+",
        metavar="TASK=VALUE",
        help="dark peak/step current (pA; space-separated TASK=VALUE); "
             "tasks: spot_dark, moving_bar_dark (aliases spot, moving_bar)",
    )
    add_stimulus_timing_arguments(parser)
    parser.add_argument(
        "--cost-interval-ms",
        type=float,
        default=None,
        metavar="MS",
        help="spot: train on post-onset times 0, interval, 2*interval, ... "
             "through response window; omit = every post-onset t (#4)",
    )
    parser.add_argument(
        "--cost-norm",
        default=COST_NORM,
        choices=list(COST_NORMS),
        help="waveform MSE normalization: gt_power = 100*SSE/Σw(a_gt·gt)²; "
             f"a_gt2 = SSE/a_gt² (default: {COST_NORM})",
    )


def add_stimulus_timing_arguments(
    parser,
    *,
    default_ms_pre=MS_PRE,
    default_ms_response=MS_RESPONSE,
    default_ms_post=MS_POST,
    default_ms_pulse=MS_PULSE,
    default_delta_ms=DELTA_MS,
):
    """Register ``--ms-pre`` / ``--ms-response`` / ``--ms-post`` / ``--ms-pulse`` / ``--delta-ms``.

    Train uses ``param_defaults`` values. Plot / analyze pass ``None`` so
    omitted flags keep the run's ``train_opts.json``.
    """
    if default_ms_pre is None:
        pre_help = (
            "override pre-stimulus baseline in ms (spot + moving_bar; "
            "keep train if omitted)"
        )
    else:
        pre_help = (
            f"pre-stimulus baseline duration in ms (default: {default_ms_pre}; "
            "t_onset = ms_to_t(ms_pre); "
            "n_t = ms_to_t(ms_pre)+ms_to_t(ms_response)+ms_to_t(ms_post)+1)"
        )
    if default_ms_response is None:
        response_help = (
            "override spot post-onset cost/gt window in ms (keep train if omitted)"
        )
    else:
        response_help = (
            f"spot: post-onset cost/gt window in ms (default: {default_ms_response}; "
            "excludes ms_post)"
        )
    if default_ms_post is None:
        post_help = (
            "override spot forward-only tail after response in ms "
            "(not in gt/cost; keep train if omitted)"
        )
    else:
        post_help = (
            f"spot: forward-only tail after response in ms (default: {default_ms_post}; "
            "not in gt/cost)"
        )
    if default_ms_pulse is None:
        pulse_help = (
            "override spot pulse width in ms "
            "(keep train if omitted; raises ms_response if shorter)"
        )
    else:
        pulse_help = (
            "spot: bright/dark PR pulse duration in ms from onset "
            f"(default: {default_ms_pulse}; raises ms_response if shorter)"
        )
    if default_delta_ms is None:
        delta_help = (
            "override simulation / stimulus time step in ms "
            "(writes delta_ms into all stimulus opts; keep train if omitted)"
        )
    else:
        delta_help = (
            f"simulation / stimulus time step in ms (default: {default_delta_ms}; "
            "writes delta_ms into all stimulus opts)"
        )
    parser.add_argument(
        "--ms-pre",
        type=float,
        default=default_ms_pre,
        metavar="MS",
        help=pre_help,
    )
    parser.add_argument(
        "--ms-response",
        type=float,
        default=default_ms_response,
        metavar="MS",
        help=response_help,
    )
    parser.add_argument(
        "--ms-post",
        type=float,
        default=default_ms_post,
        metavar="MS",
        help=post_help,
    )
    parser.add_argument(
        "--ms-pulse",
        type=float,
        default=default_ms_pulse,
        metavar="MS",
        help=pulse_help,
    )
    parser.add_argument(
        "--delta-ms",
        type=float,
        default=default_delta_ms,
        metavar="MS",
        help=delta_help,
    )


def apply_train_opts_timing(
    opts,
    *,
    ms_pre=None,
    ms_response=None,
    ms_post=None,
    ms_pulse=None,
    delta_ms=None,
):
    """Merge timing overrides into train-opts spot/bar stimulus dicts.

    Spot opts go through :func:`task.spot.input.apply_spot_timing_overrides`
    (normalize + drop derived ``t_onset``/``n_t``). Returns timing keys that
    changed on spot opts (for filename suffixes); bar-only ``ms_pre`` /
    ``delta_ms`` changes are included when no spot opts are present.
    """
    from task.spot.input import apply_spot_timing_overrides

    changed = {}
    for key in ("spot_bright_stimulus_opts", "spot_dark_stimulus_opts"):
        so = opts.get(key)
        if so is None:
            continue
        changed = apply_spot_timing_overrides(
            so,
            ms_pre=ms_pre,
            ms_response=ms_response,
            ms_post=ms_post,
            ms_pulse=ms_pulse,
            delta_ms=delta_ms,
        )
    if ms_pre is not None or delta_ms is not None:
        for key in (
            "moving_bar_bright_stimulus_opts",
            "moving_bar_dark_stimulus_opts",
        ):
            so = opts.get(key)
            if so is None:
                continue
            before_pre = so.get("ms_pre")
            before_dt = so.get("delta_ms")
            if ms_pre is not None:
                so["ms_pre"] = float(ms_pre)
            if delta_ms is not None:
                so["delta_ms"] = float(delta_ms)
            so.pop("t_onset", None)
            so.pop("n_t", None)
            if not changed:
                if ms_pre is not None and (
                    before_pre is None or float(before_pre) != float(ms_pre)
                ):
                    changed["ms_pre"] = float(ms_pre)
                if delta_ms is not None and (
                    before_dt is None or float(before_dt) != float(delta_ms)
                ):
                    changed["delta_ms"] = float(delta_ms)
    return changed


def stimulus_timing_kwargs_from_args(args):
    """Map parsed timing flags to kwargs for :func:`figure.plot_run.maybe_override_stimulus_timing`."""
    return dict(
        ms_pre=args.ms_pre,
        ms_response=args.ms_response,
        ms_post=args.ms_post,
        ms_pulse=args.ms_pulse,
        delta_ms=args.delta_ms,
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


def parse_kv_tokens(tokens, cast=str):
    """Parse space-separated ``NAME=VALUE`` tokens (``nargs='+'``)."""
    if not tokens:
        return {}
    out = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"expected NAME=VALUE, got {tok!r}")
        name, val = tok.split("=", 1)
        out[name.strip()] = cast(val.strip())
    return out


def parse_tasks(text):
    """Parse comma-separated training tasks (with alias expansion)."""
    return training.normalize_tasks(parse_comma_list(text))


def parse_cost_extent(tokens):
    """Parse ``--cost-extent``: optional bare ``N`` plus ``TASK=N`` tokens."""
    if not tokens:
        return None, {}
    default = None
    by_task = {}
    for tok in tokens:
        if "=" in tok:
            name, val = tok.split("=", 1)
            by_task[name.strip()] = int(val.strip())
        else:
            if default is not None:
                raise ValueError("only one bare extent allowed in --cost-extent")
            default = int(tok)
    return default, by_task


def parse_gt(tokens):
    """Parse ``--gt`` space-separated ``TASK=CELLS`` tokens (CELLS comma-separated).

    Values are the final keep-set (not a remove list). Returns ``None`` when
    omitted; otherwise a concrete-task → type-list map.
    """
    if tokens is None:
        return None
    raw = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"expected TASK=CELLS, got {tok!r}")
        name, val = tok.split("=", 1)
        name = name.strip()
        types = parse_comma_list(val)
        if not types:
            raise ValueError(f"--gt {name}=... must list at least one type")
        raw[name] = types
    return training.resolve_gt_cells_by_task(raw)


def parse_cost_weight(tokens, tasks):
    """Parse ``--cost-weight``: bare alias exclusive, ``NAME=VALUE`` merge.

    Bare tokens (aliases or concrete part keys) zero every cost part for
    ``tasks``, then set those names to ``1``. Explicit ``NAME=VALUE``
    always applied last (merge onto defaults / exclusive map). Empty → ``{}``
    (runtime default weight 1).
    """
    if not tokens:
        return {}
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
        weights = {key: 0.0 for key in training.session_cost_part_keys(tasks)}
        weights.update(training.expand_cost_weight_dict({name: 1.0 for name in bare}))
    weights.update(training.expand_cost_weight_dict(explicit))
    return weights


def _train_mode_cli_text(parts):
    """Join space-separated train_mode CLI tokens into one parse string."""
    if parts is None:
        return None
    return ' '.join(parts)


def _train_mode_cli_map(args):
    """Build ``{seg_name: {indi/shared/fixed/frozen: tokens}}`` from train_mode CLI flags.

    Precedence: ``--all-param`` → ``--ih-shape`` → per-param flags.
    Omitted segments keep schema defaults (not listed here).
    """
    syn_mode = training.normalize_syn_mode(getattr(args, "syn_mode", SYN_MODE))
    syn_cell_text = _train_mode_cli_text(getattr(args, "syn_strength_cell", None))
    syn_edge_text = _train_mode_cli_text(getattr(args, "syn_strength_edge", None))
    if syn_mode == "per_edge" and syn_cell_text is not None:
        raise ValueError("--syn-strength-cell requires --syn-mode per_cell")
    if syn_mode == "per_cell" and syn_edge_text is not None:
        raise ValueError("--syn-strength-edge requires --syn-mode per_edge")
    texts = {}
    all_param = _train_mode_cli_text(getattr(args, "all_param", None))
    if all_param is not None:
        for name in training.ALL_PARAM_NAMES:
            if name == "syn_strength_cell" and syn_mode != "per_cell":
                continue
            if name == "syn_strength_edge" and syn_mode != "per_edge":
                continue
            texts[name] = all_param
    shape_text = _train_mode_cli_text(getattr(args, "ih_shape", None))
    if shape_text is not None:
        for name in training.IH_SHAPE_PARAM_NAMES:
            texts[name] = shape_text
    per_param = {
        "a_in": _train_mode_cli_text(getattr(args, "a_in", None)),
        "a_out": _train_mode_cli_text(getattr(args, "a_out", None)),
        "bias_out": _train_mode_cli_text(getattr(args, "bias_out", None)),
        "a_gt": _train_mode_cli_text(getattr(args, "a_gt", None)),
        "bias_gt": _train_mode_cli_text(getattr(args, "bias_gt", None)),
        "syn_strength_cell": syn_cell_text,
        "syn_strength_edge": syn_edge_text,
        "v_th": _train_mode_cli_text(getattr(args, "v_th", None)),
        "Ih_gmax": _train_mode_cli_text(getattr(args, "ih_gmax", None)),
        "Ih_gmax_off": _train_mode_cli_text(getattr(args, "ih_gmax_off", None)),
        "Ih_midv": _train_mode_cli_text(getattr(args, "ih_midv", None)),
        "Ih_slope": _train_mode_cli_text(getattr(args, "ih_slope", None)),
        "tau_midv": _train_mode_cli_text(getattr(args, "tau_midv", None)),
        "Ih_midv_off": _train_mode_cli_text(getattr(args, "ih_midv_off", None)),
        "Ih_slope_off": _train_mode_cli_text(getattr(args, "ih_slope_off", None)),
        "tau_midv_off": _train_mode_cli_text(getattr(args, "tau_midv_off", None)),
        "tau_lp": _train_mode_cli_text(getattr(args, "tau_lp", None)),
        "v_rest": _train_mode_cli_text(getattr(args, "v_rest", None)),
        "tau_hp": _train_mode_cli_text(getattr(args, "tau_hp", None)),
        "a_slow": _train_mode_cli_text(getattr(args, "a_slow", None)),
    }
    for name, text in per_param.items():
        if text is not None:
            texts[name] = text
    out = {name: training.parse_train_mode_text(text) for name, text in texts.items()}
    if "syn_strength_edge" in out:
        training.validate_syn_strength_edge_train_mode(out["syn_strength_edge"])
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
    train_modes = _train_mode_cli_map(args) or None
    tasks = parse_tasks(args.task)
    cost_weights = parse_cost_weight(args.cost_weight, tasks)
    default_extent, extent_kv = parse_cost_extent(args.cost_extent)
    cost_extent_by_task = training.resolve_cost_extent_by_task(
        tasks, default_extent, extent_kv,
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
    spot_cost_radius_weight = parse_spot_cost_r_w_tokens(
        args.spot_cost_r_w,
        default_weights=default_spot_cost_radius_weight(
            spot_extent,
            weights=SPOT_COST_RADIUS_WEIGHT,
            weights_extent1=SPOT_COST_RADIUS_WEIGHT_EXTENT1,
        ),
        spot_cost_radii=SPOT_COST_RADII,
        aliases=SPOT_COST_RADIUS_KEY_ALIASES,
    )
    multi_spot = bool(args.multi_spot)
    fully_inside = bool(args.fully_inside)
    ms_pre = float(args.ms_pre)
    ms_response = float(args.ms_response)
    ms_post = float(args.ms_post)
    ms_pulse = float(args.ms_pulse)
    delta_ms = float(args.delta_ms)
    if delta_ms <= 0:
        raise ValueError("--delta-ms must be > 0")
    if ms_post < 0:
        raise ValueError("--ms-post must be >= 0")
    multi_bar = bool(args.multi_bar)
    _timing = {
        "ms_pre": ms_pre,
        "ms_response": ms_response,
        "ms_post": ms_post,
        "delta_ms": delta_ms,
        "ms_pulse": ms_pulse,
    }
    moving_bar_bright_stimulus_opts = {
        "multi_bar": multi_bar,
        "ms_pre": ms_pre,
        "delta_ms": delta_ms,
    }
    moving_bar_dark_stimulus_opts = {
        "multi_bar": multi_bar,
        "ms_pre": ms_pre,
        "delta_ms": delta_ms,
    }
    spot_bright_stimulus_opts = dict(_timing)
    spot_dark_stimulus_opts = dict(_timing)
    if args.cost_interval_ms is not None:
        if float(args.cost_interval_ms) <= 0:
            raise ValueError("--cost-interval-ms must be > 0")
        for _o in (spot_bright_stimulus_opts, spot_dark_stimulus_opts):
            _o["cost_interval_ms"] = float(args.cost_interval_ms)
    gt_by_task = parse_gt(args.gt)
    if gt_by_task:
        _gt_opts = {
            "moving_bar_bright": moving_bar_bright_stimulus_opts,
            "moving_bar_dark": moving_bar_dark_stimulus_opts,
            "spot_bright": spot_bright_stimulus_opts,
            "spot_dark": spot_dark_stimulus_opts,
        }
        for _tname, _types in gt_by_task.items():
            _gt_opts[_tname]["gt_cells"] = list(_types)
    i_cli = training.build_i_cli_by_task({
        "i_baseline": parse_kv_tokens(args.i_baseline, float),
        "i_bright": parse_kv_tokens(args.i_bright, float),
        "i_dark": parse_kv_tokens(args.i_dark, float),
    })
    lrs = [float(x) for x in parse_comma_list(args.lrs)]
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
        train_modes=train_modes,
        syn_mode=training.normalize_syn_mode(args.syn_mode),
        network=args.network,
        tasks=tasks,
        cost_weights=cost_weights,
        cost_norm=expand_cost_norm(args.cost_norm),
        cost_extent_by_task=cost_extent_by_task,
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
        euler=args.euler,
        pre_steady=expand_pre_steady_dict(
            parse_kv_tokens(args.pre_steady, str),
            defaults=PRE_STEADY,
        ),
        pre_steady_iters=PRE_STEADY_ITERS,
        pre_steady_damp=PRE_STEADY_DAMP,
        fp=fp,
        pre_grad=bool(args.pre_grad),
        sequential=bool(args.sequential),
        init_from=init_from,
        checkpoint_interval=args.checkpoint_interval,
    )



def main():
    raise SystemExit(
        "training.implement is the pure training library; use simulation/run.py:\n"
        "  ../.venv/bin/python run.py --help"
    )


if __name__ == "__main__":
    main()
