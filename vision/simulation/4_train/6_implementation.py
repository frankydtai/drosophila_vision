"""Train implementation: integrate session, optimization, and persistence.

Wires the pieces of one train into a path that runs through before the top-level
``simulation/run.py`` driver (which may also plot). CLI parsing lives in
:mod:`train.cli`.

Import-safe: does NOT parse argv or touch CUDA.
"""
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
from path import (
    BUILT_NETWORKS_DIR,
    resolve_network_json,
)
from config import (
    MODEL,
    MOVING_BAR_INPUT_GEO,
    NETWORK_PATH,
    NEURON_FORWARD,
    NEURON_SCHEMA,
    SPOT_INPUT_GEO,
    SPREAD_PACK,
    TRAIN_CONFIG,
    TRAIN_OPTIMIZATION,
    TRAIN_SESSION,
)
from task.spread.sti_spec import t_sti_end, resolve_sti_timing
from neuron.schema import param_from_entry
from train import do_many_runs
import train
from train.config import (
    COST_NORMS,
    PARAM_CSV,
    SYN_STRENGTH_CELL_CSV,
    SYN_STRENGTH_EDGE_CSV,
    expand_cost_norm,
    expand_pre_steady,
    run_data_dir,
)


def run_dir(model, root=None, parent=None, run=None):
    """``<PARAMETER_DIR>/<model>/<run>`` (or under *parent*)."""
    from train.config import PARAMETER_DIR
    if parent is None:
        root = str(PARAMETER_DIR) if root is None else root
        parent = os.path.join(root, model)
    if run is None:
        job_id = os.environ.get('SLURM_JOB_ID')
        run = f'run_{job_id}' if job_id else time.strftime('run_%m%d_%H%M%S')
    return os.path.join(parent, run)


def resolve_run_dir(path):
    """Resolve a run folder under ``PARAMETER_DIR`` or an absolute path."""
    from train.config import PARAMETER_DIR
    pth = Path(path).expanduser()
    outdir = pth.resolve() if pth.is_absolute() else (PARAMETER_DIR / pth).resolve()
    if not outdir.is_dir():
        raise SystemExit(f'run folder not found: {path!r} -> {outdir}')
    return str(outdir)


def checkpoint_iter_token(iter):
    return f'{int(iter):05d}'


def cell_labels(session):
    return np.asarray(session.connectome.cells)


def decompose_params(z, session):
    """Return per-cell param arrays for one parameter vector.

    This powers the ``param.csv`` table. For CSV output we intentionally do
    *not* emit separate "global" scalar fields (shared-only means), because
    they can collide with per-cell csv fields and become redundant.
    """
    n = session.connectome.n_cell
    schema = train.schema_copy(session.schema)
    decoded = train.node_vals_from_z(z, schema)
    out = {}
    for param, spec in schema.items():
        if spec["kind"] in ("edge_pair", "edge"):
            continue
        if spec.get("radii") is not None:
            continue  # e.g. a_sti_radius (per-radius, not per-cell)
        arr = np.asarray(decoded[param], dtype=np.float64).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"{param}: node w {arr.shape[0]} != n_cell {n}")
        out[param] = arr
    return out


def v_spot_markers_by_cell(z, session):
    """Per-cell-type mean ``v`` markers; add ``v_ca``/``ca`` for ``filter=ca``.

    * ``v_pre``: ``t=0`` after ``pre_steady`` (``v_rows[0]``).
    * ``v_onset``: ``t=t_onset`` (``v_rows[t_onset]``), spot onset / end of pre.
    * ``v_sti_end``: ``t=t_sti_end`` inclusive last on sample
      (``t_sti_end``).
    * ``delta_v``: ``v_sti_end - v_onset``.

    Uses ``session.primary_pack`` sti (``i_sti`` + ``pack_t_onset``).
    """
    schema = train.schema_copy(session.schema)
    z = torch.as_tensor(z, dtype=session.sim_dtype, device=session.device)
    params = train.override_val_from(
        train.assign_params(z, schema, session.connectome), session,
    )
    pack = session.primary_pack
    i_sti = session.pack_i_sti(pack)
    t_onset = train.pack_t_onset(pack)
    train_opts = session.train_opts or {}
    use_ca = str(train_opts.get("filter", "none")) == "ca"
    with torch.no_grad():
        v = train.forward_v(
            session, params, i_sti.unsqueeze(0) if i_sti.dim() == 2 else i_sti, pack=pack,
        )
        v_ca = train.v_ca_from_v(v, params, session) if use_ca else None
        ca = train.ca_from_v_ca(v_ca, params, session, t_onset=t_onset) if use_ca else None
    # v: (B, T, N)
    n_t = int(v.shape[1])
    if t_onset < 0 or t_onset >= n_t:
        raise ValueError(f"t_onset={t_onset} out of range for forward T={n_t}")
    opts = (session.train_opts or {}).get(f"{pack.task}_sti_opts") or {}
    timing = resolve_sti_timing(opts)
    t_end = t_sti_end(
        t_onset, n_t, timing.ms_sti,
        delta_ms=timing.delta_ms,
    )
    if t_end < t_onset or t_end >= n_t:
        raise ValueError(
            f"t_sti_end={t_end} out of range for t_onset={t_onset}, T={n_t}"
        )
    v_pre_n = v[0, 0].detach().cpu().numpy()
    v_onset_n = v[0, t_onset].detach().cpu().numpy()
    v_sti_end_n = v[0, t_end].detach().cpu().numpy()
    node_cells = session.connectome.conn.node_cells.detach().cpu().numpy()
    n_cell = int(session.connectome.n_cell)
    v_pre = np.empty(n_cell, dtype=np.float64)
    v_onset = np.empty(n_cell, dtype=np.float64)
    v_sti_end = np.empty(n_cell, dtype=np.float64)
    delta_v = np.empty(n_cell, dtype=np.float64)
    for cell_idx in range(n_cell):
        cell_node_mask = node_cells == cell_idx
        if not np.any(cell_node_mask):
            v_pre[cell_idx] = np.nan
            v_onset[cell_idx] = np.nan
            v_sti_end[cell_idx] = np.nan
            delta_v[cell_idx] = np.nan
        else:
            v_pre[cell_idx] = float(v_pre_n[cell_node_mask].mean())
            v_onset[cell_idx] = float(v_onset_n[cell_node_mask].mean())
            v_sti_end[cell_idx] = float(v_sti_end_n[cell_node_mask].mean())
            delta_v[cell_idx] = v_sti_end[cell_idx] - v_onset[cell_idx]
    out = {
        "v_pre": v_pre,
        "v_onset": v_onset,
        "v_sti_end": v_sti_end,
        "delta_v": delta_v,
    }
    if use_ca:
        v_ca_pre_n = v_ca[0, 0].detach().cpu().numpy()
        v_ca_onset_n = v_ca[0, t_onset].detach().cpu().numpy()
        v_ca_sti_end_n = v_ca[0, t_end].detach().cpu().numpy()
        v_ca_pre = np.empty(n_cell, dtype=np.float64)
        v_ca_onset = np.empty(n_cell, dtype=np.float64)
        v_ca_sti_end = np.empty(n_cell, dtype=np.float64)
        delta_v_ca = np.empty(n_cell, dtype=np.float64)
        for cell_idx in range(n_cell):
            cell_node_mask = node_cells == cell_idx
            if not np.any(cell_node_mask):
                v_ca_pre[cell_idx] = np.nan
                v_ca_onset[cell_idx] = np.nan
                v_ca_sti_end[cell_idx] = np.nan
                delta_v_ca[cell_idx] = np.nan
            else:
                v_ca_pre[cell_idx] = float(v_ca_pre_n[cell_node_mask].mean())
                v_ca_onset[cell_idx] = float(v_ca_onset_n[cell_node_mask].mean())
                v_ca_sti_end[cell_idx] = float(v_ca_sti_end_n[cell_node_mask].mean())
                delta_v_ca[cell_idx] = v_ca_sti_end[cell_idx] - v_ca_onset[cell_idx]
        out.update(
            v_ca_pre=v_ca_pre,
            v_ca_onset=v_ca_onset,
            v_ca_sti_end=v_ca_sti_end,
            delta_v_ca=delta_v_ca,
        )
        ca_pre_n = ca[0, 0].detach().cpu().numpy()
        ca_onset_n = ca[0, t_onset].detach().cpu().numpy()
        ca_sti_end_n = ca[0, t_end].detach().cpu().numpy()
        ca_pre = np.empty(n_cell, dtype=np.float64)
        ca_onset = np.empty(n_cell, dtype=np.float64)
        ca_sti_end = np.empty(n_cell, dtype=np.float64)
        delta_ca = np.empty(n_cell, dtype=np.float64)
        for cell_idx in range(n_cell):
            cell_node_mask = node_cells == cell_idx
            if not np.any(cell_node_mask):
                ca_pre[cell_idx] = np.nan
                ca_onset[cell_idx] = np.nan
                ca_sti_end[cell_idx] = np.nan
                delta_ca[cell_idx] = np.nan
            else:
                ca_pre[cell_idx] = float(ca_pre_n[cell_node_mask].mean())
                ca_onset[cell_idx] = float(ca_onset_n[cell_node_mask].mean())
                ca_sti_end[cell_idx] = float(ca_sti_end_n[cell_node_mask].mean())
                delta_ca[cell_idx] = ca_sti_end[cell_idx] - ca_onset[cell_idx]
        out.update(
            ca_pre=ca_pre,
            ca_onset=ca_onset,
            ca_sti_end=ca_sti_end,
            delta_ca=delta_ca,
        )
    if train.val_from_enabled(train_opts, "bias_gt"):
        if use_ca:
            bias = train.bias_gt_from_onset_trace(
                ca, t_onset, session,
            ).detach().cpu().numpy()
        else:
            lo = param_from_entry("bias_gt", "lo", NEURON_SCHEMA['params'])
            hi = param_from_entry("bias_gt", "hi", NEURON_SCHEMA['params'])
            bias = np.clip(v_onset, lo, hi)
        out["bias_gt"] = np.asarray(bias, dtype=np.float64)
    return out


def save_param_table(z, session, table_path):
    node_vals = decompose_params(z, session)
    markers = v_spot_markers_by_cell(z, session)
    bias_gt = markers.pop("bias_gt", None)
    node_vals.update(markers)
    opts = session.train_opts or {}
    if train.val_from_enabled(opts, "v_th_ca") and "v_th" in node_vals and "v_th_ca" in node_vals:
        node_vals["v_th_ca"] = np.asarray(node_vals["v_th"], dtype=np.float64).copy()
    if train.val_from_enabled(opts, "a_ca") and "a_out" in node_vals and "a_ca" in node_vals:
        node_vals["a_ca"] = np.asarray(node_vals["a_out"], dtype=np.float64).copy()
    if bias_gt is not None:
        node_vals["bias_gt"] = bias_gt
    cells_labels = cell_labels(session)
    params = list(node_vals.keys())
    n = session.connectome.n_cell
    cell_idx = dict(zip(cells_labels, range(n)))
    with open(table_path, "w") as f:
        f.write("cell_idx,cell," + ",".join(params) + "\n")
        for cell in cells_labels:
            field_vals = [
                "%.6f" % node_vals[param][cell_idx[cell]] for param in params
            ]
            f.write(
                "%d,%s," % (cell_idx[cell], cell) + ",".join(field_vals) + "\n"
            )
    return table_path


def save_syn_strength_cell_table(z, session, table_path):
    """Write edge-pair ``syn_strength_cell`` as source×target matrix CSV.

    Rows = source types, fields = target types. Absent connectome pairs are blank.
    """
    schema = train.schema_copy(session.schema)
    spec = schema.get("syn_strength_cell")
    if spec is None or spec["kind"] != "edge_pair":
        return None
    node_vals = train.node_vals_from_z(z, schema)
    arr = np.asarray(node_vals["syn_strength_cell"], dtype=np.float64).reshape(-1)
    cells = [str(cell) for cell in cell_labels(session)]
    pairs = list(session.connectome.conn.pairs)
    if arr.shape[0] != len(pairs):
        raise ValueError(
            f"syn_strength_cell length {arr.shape[0]} != n_pair {len(pairs)}"
        )
    mat = {(int(source), int(target)): float(v) for (source, target), v in zip(pairs, arr)}
    n = len(cells)
    with open(table_path, "w") as f:
        f.write("," + ",".join(cells) + "\n")
        for source, src in enumerate(cells):
            fields = [
                ("%.6f" % mat[(source, target)]) if (source, target) in mat else ""
                for target in range(n)
            ]
            f.write(src + "," + ",".join(fields) + "\n")
    return table_path


def save_syn_strength_edge_table(z, session, table_path):
    """Write per-edge ``syn_strength_edge`` CSV (network edge order)."""
    schema = train.schema_copy(session.schema)
    spec = schema.get("syn_strength_edge")
    if spec is None or spec["kind"] != "edge":
        return None
    node_vals = train.node_vals_from_z(z, schema)
    arr = np.asarray(node_vals["syn_strength_edge"], dtype=np.float64).reshape(-1)
    conn = session.connectome.conn
    if arr.shape[0] != conn.n_edge:
        raise ValueError(
            f"syn_strength_edge length {arr.shape[0]} != n_edge {conn.n_edge}"
        )
    cells = [str(cell) for cell in cell_labels(session)]
    src = conn.source_idxs.detach().cpu().numpy()
    tar = conn.target_idxs.detach().cpu().numpy()
    node_cells = conn.node_cells.detach().cpu().numpy()
    syn_sign = torch.sign(conn.w_signed).detach().cpu().numpy()
    with open(table_path, "w") as f:
        f.write(
            "edge_idx,src_node,tar_node,source_cell,target_cell,syn_sign,syn_strength_edge\n",
        )
        for edge in range(conn.n_edge):
            si, ti = int(src[edge]), int(tar[edge])
            f.write(
                "%d,%d,%d,%s,%s,%.0f,%.6f\n"
                % (
                    edge, si, ti,
                    cells[int(node_cells[si])], cells[int(node_cells[ti])],
                    float(syn_sign[edge]), float(arr[edge]),
                )
            )
    return table_path


def save_syn_table(z, session, outdir_or_path, *, token=None):
    """Write ``syn_strength_cell.csv`` or ``syn_strength_edge.csv`` for the active syn mode."""
    if token is None:
        cell_filename, edge_filename = SYN_STRENGTH_CELL_CSV, SYN_STRENGTH_EDGE_CSV
    else:
        cell_filename = f"syn_strength_cell_{token}.csv"
        edge_filename = f"syn_strength_edge_{token}.csv"
    cell_path = save_syn_strength_cell_table(
        z, session, os.path.join(outdir_or_path, cell_filename),
    )
    edge_path = save_syn_strength_edge_table(
        z, session, os.path.join(outdir_or_path, edge_filename),
    )
    return cell_path or edge_path


def best_param_path(outdir):
    return os.path.join(run_data_dir(outdir), 'best_param.npz')


def best_adam_path(outdir):
    return os.path.join(run_data_dir(outdir), 'best_adam.npz')


def _data_file(outdir, filename):
    return os.path.join(run_data_dir(outdir), filename)


def save_node_vals(outdir, z, session, filename):
    """Write per-param full-w node vals to ``data/<filename>``."""
    schema = train.schema_copy(session.schema)
    node_vals = train.node_vals_from_z(z, schema)
    cells = np.asarray(train.cells_from_connectome(session.connectome), dtype=object)
    payload = {k: np.asarray(v, dtype=np.float64) for k, v in node_vals.items()}
    payload['cells'] = cells
    if any(spec['kind'] == 'edge_pair' for spec in schema.values()):
        payload['pairs'] = np.asarray(train.pairs_from_connectome(session.connectome), dtype=object)
    os.makedirs(run_data_dir(outdir), exist_ok=True)
    np.savez(os.path.join(run_data_dir(outdir), filename), **payload)


def save_adam(outdir, exp_avg, exp_avg_sq, iter, session, filename):
    """Write per-param adam m/v (z-space) to ``data/<filename>``."""
    schema = train.schema_copy(session.schema)
    adams_m, adams_v = train.adams_from_z(exp_avg, exp_avg_sq, schema)
    cells = np.asarray(train.cells_from_connectome(session.connectome), dtype=object)
    payload = {'iter': np.asarray(int(iter), dtype=np.int64)}
    payload['cells'] = cells
    if any(spec['kind'] == 'edge_pair' for spec in schema.values()):
        payload['pairs'] = np.asarray(train.pairs_from_connectome(session.connectome), dtype=object)
    for param, arr in adams_m.items():
        payload[f'm_{param}'] = np.asarray(arr, dtype=np.float64)
    for param, arr in adams_v.items():
        payload[f'v_{param}'] = np.asarray(arr, dtype=np.float64)
    os.makedirs(run_data_dir(outdir), exist_ok=True)
    np.savez(os.path.join(run_data_dir(outdir), filename), **payload)


def save_best_param(outdir, z, session):
    """Write per-param full-w node vals to ``data/best_param.npz``."""
    save_node_vals(outdir, z, session, 'best_param.npz')


def _checkpoint_data_filename(kind, iter, run_i=0, n_run=1):
    suffix = '' if n_run == 1 else f'_run{run_i}'
    return f'best_{kind}_iter_{checkpoint_iter_token(iter)}{suffix}.npz'


def save_checkpoint_csv(outdir, iter, z_best, session):
    token = checkpoint_iter_token(iter)
    csv_dir = os.path.join(outdir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    param_path = os.path.join(csv_dir, f'param_{token}.csv')
    save_param_table(z_best, session, param_path)
    syn_path = save_syn_table(z_best, session, csv_dir, token=token)
    print(f'wrote checkpoint csv: {param_path}')
    if syn_path is not None:
        print(f'wrote checkpoint csv: {syn_path}')


def build_checkpoint_callback(outdir, session, *, run_i=0, n_run=1, on_png=None):
    """Write interval-best npz/csv; optional *on_png* for plot layer (from ``run.py``)."""

    def on_interval_best(iter, z_best, cost_best, adam=None):
        filename = _checkpoint_data_filename('param', iter, run_i=run_i, n_run=n_run)
        save_node_vals(outdir, z_best, session, filename)
        if adam is not None:
            adam_filename = _checkpoint_data_filename(
                'adam', iter, run_i=run_i, n_run=n_run,
            )
            save_adam(
                outdir,
                adam['exp_avg'].detach().cpu().numpy(),
                adam['exp_avg_sq'].detach().cpu().numpy(),
                adam['iter'],
                session,
                adam_filename,
            )
            print(f'wrote checkpoint {adam_filename}')
        save_checkpoint_csv(outdir, iter, z_best, session)
        if on_png is not None:
            on_png(outdir, iter, z_best, cost_best, session)
        print(f'wrote checkpoint {filename} (cost={cost_best:.4f})')
    return on_interval_best


def load_best_node_vals(outdir):
    """Load ``data/best_param.npz`` → (node_vals, cells, pairs|None)."""
    fp = best_param_path(outdir)
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)
    with np.load(fp, allow_pickle=True) as d:
        cells = [str(x) for x in d['cells'].tolist()]
        pairs = None
        if 'pairs' in d.files:
            pairs = [str(x) for x in d['pairs'].tolist()]
        node_vals = {
            k: np.asarray(d[k], dtype=np.float64)
            for k in d.files
            if k not in ('cells', 'pairs')
        }
    return node_vals, cells, pairs


def load_best_adam(outdir):
    """Load ``data/best_adam.npz`` → (adams_m, adams_v, adam_iter, cells, pairs)."""
    fp = best_adam_path(outdir)
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)
    with np.load(fp, allow_pickle=True) as d:
        cells = [str(x) for x in d['cells'].tolist()]
        pairs = None
        if 'pairs' in d.files:
            pairs = [str(x) for x in d['pairs'].tolist()]
        if 'iter' not in d.files:
            raise ValueError(f'{fp}: missing iter')
        adam_iter = int(np.asarray(d['iter']).reshape(-1)[0])
        adams_m = {}
        adams_v = {}
        for k in d.files:
            if k.startswith('m_'):
                adams_m[k[2:]] = np.asarray(d[k], dtype=np.float64)
            elif k.startswith('v_'):
                adams_v[k[2:]] = np.asarray(d[k], dtype=np.float64)
    return adams_m, adams_v, adam_iter, cells, pairs


def load_best_param(outdir, session):
    """Load best params as 1-D z for *session* (remap from per-param npz)."""
    node_vals, cells, pairs = load_best_node_vals(outdir)
    schema = train.schema_with_param_carry(
        train.schema_copy(session.schema),
        train.remap_node_vals(
            node_vals, cells, pairs, train.schema_copy(session.schema), session.connectome,
        ),
    )
    remapped = train.remap_node_vals(
        node_vals, cells, pairs, schema, session.connectome,
    )
    z = train.z_from_node_vals(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )
    return z.detach().cpu().numpy().astype(np.float64)


def save_best_data(outdir, session, run_params, final_costs, adam=None):
    """Write ``best_param.npz``, ``param.csv``, and syn/edge CSV for ``argmin(final_costs)``.

    *adam* is the best-run bag ``{exp_avg, exp_avg_sq, iter}`` (required for a
    fresh write from train; omit when regenerating tables from saved params only).
    """
    run_params = np.atleast_2d(run_params)
    final_costs = np.asarray(final_costs, dtype=np.float64)
    run_i = int(np.argmin(final_costs))
    best = run_params[run_i]
    os.makedirs(run_data_dir(outdir), exist_ok=True)
    z_best = torch.tensor(best, dtype=session.sim_dtype, device=session.device)
    save_best_param(outdir, z_best, session)
    if adam is not None:
        save_adam(
            outdir, adam['exp_avg'], adam['exp_avg_sq'], adam['iter'], session,
            'best_adam.npz',
        )
        print(f"wrote {best_adam_path(outdir)} (best run #{run_i})")
    table_path = os.path.join(outdir, PARAM_CSV)
    save_param_table(z_best, session, table_path)
    print("wrote table: %s (best run #%d, cost=%.4f)" % (
        table_path, run_i, final_costs[run_i]))
    syn_path = save_syn_table(z_best, session, outdir)
    if syn_path is not None:
        print("wrote table: %s" % syn_path)
    return best


def load_stored_costs(outdir):
    """Load ``costs.npy``, ``best_costs.npy``, and per-part npz when present."""
    final_costs = None
    cost_curve = None
    costs_by_part = None
    final_costs_by_part = None
    fp = _data_file(outdir, 'costs.npy')
    if os.path.isfile(fp):
        final_costs = np.load(fp)
    cp = _data_file(outdir, 'best_costs.npy')
    if os.path.isfile(cp):
        cost_curve = np.load(cp)
    cbt = _data_file(outdir, 'best_costs_by_part.npz')
    if os.path.isfile(cbt):
        with np.load(cbt) as d:
            costs_by_part = {k: np.asarray(d[k]) for k in d.files}
    fbt = _data_file(outdir, 'costs_by_part.npz')
    if os.path.isfile(fbt):
        with np.load(fbt) as d:
            final_costs_by_part = {k: np.asarray(d[k]) for k in d.files}
    return final_costs, cost_curve, costs_by_part, final_costs_by_part


def load_init_z(init_from, session):
    """Load per-param best params + adams; return ``(session, z, adam_init)``.

    Trainable ``z`` comes from remapped node vals; fixed nodes are seeded via ``inits``
    (still not in z); frozen nodes use ``carry``.
    """
    try:
        outdir = resolve_run_dir(init_from)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc
    node_vals, cells, pairs = load_best_node_vals(outdir)
    adams_m, adams_v, adam_iter, _, _ = load_best_adam(outdir)
    schema = train.schema_copy(session.schema)
    remapped = train.remap_node_vals(
        node_vals, cells, pairs, schema, session.connectome,
    )
    schema = train.inits_from_node_vals(schema, remapped)
    schema = train.schema_with_param_carry(schema, remapped)
    opts = dict(session.train_opts or {})
    opts['param_modes'] = train.param_modes_from_schema(
        schema, session.connectome,
    )
    session = replace(session, schema=schema, train_opts=opts)
    z = train.z_from_node_vals(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )
    remapped_m = train.remap_node_vals_adams(
        adams_m, cells, pairs, schema, session.connectome,
    )
    remapped_v = train.remap_node_vals_adams(
        adams_v, cells, pairs, schema, session.connectome,
    )
    exp_avg, exp_avg_sq = train.z_adams_from_node_vals(
        remapped_m, remapped_v, schema,
        dtype=session.sim_dtype, device=session.device,
    )
    adam_init = {
        'exp_avg': exp_avg,
        'exp_avg_sq': exp_avg_sq,
        'iter': int(adam_iter),
    }
    print(
        f'from {outdir!r} -> {best_param_path(outdir)!r} + {best_adam_path(outdir)!r} '
        f'({train.schema_n_z(schema)} z, adam_iter={adam_iter})'
    )
    return session, z, adam_init


def save_train_outputs(fname, outdir, session, result):
    """Write the full run data set (convention §5)."""
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(run_data_dir(outdir), exist_ok=True)
    if session.train_opts is not None:
        with open(os.path.join(run_data_dir(outdir), "train_opts.json"), 'w') as f:
            json.dump(session.train_opts, f, indent=2)
            f.write('\n')
    np.save(_data_file(outdir, fname), result.run_params)
    np.save(_data_file(outdir, 'best_costs.npy'), result.cost_curve)
    np.save(_data_file(outdir, 'costs.npy'), result.final_costs)
    if result.cost_curves_by_part:
        np.savez(_data_file(outdir, 'best_costs_by_part.npz'), **result.cost_curves_by_part)
    if result.final_costs_by_part:
        np.savez(_data_file(outdir, 'costs_by_part.npz'), **result.final_costs_by_part)
    run_i = int(np.argmin(result.final_costs)) if len(result.final_costs) else 0
    adam = result.run_adams[run_i] if result.run_adams else None
    save_best_data(
        outdir, session, result.run_params, result.final_costs, adam=adam,
    )


def print_param_modes(session):
    """Print one schema param per line: indi/shared/fixed/frozen counts and ntrain."""
    schema = train.schema_copy(session.schema)
    if not schema:
        return
    print("param_modes:")
    w = max(len(param) for param in schema)
    for param, spec in schema.items():
        print(
            f"  {param:<{w}}  "
            f"indi={len(spec.get('indi') or [])}/"
            f"shared={len(spec.get('shared') or [])}/"
            f"fixed={len(spec.get('fixed') or [])}/"
            f"frozen={len(spec.get('frozen') or [])} "
            f"({train.param_n_z(spec)})"
        )


def build_session(
    model,
    *,
    network=NETWORK_PATH['network'],
    sequential=TRAIN_SESSION['sequential'],
    tasks=None,
    contrasts=None,
    part_cost_scales=None,
    cost_norm=TRAIN_OPTIMIZATION['cost_norm'],
    cost_interval_ms=TRAIN_OPTIMIZATION['cost_interval_ms'],
    cost_ms=None,
    cost_radius_by_task=None,
    shift_radius=SPOT_INPUT_GEO['shift_radius'],
    spot_radius=SPOT_INPUT_GEO['spot_radius'],
    multi_spot=SPOT_INPUT_GEO['multi_spot'],
    fully_inside=SPOT_INPUT_GEO['fully_inside'],
    spot_cost_radius_scale=None,
    i_sti=None,
    moving_bar_sti_opts=None,
    spread_sti_opts=None,
    spot_sti_opts=None,
    param_modes=None,
    param_init=None,
    param_clamps=None,
    param_jits=None,
    syn_mode=NEURON_SCHEMA['syn_mode'],
    euler=MODEL['euler'],
    pre_steady=None,
    pre_steady_n_iter=TRAIN_OPTIMIZATION['pre_steady_n_iter'],
    pre_steady_damp=TRAIN_OPTIMIZATION['pre_steady_damp'],
    fp=TRAIN_SESSION['fp'],
    pre_grad=NEURON_FORWARD['pre_grad'],
    val_from=None,
    filter=NEURON_SCHEMA['filter'],
    spread_gt_mode=SPREAD_PACK['spread_gt_mode'],
    connectome=None,
    schema=None,
):
    """Create a :class:`TrainSession` from run options."""
    tasks = list(tasks) if tasks is not None else list(
        train.parse_tasks([TRAIN_CONFIG['task']])
    )
    device = train.active_device()
    session_kwargs = dict(
        tasks=tasks,
        contrasts=contrasts,
        part_cost_scales=part_cost_scales,
        cost_norm=expand_cost_norm(cost_norm),
        cost_interval_ms=cost_interval_ms,
        cost_ms=cost_ms,
        sequential=sequential,
        cost_radius_by_task=cost_radius_by_task,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_scale=spot_cost_radius_scale,
        i_sti=i_sti,
        spot_sti_opts=spot_sti_opts,
        spread_sti_opts=spread_sti_opts,
        moving_bar_sti_opts=moving_bar_sti_opts,
    )
    if not network:
        raise ValueError("build_session requires network")
    network = str(resolve_network_json(network))
    opts = train.resolve_train_opts(
        network_json=network,
        device=device,
        euler=euler,
        pre_steady=pre_steady,
        pre_steady_n_iter=pre_steady_n_iter,
        pre_steady_damp=pre_steady_damp,
        param_modes=param_modes,
        param_init=param_init,
        param_clamps=param_clamps,
        param_jits=param_jits,
        syn_mode=syn_mode,
        fp=fp,
        pre_grad=pre_grad,
        val_from=val_from,
        filter=filter,
        spread_gt_mode=spread_gt_mode,
        **session_kwargs,
    )
    return train.open_session(opts, model, schema=schema, connectome=connectome)


def run_train(model, n_run, n_iter, lrs, fname=None, outdir=None,
                 param_modes=None,
                 param_init=None,
                 param_clamps=None,
                 param_jits=None,
                 syn_mode=NEURON_SCHEMA['syn_mode'],
                 euler=MODEL['euler'],
                 pre_steady=None,
                 pre_steady_n_iter=TRAIN_OPTIMIZATION['pre_steady_n_iter'],
                 pre_steady_damp=TRAIN_OPTIMIZATION['pre_steady_damp'],
                 network=NETWORK_PATH['network'], sequential=TRAIN_SESSION['sequential'],
                 tasks=None, contrasts=None, part_cost_scales=None,
                 cost_norm=TRAIN_OPTIMIZATION['cost_norm'],
                 cost_interval_ms=TRAIN_OPTIMIZATION['cost_interval_ms'],
                 cost_ms=None,
                 cost_radius_by_task=None, shift_radius=SPOT_INPUT_GEO['shift_radius'],
                 spot_radius=SPOT_INPUT_GEO['spot_radius'],
                 multi_spot=SPOT_INPUT_GEO['multi_spot'],
                 fully_inside=SPOT_INPUT_GEO['fully_inside'],
                 spot_cost_radius_scale=None,
                 i_sti=None,
                 moving_bar_sti_opts=None,
                 spread_sti_opts=None,
                 spot_sti_opts=None,
                 connectome=None, schema=None,
                 fp=TRAIN_SESSION['fp'],
                 pre_grad=NEURON_FORWARD['pre_grad'],
                 val_from=None,
                 filter=NEURON_SCHEMA['filter'],
                 spread_gt_mode=SPREAD_PACK['spread_gt_mode'],
                 init_from=None,
                 checkpoint_interval=None,
                 build_checkpoint_callback=build_checkpoint_callback,
                 checkpoint_on_png=None):
    """Run train + save data (no plotting). Returns ``(fname, outdir, session, result)``.

    Plotting belongs in ``run.py``. Pass *checkpoint_on_png* from the run layer
    when ``--checkpoint-interval`` should also write PNGs.
    """
    session = build_session(
        model,
        network=network,
        sequential=sequential,
        tasks=tasks,
        contrasts=contrasts,
        part_cost_scales=part_cost_scales,
        cost_norm=cost_norm,
        cost_interval_ms=cost_interval_ms,
        cost_ms=cost_ms,
        cost_radius_by_task=cost_radius_by_task,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_scale=spot_cost_radius_scale,
        i_sti=i_sti,
        moving_bar_sti_opts=moving_bar_sti_opts,
        spread_sti_opts=spread_sti_opts,
        spot_sti_opts=spot_sti_opts,
        param_modes=param_modes,
        param_init=param_init,
        param_clamps=param_clamps,
        param_jits=param_jits,
        syn_mode=syn_mode,
        euler=euler,
        pre_steady=pre_steady,
        pre_steady_n_iter=pre_steady_n_iter,
        pre_steady_damp=pre_steady_damp,
        connectome=connectome,
        schema=schema,
        fp=fp,
        pre_grad=pre_grad,
        val_from=val_from,
        filter=filter,
        spread_gt_mode=spread_gt_mode,
    )
    suffix = "" if model == "borst" else f"_{model}"
    fname = fname or f"train{suffix or '_with_i_h'}.npy"
    outdir = outdir or run_dir(model)

    print_param_modes(session)
    syn_mode = (session.train_opts or {}).get("syn_mode", NEURON_SCHEMA['syn_mode'])
    print(f"device={session.device}, model={model}, syn_mode={syn_mode}, euler={session.euler}, "
          f"pre_steady={session.pre_steady}, "
          f"n_run={n_run}, n_iter={n_iter}, "
          f"lrs={lrs}, n_z={train.schema_n_z(train.schema_copy(session.schema))}, fname={fname}")
    if checkpoint_interval is not None:
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be a positive integer")
        print(f"checkpoint_interval={checkpoint_interval}")
    z_init = None
    adam_init = None
    if init_from:
        session, z_init, adam_init = load_init_z(init_from, session)
    t0 = time.time()
    result = do_many_runs(
        session, n_run, n_iter, lrs=lrs, z_init=z_init, adam_init=adam_init,
        checkpoint_interval=checkpoint_interval,
        checkpoint_outdir=outdir if checkpoint_interval is not None else None,
        build_checkpoint_callback=build_checkpoint_callback,
        checkpoint_on_png=checkpoint_on_png if checkpoint_interval is not None else None,
    )
    print(f"done in {(time.time() - t0) / 3600:.2f} hours")

    save_train_outputs(fname, outdir, session, result)
    return fname, outdir, session, result


