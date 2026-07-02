#!/usr/bin/env python3
"""Test plot: moving-bar model traces on the full stimulus horizon (``maxtime`` steps).

Reads trained params from ``--rundir``; writes ``model_all_cells_fulltime.png`` and
``moving_bar_full_plot_cache.npz`` under ``test/moving_bar_fulltime/`` (default).
Does not modify :mod:`plot_trained` or the run folder.

Cache hit: load npz + matplotlib only (no network, no forward, no cost).
Cache miss: restore network once, forward once, save cache, then plot.

Usage:
    ../.venv/bin/python test/run_moving_bar_fulltime_plot.py
    ../.venv/bin/python test/run_moving_bar_fulltime_plot.py \\
        --rundir FiveCol_Parameter/conductance/run_26693975
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import network_io  # noqa: F401
import FiveCol_MedSim_Pytorch as fc
from plot_trained import (
    _moving_bar_center_only,
    _moving_bar_right_spec_names,
    _moving_bar_t0_grid,
    _network_type_ids,
    _nice_ylim,
    resolve_model_type,
    restore_fc_context,
)
from training_io import COST_HALF_WINDOW_STEPS, COST_WINDOW_STEPS
from t4_t5_preference import READOUT_SUBTYPES, fig1_key_for_stimulus, normalize_side
from visual_stimulus.moving_bar_stimulus import (
    column_bar_center_step,
    gruntman_moving_bar_specs,
)

MOVING_BAR_FULL_PLOT_CACHE_FILE = 'moving_bar_full_plot_cache.npz'
PNG_NAME = 'model_all_cells_fulltime.png'
DELTAT_MS = 10.0


def _param_fingerprint(best: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(best, dtype=np.float64).tobytes()).hexdigest()


def _pad_model_full_absolute(model_full, maxtime, t_on):
    n_batch, _, n_units = model_full.shape
    out = np.zeros((n_batch, maxtime, n_units), dtype=np.float64)
    out[:, t_on:maxtime, :] = model_full
    return out


def _full_cache_context():
    # Legacy helper; cache is disabled in this test script.
    return (int(fc.maxtime), int(fc.t_on))


def _aggregate_full_traces(windows, t0_bn, type_ids, types, spec_names, center_only):
    model_mean, model_sem = {}, {}
    n_steps = windows.shape[-1]
    valid = t0_bn >= 0
    for ti, tname in enumerate(types):
        type_mask = type_ids == ti
        if not type_mask.any():
            continue
        for bi, sname in enumerate(spec_names):
            unit_mask = valid[bi] & type_mask
            if not unit_mask.any():
                continue
            arr = windows[bi, unit_mask]
            key = (tname, sname)
            model_mean[key] = arr.mean(axis=0)
            if center_only or arr.shape[0] == 1:
                model_sem[key] = np.zeros(n_steps, dtype=np.float64)
            else:
                model_sem[key] = arr.std(axis=0) / np.sqrt(arr.shape[0])
    return model_mean, model_sem


@torch.no_grad()
def _compute_full_type_traces(z):
    from network.moving_bar_target import load_fig1_trace
    from network.stimulus import build_moving_bar_signals, center_photo_column, photo_columns

    specs = gruntman_moving_bar_specs()
    spec_names = [s.name for s in specs]
    C = fc.NETWORK
    side = normalize_side(C.meta.get('side', 'right'))
    center_only = _moving_bar_center_only()
    cols = [center_photo_column(C)] if center_only else photo_columns(C)

    p = fc.assign_params(z, fc.CONDUCTANCE_SCHEMA)
    model_full = fc._run_conductance_full(p, fc.signal).cpu().numpy()
    mt, ton = int(fc.maxtime), int(fc.t_on)
    padded = _pad_model_full_absolute(model_full, mt, ton)
    windows = np.transpose(padded, (0, 2, 1))

    field_deg = C.meta.get('field_deg')
    if field_deg is None:
        field_deg = build_moving_bar_signals(
            C, t_on=ton, deltat_ms=fc.deltat, device=fc.device,
        ).info['field_deg']

    t0_map = {}
    for bi, spec in enumerate(specs):
        for c in cols:
            t_center = column_bar_center_step(
                c.x, c.y, spec, field_deg, t_on=ton, deltat_ms=fc.deltat,
            )
            t0_map[(bi, int(c.u), int(c.v))] = int(t_center - COST_HALF_WINDOW_STEPS)

    types = list(C.type_names)
    type_ids = _network_type_ids(C)
    t0_bn = _moving_bar_t0_grid(C, cols, len(specs), t0_map)
    model_mean, model_sem = _aggregate_full_traces(
        windows, t0_bn, type_ids, types, spec_names, center_only,
    )
    data_mean = {}
    wlen = COST_WINDOW_STEPS
    w0 = int(ton)
    w1 = min(int(mt), w0 + wlen)
    n_fill = max(0, w1 - w0)
    for subtype in READOUT_SUBTYPES:
        if subtype not in types:
            continue
        for spec in specs:
            trace_id = fig1_key_for_stimulus(side, subtype, spec)
            if trace_id is None:
                continue
            win = load_fig1_trace(trace_id, n_steps=COST_WINDOW_STEPS, half_window_steps=COST_HALF_WINDOW_STEPS, deltat_ms=fc.deltat)
            full = np.full((mt,), np.nan, dtype=np.float64)
            if n_fill > 0:
                full[w0:w1] = win[:n_fill]
            data_mean[(subtype, spec.name)] = full
    if center_only:
        col = center_photo_column(C)
        scope = f'centre column (u,v)=({col.u},{col.v})'
    else:
        scope = f'avg over {len(photo_columns(C))} photo columns'
    meta = {'maxtime': mt, 't_on': ton, 'deltat_ms': float(fc.deltat), 'scope': scope,
            'center_only': center_only}
    return types, spec_names, model_mean, model_sem, data_mean, meta


def _save_full_cache(cache_dir, z, result):
    # Cache disabled in this test script to avoid depending on helpers removed
    # from plot_trained; keep signature for forward-compatibility.
    return None


def _load_full_cache(cache_dir, fingerprint: str):
    # Cache disabled; always force recompute + fresh plot.
    return None


@torch.no_grad()
def _full_type_traces(z, cache_dir=None):
    # Cache disabled; always recompute.
    result = _compute_full_type_traces(z)
    if cache_dir:
        _save_full_cache(cache_dir, z, result)
    return result


def _style_horizon_axis(ax, maxtime, deltat_ms, show_xlabel):
    t_end = maxtime * deltat_ms / 1000.0
    t_mid = t_end / 2.0
    ax.set_xlim(0, maxtime)
    ax.set_xticks([0, maxtime // 2, maxtime])
    ax.set_xticklabels(['0', f'{t_mid:g}', f'{t_end:g}'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('time [s]', fontsize=7)


def _plot_cell(ax, model_trace, sem_trace, data_trace, title, maxtime, t_on, deltat_ms,
               show_ylabel=False, show_sem=True):
    t = np.arange(maxtime)
    curves = [model_trace]
    if data_trace is not None:
        finite = np.isfinite(data_trace)
        if np.any(finite):
            curves.append(data_trace[finite])
    if show_sem:
        curves.extend([model_trace + sem_trace, model_trace - sem_trace])
    ylo, yhi = _nice_ylim(*curves)
    if data_trace is not None:
        ax.plot(data_trace, color='gray', linewidth=1.5)
    if show_sem and np.any(sem_trace):
        ax.fill_between(t, model_trace - sem_trace, model_trace + sem_trace,
                        color='pink', alpha=0.8, linewidth=0)
    ax.plot(model_trace, color='red', linewidth=1.5)
    ax.axvline(t_on, color='0.75', linewidth=0.6, linestyle='--', zorder=0)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_ylim(ylo, yhi)
    # x-limits only here; ticks/labels are applied on the bottom row once.
    ax.set_xlim(0, maxtime)
    if show_ylabel:
        ax.set_ylabel('mV', fontsize=7)
    ax.tick_params(labelsize=6)


def plot_all_celltypes_fulltime(path, types, spec_names, model_mean, model_sem, data_mean, meta,
                                title=None):
    maxtime = int(meta['maxtime'])
    t_on = int(meta['t_on'])
    deltat_ms = float(meta['deltat_ms'])
    center_only = bool(meta['center_only'])
    t_end_s = maxtime * deltat_ms / 1000.0

    # Match `model_all_cells`: 32 rows (cell types) × 4 columns (right stimuli).
    spec_names = _moving_bar_right_spec_names(spec_names)
    nrows = len(types)
    ncols = len(spec_names)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(1.4 * ncols, 0.85 * nrows), sharex=True,
    )
    if nrows == 1:
        axes = np.asarray([axes])
    if ncols == 1:
        axes = axes[:, None]

    for ri, tname in enumerate(types):
        for ci, sname in enumerate(spec_names):
            ax = axes[ri, ci]
            key = (tname, sname)
            if key not in model_mean:
                ax.axis('off')
                continue
            _plot_cell(
                ax, model_mean[key], model_sem[key], data_mean.get(key), sname,
                maxtime, t_on, deltat_ms,
                show_ylabel=(ci == 0),
                show_sem=not center_only and np.any(model_sem[key]),
            )
        axes[ri, 0].set_ylabel(tname, fontsize=6, labelpad=4)

    # Apply x-axis ticks/labels only on the bottom row to reduce cost.
    for ci in range(ncols):
        _style_horizon_axis(axes[-1, ci], maxtime, deltat_ms, show_xlabel=False)

    if title is None:
        title = 'Moving-bar all cell types (full horizon)'
    fig.suptitle(
        title + f'  [{meta["scope"]}, 0–{t_end_s:g} s; dashed = t_on]',
        fontsize=10,
    )
    fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.55, wspace=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _load_best_param(rundir):
    best_path = os.path.join(rundir, 'best_param.npy')
    if os.path.isfile(best_path):
        return np.load(best_path)
    params_path = os.path.join(rundir, 'training_with_Ih.npy')
    params = np.load(params_path)
    valid = params[np.any(params != 0, axis=1)]
    if len(valid) == 0:
        raise SystemExit('no trained parameter sets found')
    print(f'warning: {best_path} missing; using first valid row from training_with_Ih.npy')
    return valid[0]


def main():
    default_rundir = os.path.join('FiveCol_Parameter', 'conductance', 'run_26693975')
    default_outdir = os.path.join(HERE, 'moving_bar_fulltime')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rundir', default=default_rundir, help='trained run folder (read-only)')
    ap.add_argument('--outdir', default=default_outdir, help='test output folder for png + cache')
    args = ap.parse_args()

    rundir = os.path.abspath(args.rundir)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, PNG_NAME)
    cache_path = os.path.join(outdir, MOVING_BAR_FULL_PLOT_CACHE_FILE)

    best = _load_best_param(rundir)
    fp = _param_fingerprint(best)

    cached = _load_full_cache(outdir, fp)
    if cached is not None:
        types, spec_names, model_mean, model_sem, data_mean, meta = cached
        plot_all_celltypes_fulltime(
            png, types, spec_names, model_mean, model_sem, data_mean, meta,
            title='Moving-bar all cells (trained, full horizon)',
        )
        print(f'mode: cache-only (no network)')
        print(f'png: {png}')
        print('ok')
        return

    params_path = os.path.join(rundir, 'training_with_Ih.npy')
    if not os.path.isfile(params_path):
        raise SystemExit(f'missing params: {params_path}')
    if not restore_fc_context(rundir):
        raise SystemExit(f'cannot restore network context from {rundir}')

    fc.MODEL_TYPE = resolve_model_type(params_path, None)
    z = torch.tensor(best, dtype=torch.float64, device=fc.device)
    types, spec_names, model_mean, model_sem, data_mean, meta = _full_type_traces(z, cache_dir=outdir)
    plot_all_celltypes_fulltime(
        png, types, spec_names, model_mean, model_sem, data_mean, meta,
        title='Moving-bar all cells (trained, full horizon)',
    )

    assert os.path.isfile(png), png
    # Cache is disabled in this variant; only confirm PNG.
    print(f'mode: forward only (no cache)')
    print(f'rundir (read): {rundir}')
    print(f'outdir: {outdir}')
    print(f'maxtime={meta["maxtime"]} steps, t_on={meta["t_on"]}, '
          f'horizon={meta["maxtime"] * meta["deltat_ms"] / 1000:g} s')
    print(f'png: {png}')
    print(f'cache: {cache_path}')
    print('ok')


if __name__ == '__main__':
    main()
