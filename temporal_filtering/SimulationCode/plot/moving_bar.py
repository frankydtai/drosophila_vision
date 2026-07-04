#!/usr/bin/env python
"""Moving-bar plotting utilities extracted from plot_trained."""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import FiveCol_MedSim_Pytorch as fc
import Medulla_Library as ml
from plot.readout import moving_bar_row_types
from plot.utils import nice_ylim as _nice_ylim
from FiveCol_MedSim_Pytorch import t_on
from network.moving_bar_target import _borst_hex_columns
from t4_t5_preference import (
    active_stimuli_for_subtype,
    fig1_key_for_stimulus,
    normalize_side,
)
from training_config import COST_HALF_WINDOW_STEPS, COST_WINDOW_STEPS
from visual_stimulus.moving_bar_stimulus import column_bar_center_step, field_bounds, gruntman_moving_bar_specs

MOVING_BAR_GRID_DPI = 100
MOVING_BAR_MVD_DPI = 100
_MOVING_BAR_T = np.arange(COST_WINDOW_STEPS)


def _moving_bar_center_only(session, target=None):
    target = target or _bar_target(session)
    return bool(session.pack_for(target).center_column)


def _bar_target(session):
    for t in session.target_list:
        if t in fc.MOVING_BAR_TARGETS:
            return t
    raise ValueError(f"session has no moving-bar target in {session.target_list!r}")


def _bar_contrast(target):
    return "bright" if "bright" in target else "dark"


def _bar_specs_for_session(session, target):
    contrast = _bar_contrast(target)
    C = session.backend.network
    if C is not None:
        return list(gruntman_moving_bar_specs(contrasts=(contrast,)))
    return list(gruntman_moving_bar_specs(directions=("right", "left"), contrasts=(contrast,)))


def _borst_type_ids(session):
    node_type = session.backend.conn.node_type
    if torch.is_tensor(node_type):
        node_type = node_type.detach().cpu().numpy()
    return np.asarray(node_type, dtype=np.int64)


def _moving_bar_ylim(model_mean, model_sem, data_mean, keys, show_sem=False):
    curves = []
    for key in keys:
        m = model_mean[key]
        curves.append(m)
        d = data_mean.get(key) if data_mean else None
        if d is not None:
            curves.append(d)
        if show_sem and key in model_sem:
            s = model_sem[key]
            if np.any(s):
                curves.extend([m + s, m - s])
    return _nice_ylim(*curves)


def _save_moving_bar_fig(fig, path, dpi, rasterize=True):
    if rasterize:
        for ax in fig.axes:
            ax.set_rasterized(True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _network_uv_np(C):
    u = C.u.detach().cpu().numpy() if torch.is_tensor(C.u) else np.asarray(C.u)
    v = C.v.detach().cpu().numpy() if torch.is_tensor(C.v) else np.asarray(C.v)
    return u.astype(np.int64), v.astype(np.int64)


def _network_type_ids(C):
    node_type = C.node_type
    if torch.is_tensor(node_type):
        node_type = node_type.detach().cpu().numpy()
    return np.asarray(node_type, dtype=np.int64)


def _moving_bar_t0_grid(C, cols, n_batch, t0_map):
    n_units = C.n_units
    u_np, v_np = _network_uv_np(C)
    t0_bn = np.full((n_batch, n_units), -1, dtype=np.int64)
    for bi in range(n_batch):
        for c in cols:
            t0 = t0_map.get((bi, int(c.u), int(c.v)))
            if t0 is None:
                continue
            on_col = (u_np == int(c.u)) & (v_np == int(c.v))
            t0_bn[bi, on_col] = t0
    return t0_bn


def _extract_moving_bar_windows(model_full, t0_bn):
    n_batch, t_len, n_units = model_full.shape
    win = np.arange(COST_WINDOW_STEPS, dtype=np.int64)
    t_rel = t0_bn[:, :, None].astype(np.int64) - int(t_on) + win[None, None, :]
    t_max = t_len - 1
    pre = t_rel < 0
    t_safe = np.clip(t_rel, 0, t_max)
    b_ix = np.arange(n_batch, dtype=np.int64)[:, None, None]
    u_ix = np.arange(n_units, dtype=np.int64)[None, :, None]
    out = model_full[b_ix, t_safe, u_ix].astype(np.float64, copy=False)
    out[pre] = 0.0
    return out


def _aggregate_moving_bar_traces(windows, t0_bn, type_ids, types, spec_names, center_only):
    model_mean, model_sem = {}, {}
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
                model_sem[key] = np.zeros(COST_WINDOW_STEPS, dtype=np.float64)
            else:
                model_sem[key] = arr.std(axis=0) / np.sqrt(arr.shape[0])
    return model_mean, model_sem


@torch.no_grad()
def _compute_moving_bar_all_type_traces(session, z, target=None):
    from network.moving_bar_target import load_fig1_trace
    from network.stimulus import build_moving_bar_signals, center_photo_column, photo_columns

    target = target or _bar_target(session)
    pack = session.pack_for(target)
    schema = list(session.schema)
    p = fc.assign_params(z, schema, session.backend)
    model_full = fc._run_conductance_full(session, p, pack.signal).cpu().numpy()
    specs = _bar_specs_for_session(session, target)
    spec_names = [s.name for s in specs]
    C = session.backend.network
    if C is not None:
        side = normalize_side(C.meta.get('side', 'right'))
        center_only = _moving_bar_center_only(session)
        center_col = center_photo_column(C)
        cols = [center_col] if center_only else photo_columns(C)
        field_deg = C.meta.get('field_deg')
        if field_deg is None:
            field_deg = build_moving_bar_signals(
                C, t_on=t_on, deltat_ms=fc.deltat, device=session.device,
            ).info['field_deg']

        t0_map = {}
        for bi, spec in enumerate(specs):
            for c in cols:
                t_center = column_bar_center_step(
                    c.x, c.y, spec, field_deg, t_on=t_on, deltat_ms=fc.deltat,
                )
                t0_map[(bi, int(c.u), int(c.v))] = int(t_center - COST_HALF_WINDOW_STEPS)

        types = list(C.type_names)
        type_ids = _network_type_ids(C)
        t0_bn = _moving_bar_t0_grid(C, cols, len(specs), t0_map)
        windows = _extract_moving_bar_windows(model_full, t0_bn)
        model_mean, model_sem = _aggregate_moving_bar_traces(
            windows, t0_bn, type_ids, types, spec_names, center_only,
        )
        data_mean = {}
        row_types = moving_bar_row_types(session, target)
        for subtype in row_types:
            if subtype not in types:
                continue
            for spec in specs:
                trace_id = fig1_key_for_stimulus(side, subtype, spec)
                if trace_id is None:
                    continue
                data_mean[(subtype, spec.name)] = load_fig1_trace(trace_id)
        return types, spec_names, model_mean, model_sem, data_mean

    cols_all = _borst_hex_columns()
    center_only = _moving_bar_center_only(session)
    col_ids = [ml.CENTER_COL] if center_only else list(range(ml.nofcols))
    cols = [cols_all[i] for i in col_ids]
    field_deg = field_bounds(cols_all)
    t0_bn = np.full((len(specs), ml.n_state_units()), -1, dtype=np.int64)
    for bi, spec in enumerate(specs):
        for col_id, col in zip(col_ids, cols):
            t_center = column_bar_center_step(
                col.x, col.y, spec, field_deg, t_on=t_on, deltat_ms=fc.deltat,
            )
            t0 = int(t_center - COST_HALF_WINDOW_STEPS)
            t0_bn[bi, ml.column_slice(col_id)] = t0
    type_ids = _borst_type_ids(session)
    types = list(ml.ctype.tolist())
    windows = _extract_moving_bar_windows(model_full, t0_bn)
    model_mean, model_sem = _aggregate_moving_bar_traces(
        windows, t0_bn, type_ids, types, spec_names, center_only,
    )
    data_mean = {}
    row_types = moving_bar_row_types(session, target)
    for subtype in row_types:
        for spec in specs:
            trace_id = fig1_key_for_stimulus("right", subtype, spec)
            if trace_id is None:
                continue
            data_mean[(subtype, spec.name)] = load_fig1_trace(trace_id)
    return types, spec_names, model_mean, model_sem, data_mean


@torch.no_grad()
def _moving_bar_mean_traces(session, z, target=None):
    target = target or _bar_target(session)
    C = session.backend.network
    side = normalize_side(C.meta.get('side', 'right')) if C is not None else "right"
    _, _, model_mean, model_sem, data_mean = _compute_moving_bar_all_type_traces(session, z, target)
    readout_subtypes = moving_bar_row_types(session, target)
    contrast = _bar_contrast(target)
    row_specs = {
        st: [f'{d}_{c}_{w}' for d, c, w in active_stimuli_for_subtype(side, st) if c == contrast]
        for st in readout_subtypes
    }
    return row_specs, model_mean, model_sem, data_mean


def _moving_bar_scope_label(session):
    center_only = _moving_bar_center_only(session)
    C = session.backend.network
    if center_only:
        if C is not None:
            from network.stimulus import center_photo_column
            col = center_photo_column(C)
            return f'centre column (u,v)=({col.u},{col.v})'
        return f'centre column (col={ml.CENTER_COL})'
    if C is not None:
        from network.stimulus import photo_columns
        return f'avg over {len(photo_columns(C))} photo columns'
    return f'avg over {ml.nofcols} Borst columns'


def _set_moving_bar_xlim(ax):
    ax.set_xlim(0, COST_WINDOW_STEPS)


def _set_moving_bar_xticks(ax):
    mid = COST_HALF_WINDOW_STEPS
    end = COST_WINDOW_STEPS
    ax.set_xticks([0, mid, end])
    ax.set_xticklabels(['-0.45', '0', '0.45'], fontsize=6)


def _moving_bar_right_spec_names(spec_names):
    return [s for s in spec_names if s.startswith('right_')]


def _style_moving_bar_time_axis(ax, show_xlabel=False):
    _set_moving_bar_xlim(ax)
    _set_moving_bar_xticks(ax)
    if show_xlabel:
        ax.set_xlabel('time [s]', fontsize=7)


def _plot_moving_bar_cell(
    ax,
    model_trace,
    sem_trace,
    data_trace,
    title,
    show_ylabel=False,
    show_sem=True,
    ylim=None,
    cell_ticks=True,
    show_xticks=True,
):
    curves = [model_trace]
    if data_trace is not None:
        curves.append(data_trace)
    if show_sem:
        curves.extend([model_trace + sem_trace, model_trace - sem_trace])
    if ylim is None:
        ylo, yhi = _nice_ylim(*curves)
    else:
        ylo, yhi = ylim
    if data_trace is not None:
        ax.plot(data_trace, color='gray', linewidth=1.5)
    if show_sem and np.any(sem_trace):
        ax.fill_between(
            _MOVING_BAR_T, model_trace - sem_trace, model_trace + sem_trace,
            color='pink', alpha=0.8, linewidth=0,
        )
    ax.plot(model_trace, color='red', linewidth=1.5)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_ylim(ylo, yhi)
    _set_moving_bar_xlim(ax)
    if show_xticks:
        _set_moving_bar_xticks(ax)
    if show_ylabel:
        ax.set_ylabel('mV', fontsize=7)
    if cell_ticks:
        ax.tick_params(labelsize=6)


def plot_moving_bar_data(session, z, path, session_off=None, title=None):
    center_only = _moving_bar_center_only(session)
    target = _bar_target(session)
    row_specs, model_mean, model_sem, data_mean = _moving_bar_mean_traces(session, z, target)
    readout_subtypes = list(row_specs.keys())
    ncols_half = max((len(v) for v in row_specs.values()), default=8)
    if session_off is not None:
        target_off = _bar_target(session_off)
        row_specs_off, model_mean_off, model_sem_off, data_mean_off = _moving_bar_mean_traces(
            session_off, z, target_off,
        )
        ncols_half = max(ncols_half, max((len(v) for v in row_specs_off.values()), default=8))
        ncols = ncols_half * 2
    else:
        ncols = ncols_half
    nrows = len(readout_subtypes)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.2 * ncols, 1.8 * nrows), sharex=True,
    )
    if nrows == 1:
        axes = np.asarray([axes])
    if ncols == 1:
        axes = axes[:, None]

    def _plot_row(ri, subtype, specs, col_offset, mm, ms, dm):
        for ci, sname in enumerate(specs):
            ax = axes[ri, col_offset + ci]
            key = (subtype, sname)
            if key not in mm:
                ax.axis('off')
                continue
            _plot_moving_bar_cell(
                ax, mm[key], ms[key], dm.get(key),
                sname, show_ylabel=(col_offset + ci == 0), show_sem=not center_only,
            )

    for ri, subtype in enumerate(readout_subtypes):
        _plot_row(ri, subtype, row_specs[subtype], 0, model_mean, model_sem, data_mean)
        if session_off is not None:
            _plot_row(
                ri, subtype, row_specs_off[subtype], ncols_half,
                model_mean_off, model_sem_off, data_mean_off,
            )
        axes[ri, 0].set_ylabel(subtype, fontsize=8, labelpad=12)
    if title is None:
        title = 'Moving-bar model-data'
    scope = _moving_bar_scope_label(session)
    fig.suptitle(title + f'  [{scope}, t_center ± 0.45 s]', fontsize=12)
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.45, wspace=0.35)
    _save_moving_bar_fig(fig, path, MOVING_BAR_MVD_DPI)


@torch.no_grad()
def plot_moving_bar_all(session, z, path, session_off=None, title=None):
    t0 = time.perf_counter()
    center_only = _moving_bar_center_only(session)
    target = _bar_target(session)
    types, all_spec_names, model_mean, model_sem, data_mean = _compute_moving_bar_all_type_traces(
        session, z, target,
    )
    spec_names = _moving_bar_right_spec_names(all_spec_names)
    if session_off is not None:
        target_off = _bar_target(session_off)
        _, spec_names_off, mm_off, ms_off, dm_off = _compute_moving_bar_all_type_traces(
            session_off, z, target_off,
        )
        spec_names_off = _moving_bar_right_spec_names(spec_names_off)
        spec_names = list(spec_names) + list(spec_names_off)
        model_mean = {**model_mean, **mm_off}
        model_sem = {**model_sem, **ms_off}
        data_mean = {**data_mean, **dm_off}
    t_traces = time.perf_counter() - t0

    keys = [(t, s) for t in types for s in spec_names if (t, s) in model_mean]
    show_sem = not center_only
    ylim = _moving_bar_ylim(model_mean, model_sem, data_mean, keys, show_sem=show_sem)

    nrows = len(types)
    ncols = len(spec_names)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(1.4 * ncols, 0.85 * nrows), sharex=True, sharey=True,
    )
    if nrows == 1:
        axes = np.asarray([axes])
    if ncols == 1:
        axes = axes[:, None]

    t1 = time.perf_counter()
    for ri, tname in enumerate(types):
        for ci, sname in enumerate(spec_names):
            ax = axes[ri, ci]
            key = (tname, sname)
            if key not in model_mean:
                ax.axis('off')
                continue
            _plot_moving_bar_cell(
                ax, model_mean[key], model_sem[key], data_mean.get(key),
                sname if ri == 0 else sname,
                show_ylabel=(ci == 0),
                show_sem=show_sem and key in model_sem and np.any(model_sem[key]),
                ylim=ylim,
                cell_ticks=False,
                show_xticks=(ri == nrows - 1),
            )
        if ncols:
            axes[ri, 0].set_ylabel(tname, fontsize=6, labelpad=4)
    if title is None:
        title = 'Moving-bar model-all (right only)'
    scope = _moving_bar_scope_label(session)
    fig.suptitle(title + f'  [{scope}, t_center ± 0.45 s]', fontsize=10)
    fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.55, wspace=0.3)
    t_draw = time.perf_counter() - t1
    t2 = time.perf_counter()
    _save_moving_bar_fig(fig, path, MOVING_BAR_GRID_DPI)
    t_save = time.perf_counter() - t2
    print(
        f'plot_moving_bar_all: traces={t_traces:.1f}s  '
        f'draw={t_draw:.1f}s  savefig={t_save:.1f}s  total={t_traces+t_draw+t_save:.1f}s'
    )

