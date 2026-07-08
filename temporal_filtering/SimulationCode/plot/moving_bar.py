#!/usr/bin/env python
"""Moving-bar plotting utilities extracted from plot_trained."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

import FiveCol_MedSim_Pytorch as fc
import Medulla_Library as ml
from Medulla_Library import I_BASELINE
from plot.readout import moving_bar_row_types
from plot.utils import (
    DATA_COLOR,
    TRACE_LW,
    annotate_baseline,
    baselines_for_types,
    column_at_scope_tag,
    filter_borst_sti_columns,
    filter_sti_columns,
    overlay_model_reds,
    plot_timecourse,
    save_figure,
    save_forward_trace_csvs,
    sem_from_traces,
    slice_axis_label,
    slice_xy_label,
    suppress_cost_sem,
    ylim_for_traces,
)
from FiveCol_MedSim_Pytorch import t_on
import network_bootstrap  # noqa: F401  # ensure FAFBv783 modules are importable
from network.moving_bar_target import _borst_moving_bar_specs, load_fig1_trace
from t4_t5_preference import (
    READOUT_SUBTYPES,
    active_stimuli_for_subtype,
    fig1_key_for_stimulus,
    motion_preference,
    normalize_side,
)
from training_config import (
    COST_WINDOW_AFTER,
    COST_WINDOW_BEFORE,
    DELTAT_MS,
)
from visual_stimulus.moving_bar_stimulus import (
    build_batched_column_current,
    column_first_stim_step,
    gruntman_moving_bar_specs,
)

MOVING_BAR_DPI = 100


def _borst_sti_columns():
    return import_module("column_mapper").borst_sti_columns()


@dataclass
class MovingBarWindowTraces:
    model_mean: dict
    model_sem: dict
    before_steps: dict[str, int] | None = None
    after_steps: dict[str, int] | None = None


@dataclass
class MovingBarTraceBundle:
    """One forward pass; t_first_sti-aligned full-window model traces."""

    target: str
    types: list
    spec_names: list
    side: str
    single_column: bool
    baselines: dict
    data_mean: dict
    maxtime: int
    traces: MovingBarWindowTraces
    session: object = field(default=None, repr=False, compare=False)
    at_x: float | None = None
    at_y: float | None = None
    n_filter_cols: int | None = None
    slice_overlay: dict[str, MovingBarWindowTraces] | None = None
    slice_axis: str | None = None
    slice_x_list: list | None = None
    slice_y_list: list | None = None


def _moving_bar_figure(nrows, ncols, *, sharex='col'):
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.2 * ncols, 1.8 * nrows), sharex=sharex,
    )
    if nrows == 1:
        axes = np.asarray([axes])
    if ncols == 1:
        axes = axes[:, None]
    return fig, axes


def _moving_bar_figure_adjust(fig):
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.45, wspace=0.35)


def _tensor_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _type_ids_np(node_type):
    return np.asarray(_tensor_np(node_type), dtype=np.int64)


def _network_uv_np(C):
    return _type_ids_np(C.u), _type_ids_np(C.v)


def _rel_window_seconds(before_steps, after_steps):
    scale = DELTAT_MS / 1000.0
    return before_steps * scale, after_steps * scale


def _record_spec_full_horizon(t_first_stis, maxtime, spec_name, full_before, full_after):
    fb = min(t_first_stis)
    full_before[spec_name] = fb
    full_after[spec_name] = int(maxtime) - max(t_first_stis)
    return fb


def _filter_right_specs(spec_names, right_only):
    if right_only:
        return [s for s in spec_names if s.startswith('right_')]
    return list(spec_names)


def _bar_specs_for_session(session, target):
    contrast = "bright" if "bright" in target else "dark"
    C = session.backend.network
    if C is not None:
        return list(gruntman_moving_bar_specs(contrasts=(contrast,)))
    return list(_borst_moving_bar_specs(contrasts=(contrast,)))


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


def _scale_model_full(model_full, p, backend):
    """Apply per-unit ``out_scale`` to ``model_full`` (B, T', N), matching cost."""
    n_units = model_full.shape[2]
    unit_ix = torch.arange(n_units, dtype=torch.long)
    scale = fc.out_scale_for_units(p, unit_ix, backend).cpu().numpy()
    return model_full * scale[np.newaxis, np.newaxis, :]


def _windows_by_batch(model_full, t0_bn, win_lens):
    """``win_lens``: int (uniform) or length-``B`` sequence of window lengths."""
    n_batch = model_full.shape[0]
    if isinstance(win_lens, int):
        win_lens = (win_lens,) * n_batch
    out = []
    for bi in range(n_batch):
        wl = int(win_lens[bi])
        sl = model_full[bi:bi + 1]
        t0 = t0_bn[bi:bi + 1]
        win = np.arange(wl, dtype=np.int64)
        t_len = sl.shape[1]
        n_units = sl.shape[2]
        t_rel = t0[:, :, None].astype(np.int64) - int(t_on) + win[None, None, :]
        pre = t_rel < 0
        t_safe = np.clip(t_rel, 0, t_len - 1)
        b_ix = np.zeros(1, dtype=np.int64)[:, None, None]
        u_ix = np.arange(n_units, dtype=np.int64)[None, :, None]
        batch = sl[b_ix, t_safe, u_ix].astype(np.float64, copy=False)
        batch[pre] = 0.0
        out.append(batch[0])
    return out


def _aggregate_moving_bar_traces(windows_by_batch, t0_bn, type_ids, types, spec_names, single_column):
    """``windows_by_batch[bi]`` shape ``(n_units, W_bi)``."""
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
            arr = windows_by_batch[bi][unit_mask]
            key = (tname, sname)
            model_mean[key] = arr.mean(axis=0)
            model_sem[key] = sem_from_traces(arr, single_column=single_column)
    return model_mean, model_sem


def _aggregate_moving_bar_traces_colmask(
    windows_by_batch, t0_bn, type_ids, types, spec_names, col_mask, single_column,
):
    """Like :func:`_aggregate_moving_bar_traces` but only units in ``col_mask``."""
    model_mean, model_sem = {}, {}
    valid = t0_bn >= 0
    for ti, tname in enumerate(types):
        type_mask = type_ids == ti
        if not type_mask.any():
            continue
        for bi, sname in enumerate(spec_names):
            unit_mask = valid[bi] & type_mask & col_mask[bi]
            if not unit_mask.any():
                continue
            arr = windows_by_batch[bi][unit_mask]
            key = (tname, sname)
            model_mean[key] = arr.mean(axis=0)
            model_sem[key] = sem_from_traces(arr, single_column=single_column)
    return model_mean, model_sem


def _network_column_unit_mask(C, filt_cols, n_batch):
    u_np, v_np = _network_uv_np(C)
    col_uv = {(int(c.u), int(c.v)) for c in filt_cols}
    unit_in_col = np.array(
        [(int(u), int(v)) in col_uv for u, v in zip(u_np, v_np)],
        dtype=bool,
    )
    return np.broadcast_to(unit_in_col, (n_batch, C.n_units)).copy()


def _borst_column_unit_mask(filt_cols, n_batch):
    mask = np.zeros((n_batch, ml.n_state_units()), dtype=bool)
    for col in filt_cols:
        mask[:, ml.column_slice(col.col)] = True
    return mask


def _moving_bar_slice_overlay_traces(
    session, target, trace_full, base_wt, specs, spec_names, *, at_x=None, at_y=None,
):
    """Per-axis slice traces aligned to ``base_wt`` window geometry."""
    from network.moving_bar_target import moving_bar_cost_columns

    pack = session.pack_for(target)
    cost_extent = pack.cost_extent
    maxtime = int(session.maxtime)
    C = session.backend.network
    types, type_ids, t0_full_bn, _, _, _, _ = _moving_bar_t0_grids(
        session, specs, cost_extent, maxtime,
    )
    n_batch = len(spec_names)
    if C is not None:
        all_cols = moving_bar_cost_columns(C, cost_extent=cost_extent)
        filt_cols = filter_sti_columns(all_cols, at_x=at_x, at_y=at_y)
        if not filt_cols:
            return None
        col_mask = _network_column_unit_mask(C, filt_cols, n_batch)
    else:
        cols_all = list(_borst_sti_columns())
        try:
            filt_cols = filter_borst_sti_columns(cols_all, at_x=at_x, at_y=at_y)
        except ValueError:
            return None
        if not filt_cols:
            return None
        col_mask = _borst_column_unit_mask(filt_cols, n_batch)
    win_lens = [
        base_wt.before_steps[sname] + base_wt.after_steps[sname] + 1
        for sname in spec_names
    ]
    windows_full = _windows_by_batch(trace_full, t0_full_bn, win_lens)
    model_mean, model_sem = _aggregate_moving_bar_traces_colmask(
        windows_full, t0_full_bn, type_ids, types, spec_names, col_mask, True,
    )
    return MovingBarWindowTraces(
        model_mean=model_mean,
        model_sem=model_sem,
        before_steps=base_wt.before_steps,
        after_steps=base_wt.after_steps,
    )


def _column_t_first_stis(col_curr_bti, batch_idx, col_idxs, i_baseline):
    return [
        column_first_stim_step(col_curr_bti[batch_idx, :, col_idx], i_baseline=i_baseline)
        for col_idx in col_idxs
    ]


def _load_moving_bar_data_mean(session, target, types, specs, side):
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
    return data_mean


def _moving_bar_baselines(C, vm_ref, types, type_ids, type_names, cost_extent, *, at_x=None, at_y=None):
    """Mean ``Vm_ref`` per type over units on moving-bar cost columns (matches trace scope)."""
    from network.moving_bar_target import moving_bar_cost_columns

    cols = moving_bar_cost_columns(C, cost_extent=cost_extent)
    if at_x is not None or at_y is not None:
        cols = filter_sti_columns(cols, at_x=at_x, at_y=at_y)
    u_np, v_np = _network_uv_np(C)
    vm_ref = np.asarray(vm_ref, dtype=np.float64)
    out = {}
    for tname in types:
        ti = type_names.index(tname)
        units = []
        for c in cols:
            on_col = (u_np == int(c.u)) & (v_np == int(c.v))
            units.extend(np.where(on_col & (type_ids == ti))[0])
        u = np.unique(units)
        out[tname] = float(vm_ref[u].mean()) if u.size else np.nan
    return out


def _moving_bar_t0_grids(session, specs, cost_extent, maxtime, *, at_x=None, at_y=None):
    from network.moving_bar_target import build_moving_bar_signals, moving_bar_cost_columns, sti_columns

    C = session.backend.network
    n_batch = len(specs)
    full_before_steps = {}
    full_after_steps = {}
    t0_full_map = {}

    if C is not None:
        side = normalize_side(C.meta.get('side', 'right'))
        cols = moving_bar_cost_columns(C, cost_extent=cost_extent)
        if at_x is not None or at_y is not None:
            cols = filter_sti_columns(cols, at_x=at_x, at_y=at_y)
            if not cols:
                raise SystemExit(
                    f'no sti columns match x={at_x!r} y={at_y!r} within cost_extent',
                )
        stim = build_moving_bar_signals(
            C, specs=specs, maxtime=maxtime, t_on=t_on, deltat_ms=fc.deltat,
            device=C.node_type.device,
        )
        i_baseline = float(stim.info.get("i_baseline", 0.0))
        uv_to_idx = {
            (int(col.u), int(col.v)): j
            for j, col in enumerate(sti_columns(C))
        }
        col_idxs = [uv_to_idx[(int(c.u), int(c.v))] for c in cols]
        types = list(C.type_names)
        type_ids = _type_ids_np(C.node_type)
        for bi, spec in enumerate(specs):
            t_first_stis = _column_t_first_stis(
                stim.column_current, bi, col_idxs, i_baseline,
            )
            fb = _record_spec_full_horizon(
                t_first_stis, maxtime, spec.name, full_before_steps, full_after_steps,
            )
            for c, tc in zip(cols, t_first_stis):
                uv = (bi, int(c.u), int(c.v))
                t0_full_map[uv] = tc - fb
        t0_full_bn = _moving_bar_t0_grid(C, cols, n_batch, t0_full_map)
    else:
        side = "right"
        cols_all = list(_borst_sti_columns())
        if at_x is not None or at_y is not None:
            cols = filter_borst_sti_columns(cols_all, at_x=at_x, at_y=at_y)
            if not cols:
                raise SystemExit(
                    f'no Borst columns match x={at_x!r} y={at_y!r}',
                )
            col_ids = [col.col for col in cols]
        else:
            col_ids = list(range(ml.nofcols))
            cols = [cols_all[i] for i in col_ids]
        i_baseline = I_BASELINE
        col_curr = build_batched_column_current(
            cols_all, specs, maxtime, t_on=t_on, deltat_ms=fc.deltat,
        )
        types = list(ml.ctype.tolist())
        type_ids = _type_ids_np(session.backend.conn.node_type)
        t0_full_bn = np.full((n_batch, ml.n_state_units()), -1, dtype=np.int64)
        for bi, spec in enumerate(specs):
            t_first_stis = _column_t_first_stis(
                col_curr, bi, col_ids, i_baseline,
            )
            fb = _record_spec_full_horizon(
                t_first_stis, maxtime, spec.name, full_before_steps, full_after_steps,
            )
            for col_id, col, tc in zip(col_ids, cols, t_first_stis):
                t0_full_bn[bi, ml.column_slice(col_id)] = tc - fb

    n_filter_cols = len(cols)
    return types, type_ids, t0_full_bn, full_before_steps, full_after_steps, side, n_filter_cols


def _moving_bar_row_specs(session, target, side):
    readout_subtypes = moving_bar_row_types(session, target)
    contrast = "bright" if "bright" in target else "dark"
    return {
        st: [f'{d}_{c}_{w}' for d, c, w in active_stimuli_for_subtype(side, st) if c == contrast]
        for st in readout_subtypes
    }


def _moving_bar_traces_from_forward(
    session, target, trace_full, vm_ref_np, specs, spec_names, *,
    at_x=None, at_y=None,
):
    pack = session.pack_for(target)
    cost_extent = pack.cost_extent
    maxtime = int(session.maxtime)
    C = session.backend.network
    types, type_ids, t0_full_bn, full_before_steps, full_after_steps, side, n_filter_cols = (
        _moving_bar_t0_grids(session, specs, cost_extent, maxtime, at_x=at_x, at_y=at_y)
    )
    single_column = suppress_cost_sem(session, target) or n_filter_cols == 1
    win_lens = [
        full_before_steps[sname] + full_after_steps[sname] + 1
        for sname in spec_names
    ]
    windows_full = _windows_by_batch(trace_full, t0_full_bn, win_lens)
    trace_mean, trace_sem = _aggregate_moving_bar_traces(
        windows_full, t0_full_bn, type_ids, types, spec_names, single_column,
    )
    return MovingBarWindowTraces(
        model_mean=trace_mean,
        model_sem=trace_sem,
        before_steps=full_before_steps,
        after_steps=full_after_steps,
    ), types, side, n_filter_cols


@torch.no_grad()
def moving_bar_trace_bundle(session, z, target, *, at_x=None, at_y=None,
                            at_x_list=None, at_y_list=None,
                            trace_kind: Literal['model', 'vm'] = 'model',
                            save_trace_csv_dir: str | None = None):
    """Run one forward; t_first_sti-aligned full-window model traces."""
    pack = session.pack_for(target)
    schema = list(session.schema)
    p = fc.assign_params(z, schema, session.backend)
    if trace_kind == 'vm':
        model_full, vm_ref, vm_full = fc._run_conductance_full(
            session, p, pack.signal, return_ref=True, return_vm=True,
        )
        vm_ref_np = vm_ref[0].cpu().numpy()
        trace_full = (vm_full - vm_ref[:, None, :]).cpu().numpy()
        save_forward_trace_csvs(
            save_trace_csv_dir, target,
            trace_kind=trace_kind, ref=vm_ref_np, trace_full=trace_full,
            ref_stem='moving_bar_ref_vm' if trace_kind == 'vm' else None,
        )
        data_mean = {}
    else:
        model_full, vm_ref = fc._run_conductance_full(session, p, pack.signal, return_ref=True)
        vm_ref_np = vm_ref[0].cpu().numpy()
        trace_full = _scale_model_full(model_full.cpu().numpy(), p, session.backend)
        save_forward_trace_csvs(
            save_trace_csv_dir, target,
            trace_kind=trace_kind, ref=vm_ref_np, trace_full=trace_full,
            ref_stem='moving_bar_ref_vm' if trace_kind == 'vm' else None,
        )
        data_mean = None
    specs = _bar_specs_for_session(session, target)
    spec_names = [s.name for s in specs]
    maxtime = int(session.maxtime)
    C = session.backend.network
    traces, types, side, n_filter_cols = _moving_bar_traces_from_forward(
        session, target, trace_full, vm_ref_np, specs, spec_names,
    )
    if C is not None:
        type_names = list(C.type_names)
        type_ids = _type_ids_np(C.node_type)
        baselines = _moving_bar_baselines(
            C, vm_ref_np, types, type_ids, type_names, pack.cost_extent,
            at_x=at_x, at_y=at_y,
        )
    else:
        type_ids = _type_ids_np(session.backend.conn.node_type)
        baselines = baselines_for_types(
            pack, session.backend, vm_ref_np, types, type_ids, types,
        )
    single_column = suppress_cost_sem(session, target) or n_filter_cols == 1
    if data_mean is None:
        data_mean = _load_moving_bar_data_mean(session, target, types, specs, side)
    slice_overlay = None
    slice_axis = None
    slice_x_list = None
    slice_y_list = None
    if at_x_list is not None and at_y_list is not None:
        slice_axis = 'xy'
        slice_x_list = list(at_x_list)
        slice_y_list = list(at_y_list)
        slice_overlay = {}
        for xv in slice_x_list:
            for yv in slice_y_list:
                label = slice_xy_label(xv, yv)
                wt = _moving_bar_slice_overlay_traces(
                    session, target, trace_full, traces, specs, spec_names,
                    at_x=xv, at_y=yv,
                )
                if wt is None:
                    print(f'skip slice overlay {label}: no column within cost_extent')
                    continue
                slice_overlay[label] = wt
        if not slice_overlay:
            slice_overlay = None
            slice_axis = None
            slice_x_list = None
            slice_y_list = None
    elif at_x_list is not None:
        slice_axis = 'x'
        slice_x_list = list(at_x_list)
        slice_overlay = {}
        for xv in slice_x_list:
            slice_overlay[slice_axis_label(xv)] = _moving_bar_slice_overlay_traces(
                session, target, trace_full, traces, specs, spec_names, at_x=xv,
            )
    elif at_y_list is not None:
        slice_axis = 'y'
        slice_y_list = list(at_y_list)
        slice_overlay = {}
        for yv in slice_y_list:
            slice_overlay[slice_axis_label(yv)] = _moving_bar_slice_overlay_traces(
                session, target, trace_full, traces, specs, spec_names, at_y=yv,
            )
    return MovingBarTraceBundle(
        target=target,
        types=types,
        spec_names=spec_names,
        side=side,
        single_column=single_column,
        baselines=baselines,
        data_mean=data_mean,
        maxtime=maxtime,
        traces=traces,
        session=session,
        at_x=at_x,
        at_y=at_y,
        n_filter_cols=n_filter_cols,
        slice_overlay=slice_overlay,
        slice_axis=slice_axis,
        slice_x_list=slice_x_list,
        slice_y_list=slice_y_list,
    )


def _window_steps(wt, sname):
    return wt.before_steps[sname], wt.after_steps[sname]


def _cost_window_overlay(cost_trace, before_steps):
    """Fig1 overlay x/y within full-window coordinates (cost window only)."""
    i0 = before_steps - COST_WINDOW_BEFORE
    trace = np.asarray(cost_trace, dtype=np.float64)
    if i0 < 0:
        trace = trace[-i0:]
        i0 = 0
    x = np.arange(i0, i0 + len(trace), dtype=np.int64)
    return x, trace


def _moving_bar_scope_label(session, *, at_x=None, at_y=None, n_filter_cols=None):
    pack = session.primary_pack
    cost_extent = pack.cost_extent
    C = session.backend.network
    if at_x is not None or at_y is not None:
        ncol_part = f'{n_filter_cols} sti column'
        if n_filter_cols != 1:
            ncol_part += 's'
        parts = [column_at_scope_tag(at_x, at_y), ncol_part]
        if cost_extent is not None:
            parts.insert(0, f'cost_extent={cost_extent}')
        return ', '.join(parts)
    if cost_extent is not None:
        if C is not None:
            from network.moving_bar_target import moving_bar_cost_columns
            ncols = len(moving_bar_cost_columns(C, cost_extent=cost_extent))
            return f'cost_extent={cost_extent} ({ncols} sti columns)'
        return f'cost_extent={cost_extent}'
    if C is not None:
        from network.moving_bar_target import sti_columns
        return f'avg over {len(sti_columns(C))} sti columns'
    return f'avg over {ml.nofcols} Borst columns'


def _style_moving_bar_relative_axis(
    ax, before_steps, after_steps, win_len, *,
    show_tick_labels=True, mark_cost_window=False,
):
    end = win_len - 1
    ax.set_xlim(0, end)
    ax.set_xticks([0, before_steps, end])
    before_s, after_s = _rel_window_seconds(before_steps, after_steps)
    ax.set_xticklabels([f'{-before_s:g}', '0', f'{after_s:g}'], fontsize=6)
    if not show_tick_labels:
        ax.tick_params(labelbottom=False)
    if mark_cost_window:
        for x in (before_steps - COST_WINDOW_BEFORE, before_steps + COST_WINDOW_AFTER):
            ax.axvline(x, color='0.75', linewidth=0.6, linestyle='--', zorder=0)


def _moving_bar_spec_linestyle(side, subtype, sname):
    """Solid for PD stimuli, dashed for ND (Gruntman fig1 convention)."""
    if subtype not in READOUT_SUBTYPES:
        return '-'
    parts = str(sname).split('_')
    if len(parts) < 3:
        return '-'
    direction, contrast, wtag = parts[0], parts[1], parts[2]
    pref = motion_preference(side, subtype, direction, contrast)
    if pref is None:
        return '-'
    return '--' if pref.pd_nd == 'ND' else '-'


def _plot_moving_bar_cell(
    ax,
    model_trace,
    sem_trace,
    title,
    before_steps,
    after_steps,
    *,
    data_trace=None,
    show_ylabel=False,
    show_sem=True,
    ylim=None,
    cell_ticks=True,
    show_tick_labels=True,
    mark_cost_window=False,
    baseline=None,
    linestyle='-',
):
    win_len = len(model_trace)
    data_x, data_y = None, None
    if data_trace is not None:
        data_x, data_y = _cost_window_overlay(data_trace, before_steps)

    def style_xaxis(ax):
        _style_moving_bar_relative_axis(
            ax, before_steps, after_steps, win_len,
            show_tick_labels=show_tick_labels,
            mark_cost_window=mark_cost_window,
        )

    if ylim is None:
        ylo, yhi = ylim_for_traces(
            model_trace, data=data_y, sem=sem_trace,
            show_sem=show_sem and sem_trace is not None and np.any(sem_trace),
        )
    else:
        ylo, yhi = ylim

    plot_timecourse(
        ax, np.arange(win_len), model_trace,
        data=None,
        sem=sem_trace,
        show_sem=show_sem and sem_trace is not None and np.any(sem_trace),
        title=title,
        ylim=(ylo, yhi),
        baseline=baseline,
        show_ylabel=show_ylabel,
        ticksize=6 if cell_ticks else 5,
        style_xaxis=style_xaxis,
        linestyle=linestyle,
    )
    if data_x is not None:
        ax.plot(data_x, data_y, color=DATA_COLOR, linewidth=TRACE_LW, linestyle=linestyle)


def _plot_moving_bar_cell_slices(
    ax,
    total_trace,
    sem_trace,
    slice_traces,
    slice_labels,
    title,
    before_steps,
    after_steps,
    *,
    data_trace=None,
    show_ylabel=False,
    show_sem=True,
    show_legend=False,
    cell_ticks=True,
    show_tick_labels=True,
    mark_cost_window=False,
    baseline=None,
):
    win_len = len(total_trace)
    data_x, data_y = None, None
    if data_trace is not None:
        data_x, data_y = _cost_window_overlay(data_trace, before_steps)

    def style_xaxis(ax):
        _style_moving_bar_relative_axis(
            ax, before_steps, after_steps, win_len,
            show_tick_labels=show_tick_labels,
            mark_cost_window=mark_cost_window,
        )

    extra = [total_trace]
    for label in slice_labels:
        extra.append(slice_traces[label])
    ylo, yhi = ylim_for_traces(
        total_trace, data=data_y, sem=sem_trace,
        show_sem=show_sem and sem_trace is not None and np.any(sem_trace),
        extra=extra[1:],
    )
    t = np.arange(win_len)
    if data_x is not None:
        ax.plot(data_x, data_y, color=DATA_COLOR, linewidth=TRACE_LW)
    colors = overlay_model_reds(len(slice_labels))
    for i, label in enumerate(slice_labels):
        ax.plot(t, slice_traces[label], color=colors[i], linewidth=TRACE_LW, label=label)
    if show_sem and sem_trace is not None and np.any(sem_trace):
        from plot.utils import plot_sem_band
        plot_sem_band(ax, t, total_trace, sem_trace)
    ax.plot(t, total_trace, color=colors[-1], linewidth=TRACE_LW, label='total')
    if title is not None:
        ax.set_title(title, fontsize=7, pad=2)
    ax.set_ylim(ylo, yhi)
    style_xaxis(ax)
    if show_ylabel:
        ax.set_ylabel('mV', fontsize=7)
    ax.tick_params(labelsize=6 if cell_ticks else 5)
    annotate_baseline(ax, baseline)
    if show_legend:
        ax.legend(fontsize=5, loc='upper right', framealpha=0.85)


def _bundle_slice_labels(bundle):
    if bundle.slice_overlay is None:
        return []
    return list(bundle.slice_overlay.keys())


def _bundle_slice_trace(bundle, label, key):
    return bundle.slice_overlay[label].model_mean[key]


def _moving_bar_all_scope_label(b_on):
    if b_on.slice_overlay is not None:
        pack = b_on.session.primary_pack
        cost_extent = pack.cost_extent
        at_x = b_on.slice_x_list if b_on.slice_axis in ('x', 'xy') else None
        at_y = b_on.slice_y_list if b_on.slice_axis in ('y', 'xy') else None
        parts = [column_at_scope_tag(at_x, at_y), 'overlay + total']
        if cost_extent is not None:
            parts.insert(0, f'cost_extent={cost_extent}')
        return ', '.join(parts)
    return _moving_bar_scope_label(b_on.session)


def _plot_moving_bar_all_from_bundles(path, b_on, b_2, title, *, right_only=True):
    t0 = time.perf_counter()
    single_column = b_on.single_column
    types = b_on.types
    wt_on = b_on.traces
    spec_names = _filter_right_specs(b_on.spec_names, right_only)
    ncols_on = len(spec_names)
    model_mean, model_sem = wt_on.model_mean, wt_on.model_sem
    data_mean = b_on.data_mean
    baselines = b_on.baselines
    baselines_2 = None
    wt_2 = None
    slice_labels = _bundle_slice_labels(b_on)
    has_slices = bool(slice_labels)
    if b_2 is not None:
        wt_2 = b_2.traces
        spec_2 = _filter_right_specs(b_2.spec_names, right_only)
        spec_names = list(spec_names) + list(spec_2)
        model_mean = {**model_mean, **wt_2.model_mean}
        model_sem = {**model_sem, **wt_2.model_sem}
        data_mean = {**data_mean, **b_2.data_mean}
        baselines_2 = b_2.baselines
    t_traces = time.perf_counter() - t0

    show_sem = not single_column and not has_slices
    nrows = len(types)
    ncols = len(spec_names)
    fig, axes = _moving_bar_figure(nrows, ncols)
    t1 = time.perf_counter()
    for ri, tname in enumerate(types):
        for ci, sname in enumerate(spec_names):
            ax = axes[ri, ci]
            key = (tname, sname)
            if key not in model_mean:
                ax.axis('off')
                continue
            bl = baselines.get(tname)
            b_src = b_on if ci < ncols_on else b_2
            wt = wt_on if ci < ncols_on else wt_2
            if baselines_2 is not None and ci >= ncols_on:
                bl = baselines_2.get(tname)
            before_steps, after_steps = _window_steps(wt, sname)
            if has_slices and b_src is not None and b_src.slice_overlay is not None:
                slice_traces = {
                    label: _bundle_slice_trace(b_src, label, key)
                    for label in slice_labels
                    if key in b_src.slice_overlay[label].model_mean
                }
                if not slice_traces:
                    ax.axis('off')
                    continue
                plot_labels = [label for label in slice_labels if label in slice_traces]
                _plot_moving_bar_cell_slices(
                    ax, model_mean[key], model_sem.get(key),
                    slice_traces, plot_labels,
                    sname, before_steps, after_steps,
                    data_trace=data_mean.get(key),
                    show_ylabel=(ci == 0),
                    show_sem=show_sem and key in model_sem and np.any(model_sem[key]),
                    show_legend=(ri == 0 and ci == 0),
                    cell_ticks=False,
                    show_tick_labels=(ri == nrows - 1),
                    mark_cost_window=True,
                    baseline=bl,
                )
            else:
                _plot_moving_bar_cell(
                    ax, model_mean[key], model_sem.get(key),
                    sname, before_steps, after_steps,
                    data_trace=data_mean.get(key),
                    show_ylabel=(ci == 0),
                    show_sem=show_sem and key in model_sem and np.any(model_sem[key]),
                    cell_ticks=False,
                    show_tick_labels=(ri == nrows - 1),
                    mark_cost_window=True,
                    baseline=bl,
                )
        axes[ri, 0].set_ylabel(tname, fontsize=8, labelpad=12)
    if title is None:
        title = 'Moving-bar model-all (right only)' if right_only else 'Moving-bar model-all'
    scope = _moving_bar_all_scope_label(b_on)
    fig.suptitle(title + f'  [{scope}, t_first_sti-aligned full window]', fontsize=12)
    _moving_bar_figure_adjust(fig)
    t_draw = time.perf_counter() - t1
    t2 = time.perf_counter()
    save_figure(fig, path, dpi=MOVING_BAR_DPI, rasterize=True)
    t_save = time.perf_counter() - t2
    print(
        f'plot_moving_bar_all: {path} traces={t_traces:.1f}s  '
        f'draw={t_draw:.1f}s  savefig={t_save:.1f}s  total={t_traces+t_draw+t_save:.1f}s'
    )


@torch.no_grad()
def plot_moving_bar_data(session_1, z, path, target, session_2=None, title=None, *,
                         bundle=None, bundle_2=None, save_trace_csv_dir: str | None = None):
    b_on = bundle or moving_bar_trace_bundle(
        session_1, z, target, save_trace_csv_dir=save_trace_csv_dir,
    )
    b_2 = None
    if session_2 is not None:
        b_2 = bundle_2 or moving_bar_trace_bundle(
            session_2, z, target, save_trace_csv_dir=save_trace_csv_dir,
        )
    single_column = b_on.single_column
    row_specs = _moving_bar_row_specs(b_on.session, b_on.target, b_on.side)
    readout_subtypes = list(row_specs.keys())
    ncols_half = max((len(v) for v in row_specs.values()), default=8)
    if b_2 is not None:
        row_specs_2 = _moving_bar_row_specs(b_2.session, b_2.target, b_2.side)
        ncols_half = max(ncols_half, max((len(v) for v in row_specs_2.values()), default=8))
        ncols = ncols_half * 2
    else:
        row_specs_2 = None
        ncols = ncols_half
    nrows = len(readout_subtypes)
    fig, axes = _moving_bar_figure(nrows, ncols)

    def _plot_row(ri, subtype, specs, col_offset, b, plot_side):
        wt = b.traces
        for ci, sname in enumerate(specs):
            ax = axes[ri, col_offset + ci]
            key = (subtype, sname)
            if key not in wt.model_mean:
                ax.axis('off')
                continue
            before_steps, after_steps = _window_steps(wt, sname)
            _plot_moving_bar_cell(
                ax, wt.model_mean[key], wt.model_sem[key],
                sname, before_steps, after_steps,
                data_trace=b.data_mean.get(key),
                show_ylabel=(col_offset + ci == 0), show_sem=not single_column,
                mark_cost_window=True,
                baseline=b.baselines.get(subtype),
                linestyle=_moving_bar_spec_linestyle(plot_side, subtype, sname),
            )

    for ri, subtype in enumerate(readout_subtypes):
        _plot_row(ri, subtype, row_specs[subtype], 0, b_on, b_on.side)
        if b_2 is not None:
            _plot_row(ri, subtype, row_specs_2[subtype], ncols_half, b_2, b_2.side)
        axes[ri, 0].set_ylabel(subtype, fontsize=8, labelpad=12)
    if title is None:
        title = 'Moving-bar model-data'
    scope = _moving_bar_scope_label(b_on.session)
    fig.suptitle(
        title + f'  [{scope}, t_first_sti-aligned full window]',
        fontsize=12,
    )
    _moving_bar_figure_adjust(fig)
    save_figure(fig, path, dpi=MOVING_BAR_DPI, rasterize=True)


@torch.no_grad()
def plot_moving_bar_all(session_1, z, path, target, session_2=None, title=None, *,
                        right_only=True, bundle=None, bundle_2=None,
                        at_x_list=None, at_y_list=None,
                        trace_kind: Literal['model', 'vm'] = 'model',
                        save_trace_csv_dir: str | None = None):
    b_on = bundle or moving_bar_trace_bundle(
        session_1, z, target, at_x_list=at_x_list, at_y_list=at_y_list,
        trace_kind=trace_kind,
        save_trace_csv_dir=save_trace_csv_dir,
    )
    b_2 = None
    if session_2 is not None:
        b_2 = bundle_2 or moving_bar_trace_bundle(
            session_2, z, target, at_x_list=at_x_list, at_y_list=at_y_list,
            trace_kind=trace_kind,
            save_trace_csv_dir=save_trace_csv_dir,
        )
    _plot_moving_bar_all_from_bundles(path, b_on, b_2, title, right_only=right_only)
