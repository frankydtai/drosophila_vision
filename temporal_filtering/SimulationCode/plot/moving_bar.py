#!/usr/bin/env python
"""Moving-bar plotting utilities extracted from plot_trained."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

import FiveCol_MedSim_Pytorch as fc
import Medulla_Library as ml
from plot.readout import moving_bar_row_types
from plot.utils import (
    baselines_for_types,
    plot_timecourse,
    save_figure,
    sem_from_traces,
    suppress_cost_sem,
)
from FiveCol_MedSim_Pytorch import t_on
from network.moving_bar_target import _borst_moving_bar_specs, load_fig1_trace
from column_mapper import borst_sti_columns
from t4_t5_preference import (
    READOUT_SUBTYPES,
    active_stimuli_for_subtype,
    fig1_key_for_stimulus,
    motion_preference,
    normalize_side,
)
from training_config import (
    COST_WINDOW,
    COST_WINDOW_AFTER,
    COST_WINDOW_BEFORE,
    DELTAT_MS,
)
from network.tiling import tile_stimulus_batches, tiling_from_stimulus_opts, unit_ring_layout
from visual_stimulus.moving_bar_stimulus import column_bar_center_step, field_bounds, gruntman_moving_bar_specs

MOVING_BAR_DPI = 100


@dataclass
class MovingBarWindowTraces:
    model_mean: dict
    model_sem: dict
    before_steps: dict[str, int] | None = None
    after_steps: dict[str, int] | None = None


@dataclass
class MovingBarTraceBundle:
    """One forward pass; ``cost`` always present, ``full`` when built with ``full=True``."""

    target: str
    types: list
    spec_names: list
    side: str
    single_column: bool
    baselines: dict
    data_mean: dict
    maxtime: int
    cost: MovingBarWindowTraces
    full: MovingBarWindowTraces | None = None
    session: object = field(default=None, repr=False, compare=False)


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


def _record_spec_full_horizon(t_centers, maxtime, spec_name, full_before, full_after, *, full):
    fb = min(t_centers)
    if full:
        full_before[spec_name] = fb
        full_after[spec_name] = int(maxtime) - max(t_centers)
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


def _column_t_centers(cols, spec, field_deg):
    return [
        int(column_bar_center_step(
            c.x_deg, c.y_deg, spec, field_deg, t_on=t_on, deltat_ms=fc.deltat,
        ))
        for c in cols
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


def _moving_bar_t0_grids(session, specs, cost_extent, maxtime, *, full):
    from network.stimulus import build_moving_bar_signals, cost_sti_columns

    C = session.backend.network
    n_batch = len(specs)
    full_before_steps = {}
    full_after_steps = {}
    t0_cost_map = {}
    t0_full_map = {}

    if C is not None:
        side = normalize_side(C.meta.get('side', 'right'))
        cols = cost_sti_columns(C, cost_extent=cost_extent)
        field_deg = C.meta.get('field_deg')
        if field_deg is None:
            field_deg = build_moving_bar_signals(
                C, t_on=t_on, deltat_ms=fc.deltat, device=C.node_type.device,
            ).info['field_deg']
        types = list(C.type_names)
        type_ids = _type_ids_np(C.node_type)
        for bi, spec in enumerate(specs):
            t_centers = _column_t_centers(cols, spec, field_deg)
            fb = _record_spec_full_horizon(
                t_centers, maxtime, spec.name, full_before_steps, full_after_steps, full=full,
            )
            for c, tc in zip(cols, t_centers):
                uv = (bi, int(c.u), int(c.v))
                t0_cost_map[uv] = tc - COST_WINDOW_BEFORE
                if full:
                    t0_full_map[uv] = tc - fb
        t0_cost_bn = _moving_bar_t0_grid(C, cols, n_batch, t0_cost_map)
        t0_full_bn = _moving_bar_t0_grid(C, cols, n_batch, t0_full_map) if full else None
    else:
        side = "right"
        cols_all = list(borst_sti_columns())
        col_ids = list(range(ml.nofcols))
        cols = [cols_all[i] for i in col_ids]
        field_deg = field_bounds(cols_all)
        types = list(ml.ctype.tolist())
        type_ids = _type_ids_np(session.backend.conn.node_type)
        t0_cost_bn = np.full((n_batch, ml.n_state_units()), -1, dtype=np.int64)
        t0_full_bn = np.full((n_batch, ml.n_state_units()), -1, dtype=np.int64) if full else None
        for bi, spec in enumerate(specs):
            t_centers = _column_t_centers(cols, spec, field_deg)
            fb = _record_spec_full_horizon(
                t_centers, maxtime, spec.name, full_before_steps, full_after_steps, full=full,
            )
            for col_id, col, tc in zip(col_ids, cols, t_centers):
                t0_cost_bn[bi, ml.column_slice(col_id)] = tc - COST_WINDOW_BEFORE
                if full:
                    t0_full_bn[bi, ml.column_slice(col_id)] = tc - fb

    return types, type_ids, t0_cost_bn, t0_full_bn, full_before_steps, full_after_steps, side


def _moving_bar_row_specs(session, target, side):
    readout_subtypes = moving_bar_row_types(session, target)
    contrast = "bright" if "bright" in target else "dark"
    return {
        st: [f'{d}_{c}_{w}' for d, c, w in active_stimuli_for_subtype(side, st) if c == contrast]
        for st in readout_subtypes
    }


@torch.no_grad()
def moving_bar_trace_bundle(session, z, target, *, full=False):
    """Run one forward; ``cost`` traces always; ``full`` when ``full=True``."""
    pack = session.pack_for(target)
    schema = list(session.schema)
    p = fc.assign_params(z, schema, session.backend)
    model_full, vm_ref = fc._run_conductance_full(session, p, pack.signal, return_ref=True)
    model_full = _scale_model_full(model_full.cpu().numpy(), p, session.backend)
    specs = _bar_specs_for_session(session, target)
    spec_names = [s.name for s in specs]
    single_column = suppress_cost_sem(session, target)
    cost_extent = pack.cost_extent
    maxtime = int(session.maxtime)
    vm_ref_np = vm_ref[0].cpu().numpy()
    C = session.backend.network
    types, type_ids, t0_cost_bn, t0_full_bn, full_before_steps, full_after_steps, side = (
        _moving_bar_t0_grids(session, specs, cost_extent, maxtime, full=full)
    )
    if C is not None:
        type_names = list(C.type_names)
        opts = dict((session.train_opts or {}).get(f"{target}_stimulus_opts") or {})
        batches = tile_stimulus_batches(tiling_from_stimulus_opts(C, opts))
        batch_idx, unit_idx, radius, type_idx = unit_ring_layout(C, batches)
        baselines = baselines_for_types(
            pack, session.backend, vm_ref_np, types, type_ids, type_names,
            ring_layout=(batch_idx, unit_idx, type_idx, radius),
        )
    else:
        baselines = baselines_for_types(
            pack, session.backend, vm_ref_np, types, type_ids, types,
        )
    windows_cost = _windows_by_batch(model_full, t0_cost_bn, COST_WINDOW)
    cost_mean, cost_sem = _aggregate_moving_bar_traces(
        windows_cost, t0_cost_bn, type_ids, types, spec_names, single_column,
    )
    data_mean = _load_moving_bar_data_mean(session, target, types, specs, side)
    cost = MovingBarWindowTraces(model_mean=cost_mean, model_sem=cost_sem)
    full_traces = None
    if full:
        win_lens = [
            full_before_steps[sname] + full_after_steps[sname] + 1
            for sname in spec_names
        ]
        windows_full = _windows_by_batch(model_full, t0_full_bn, win_lens)
        full_mean, full_sem = _aggregate_moving_bar_traces(
            windows_full, t0_full_bn, type_ids, types, spec_names, single_column,
        )
        full_traces = MovingBarWindowTraces(
            model_mean=full_mean,
            model_sem=full_sem,
            before_steps=full_before_steps,
            after_steps=full_after_steps,
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
        cost=cost,
        full=full_traces,
        session=session,
    )


def _bundle_window(b, full):
    if full:
        if b.full is None:
            raise ValueError('full traces missing; build bundle with full=True')
        return b.full
    return b.cost


def _window_steps(wt, sname, full):
    if full:
        return wt.before_steps[sname], wt.after_steps[sname]
    return COST_WINDOW_BEFORE, COST_WINDOW_AFTER


def _moving_bar_scope_label(session):
    pack = session.primary_pack
    cost_extent = pack.cost_extent
    C = session.backend.network
    if cost_extent is not None:
        if C is not None:
            from network.stimulus import cost_sti_columns
            ncols = len(cost_sti_columns(C, cost_extent=cost_extent))
            return f'cost_extent={cost_extent} ({ncols} sti columns)'
        return f'cost_extent={cost_extent}'
    if C is not None:
        from network.stimulus import sti_columns
        return f'avg over {len(sti_columns(C))} sti columns'
    return f'avg over {ml.nofcols} Borst columns'


def _moving_bar_window_label(before_steps, after_steps) -> str:
    before_s, after_s = _rel_window_seconds(before_steps, after_steps)
    return f't_center - {before_s:g} .. + {after_s:g} s'


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

    def style_xaxis(ax):
        _style_moving_bar_relative_axis(
            ax, before_steps, after_steps, win_len,
            show_tick_labels=show_tick_labels,
            mark_cost_window=mark_cost_window,
        )

    plot_timecourse(
        ax, np.arange(win_len), model_trace,
        data=data_trace,
        sem=sem_trace,
        show_sem=show_sem and sem_trace is not None and np.any(sem_trace),
        title=title,
        ylim=ylim,
        baseline=baseline,
        show_ylabel=show_ylabel,
        ticksize=6 if cell_ticks else 5,
        style_xaxis=style_xaxis,
        linestyle=linestyle,
    )


@torch.no_grad()
def plot_moving_bar_data(session, z, path, target, session_off=None, title=None, *,
                         bundle=None, bundle_off=None):
    b_on = bundle or moving_bar_trace_bundle(session, z, target)
    b_off = None
    if session_off is not None:
        b_off = bundle_off or moving_bar_trace_bundle(session_off, z, target)
    single_column = b_on.single_column
    row_specs = _moving_bar_row_specs(b_on.session, b_on.target, b_on.side)
    readout_subtypes = list(row_specs.keys())
    ncols_half = max((len(v) for v in row_specs.values()), default=8)
    if b_off is not None:
        row_specs_off = _moving_bar_row_specs(b_off.session, b_off.target, b_off.side)
        ncols_half = max(ncols_half, max((len(v) for v in row_specs_off.values()), default=8))
        ncols = ncols_half * 2
    else:
        row_specs_off = None
        ncols = ncols_half
    nrows = len(readout_subtypes)
    fig, axes = _moving_bar_figure(nrows, ncols)

    def _plot_row(ri, subtype, specs, col_offset, b, plot_side):
        for ci, sname in enumerate(specs):
            ax = axes[ri, col_offset + ci]
            key = (subtype, sname)
            if key not in b.cost.model_mean:
                ax.axis('off')
                continue
            _plot_moving_bar_cell(
                ax, b.cost.model_mean[key], b.cost.model_sem[key],
                sname, COST_WINDOW_BEFORE, COST_WINDOW_AFTER,
                data_trace=b.data_mean.get(key),
                show_ylabel=(col_offset + ci == 0), show_sem=not single_column,
                baseline=b.baselines.get(subtype),
                linestyle=_moving_bar_spec_linestyle(plot_side, subtype, sname),
            )

    for ri, subtype in enumerate(readout_subtypes):
        _plot_row(ri, subtype, row_specs[subtype], 0, b_on, b_on.side)
        if b_off is not None:
            _plot_row(ri, subtype, row_specs_off[subtype], ncols_half, b_off, b_off.side)
        axes[ri, 0].set_ylabel(subtype, fontsize=8, labelpad=12)
    if title is None:
        title = 'Moving-bar model-data'
    scope = _moving_bar_scope_label(b_on.session)
    fig.suptitle(
        title + f'  [{scope}, {_moving_bar_window_label(COST_WINDOW_BEFORE, COST_WINDOW_AFTER)}]',
        fontsize=12,
    )
    _moving_bar_figure_adjust(fig)
    save_figure(fig, path, dpi=MOVING_BAR_DPI, rasterize=True)


def _plot_moving_bar_all_from_bundles(path, b_on, b_off, title, *, right_only=True, full=False):
    t0 = time.perf_counter()
    single_column = b_on.single_column
    types = b_on.types
    wt_on = _bundle_window(b_on, full)
    spec_names = _filter_right_specs(b_on.spec_names, right_only)
    ncols_on = len(spec_names)
    model_mean, model_sem = wt_on.model_mean, wt_on.model_sem
    data_mean = b_on.data_mean
    baselines = b_on.baselines
    baselines_off = None
    wt_off = None
    if b_off is not None:
        wt_off = _bundle_window(b_off, full)
        spec_off = _filter_right_specs(b_off.spec_names, right_only)
        spec_names = list(spec_names) + list(spec_off)
        model_mean = {**model_mean, **wt_off.model_mean}
        model_sem = {**model_sem, **wt_off.model_sem}
        data_mean = {**data_mean, **b_off.data_mean}
        baselines_off = b_off.baselines
    t_traces = time.perf_counter() - t0

    show_sem = not single_column
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
            wt = wt_on if ci < ncols_on else wt_off
            if baselines_off is not None and ci >= ncols_on:
                bl = baselines_off.get(tname)
            before_steps, after_steps = _window_steps(wt, sname, full)
            _plot_moving_bar_cell(
                ax, model_mean[key], model_sem.get(key),
                sname, before_steps, after_steps,
                data_trace=None if full else data_mean.get(key),
                show_ylabel=(ci == 0),
                show_sem=show_sem and key in model_sem and np.any(model_sem[key]),
                cell_ticks=False,
                show_tick_labels=(ri == nrows - 1),
                mark_cost_window=full,
                baseline=bl,
            )
        axes[ri, 0].set_ylabel(tname, fontsize=8, labelpad=12)
    if title is None:
        if full:
            title = 'Moving-bar model-all full horizon'
        else:
            title = 'Moving-bar model-all (right only)' if right_only else 'Moving-bar model-all'
    scope = _moving_bar_scope_label(b_on.session)
    if full:
        window_label = 't_center-aligned full window'
    else:
        window_label = _moving_bar_window_label(COST_WINDOW_BEFORE, COST_WINDOW_AFTER)
    fig.suptitle(title + f'  [{scope}, {window_label}]', fontsize=12)
    _moving_bar_figure_adjust(fig)
    t_draw = time.perf_counter() - t1
    t2 = time.perf_counter()
    save_figure(fig, path, dpi=MOVING_BAR_DPI, rasterize=True)
    t_save = time.perf_counter() - t2
    label = 'plot_moving_bar_all_full' if full else 'plot_moving_bar_all'
    print(
        f'{label}: traces={t_traces:.1f}s  '
        f'draw={t_draw:.1f}s  savefig={t_save:.1f}s  total={t_traces+t_draw+t_save:.1f}s'
    )


@torch.no_grad()
def plot_moving_bar_all(session, z, path, target, session_off=None, title=None, *,
                        right_only=True, full=False, bundle=None, bundle_off=None):
    b_on = bundle or moving_bar_trace_bundle(session, z, target, full=full)
    b_off = None
    if session_off is not None:
        b_off = bundle_off or moving_bar_trace_bundle(session_off, z, target, full=full)
    _plot_moving_bar_all_from_bundles(path, b_on, b_off, title, right_only=right_only, full=full)
