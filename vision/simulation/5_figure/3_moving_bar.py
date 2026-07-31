"""Moving-bar plotting utilities extracted from ``figure.plot_run``."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

import training
from task.moving_bar.data import (
    READOUT_SUBTYPES,
    fig1_key_for_stimulus,
    motion_preference,
    moving_bar_cell_title,
    moving_bar_dsi_lookup,
)
from figure.readout import pack_readout_types, plot_types_in_order
from figure.util import (
    DATA_COLOR,
    TRACE_LW,
    TRACE_YLIM,
    PlotTimer,
    annotate_baseline,
    apply_out_scale,
    as_numpy,
    baselines_for_types,
    bundle_panel_title,
    bundle_prep_s,
    column_at_scope_tag,
    cell_ylabel,
    overlay_model_reds,
    plot_pre_post_line,
    plot_sem_band,
    plot_timecourse,
    save_figure,
    save_forward_trace_csvs,
    sem_from_traces,
    slice_axis_name,
    slice_coord_specs,
    suppress_cost_sem,
    v_th_by_type_name,
)
import network.path  # noqa: F401  # ensure FAFBv783 modules are importable
from task.moving_bar.data import (
    bar_specs_for_session,
    load_fig1_trace,
    moving_bar_row_specs,
    moving_bar_session_t0_grids,
    moving_bar_units_on_columns,
)
from task.moving_bar.input import (
    filter_sti_columns,
    moving_bar_cost_columns,
    network_uv_np,
)
from task.moving_bar.input import (
    cost_window_after_t,
    cost_window_before_t,
)

MOVING_BAR_DPI = 100



@dataclass
class MovingBarWindowTraces:
    ca_mean: dict
    ca_sem: dict
    ca_n: dict
    before_t: dict
    after_t: dict
    t0_bn: np.ndarray
    type_ids: np.ndarray
    types: list


@dataclass
class MovingBarTraceBundle:
    target: str
    types: list
    spec_names: list
    side: str
    single_column: bool
    baselines: dict
    data_mean: dict
    n_t: int
    traces: MovingBarWindowTraces
    session: object
    at_x: int | None = None
    at_y: int | None = None
    n_filter_cols: int = 0
    slice_overlay: dict | None = None
    slice_axis: str | None = None
    slice_x_list: list | None = None
    slice_y_list: list | None = None
    align_at_x: int | None = None
    align_at_y: int | None = None
    prep_s: float = 0.0
    v_th_by_name: dict = field(default_factory=dict)
    show_pre: bool = True
    t_onset: int | None = None

    @property
    def has_slices(self):
        return bool(self.slice_overlay)


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
    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.50, wspace=0.35)


def _moving_bar_cell_title(
    bundle,
    sname,
    ca_mean,
    data_mean,
    ca_dsi_lookup,
    data_dsi_lookup,
    key,
    *,
    type_name,
):
    return moving_bar_cell_title(
        bundle_panel_title(bundle, sname, type_name=type_name),
        ca_dsi=ca_dsi_lookup.get(key),
        data_dsi=data_dsi_lookup.get(key),
        has_data=data_mean.get(key) is not None,
    )


def _type_ids_np(node_type):
    return np.asarray(as_numpy(node_type), dtype=np.int64)


def _type_ids_for_plot_order(connectome_type_names, node_type, plot_types):
    """Map unit ``node_type`` (index into connectome names) to ``plot_types`` index.

    ``plot_types_in_order`` reorders names; aggregating with raw ``node_type`` against
    ``enumerate(plot_types)`` mislabels every type whose plot index ≠ connectome index.
    """
    c_ids = _type_ids_np(node_type)
    name_to_plot = {str(n): i for i, n in enumerate(plot_types)}
    out = np.full(c_ids.shape, -1, dtype=np.int64)
    for ci, name in enumerate(connectome_type_names):
        pi = name_to_plot.get(str(name))
        if pi is None:
            continue
        out[c_ids == int(ci)] = int(pi)
    return out


def _plot_types_and_ids(session):
    """Plot-family type order + remapped ``type_ids`` for bar aggregation."""
    C = session.backend.network
    if C is None:
        raise ValueError("_plot_types_and_ids requires session.backend.network")
    types = plot_types_in_order(C.type_names)
    type_ids = _type_ids_for_plot_order(C.type_names, C.node_type, types)
    return types, type_ids


def _rel_window_seconds(before_t, after_t, delta_ms):
    scale = float(delta_ms) / 1000.0
    return before_t * scale, after_t * scale


def _filter_right_specs(spec_names, right_only):
    if right_only:
        return [s for s in spec_names if s.startswith('right_')]
    return list(spec_names)


def _windows_by_batch(ca_full, t0_bn, win_lens):
    """``win_lens``: int (uniform) or length-``B`` sequence of window lengths."""
    n_batch = ca_full.shape[0]
    if isinstance(win_lens, int):
        win_lens = (win_lens,) * n_batch
    out = []
    for bi in range(n_batch):
        wl = int(win_lens[bi])
        sl = ca_full[bi:bi + 1]
        t0 = t0_bn[bi:bi + 1]
        t_len = sl.shape[1]
        n_units = sl.shape[2]
        win_ix = np.arange(wl, dtype=np.int64)
        t_idx = t0[..., None] + win_ix[None, None, :]
        t_safe = np.clip(t_idx, 0, t_len - 1)
        b_ix = np.zeros(1, dtype=np.int64)[:, None, None]
        u_ix = np.arange(n_units, dtype=np.int64)[None, :, None]
        batch = sl[b_ix, t_safe, u_ix].astype(np.float64, copy=False)
        batch[t_idx < 0] = 0.0
        out.append(batch[0])
    return out


def _aggregate_moving_bar_traces(
    windows_by_batch, t0_bn, type_ids, types, spec_names, single_column, *,
    col_mask=None,
):
    """``windows_by_batch[bi]`` shape ``(n_units, W_bi)``."""
    ca_mean, ca_sem, ca_n = {}, {}, {}
    valid = t0_bn >= 0
    for ti, tname in enumerate(types):
        type_mask = type_ids == ti
        if not type_mask.any():
            continue
        for bi, sname in enumerate(spec_names):
            unit_mask = valid[bi] & type_mask
            if col_mask is not None:
                unit_mask = unit_mask & col_mask[bi]
            if not unit_mask.any():
                continue
            uids = np.nonzero(unit_mask)[0]
            arr = windows_by_batch[bi][uids]
            key = (tname, sname)
            ca_mean[key] = arr.mean(axis=0)
            ca_sem[key] = sem_from_traces(arr, single_column=single_column)
            ca_n[key] = int(np.unique(uids).size)
    return ca_mean, ca_sem, ca_n


def _network_column_unit_mask(C, filt_cols, n_batch):
    u_np, v_np = network_uv_np(C)
    col_uv = {(int(c.u), int(c.v)) for c in filt_cols}
    unit_in_col = np.array(
        [(int(u), int(v)) in col_uv for u, v in zip(u_np, v_np)],
        dtype=bool,
    )
    return np.broadcast_to(unit_in_col, (n_batch, C.n_units)).copy()



def _t0_ref_for_align_column(t0_bn, bi, ref_col, *, C):
    if C is None:
        raise ValueError("_t0_ref_for_align_column requires network C")
    u_np, v_np = network_uv_np(C)
    on_ref = (u_np == int(ref_col.u)) & (v_np == int(ref_col.v))
    t0_ref = int(t0_bn[bi, on_ref][0])
    if t0_ref < 0:
        loc = f'({int(ref_col.u)},{int(ref_col.v)})'
        raise SystemExit(f'--align-xy ref column {loc} has no valid t0')
    return t0_ref


def _t0_bn_slice_aligned_to_ref(
    t0_bn, n_batch, filt_cols, align_at_x, align_at_y, *,
    session, cost_extent,
):
    """Copy ``t0_bn`` with slice units forced to the ref column ``t0`` (plot only)."""
    C = session.backend.network
    if C is None:
        raise ValueError("_t0_bn_slice_aligned_to_ref requires session.backend.network")
    all_cols = moving_bar_cost_columns(C, cost_extent=cost_extent)
    ref_cols = filter_sti_columns(all_cols, at_x=align_at_x, at_y=align_at_y)
    if len(ref_cols) != 1:
        raise SystemExit(
            f'--align-xy must match exactly one sti column within cost_extent, '
            f'got {len(ref_cols)} for x={align_at_x!r} y={align_at_y!r}',
        )
    ref_col = ref_cols[0]
    u_np, v_np = network_uv_np(C)
    out = t0_bn.copy()
    for bi in range(n_batch):
        t0_ref = _t0_ref_for_align_column(out, bi, ref_col, C=C)
        for col in filt_cols:
            on_col = (u_np == int(col.u)) & (v_np == int(col.v))
            out[bi, on_col] = t0_ref
    return out


def _moving_bar_slice_overlay_traces(
    session, target, trace_full, base_wt, spec_names, *, at_x=None, at_y=None,
    align_at_x=None, align_at_y=None,
):
    """Per-axis slice traces aligned to ``base_wt`` window geometry."""
    if base_wt.t0_bn is None or base_wt.type_ids is None or base_wt.types is None:
        raise ValueError("base_wt missing cached t0_bn/types for slice overlay")
    pack = session.pack_for(target)
    types = base_wt.types
    type_ids = base_wt.type_ids
    t0_full_bn = base_wt.t0_bn
    n_batch = len(spec_names)
    C = session.backend.network
    if C is None:
        raise ValueError("_moving_bar_slice_overlay_traces requires session.backend.network")
    all_cols = moving_bar_cost_columns(C, cost_extent=pack.cost_extent)
    filt_cols = filter_sti_columns(all_cols, at_x=at_x, at_y=at_y)
    if not filt_cols:
        return None
    col_mask = _network_column_unit_mask(C, filt_cols, n_batch)
    win_lens = [
        base_wt.before_t[sname] + base_wt.after_t[sname] + 1
        for sname in spec_names
    ]
    t0_use = t0_full_bn
    if align_at_x is not None and align_at_y is not None:
        t0_use = _t0_bn_slice_aligned_to_ref(
            t0_full_bn, n_batch, filt_cols, align_at_x, align_at_y,
            session=session, cost_extent=pack.cost_extent,
        )
    windows_full = _windows_by_batch(trace_full, t0_use, win_lens)
    ca_mean, ca_sem, ca_n = _aggregate_moving_bar_traces(
        windows_full, t0_use, type_ids, types, spec_names, True, col_mask=col_mask,
    )
    return MovingBarWindowTraces(
        ca_mean=ca_mean,
        ca_sem=ca_sem,
        ca_n=ca_n,
        before_t=base_wt.before_t,
        after_t=base_wt.after_t,
        t0_bn=t0_full_bn,
        type_ids=type_ids,
        types=types,
    )


def _fig1_trace_delta(trace: np.ndarray, delta_ms: float) -> np.ndarray:
    """ΔVm for fig1 cost-window traces (subtract pre-stimulus mean)."""
    trace = np.asarray(trace, dtype=np.float64)
    i_on = cost_window_before_t(delta_ms)
    if i_on > 0 and i_on < len(trace):
        return trace - float(np.mean(trace[:i_on]))
    return trace - float(trace[0])


def _load_moving_bar_data_mean(session, target, types, specs, side):
    data_mean = {}
    row_types = plot_types_in_order(pack_readout_types(session, target))
    for subtype in row_types:
        if subtype not in types:
            continue
        for spec in specs:
            trace_id = fig1_key_for_stimulus(side, subtype, spec)
            if trace_id is None:
                continue
            trace = _fig1_trace_delta(load_fig1_trace(trace_id), session.delta_ms)
            data_mean[(subtype, spec.name)] = trace
    return data_mean


def _moving_bar_baselines(C, v_onset, types, type_ids, type_names, cost_extent, *, at_x=None, at_y=None):
    """Mean ``v_onset`` per type over units on moving-bar cost columns (matches trace scope)."""
    cols = moving_bar_cost_columns(C, cost_extent=cost_extent)
    if at_x is not None or at_y is not None:
        cols = filter_sti_columns(cols, at_x=at_x, at_y=at_y)
    v_onset = np.asarray(v_onset, dtype=np.float64)
    out = {}
    for tname in types:
        u = moving_bar_units_on_columns(C, tname, cols)
        out[tname] = float(v_onset[u].mean()) if u.size else np.nan
    return out


def _moving_bar_traces_from_forward(
    session, target, trace_full, v_onset_np, specs, spec_names, *,
    at_x=None, at_y=None,
):
    pack = session.pack_for(target)
    cost_extent = pack.cost_extent
    n_t = int(session.n_t)
    _t_onset = int(pack.signal.shape[1] - pack.data.shape[1])
    grids = moving_bar_session_t0_grids(
        session, specs, cost_extent, n_t, at_x=at_x, at_y=at_y,
        t_onset=_t_onset, delta_ms=session.delta_ms,
    )
    types, type_ids = _plot_types_and_ids(session)
    t0_full_bn = grids.t0_bn
    full_before_t = grids.before_t
    full_after_t = grids.after_t
    side = grids.side
    n_filter_cols = grids.n_filter_cols
    single_column = suppress_cost_sem(session, target) or n_filter_cols == 1
    win_lens = [
        full_before_t[sname] + full_after_t[sname] + 1
        for sname in spec_names
    ]
    windows_full = _windows_by_batch(trace_full, t0_full_bn, win_lens)
    trace_mean, trace_sem, trace_n = _aggregate_moving_bar_traces(
        windows_full, t0_full_bn, type_ids, types, spec_names, single_column,
    )
    return MovingBarWindowTraces(
        ca_mean=trace_mean,
        ca_sem=trace_sem,
        ca_n=trace_n,
        before_t=full_before_t,
        after_t=full_after_t,
        t0_bn=t0_full_bn,
        type_ids=type_ids,
        types=types,
    ), types, side, n_filter_cols, _t_onset


@torch.no_grad()
def moving_bar_trace_bundle(session, z, target, *, at_x=None, at_y=None,
                            at_x_list=None, at_y_list=None,
                            align_at_x=None, align_at_y=None,
                            save_trace_csv_dir: str | None = None, show_pre=True):
    """Run one forward; t_first_sti-aligned full-window model traces."""
    t_prep0 = time.perf_counter()
    pack = session.pack_for(target)
    schema = list(session.schema)
    p = training.assign_params(z, schema, session.backend)
    v_delta, v_onset, _v_full = training.run_full(
        session, p, pack.signal, return_v_onset=True, pack=pack,
    )
    v_onset_np = v_onset[0].cpu().numpy()
    save_forward_trace_csvs(
        save_trace_csv_dir, target,
        ref=v_onset_np, trace_full=v_delta,
        ref_stem='moving_bar_v_onset',
    )
    trace_full = apply_out_scale(
        p, v_delta, None, session.backend,
    ).cpu().numpy()
    specs = bar_specs_for_session(session, target)
    spec_names = [s.name for s in specs]
    n_t = int(session.n_t)
    C = session.backend.network
    traces, types, side, n_filter_cols, t_onset = _moving_bar_traces_from_forward(
        session, target, trace_full, v_onset_np, specs, spec_names,
    )
    if C is not None:
        type_names = list(C.type_names)
        type_ids = _type_ids_np(C.node_type)
        baselines = _moving_bar_baselines(
            C, v_onset_np, types, type_ids, type_names, pack.cost_extent,
            at_x=at_x, at_y=at_y,
        )
    else:
        type_ids = _type_ids_np(session.backend.conn.node_type)
        baselines = baselines_for_types(
            pack, session.backend, v_onset_np, types, type_ids, types,
        )
    single_column = suppress_cost_sem(session, target) or n_filter_cols == 1
    data_mean = _load_moving_bar_data_mean(
        session, target, types, specs, side,
    )
    slice_overlay = None
    slice_axis = None
    slice_x_list = None
    slice_y_list = None
    specs = slice_coord_specs(at_x_list, at_y_list)
    if specs:
        slice_axis = slice_axis_name(at_x_list, at_y_list)
        slice_x_list = list(at_x_list) if at_x_list is not None else None
        slice_y_list = list(at_y_list) if at_y_list is not None else None
        slice_overlay = {}
        for label, xv, yv in specs:
            wt = _moving_bar_slice_overlay_traces(
                session, target, trace_full, traces, spec_names,
                at_x=xv, at_y=yv,
                align_at_x=align_at_x, align_at_y=align_at_y,
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
    return MovingBarTraceBundle(
        target=target,
        types=types,
        spec_names=spec_names,
        side=side,
        single_column=single_column,
        baselines=baselines,
        data_mean=data_mean,
        n_t=n_t,
        traces=traces,
        session=session,
        at_x=at_x,
        at_y=at_y,
        n_filter_cols=n_filter_cols,
        slice_overlay=slice_overlay,
        slice_axis=slice_axis,
        slice_x_list=slice_x_list,
        slice_y_list=slice_y_list,
        align_at_x=align_at_x,
        align_at_y=align_at_y,
        prep_s=time.perf_counter() - t_prep0,
        v_th_by_name=v_th_by_type_name(z, session),
        show_pre=bool(show_pre),
        t_onset=int(t_onset),
    )


def _window_t(wt, sname):
    return wt.before_t[sname], wt.after_t[sname]


def _moving_bar_pre_end(bundle, subtype, sname):
    """Median relative index of global ``t_onset`` within the plotted window."""
    t_onset = getattr(bundle, "t_onset", None)
    wt = bundle.traces
    if t_onset is None or wt.t0_bn is None:
        return 0
    try:
        bi = bundle.spec_names.index(sname)
        ti = wt.types.index(subtype)
    except ValueError:
        return 0
    key = (subtype, sname)
    if key not in wt.ca_mean:
        return 0
    win_len = int(np.asarray(wt.ca_mean[key]).shape[0])
    type_mask = wt.type_ids == ti
    valid = (wt.t0_bn[bi] >= 0) & type_mask
    if not bool(valid.any()):
        return 0
    pre = np.clip(int(t_onset) - wt.t0_bn[bi, valid], 0, win_len)
    return int(np.median(pre))


def _cost_window_overlay(cost_trace, before_t, delta_ms):
    """Fig1 overlay x/y within full-window coordinates (cost window only)."""
    i0 = before_t - cost_window_before_t(delta_ms)
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
    if C is None:
        raise ValueError("_moving_bar_scope_label requires session.backend.network")
    if at_x is not None or at_y is not None:
        ncol_part = f'{n_filter_cols} sti column'
        if n_filter_cols != 1:
            ncol_part += 's'
        parts = [column_at_scope_tag(at_x, at_y), ncol_part]
        if cost_extent is not None:
            parts.insert(0, f'cost_extent={cost_extent}')
        return ', '.join(parts)
    if cost_extent is not None:
        from task.moving_bar.input import moving_bar_cost_columns
        ncols = len(moving_bar_cost_columns(C, cost_extent=cost_extent))
        return f'cost_extent={cost_extent} ({ncols} sti columns)'
    from task.moving_bar.input import sti_columns
    return f'avg over {len(sti_columns(C))} sti columns'


def _style_moving_bar_relative_axis(
    ax, before_t, after_t, win_len, *,
    delta_ms,
    show_tick_labels=True, mark_cost_window=False,
):
    end = win_len - 1
    ax.set_xlim(0, end)
    ax.set_xticks([0, before_t, end])
    before_s, after_s = _rel_window_seconds(before_t, after_t, delta_ms)
    ax.set_xticklabels([f'{-before_s:g}', '0', f'{after_s:g}'], fontsize=6)
    if not show_tick_labels:
        ax.tick_params(labelbottom=False)
    if mark_cost_window:
        cw_before = cost_window_before_t(delta_ms)
        cw_after = cost_window_after_t(delta_ms)
        for x in (before_t - cw_before, before_t + cw_after):
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
    ca_trace,
    sem_trace,
    title,
    before_t,
    after_t,
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
    pre_end=0,
    show_pre=False,
    delta_ms=None,
):
    win_len = len(ca_trace)
    data_x, data_y = None, None
    if data_trace is not None:
        data_x, data_y = _cost_window_overlay(data_trace, before_t, delta_ms)

    def style_xaxis(ax):
        _style_moving_bar_relative_axis(
            ax, before_t, after_t, win_len,
            delta_ms=delta_ms,
            show_tick_labels=show_tick_labels,
            mark_cost_window=mark_cost_window,
        )

    if ylim is None:
        ylo, yhi = TRACE_YLIM
    else:
        ylo, yhi = ylim

    plot_timecourse(
        ax, np.arange(win_len),
        [{
            "model": ca_trace,
            "data": None,
            "sem": sem_trace,
            "linestyle": linestyle,
        }],
        show_sem=show_sem and sem_trace is not None and np.any(sem_trace),
        title=title,
        ylim=(ylo, yhi),
        baseline=baseline,
        show_ylabel=show_ylabel,
        ticksize=6 if cell_ticks else 5,
        style_xaxis=style_xaxis,
        pre_end=pre_end,
        show_pre=show_pre,
    )
    # Gray fig1 overlay is already restricted to the cost window (no global pre).
    if data_x is not None:
        ax.plot(data_x, data_y, color=DATA_COLOR, linewidth=TRACE_LW, linestyle=linestyle)


def _plot_moving_bar_cell_slices(
    ax,
    total_trace,
    sem_trace,
    slice_traces,
    slice_labels,
    title,
    before_t,
    after_t,
    *,
    data_trace=None,
    show_ylabel=False,
    show_sem=True,
    show_legend=False,
    cell_ticks=True,
    show_tick_labels=True,
    mark_cost_window=False,
    baseline=None,
    ylim=None,
    pre_end=0,
    show_pre=False,
    delta_ms=None,
):
    win_len = len(total_trace)
    data_x, data_y = None, None
    if data_trace is not None:
        data_x, data_y = _cost_window_overlay(data_trace, before_t, delta_ms)

    def style_xaxis(ax):
        _style_moving_bar_relative_axis(
            ax, before_t, after_t, win_len,
            delta_ms=delta_ms,
            show_tick_labels=show_tick_labels,
            mark_cost_window=mark_cost_window,
        )

    if ylim is None:
        ylo, yhi = TRACE_YLIM
    else:
        ylo, yhi = ylim
    t = np.arange(win_len)
    if data_x is not None:
        ax.plot(data_x, data_y, color=DATA_COLOR, linewidth=TRACE_LW)
    colors = overlay_model_reds(len(slice_labels))
    for i, label in enumerate(slice_labels):
        plot_pre_post_line(
            ax, t, slice_traces[label], pre_end=pre_end,
            show_pre=show_pre, draw_pre=True,
            color=colors[i], linestyle='-', linewidth=TRACE_LW, label=label,
        )
    if show_sem and sem_trace is not None and np.any(sem_trace):
        split = max(0, min(int(pre_end or 0), win_len))
        plot_sem_band(ax, t[split:], total_trace[split:], sem_trace[split:])
    plot_pre_post_line(
        ax, t, total_trace, pre_end=pre_end,
        show_pre=show_pre, draw_pre=True,
        color=colors[-1], linestyle='-', linewidth=TRACE_LW, label='total',
    )
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
    return bundle.slice_overlay[label].ca_mean[key]


def _moving_bar_all_scope_label(bundle_on):
    if bundle_on.has_slices:
        pack = bundle_on.session.primary_pack
        cost_extent = pack.cost_extent
        at_x = bundle_on.slice_x_list if bundle_on.slice_axis in ('x', 'xy') else None
        at_y = bundle_on.slice_y_list if bundle_on.slice_axis in ('y', 'xy') else None
        parts = [column_at_scope_tag(at_x, at_y), 'overlay + total']
        if bundle_on.align_at_x is not None and bundle_on.align_at_y is not None:
            parts.append(
                'aligned to '
                + column_at_scope_tag(bundle_on.align_at_x, bundle_on.align_at_y),
            )
        if cost_extent is not None:
            parts.insert(0, f'cost_extent={cost_extent}')
        return ', '.join(parts)
    return _moving_bar_scope_label(bundle_on.session)


def _moving_bar_all_figure(bundle_on, bundle_2, title, *, right_only=True):
    single_column = bundle_on.single_column
    types = bundle_on.types
    wt_on = bundle_on.traces
    spec_names = _filter_right_specs(bundle_on.spec_names, right_only)
    ncols_on = len(spec_names)
    ca_mean, ca_sem, ca_n = wt_on.ca_mean, wt_on.ca_sem, wt_on.ca_n
    data_mean = bundle_on.data_mean
    baselines = bundle_on.baselines
    baselines_2 = None
    wt_2 = None
    slice_labels = _bundle_slice_labels(bundle_on)
    has_slices = bundle_on.has_slices
    if bundle_2 is not None:
        wt_2 = bundle_2.traces
        spec_2 = _filter_right_specs(bundle_2.spec_names, right_only)
        spec_names = list(spec_names) + list(spec_2)
        ca_mean = {**ca_mean, **wt_2.ca_mean}
        ca_sem = {**ca_sem, **wt_2.ca_sem}
        ca_n = {**ca_n, **wt_2.ca_n}
        data_mean = {**data_mean, **bundle_2.data_mean}
        baselines_2 = bundle_2.baselines
    show_sem = not single_column and not has_slices
    nrows = len(types)
    ncols = len(spec_names)
    ca_dsi_on = moving_bar_dsi_lookup(wt_on.ca_mean, types, bundle_on.spec_names)
    data_dsi_on = moving_bar_dsi_lookup(bundle_on.data_mean, types, bundle_on.spec_names)
    ca_dsi_2 = data_dsi_2 = None
    if bundle_2 is not None:
        ca_dsi_2 = moving_bar_dsi_lookup(wt_2.ca_mean, types, bundle_2.spec_names)
        data_dsi_2 = moving_bar_dsi_lookup(bundle_2.data_mean, types, bundle_2.spec_names)
    fig, axes = _moving_bar_figure(nrows, ncols)
    for ri, tname in enumerate(types):
        for ci, sname in enumerate(spec_names):
            ax = axes[ri, ci]
            key = (tname, sname)
            if key not in ca_mean:
                ax.axis('off')
                continue
            bl = baselines.get(tname)
            bundle_src = bundle_on if ci < ncols_on else bundle_2
            wt = wt_on if ci < ncols_on else wt_2
            if baselines_2 is not None and ci >= ncols_on:
                bl = baselines_2.get(tname)
            before_t, after_t = _window_t(wt, sname)
            dsi_on = ci < ncols_on
            cell_title = _moving_bar_cell_title(
                bundle_on, sname, ca_mean, data_mean,
                ca_dsi_on if dsi_on else ca_dsi_2,
                data_dsi_on if dsi_on else data_dsi_2,
                key,
                type_name=tname,
            )
            if has_slices and bundle_src is not None and bundle_src.slice_overlay is not None:
                slice_traces = {
                    label: _bundle_slice_trace(bundle_src, label, key)
                    for label in slice_labels
                    if key in bundle_src.slice_overlay[label].ca_mean
                }
                if not slice_traces:
                    ax.axis('off')
                    continue
                plot_labels = [label for label in slice_labels if label in slice_traces]
                _plot_moving_bar_cell_slices(
                    ax, ca_mean[key], ca_sem.get(key),
                    slice_traces, plot_labels,
                    cell_title, before_t, after_t,
                    data_trace=data_mean.get(key),
                    show_ylabel=(ci == 0),
                    show_sem=show_sem and key in ca_sem and np.any(ca_sem[key]),
                    show_legend=(ri == 0 and ci == 0),
                    cell_ticks=False,
                    show_tick_labels=(ri == nrows - 1),
                    mark_cost_window=True,
                    baseline=bl,
                    pre_end=_moving_bar_pre_end(bundle_src, tname, sname),
                    show_pre=getattr(bundle_src, "show_pre", True),
                    delta_ms=bundle_src.session.delta_ms,
                )
            else:
                _plot_moving_bar_cell(
                    ax, ca_mean[key], ca_sem.get(key),
                    cell_title, before_t, after_t,
                    data_trace=data_mean.get(key),
                    show_ylabel=(ci == 0),
                    show_sem=show_sem and key in ca_sem and np.any(ca_sem[key]),
                    cell_ticks=False,
                    show_tick_labels=(ri == nrows - 1),
                    mark_cost_window=True,
                    baseline=bl,
                    pre_end=_moving_bar_pre_end(bundle_src or bundle_on, tname, sname),
                    show_pre=getattr(bundle_src or bundle_on, "show_pre", True),
                    delta_ms=(bundle_src or bundle_on).session.delta_ms,
                )
        axes[ri, 0].set_ylabel(cell_ylabel(tname, ca_n), fontsize=8, labelpad=12)
    if title is None:
        title = 'Moving-bar ca-all (right only)' if right_only else 'Moving-bar ca-all'
    scope = _moving_bar_all_scope_label(bundle_on)
    fig.suptitle(title + f'  [{scope}, t_first_sti-aligned full window]', fontsize=12)
    _moving_bar_figure_adjust(fig)
    return fig


@torch.no_grad()
def plot_moving_bar_data(path, *, bundle, bundle_2=None, title=None):
    """Draw ca-data figure from a full-scope :class:`MovingBarTraceBundle`."""
    timer = PlotTimer(prior_prep=bundle_prep_s(bundle, bundle_2))
    timer.end_prep()
    single_column = bundle.single_column
    row_specs = moving_bar_row_specs(bundle.session, bundle.target, bundle.side)
    readout_subtypes = list(row_specs.keys())
    ncols_half = max((len(v) for v in row_specs.values()), default=8)
    if bundle_2 is not None:
        row_specs_2 = moving_bar_row_specs(bundle_2.session, bundle_2.target, bundle_2.side)
        ncols_half = max(ncols_half, max((len(v) for v in row_specs_2.values()), default=8))
        ncols = ncols_half * 2
    else:
        row_specs_2 = None
        ncols = ncols_half
    nrows = len(readout_subtypes)
    fig, axes = _moving_bar_figure(nrows, ncols)

    ca_dsi_on = moving_bar_dsi_lookup(bundle.traces.ca_mean, readout_subtypes, bundle.spec_names)
    data_dsi_on = moving_bar_dsi_lookup(bundle.data_mean, readout_subtypes, bundle.spec_names)
    ca_dsi_2 = data_dsi_2 = None
    if bundle_2 is not None:
        ca_dsi_2 = moving_bar_dsi_lookup(bundle_2.traces.ca_mean, readout_subtypes, bundle_2.spec_names)
        data_dsi_2 = moving_bar_dsi_lookup(bundle_2.data_mean, readout_subtypes, bundle_2.spec_names)

    def _plot_row(ri, subtype, specs, col_offset, row_bundle, plot_side, ca_dsi, data_dsi):
        wt = row_bundle.traces
        for ci, sname in enumerate(specs):
            ax = axes[ri, col_offset + ci]
            key = (subtype, sname)
            if key not in wt.ca_mean:
                ax.axis('off')
                continue
            before_t, after_t = _window_t(wt, sname)
            cell_title = _moving_bar_cell_title(
                bundle, sname, wt.ca_mean, row_bundle.data_mean,
                ca_dsi, data_dsi, key,
                type_name=subtype,
            )
            _plot_moving_bar_cell(
                ax, wt.ca_mean[key], wt.ca_sem[key],
                cell_title, before_t, after_t,
                data_trace=row_bundle.data_mean.get(key),
                show_ylabel=(col_offset + ci == 0), show_sem=not single_column,
                mark_cost_window=True,
                baseline=row_bundle.baselines.get(subtype),
                linestyle=_moving_bar_spec_linestyle(plot_side, subtype, sname),
                pre_end=_moving_bar_pre_end(row_bundle, subtype, sname),
                show_pre=getattr(row_bundle, "show_pre", True),
                delta_ms=row_bundle.session.delta_ms,
            )

    for ri, subtype in enumerate(readout_subtypes):
        _plot_row(ri, subtype, row_specs[subtype], 0, bundle, bundle.side, ca_dsi_on, data_dsi_on)
        if bundle_2 is not None:
            _plot_row(
                ri, subtype, row_specs_2[subtype], ncols_half, bundle_2, bundle_2.side,
                ca_dsi_2, data_dsi_2,
            )
        axes[ri, 0].set_ylabel(
            cell_ylabel(subtype, bundle.traces.ca_n), fontsize=8, labelpad=12,
        )
    if title is None:
        title = 'Moving-bar ca-data'
    scope = _moving_bar_scope_label(bundle.session)
    fig.suptitle(
        title + f'  [{scope}, t_first_sti-aligned full window]',
        fontsize=12,
    )
    _moving_bar_figure_adjust(fig)
    timer.end_draw()
    save_figure(fig, path, dpi=MOVING_BAR_DPI, rasterize=True)
    timer.log(path)


@torch.no_grad()
def plot_moving_bar_all(path, *, bundle, bundle_2=None, title=None, right_only=True):
    """Draw ca-all figure from a full-scope :class:`MovingBarTraceBundle`."""
    timer = PlotTimer(prior_prep=bundle_prep_s(bundle, bundle_2))
    timer.end_prep()
    fig = _moving_bar_all_figure(bundle, bundle_2, title, right_only=right_only)
    timer.end_draw()
    save_figure(fig, path, dpi=MOVING_BAR_DPI, rasterize=True)
    timer.log(path)
