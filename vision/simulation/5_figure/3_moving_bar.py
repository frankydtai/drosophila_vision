"""Moving-bar plotting utilities extracted from ``figure.plot_run``."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

import training
from task.moving_bar.data import (
    GT_CELLS,
    fig1_key_for_stimulus,
    motion_preference,
    moving_bar_cell_title,
    moving_bar_dsi_lookup,
)
from figure.readout import pack_readout_cells, plot_cells_in_order
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
    hex_at_scope_tag,
    cell_ylabel,
    overlay_model_reds,
    plot_pre_post_line,
    plot_sem_band,
    plot_timecourse,
    readout_center_mask,
    save_figure,
    save_forward_trace_csvs,
    sem_from_traces,
    slice_axis_name,
    slice_coord_specs,
    suppress_cost_sem,
    v_ref_by_type_name,
    v_ref_schema_name,
)
import network.path  # noqa: F401  # ensure FAFBv783 modules are importable
from task.moving_bar.data import (
    bar_specs_for_session,
    load_fig1_trace,
    moving_bar_row_specs,
    moving_bar_session_t0_grids,
    moving_bar_nodes_on_hexes,
)
from task.moving_bar.input import (
    filter_sti_hexes,
    moving_bar_cost_hexes,
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
    cell_ids: np.ndarray
    cells: list


@dataclass
class MovingBarTraceBundle:
    task: str
    cells: list
    spec_names: list
    side: str
    single_hex: bool
    baselines: dict
    data_mean: dict
    n_t: int
    traces: MovingBarWindowTraces
    session: object
    at_x: int | None = None
    at_y: int | None = None
    n_filter_hexes: int = 0
    slice_overlay: dict | None = None
    slice_axis: str | None = None
    slice_xs: list | None = None
    slice_ys: list | None = None
    align_at_x: int | None = None
    align_at_y: int | None = None
    prep_s: float = 0.0
    v_ref_by_name: dict = field(default_factory=dict)
    v_ref_name: str | None = None
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


def _cell_ids_np(node_cell):
    return np.asarray(as_numpy(node_cell), dtype=np.int64)


def _cell_ids_for_plot_order(connectome_cell_names, node_cell, plot_cells):
    """Map node ``node_cell`` (index into connectome names) to ``plot_cells`` index.

    ``plot_cells_in_order`` reorders names; aggregating with raw ``node_cell`` against
    ``enumerate(plot_cells)`` mislabels every cell whose plot index ≠ connectome index.
    """
    c_ids = _cell_ids_np(node_cell)
    name_to_plot = {str(n): i for i, n in enumerate(plot_cells)}
    out = np.full(c_ids.shape, -1, dtype=np.int64)
    for ci, name in enumerate(connectome_cell_names):
        pi = name_to_plot.get(str(name))
        if pi is None:
            continue
        out[c_ids == int(ci)] = int(pi)
    return out


def _plot_cells_and_ids(session):
    """Plot-family cell order + remapped ``cell_ids`` for bar aggregation."""
    C = session.backend.network
    if C is None:
        raise ValueError("_plot_cells_and_ids requires session.backend.network")
    cells = plot_cells_in_order(C.cell_names)
    cell_ids = _cell_ids_for_plot_order(C.cell_names, C.node_cell, cells)
    return cells, cell_ids


def _t_rel_window_seconds(before_t, after_t, delta_ms):
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
        n_nodes = sl.shape[2]
        win_ix = np.arange(wl, dtype=np.int64)
        t_idx = t0[..., None] + win_ix[None, None, :]
        t_safe = np.clip(t_idx, 0, t_len - 1)
        b_ix = np.zeros(1, dtype=np.int64)[:, None, None]
        u_ix = np.arange(n_nodes, dtype=np.int64)[None, :, None]
        batch = sl[b_ix, t_safe, u_ix].astype(np.float64, copy=False)
        batch[t_idx < 0] = 0.0
        out.append(batch[0])
    return out


def _aggregate_moving_bar_traces(
    windows_by_batch, t0_bn, cell_ids, cells, spec_names, single_hex, *,
    hex_mask=None,
):
    """``windows_by_batch[bi]`` shape ``(n_nodes, W_bi)``."""
    ca_mean, ca_sem, ca_n = {}, {}, {}
    valid = t0_bn >= 0
    for ti, tname in enumerate(cells):
        cell_mask = cell_ids == ti
        if not cell_mask.any():
            continue
        for bi, sname in enumerate(spec_names):
            node_mask = valid[bi] & cell_mask
            if hex_mask is not None:
                node_mask = node_mask & hex_mask[bi]
            if not node_mask.any():
                continue
            uids = np.nonzero(node_mask)[0]
            arr = windows_by_batch[bi][uids]
            key = (tname, sname)
            ca_mean[key] = arr.mean(axis=0)
            ca_sem[key] = sem_from_traces(arr, single_hex=single_hex)
            ca_n[key] = int(np.unique(uids).size)
    return ca_mean, ca_sem, ca_n


def _network_hex_node_mask(C, filt_hexes, n_batch):
    u_np, v_np = network_uv_np(C)
    col_uv = {(int(c.u), int(c.v)) for c in filt_hexes}
    node_in_col = np.array(
        [(int(u), int(v)) in col_uv for u, v in zip(u_np, v_np)],
        dtype=bool,
    )
    return np.broadcast_to(node_in_col, (n_batch, C.n_nodes)).copy()



def _t0_ref_for_align_hex(t0_bn, bi, ref_hex, *, C):
    if C is None:
        raise ValueError("_t0_ref_for_align_hex requires network C")
    u_np, v_np = network_uv_np(C)
    on_ref = (u_np == int(ref_hex.u)) & (v_np == int(ref_hex.v))
    t0_ref = int(t0_bn[bi, on_ref][0])
    if t0_ref < 0:
        loc = f'({int(ref_hex.u)},{int(ref_hex.v)})'
        raise SystemExit(f'--align-xy ref hex {loc} has no valid t0')
    return t0_ref


def _t0_bn_slice_aligned_to_ref(
    t0_bn, n_batch, filt_hexes, align_at_x, align_at_y, *,
    session, cost_extent,
):
    """Copy ``t0_bn`` with slice nodes forced to the ref hex ``t0`` (plot only)."""
    C = session.backend.network
    if C is None:
        raise ValueError("_t0_bn_slice_aligned_to_ref requires session.backend.network")
    all_hexes = moving_bar_cost_hexes(C, cost_extent=cost_extent)
    ref_hexes = filter_sti_hexes(all_hexes, at_x=align_at_x, at_y=align_at_y)
    if len(ref_hexes) != 1:
        raise SystemExit(
            f'--align-xy must match exactly one sti hex within cost_extent, '
            f'got {len(ref_hexes)} for x={align_at_x!r} y={align_at_y!r}',
        )
    ref_hex = ref_hexes[0]
    u_np, v_np = network_uv_np(C)
    out = t0_bn.copy()
    for bi in range(n_batch):
        t0_ref = _t0_ref_for_align_hex(out, bi, ref_hex, C=C)
        for col in filt_hexes:
            on_col = (u_np == int(col.u)) & (v_np == int(col.v))
            out[bi, on_col] = t0_ref
    return out


def _moving_bar_slice_overlay_traces(
    session, task, trace_full, base_wt, spec_names, *, at_x=None, at_y=None,
    align_at_x=None, align_at_y=None,
):
    """Per-axis slice traces aligned to ``base_wt`` window geometry."""
    if base_wt.t0_bn is None or base_wt.cell_ids is None or base_wt.cells is None:
        raise ValueError("base_wt missing cached t0_bn/cells for slice overlay")
    pack = session.pack_for(task)
    cells = base_wt.cells
    cell_ids = base_wt.cell_ids
    t0_full_bn = base_wt.t0_bn
    n_batch = len(spec_names)
    C = session.backend.network
    if C is None:
        raise ValueError("_moving_bar_slice_overlay_traces requires session.backend.network")
    all_hexes = moving_bar_cost_hexes(C, cost_extent=pack.cost_extent)
    filt_hexes = filter_sti_hexes(all_hexes, at_x=at_x, at_y=at_y)
    if not filt_hexes:
        return None
    hex_mask = _network_hex_node_mask(C, filt_hexes, n_batch)
    win_lens = [
        base_wt.before_t[sname] + base_wt.after_t[sname] + 1
        for sname in spec_names
    ]
    t0_use = t0_full_bn
    if align_at_x is not None and align_at_y is not None:
        t0_use = _t0_bn_slice_aligned_to_ref(
            t0_full_bn, n_batch, filt_hexes, align_at_x, align_at_y,
            session=session, cost_extent=pack.cost_extent,
        )
    windows_full = _windows_by_batch(trace_full, t0_use, win_lens)
    ca_mean, ca_sem, ca_n = _aggregate_moving_bar_traces(
        windows_full, t0_use, cell_ids, cells, spec_names, True, hex_mask=hex_mask,
    )
    return MovingBarWindowTraces(
        ca_mean=ca_mean,
        ca_sem=ca_sem,
        ca_n=ca_n,
        before_t=base_wt.before_t,
        after_t=base_wt.after_t,
        t0_bn=t0_full_bn,
        cell_ids=cell_ids,
        cells=cells,
    )


def _fig1_trace_delta(trace: np.ndarray, delta_ms: float) -> np.ndarray:
    """ΔVm for fig1 cost-window traces (subtract pre-stimulus mean)."""
    trace = np.asarray(trace, dtype=np.float64)
    i_on = cost_window_before_t(delta_ms)
    if i_on > 0 and i_on < len(trace):
        return trace - float(np.mean(trace[:i_on]))
    return trace - float(trace[0])


def _load_moving_bar_data_mean(session, task, cells, specs, side):
    data_mean = {}
    row_cells = plot_cells_in_order(pack_readout_cells(session, task))
    for subtype in row_cells:
        if subtype not in cells:
            continue
        for spec in specs:
            trace_id = fig1_key_for_stimulus(side, subtype, spec)
            if trace_id is None:
                continue
            trace = _fig1_trace_delta(load_fig1_trace(trace_id), session.delta_ms)
            data_mean[(subtype, spec.name)] = trace
    return data_mean


def _moving_bar_traces_from_forward(
    session, task, trace_full, v_onset_np, specs, spec_names, *,
    at_x=None, at_y=None,
):
    pack = session.pack_for(task)
    cost_extent = pack.cost_extent
    n_t = int(session.n_t)
    _t_onset = int(pack.i_sti.shape[1] - pack.data.shape[1])
    grids = moving_bar_session_t0_grids(
        session, specs, cost_extent, n_t, at_x=at_x, at_y=at_y,
        t_onset=_t_onset, delta_ms=session.delta_ms,
    )
    cells, cell_ids = _plot_cells_and_ids(session)
    t0_full_bn = grids.t0_bn
    full_before_t = grids.before_t
    full_after_t = grids.after_t
    side = grids.side
    n_filter_hexes = grids.n_filter_hexes
    single_hex = suppress_cost_sem(session, task) or n_filter_hexes == 1
    win_lens = [
        full_before_t[sname] + full_after_t[sname] + 1
        for sname in spec_names
    ]
    windows_full = _windows_by_batch(trace_full, t0_full_bn, win_lens)
    trace_mean, trace_sem, trace_n = _aggregate_moving_bar_traces(
        windows_full, t0_full_bn, cell_ids, cells, spec_names, single_hex,
    )
    return MovingBarWindowTraces(
        ca_mean=trace_mean,
        ca_sem=trace_sem,
        ca_n=trace_n,
        before_t=full_before_t,
        after_t=full_after_t,
        t0_bn=t0_full_bn,
        cell_ids=cell_ids,
        cells=cells,
    ), cells, side, n_filter_hexes, _t_onset


@torch.no_grad()
def moving_bar_trace_bundle(session, z, task, *, at_x=None, at_y=None,
                            at_xs=None, at_ys=None,
                            align_at_x=None, align_at_y=None,
                            save_trace_csv_dir: str | None = None, show_pre=True):
    """Run one forward; t_first_sti-aligned full-window model traces."""
    t_prep0 = time.perf_counter()
    pack = session.pack_for(task)
    schema = list(session.schema)
    p = training.assign_params(z, schema, session.backend)
    v_delta, v_onset, _v_full = training.forward_full(
        session, p, pack.i_sti, return_v_onset=True, pack=pack,
    )
    v_onset_np = v_onset[0].cpu().numpy()
    save_forward_trace_csvs(
        save_trace_csv_dir, task,
        ref=v_onset_np, trace_full=v_delta,
        ref_stem='moving_bar_v_onset',
    )
    trace_full = apply_out_scale(
        p, v_delta, None, session.backend,
    ).cpu().numpy()
    specs = bar_specs_for_session(session, task)
    spec_names = [s.name for s in specs]
    n_t = int(session.n_t)
    C = session.backend.network
    traces, cells, side, n_filter_hexes, t_onset = _moving_bar_traces_from_forward(
        session, task, trace_full, v_onset_np, specs, spec_names,
    )
    v_ref = v_ref_by_type_name(z, session)
    if C is not None:
        hexes = moving_bar_cost_hexes(C, cost_extent=pack.cost_extent)
        if at_x is not None or at_y is not None:
            hexes = filter_sti_hexes(hexes, at_x=at_x, at_y=at_y)
        nodes_by_name = {
            tname: moving_bar_nodes_on_hexes(C, tname, hexes) for tname in cells
        }
    else:
        cell_ids = _cell_ids_np(session.backend.conn.node_cell)
        readout = pack.readout_node.cpu().numpy()
        center = readout_center_mask(pack, session.backend)
        node_cells = cell_ids[readout]
        nodes_by_name = {
            name: readout[center & (node_cells == cells.index(name))]
            for name in cells
        }
    baselines = baselines_for_types(v_onset_np, nodes_by_name, v_ref)
    single_hex = suppress_cost_sem(session, task) or n_filter_hexes == 1
    data_mean = _load_moving_bar_data_mean(
        session, task, cells, specs, side,
    )
    slice_overlay = None
    slice_axis = None
    slice_xs = None
    slice_ys = None
    specs = slice_coord_specs(at_xs, at_ys)
    if specs:
        slice_axis = slice_axis_name(at_xs, at_ys)
        slice_xs = list(at_xs) if at_xs is not None else None
        slice_ys = list(at_ys) if at_ys is not None else None
        slice_overlay = {}
        for label, xv, yv in specs:
            wt = _moving_bar_slice_overlay_traces(
                session, task, trace_full, traces, spec_names,
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
            slice_xs = None
            slice_ys = None
    return MovingBarTraceBundle(
        task=task,
        cells=cells,
        spec_names=spec_names,
        side=side,
        single_hex=single_hex,
        baselines=baselines,
        data_mean=data_mean,
        n_t=n_t,
        traces=traces,
        session=session,
        at_x=at_x,
        at_y=at_y,
        n_filter_hexes=n_filter_hexes,
        slice_overlay=slice_overlay,
        slice_axis=slice_axis,
        slice_xs=slice_xs,
        slice_ys=slice_ys,
        align_at_x=align_at_x,
        align_at_y=align_at_y,
        prep_s=time.perf_counter() - t_prep0,
        v_ref_by_name=v_ref,
        v_ref_name=v_ref_schema_name(session.schema),
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
        ti = wt.cells.index(subtype)
    except ValueError:
        return 0
    key = (subtype, sname)
    if key not in wt.ca_mean:
        return 0
    win_len = int(np.asarray(wt.ca_mean[key]).shape[0])
    cell_mask = wt.cell_ids == ti
    valid = (wt.t0_bn[bi] >= 0) & cell_mask
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


def _moving_bar_scope_label(session, *, at_x=None, at_y=None, n_filter_hexes=None):
    pack = session.primary_readout
    cost_extent = pack.cost_extent
    C = session.backend.network
    if C is None:
        raise ValueError("_moving_bar_scope_label requires session.backend.network")
    if at_x is not None or at_y is not None:
        ncol_part = f'{n_filter_hexes} sti hex'
        if n_filter_hexes != 1:
            ncol_part += 's'
        parts = [hex_at_scope_tag(at_x, at_y), ncol_part]
        if cost_extent is not None:
            parts.insert(0, f'cost_extent={cost_extent}')
        return ', '.join(parts)
    if cost_extent is not None:
        from task.moving_bar.input import moving_bar_cost_hexes
        ncols = len(moving_bar_cost_hexes(C, cost_extent=cost_extent))
        return f'cost_extent={cost_extent} ({ncols} sti hexes)'
    from task.moving_bar.input import sti_hexes
    return f'avg over {len(sti_hexes(C))} sti hexes'


def _style_moving_bar_relative_axis(
    ax, before_t, after_t, win_len, *,
    delta_ms,
    show_tick_labels=True, mark_cost_window=False,
):
    end = win_len - 1
    ax.set_xlim(0, end)
    ax.set_xticks([0, before_t, end])
    before_s, after_s = _t_rel_window_seconds(before_t, after_t, delta_ms)
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
    if subtype not in GT_CELLS:
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
        pack = bundle_on.session.primary_readout
        cost_extent = pack.cost_extent
        at_x = bundle_on.slice_xs if bundle_on.slice_axis in ('x', 'xy') else None
        at_y = bundle_on.slice_ys if bundle_on.slice_axis in ('y', 'xy') else None
        parts = [hex_at_scope_tag(at_x, at_y), 'overlay + total']
        if bundle_on.align_at_x is not None and bundle_on.align_at_y is not None:
            parts.append(
                'aligned to '
                + hex_at_scope_tag(bundle_on.align_at_x, bundle_on.align_at_y),
            )
        if cost_extent is not None:
            parts.insert(0, f'cost_extent={cost_extent}')
        return ', '.join(parts)
    return _moving_bar_scope_label(bundle_on.session)


def _moving_bar_all_figure(bundle_on, bundle_2, title, *, right_only=True):
    single_hex = bundle_on.single_hex
    cells = bundle_on.cells
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
    show_sem = not single_hex and not has_slices
    nrows = len(cells)
    ncols = len(spec_names)
    ca_dsi_on = moving_bar_dsi_lookup(wt_on.ca_mean, cells, bundle_on.spec_names)
    data_dsi_on = moving_bar_dsi_lookup(bundle_on.data_mean, cells, bundle_on.spec_names)
    ca_dsi_2 = data_dsi_2 = None
    if bundle_2 is not None:
        ca_dsi_2 = moving_bar_dsi_lookup(wt_2.ca_mean, cells, bundle_2.spec_names)
        data_dsi_2 = moving_bar_dsi_lookup(bundle_2.data_mean, cells, bundle_2.spec_names)
    fig, axes = _moving_bar_figure(nrows, ncols)
    for ri, tname in enumerate(cells):
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
    single_hex = bundle.single_hex
    row_specs = moving_bar_row_specs(bundle.session, bundle.task, bundle.side)
    gt_cells = list(row_specs.keys())
    ncols_half = max((len(v) for v in row_specs.values()), default=8)
    if bundle_2 is not None:
        row_specs_2 = moving_bar_row_specs(bundle_2.session, bundle_2.task, bundle_2.side)
        ncols_half = max(ncols_half, max((len(v) for v in row_specs_2.values()), default=8))
        ncols = ncols_half * 2
    else:
        row_specs_2 = None
        ncols = ncols_half
    nrows = len(gt_cells)
    fig, axes = _moving_bar_figure(nrows, ncols)

    ca_dsi_on = moving_bar_dsi_lookup(bundle.traces.ca_mean, gt_cells, bundle.spec_names)
    data_dsi_on = moving_bar_dsi_lookup(bundle.data_mean, gt_cells, bundle.spec_names)
    ca_dsi_2 = data_dsi_2 = None
    if bundle_2 is not None:
        ca_dsi_2 = moving_bar_dsi_lookup(bundle_2.traces.ca_mean, gt_cells, bundle_2.spec_names)
        data_dsi_2 = moving_bar_dsi_lookup(bundle_2.data_mean, gt_cells, bundle_2.spec_names)

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
                show_ylabel=(col_offset + ci == 0), show_sem=not single_hex,
                mark_cost_window=True,
                baseline=row_bundle.baselines.get(subtype),
                linestyle=_moving_bar_spec_linestyle(plot_side, subtype, sname),
                pre_end=_moving_bar_pre_end(row_bundle, subtype, sname),
                show_pre=getattr(row_bundle, "show_pre", True),
                delta_ms=row_bundle.session.delta_ms,
            )

    for ri, subtype in enumerate(gt_cells):
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
