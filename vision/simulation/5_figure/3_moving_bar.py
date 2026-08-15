"""Moving-bar plotting utilities extracted from ``figure.plot``."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

import train
from task.moving_bar.gt import (
    GT_CELLS,
    fig1_trace_for_sti,
    load_fig1_trace,
    motion_preference,
    moving_bar_cell_title,
    dsi_from_trace_map,
)
from task.moving_bar.pack import (
    bar_specs_for_session,
    moving_bar_nodes_on_hexes,
    moving_bar_specs_by_cell,
    moving_bar_session_t0_grids,
)
from figure.gt import pack_cells
from network.construction import cells_in_order
from figure.util import (
    GT_COLOR,
    TRACE_LW,
    ElapsedTimer,
    annotate_v_th,
    as_numpy,
    params_for_types,
    e_leak_by_cell_from_z,
    format_moving_bar_cell_cost_lines,
    readout_prep_s,
    hex_at_scope_tag,
    cell_ylabel,
    gt_affine_scalars_for_cell,
    gt_trace_affine,
    ms_shown_axis_xlim,
    overlay_v_readout_reds,
    plot_pre_post_line,
    plot_std_band,
    plot_timecourse,
    pack_center_mask,
    save_figure,
    std_from_traces,
    slice_axis,
    slice_coord_specs,
    suppress_cost_std,
    v_th_by_cell_from_z,
)
import network.path  # noqa: F401  # ensure FAFBv783 modules are importable
from task.moving_bar.sti_geo import (
    filter_sti_hexes,
    moving_bar_cost_hexes,
    network_uv_np,
    sti_hexes,
)
from task.moving_bar.sti_spec import (
    cost_window_after_t,
    cost_window_before_t,
)

MOVING_BAR_DPI = 100


@dataclass
class MovingBarWindowTraces:
    ca_mean: dict
    ca_std: dict
    ca_n: dict
    before_t: dict
    after_t: dict
    t0_bn: np.ndarray
    cell_ids: np.ndarray
    cells: list


@dataclass
class MovingBarTraceReadout:
    task: str
    cells: list
    spec_tokens: list
    side: str
    single_hex: bool
    v_th_by_cell: dict
    e_leaks: dict
    gt_mean: dict
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
    show_pre: bool = True
    t_onset: int | None = None
    gt_affine_by_cell: dict = field(default_factory=dict)
    ms_shown: tuple[float, float] | None = None

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


def _moving_bar_cell_title(
    token,
    ca_mean,
    gt_mean,
    ca_dsi_from_trace,
    gt_dsi_from_trace,
    key,
    *,
    cell,
    cost_parts=None,
    cost_tasks=None,
):
    head = token
    if cost_parts is not None and cost_tasks:
        cost_lines = format_moving_bar_cell_cost_lines(cell, cost_parts, cost_tasks)
        if cost_lines:
            head = '\n'.join([f'{cell} Cost', *cost_lines, head])
    return moving_bar_cell_title(
        head,
        ca_dsi=ca_dsi_from_trace.get(key),
        gt_dsi=gt_dsi_from_trace.get(key),
        has_gt=gt_mean.get(key) is not None,
    )


def _cell_ids_for_order(connectome_cells, node_cells, figure_cells):
    """Map node ``node_cells`` (index into connectome cells) to ``figure_cells`` index.

    ``cells_in_order`` reorders cells; accumulating with raw ``node_cells`` against
    ``enumerate(figure_cells)`` mislabels every cell whose plot index ≠ connectome index.
    """
    c_ids = np.asarray(as_numpy(node_cells), dtype=np.int64)
    idx_from_cell = {str(n): i for i, n in enumerate(figure_cells)}
    out = np.full(c_ids.shape, -1, dtype=np.int64)
    for ci, name in enumerate(connectome_cells):
        pi = idx_from_cell.get(str(name))
        if pi is None:
            continue
        out[c_ids == int(ci)] = int(pi)
    return out


def _cells_and_ids(session):
    """Plot-order cell order + remapped ``cell_ids`` for bar."""
    connectome = session.backend.network
    if connectome is None:
        raise ValueError("_cells_and_ids requires session.backend.network")
    cells = cells_in_order(connectome.cells)
    cell_ids = _cell_ids_for_order(connectome.cells, connectome.node_cells, cells)
    return cells, cell_ids


def _filter_right_specs(spec_tokens, right_only):
    if right_only:
        return [s for s in spec_tokens if s.startswith('right_')]
    return list(spec_tokens)


def _windows_by_batch(ca_full, t0_bn, n_t_by_batch):
    """``n_t_by_batch``: int (uniform) or length-``B`` sequence of trace lengths."""
    n_batch = ca_full.shape[0]
    if isinstance(n_t_by_batch, int):
        n_t_by_batch = (n_t_by_batch,) * n_batch
    out = []
    for bi in range(n_batch):
        n_t_batch = int(n_t_by_batch[bi])
        sl = ca_full[bi:bi + 1]
        t0 = t0_bn[bi:bi + 1]
        t_len = sl.shape[1]
        n_nodes = sl.shape[2]
        t_in_window_idx = np.arange(n_t_batch, dtype=np.int64)
        t_abs = t0[..., None] + t_in_window_idx[None, None, :]
        t_safe = np.clip(t_abs, 0, t_len - 1)
        b_idx = np.zeros(1, dtype=np.int64)[:, None, None]
        u_idx = np.arange(n_nodes, dtype=np.int64)[None, :, None]
        batch = sl[b_idx, t_safe, u_idx].astype(np.float64, copy=False)
        batch[t_abs < 0] = 0.0
        out.append(batch[0])
    return out


def _accumulate_moving_bar_traces(
    windows_by_batch, t0_bn, cell_ids, cells, spec_tokens, single_hex, *,
    hex_mask=None,
):
    """``windows_by_batch[bi]`` shape ``(n_nodes, W_bi)``."""
    ca_mean, ca_std, ca_n = {}, {}, {}
    valid = t0_bn >= 0
    for ti, cell in enumerate(cells):
        cell_mask = cell_ids == ti
        if not cell_mask.any():
            continue
        for bi, token in enumerate(spec_tokens):
            node_mask = valid[bi] & cell_mask
            if hex_mask is not None:
                node_mask = node_mask & hex_mask[bi]
            if not node_mask.any():
                continue
            uids = np.nonzero(node_mask)[0]
            arr = windows_by_batch[bi][uids]
            key = (cell, token)
            ca_mean[key] = arr.mean(axis=0)
            ca_std[key] = std_from_traces(arr, single_hex=single_hex)
            ca_n[key] = int(np.unique(uids).size)
    return ca_mean, ca_std, ca_n


def _network_hex_node_mask(connectome, filt_hexes, n_batch):
    node_u_np, node_v_np = network_uv_np(connectome)
    hex_uv = {(int(hex.u), int(hex.v)) for hex in filt_hexes}
    node_in_hex = np.array(
        [(int(hex_u), int(hex_v)) in hex_uv for hex_u, hex_v in zip(node_u_np, node_v_np)],
        dtype=bool,
    )
    return np.broadcast_to(node_in_hex, (n_batch, connectome.n_nodes)).copy()


def _t0_ref_for_align_hex(t0_bn, bi, ref_hex, *, connectome):
    if connectome is None:
        raise ValueError("_t0_ref_for_align_hex requires connectome")
    node_u_np, node_v_np = network_uv_np(connectome)
    on_ref = (node_u_np == int(ref_hex.u)) & (node_v_np == int(ref_hex.v))
    t0_ref = int(t0_bn[bi, on_ref][0])
    if t0_ref < 0:
        loc = f'({int(ref_hex.u)},{int(ref_hex.v)})'
        raise SystemExit(f'--align-xy ref hex {loc} has no valid t0')
    return t0_ref


def t0_bn_slice_from_ref(
    t0_bn, n_batch, filt_hexes, align_at_x, align_at_y, *,
    session, cost_radius,
):
    """Copy ``t0_bn`` with slice nodes forced to the ref hex ``t0`` (plot only)."""
    connectome = session.backend.network
    if connectome is None:
        raise ValueError("t0_bn_slice_from_ref requires session.backend.network")
    hexes = moving_bar_cost_hexes(connectome, cost_radius=cost_radius)
    ref_hexes = filter_sti_hexes(hexes, at_x=align_at_x, at_y=align_at_y)
    if len(ref_hexes) != 1:
        raise SystemExit(
            f'--align-xy must match exactly one sti hex within cost_radius, '
            f'got {len(ref_hexes)} for x={align_at_x!r} y={align_at_y!r}',
        )
    ref_hex = ref_hexes[0]
    node_u_np, node_v_np = network_uv_np(connectome)
    out = t0_bn.copy()
    for bi in range(n_batch):
        t0_ref = _t0_ref_for_align_hex(out, bi, ref_hex, connectome=connectome)
        for hex in filt_hexes:
            on_hex = (node_u_np == int(hex.u)) & (node_v_np == int(hex.v))
            out[bi, on_hex] = t0_ref
    return out


def _moving_bar_slice_overlay_traces(
    session, task, trace_full, base_wt, spec_tokens, *, at_x=None, at_y=None,
    align_at_x=None, align_at_y=None,
):
    """Per-axis slice traces aligned to ``base_wt`` trace geometry."""
    if base_wt.t0_bn is None or base_wt.cell_ids is None or base_wt.cells is None:
        raise ValueError("base_wt missing cached t0_bn/cells for slice overlay")
    pack = session.pack_for(task)
    cells = base_wt.cells
    cell_ids = base_wt.cell_ids
    t0_full_bn = base_wt.t0_bn
    n_batch = len(spec_tokens)
    connectome = session.backend.network
    if connectome is None:
        raise ValueError("_moving_bar_slice_overlay_traces requires session.backend.network")
    hexes = moving_bar_cost_hexes(connectome, cost_radius=pack.cost_radius)
    filt_hexes = filter_sti_hexes(hexes, at_x=at_x, at_y=at_y)
    if not filt_hexes:
        return None
    hex_mask = _network_hex_node_mask(connectome, filt_hexes, n_batch)
    n_t_by_batch = [
        base_wt.before_t[token] + base_wt.after_t[token] + 1
        for token in spec_tokens
    ]
    t0_bn_aligned = t0_full_bn
    if align_at_x is not None and align_at_y is not None:
        t0_bn_aligned = t0_bn_slice_from_ref(
            t0_full_bn, n_batch, filt_hexes, align_at_x, align_at_y,
            session=session, cost_radius=pack.cost_radius,
        )
    windows_full = _windows_by_batch(trace_full, t0_bn_aligned, n_t_by_batch)
    ca_mean, ca_std, ca_n = _accumulate_moving_bar_traces(
        windows_full, t0_bn_aligned, cell_ids, cells, spec_tokens, True, hex_mask=hex_mask,
    )
    return MovingBarWindowTraces(
        ca_mean=ca_mean,
        ca_std=ca_std,
        ca_n=ca_n,
        before_t=base_wt.before_t,
        after_t=base_wt.after_t,
        t0_bn=t0_full_bn,
        cell_ids=cell_ids,
        cells=cells,
    )


def _fig1_trace_delta(trace: np.ndarray, delta_ms: float) -> np.ndarray:
    """ΔVm for fig1 cost-window traces (subtract pre-sti mean)."""
    trace = np.asarray(trace, dtype=np.float64)
    i_on = cost_window_before_t(delta_ms)
    if i_on > 0 and i_on < len(trace):
        return trace - float(np.mean(trace[:i_on]))
    return trace - float(trace[0])


def _load_moving_bar_gt_mean(session, task, cells, specs, side):
    gt_mean = {}
    row_cells = cells_in_order(pack_cells(session, task))
    for subtype in row_cells:
        if subtype not in cells:
            continue
        for spec in specs:
            trace_id = fig1_trace_for_sti(side, subtype, spec)
            if trace_id is None:
                continue
            trace = _fig1_trace_delta(load_fig1_trace(trace_id), session.delta_ms)
            gt_mean[(subtype, spec.token)] = trace
    return gt_mean


def _traces_from_forward(
    session, task, trace_full, specs, spec_tokens, *,
    at_x=None, at_y=None,
):
    pack = session.pack_for(task)
    cost_radius = pack.cost_radius
    n_t = int(session.n_t)
    _t_onset = train.pack_t_onset(pack)
    grids = moving_bar_session_t0_grids(
        session, specs, cost_radius, n_t, at_x=at_x, at_y=at_y,
        t_onset=_t_onset, delta_ms=session.delta_ms,
    )
    cells, cell_ids = _cells_and_ids(session)
    t0_full_bn = grids.t0_bn
    full_before_t = grids.before_t
    full_after_t = grids.after_t
    side = grids.side
    n_filter_hexes = grids.n_filter_hexes
    single_hex = suppress_cost_std(session, task) or n_filter_hexes == 1
    n_t_by_batch = [
        full_before_t[token] + full_after_t[token] + 1
        for token in spec_tokens
    ]
    windows_full = _windows_by_batch(trace_full, t0_full_bn, n_t_by_batch)
    trace_mean, trace_std, trace_n = _accumulate_moving_bar_traces(
        windows_full, t0_full_bn, cell_ids, cells, spec_tokens, single_hex,
    )
    return MovingBarWindowTraces(
        ca_mean=trace_mean,
        ca_std=trace_std,
        ca_n=trace_n,
        before_t=full_before_t,
        after_t=full_after_t,
        t0_bn=t0_full_bn,
        cell_ids=cell_ids,
        cells=cells,
    ), cells, side, n_filter_hexes, _t_onset, single_hex


@torch.no_grad()
def moving_bar_trace_readout(session, z, task, *, at_x=None, at_y=None,
                            at_xs=None, at_ys=None,
                            align_at_x=None, align_at_y=None,
                            show_pre=True, ms_shown=None):
    """Run one forward; t_first_sti-aligned full-trace v_readout traces."""
    t_prep0 = time.perf_counter()
    pack = session.pack_for(task)
    schema = list(session.schema)
    params = train.override_val_from(
        train.assign_params(z, schema, session.backend), session,
    )
    v = train.forward_v(session, params, pack.i_sti, pack=pack)
    t0 = train.pack_t_onset(pack)
    if str((session.train_opts or {}).get("filter", "none")) == "ca":
        v_ca = train.v_ca_from_v(v, params, session)
        plot_t = train.ca_from_v_ca(v_ca, params, session, t_onset=t0)
    else:
        v_ca = None
        plot_t = v
    train.override_val_from(params, session, onset_trace=plot_t, t_onset=t0)
    trace_full = plot_t.detach().cpu().numpy()
    specs = bar_specs_for_session(session, task)
    spec_tokens = [s.token for s in specs]
    n_t = int(session.n_t)
    connectome = session.backend.network
    traces, cells, side, n_filter_hexes, t_onset, single_hex = _traces_from_forward(
        session, task, trace_full, specs, spec_tokens,
    )
    v_th = v_th_by_cell_from_z(z, session)
    if connectome is not None:
        hexes = moving_bar_cost_hexes(connectome, cost_radius=pack.cost_radius)
        if at_x is not None or at_y is not None:
            hexes = filter_sti_hexes(hexes, at_x=at_x, at_y=at_y)
        nodes_by_cell = {
            cell: moving_bar_nodes_on_hexes(connectome, cell, hexes) for cell in cells
        }
    else:
        cell_ids = np.asarray(as_numpy(session.backend.conn.node_cells), dtype=np.int64)
        entry_nodes = pack.entry_nodes.cpu().numpy()
        center = pack_center_mask(pack, session.backend)
        node_cells = cell_ids[entry_nodes]
        nodes_by_cell = {
            name: entry_nodes[center & (node_cells == cells.index(name))]
            for name in cells
        }
    v_th_by_cell = params_for_types(nodes_by_cell, v_th)
    e_leaks = params_for_types(nodes_by_cell, e_leak_by_cell_from_z(z, session))
    gt_mean = _load_moving_bar_gt_mean(
        session, task, cells, specs, side,
    )
    gt_affine_by_cell = {
        str(name): gt_affine_scalars_for_cell(
            params, name, session.backend, session=session,
        )
        for name in cells
    }
    slice_overlay = None
    slice_axis = None
    slice_xs = None
    slice_ys = None
    specs = slice_coord_specs(at_xs, at_ys)
    if specs:
        slice_axis = slice_axis(at_xs, at_ys)
        slice_xs = list(at_xs) if at_xs is not None else None
        slice_ys = list(at_ys) if at_ys is not None else None
        slice_overlay = {}
        for label, xv, yv in specs:
            wt = _moving_bar_slice_overlay_traces(
                session, task, trace_full, traces, spec_tokens,
                at_x=xv, at_y=yv,
                align_at_x=align_at_x, align_at_y=align_at_y,
            )
            if wt is None:
                print(f'skip slice overlay {label}: no hex within cost_radius')
                continue
            slice_overlay[label] = wt
        if not slice_overlay:
            slice_overlay = None
            slice_axis = None
            slice_xs = None
            slice_ys = None
    return MovingBarTraceReadout(
        task=task,
        cells=cells,
        spec_tokens=spec_tokens,
        side=side,
        single_hex=single_hex,
        v_th_by_cell=v_th_by_cell,
        e_leaks=e_leaks,
        gt_mean=gt_mean,
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
        show_pre=bool(show_pre),
        t_onset=int(t_onset),
        gt_affine_by_cell=gt_affine_by_cell,
        ms_shown=ms_shown,
    )


def _moving_bar_pre_end(readout, subtype, token):
    """Median relative index of global ``t_onset`` within the plotted trace."""
    t_onset = readout.t_onset
    wt = readout.traces
    if t_onset is None or wt.t0_bn is None:
        return 0
    try:
        bi = readout.spec_tokens.index(token)
        ti = wt.cells.index(subtype)
    except ValueError:
        return 0
    key = (subtype, token)
    if key not in wt.ca_mean:
        return 0
    n_t_figure = int(np.asarray(wt.ca_mean[key]).shape[0])
    cell_mask = wt.cell_ids == ti
    valid = (wt.t0_bn[bi] >= 0) & cell_mask
    if not bool(valid.any()):
        return 0
    pre = np.clip(int(t_onset) - wt.t0_bn[bi, valid], 0, n_t_figure)
    return int(np.median(pre))


def _cost_window_overlay(cost_trace, before_t, delta_ms):
    """Fig1 overlay x/y within full-trace coordinates (cost_window only)."""
    i0 = before_t - cost_window_before_t(delta_ms)
    trace = np.asarray(cost_trace, dtype=np.float64)
    if i0 < 0:
        trace = trace[-i0:]
        i0 = 0
    x = np.arange(i0, i0 + len(trace), dtype=np.int64)
    return x, trace


def _moving_bar_scope_label(session, *, at_x=None, at_y=None, n_filter_hexes=None):
    pack = session.primary_pack
    cost_radius = pack.cost_radius
    connectome = session.backend.network
    if connectome is None:
        raise ValueError("_moving_bar_scope_label requires session.backend.network")
    if at_x is not None or at_y is not None:
        ncol_part = f'{n_filter_hexes} sti hex'
        if n_filter_hexes != 1:
            ncol_part += 's'
        parts = [hex_at_scope_tag(at_x, at_y), ncol_part]
        if cost_radius is not None:
            parts.insert(0, f'cost_radius={cost_radius}')
        return ', '.join(parts)
    if cost_radius is not None:
        n_sti_hexes = len(moving_bar_cost_hexes(connectome, cost_radius=cost_radius))
        return f'cost_radius={cost_radius} ({n_sti_hexes} sti hexes)'
    return f'avg over {len(sti_hexes(connectome))} sti hexes'


def _style_moving_bar_relative_axis(
    ax, before_t, n_t_figure, *,
    delta_ms,
    show_tick_labels=True, mark_cost_window=False,
    ms_shown=None,
):
    end = n_t_figure - 1
    xlim = ms_shown_axis_xlim(ms_shown, delta_ms=delta_ms, origin_t=before_t)
    if xlim is None:
        lo, hi = 0, end
    else:
        lo, hi = max(0, xlim[0]), min(end, xlim[1])
        if lo > hi:
            lo, hi = 0, end
    ax.set_xlim(lo, hi)
    ticks = [t for t in (lo, before_t, hi) if lo <= t <= hi]
    if len(ticks) < 2:
        ticks = [lo, hi]
    ax.set_xticks(ticks)
    scale = float(delta_ms) / 1000.0
    ax.set_xticklabels(
        [f'{(t - before_t) * scale:g}' for t in ticks],
        fontsize=6,
    )
    if not show_tick_labels:
        ax.tick_params(labelbottom=False)
    if mark_cost_window:
        cw_before = cost_window_before_t(delta_ms)
        cw_after = cost_window_after_t(delta_ms)
        for x in (before_t - cw_before, before_t + cw_after):
            if lo <= x <= hi:
                ax.axvline(x, color='0.75', linewidth=0.6, linestyle='--', zorder=0)


def _moving_bar_spec_linestyle(side, subtype, token):
    """Solid for PD stis, dashed for ND (Gruntman fig1 convention)."""
    if subtype not in GT_CELLS:
        return '-'
    parts = str(token).split('_')
    if len(parts) < 3:
        return '-'
    direction, contrast = parts[0], parts[1]
    pref = motion_preference(side, subtype, direction, contrast)
    if pref is None:
        return '-'
    return '--' if pref.pd_nd == 'ND' else '-'


def _plot_moving_bar_cell(
    ax,
    ca_trace,
    std_trace,
    title,
    before_t,
    *,
    gt_trace=None,
    show_ylabel=False,
    show_std=True,
    cell_ticks=True,
    show_tick_labels=True,
    mark_cost_window=False,
    v_th=None,
    e_leak=None,
    linestyle='-',
    pre_end=0,
    show_pre=False,
    delta_ms=None,
    ms_shown=None,
):
    n_t_figure = len(ca_trace)
    gt_x, gt_y = None, None
    if gt_trace is not None:
        gt_x, gt_y = _cost_window_overlay(gt_trace, before_t, delta_ms)

    def style_xaxis(ax):
        _style_moving_bar_relative_axis(
            ax, before_t, n_t_figure,
            delta_ms=delta_ms,
            show_tick_labels=show_tick_labels,
            mark_cost_window=mark_cost_window,
            ms_shown=ms_shown,
        )

    plot_timecourse(
        ax, np.arange(n_t_figure),
        [{
            "v_readout": ca_trace,
            "gt": None,
            "std": std_trace,
            "linestyle": linestyle,
        }],
        show_std=show_std and std_trace is not None and np.any(std_trace),
        title=title,
        v_th=v_th,
        e_leak=e_leak,
        show_ylabel=show_ylabel,
        ticksize=6 if cell_ticks else 5,
        style_xaxis=style_xaxis,
        pre_end=pre_end,
        show_pre=show_pre,
    )
    # Gray fig1 overlay is already restricted to cost_window (no global pre).
    if gt_x is not None:
        ax.plot(gt_x, gt_y, color=GT_COLOR, linewidth=TRACE_LW, linestyle=linestyle)


def _plot_moving_bar_cell_slices(
    ax,
    scope_trace,
    std_trace,
    slice_traces,
    slice_labels,
    title,
    before_t,
    *,
    gt_trace=None,
    show_ylabel=False,
    show_std=True,
    show_legend=False,
    cell_ticks=True,
    show_tick_labels=True,
    mark_cost_window=False,
    v_th=None,
    e_leak=None,
    pre_end=0,
    show_pre=False,
    delta_ms=None,
    ms_shown=None,
):
    n_t_figure = len(scope_trace)
    gt_x, gt_y = None, None
    if gt_trace is not None:
        gt_x, gt_y = _cost_window_overlay(gt_trace, before_t, delta_ms)

    def style_xaxis(ax):
        _style_moving_bar_relative_axis(
            ax, before_t, n_t_figure,
            delta_ms=delta_ms,
            show_tick_labels=show_tick_labels,
            mark_cost_window=mark_cost_window,
            ms_shown=ms_shown,
        )

    t = np.arange(n_t_figure)
    if gt_x is not None:
        ax.plot(gt_x, gt_y, color=GT_COLOR, linewidth=TRACE_LW)
    colors = overlay_v_readout_reds(len(slice_labels))
    for i, label in enumerate(slice_labels):
        plot_pre_post_line(
            ax, t, slice_traces[label], pre_end=pre_end,
            show_pre=show_pre, plot_pre=True,
            color=colors[i], linestyle='-', linewidth=TRACE_LW, label=label,
        )
    if show_std and std_trace is not None and np.any(std_trace):
        split = max(0, min(int(pre_end or 0), n_t_figure))
        plot_std_band(ax, t[split:], scope_trace[split:], std_trace[split:])
    plot_pre_post_line(
        ax, t, scope_trace, pre_end=pre_end,
        show_pre=show_pre, plot_pre=True,
        color=colors[-1], linestyle='-', linewidth=TRACE_LW, label='scope',
    )
    if title is not None:
        ax.set_title(title, fontsize=7, pad=2)
    style_xaxis(ax)
    if show_ylabel:
        ax.set_ylabel('mV', fontsize=7)
    ax.tick_params(labelsize=6 if cell_ticks else 5)
    annotate_v_th(ax, v_th, e_leak=e_leak)
    if show_legend:
        ax.legend(fontsize=5, loc='upper right', framealpha=0.85)


def _moving_bar_all_scope_label(readout_on):
    if readout_on.has_slices:
        pack = readout_on.session.primary_pack
        cost_radius = pack.cost_radius
        at_x = readout_on.slice_xs if readout_on.slice_axis in ('x', 'xy') else None
        at_y = readout_on.slice_ys if readout_on.slice_axis in ('y', 'xy') else None
        parts = [hex_at_scope_tag(at_x, at_y), 'overlay + scope']
        if readout_on.align_at_x is not None and readout_on.align_at_y is not None:
            parts.append(
                'aligned to '
                + hex_at_scope_tag(readout_on.align_at_x, readout_on.align_at_y),
            )
        if cost_radius is not None:
            parts.insert(0, f'cost_radius={cost_radius}')
        return ', '.join(parts)
    return _moving_bar_scope_label(readout_on.session)


def _moving_bar_cost_tasks(readout_on, readout_2=None):
    tasks = [readout_on.task]
    if readout_2 is not None and readout_2.task not in tasks:
        tasks.append(readout_2.task)
    return tasks


def _moving_bar_all_figure(readout_on, readout_2, title, *, right_only=True, cost_parts=None):
    single_hex = readout_on.single_hex
    cells = readout_on.cells
    wt_on = readout_on.traces
    spec_tokens = _filter_right_specs(readout_on.spec_tokens, right_only)
    ncols_on = len(spec_tokens)
    ca_mean, ca_std, ca_n = wt_on.ca_mean, wt_on.ca_std, wt_on.ca_n
    gt_mean = readout_on.gt_mean
    v_th_by_cell = readout_on.v_th_by_cell
    e_leaks = readout_on.e_leaks
    v_th_by_cell_2 = None
    e_leaks_2 = None
    wt_2 = None
    slice_labels = (
        list(readout_on.slice_overlay.keys()) if readout_on.slice_overlay else []
    )
    has_slices = readout_on.has_slices
    cost_tasks = _moving_bar_cost_tasks(readout_on, readout_2)
    if readout_2 is not None:
        wt_2 = readout_2.traces
        spec_2 = _filter_right_specs(readout_2.spec_tokens, right_only)
        spec_tokens = list(spec_tokens) + list(spec_2)
        ca_mean = {**ca_mean, **wt_2.ca_mean}
        ca_std = {**ca_std, **wt_2.ca_std}
        ca_n = {**ca_n, **wt_2.ca_n}
        gt_mean = {**gt_mean, **readout_2.gt_mean}
        v_th_by_cell_2 = readout_2.v_th_by_cell
        e_leaks_2 = readout_2.e_leaks
    show_std = not single_hex and not has_slices
    nrows = len(cells)
    ncols = len(spec_tokens)
    ca_dsi_on = dsi_from_trace_map(wt_on.ca_mean, cells, readout_on.spec_tokens)
    gt_dsi_on = dsi_from_trace_map(wt_on.gt_mean, cells, readout_on.spec_tokens)
    ca_dsi_2 = gt_dsi_2 = None
    if readout_2 is not None:
        ca_dsi_2 = dsi_from_trace_map(wt_2.ca_mean, cells, readout_2.spec_tokens)
        gt_dsi_2 = dsi_from_trace_map(readout_2.gt_mean, cells, readout_2.spec_tokens)
    fig, axes = _moving_bar_figure(nrows, ncols)
    for ri, cell in enumerate(cells):
        for ci, token in enumerate(spec_tokens):
            ax = axes[ri, ci]
            key = (cell, token)
            if key not in ca_mean:
                ax.axis('off')
                continue
            v_th = v_th_by_cell.get(cell)
            el = e_leaks.get(cell)
            readout_src = readout_on if ci < ncols_on else readout_2
            wt = wt_on if ci < ncols_on else wt_2
            if v_th_by_cell_2 is not None and ci >= ncols_on:
                v_th = v_th_by_cell_2.get(cell)
                el = e_leaks_2.get(cell)
            before_t = wt.before_t[token]
            dsi_on = ci < ncols_on
            cell_title = _moving_bar_cell_title(
                token, ca_mean, gt_mean,
                ca_dsi_on if dsi_on else ca_dsi_2,
                gt_dsi_on if dsi_on else gt_dsi_2,
                key,
                cell=cell,
                cost_parts=cost_parts,
                cost_tasks=cost_tasks,
            )
            if has_slices and readout_src is not None and readout_src.slice_overlay is not None:
                slice_traces = {
                    label: readout_src.slice_overlay[label].ca_mean[key]
                    for label in slice_labels
                    if key in readout_src.slice_overlay[label].ca_mean
                }
                if not slice_traces:
                    ax.axis('off')
                    continue
                plot_labels = [label for label in slice_labels if label in slice_traces]
                _plot_moving_bar_cell_slices(
                    ax, ca_mean[key], ca_std.get(key),
                    slice_traces, plot_labels,
                    cell_title, before_t,
                    gt_trace=gt_trace_affine(readout_src, cell, gt_mean.get(key)),
                    show_ylabel=(ci == 0),
                    show_std=show_std and key in ca_std and np.any(ca_std[key]),
                    show_legend=(ri == 0 and ci == 0),
                    cell_ticks=False,
                    show_tick_labels=(ri == nrows - 1),
                    mark_cost_window=True,
                    v_th=v_th,
                    e_leak=el,
                    pre_end=_moving_bar_pre_end(readout_src, cell, token),
                    show_pre=readout_src.show_pre,
                    delta_ms=readout_src.session.delta_ms,
                    ms_shown=readout_src.ms_shown,
                )
            else:
                src = readout_src or readout_on
                _plot_moving_bar_cell(
                    ax, ca_mean[key], ca_std.get(key),
                    cell_title, before_t,
                    gt_trace=gt_trace_affine(src, cell, gt_mean.get(key)),
                    show_ylabel=(ci == 0),
                    show_std=show_std and key in ca_std and np.any(ca_std[key]),
                    cell_ticks=False,
                    show_tick_labels=(ri == nrows - 1),
                    mark_cost_window=True,
                    v_th=v_th,
                    e_leak=el,
                    pre_end=_moving_bar_pre_end(src, cell, token),
                    show_pre=src.show_pre,
                    delta_ms=src.session.delta_ms,
                    ms_shown=src.ms_shown,
                )
        axes[ri, 0].set_ylabel(cell_ylabel(cell, ca_n), fontsize=8, labelpad=12)
    if title is None:
        title = 'Moving-bar ca-all (right only)' if right_only else 'Moving-bar ca-all'
    scope = _moving_bar_all_scope_label(readout_on)
    fig.suptitle(title + f'  [{scope}, t_first_sti-aligned full trace]', fontsize=12)
    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.50, wspace=0.35)
    return fig


@torch.no_grad()
def plot_moving_bar_gt(path, *, readout, readout_2=None, title=None, cost_parts=None):
    """Draw ca-gt figure from a full-scope :class:`MovingBarTraceReadout`."""
    timer = ElapsedTimer(prior_prep=readout_prep_s(readout, readout_2))
    timer.end_prep()
    single_hex = readout.single_hex
    row_specs = moving_bar_specs_by_cell(readout.session, readout.task, readout.side)
    gt_cells = list(row_specs.keys())
    ncols_half = max((len(v) for v in row_specs.values()), default=8)
    cost_tasks = _moving_bar_cost_tasks(readout, readout_2)
    if readout_2 is not None:
        row_specs_2 = moving_bar_specs_by_cell(readout_2.session, readout_2.task, readout_2.side)
        ncols_half = max(ncols_half, max((len(v) for v in row_specs_2.values()), default=8))
        ncols = ncols_half * 2
    else:
        row_specs_2 = None
        ncols = ncols_half
    nrows = len(gt_cells)
    fig, axes = _moving_bar_figure(nrows, ncols)

    ca_dsi_on = dsi_from_trace_map(readout.traces.ca_mean, gt_cells, readout.spec_tokens)
    gt_dsi_on = dsi_from_trace_map(readout.gt_mean, gt_cells, readout.spec_tokens)
    ca_dsi_2 = gt_dsi_2 = None
    if readout_2 is not None:
        ca_dsi_2 = dsi_from_trace_map(readout_2.traces.ca_mean, gt_cells, readout_2.spec_tokens)
        gt_dsi_2 = dsi_from_trace_map(readout_2.gt_mean, gt_cells, readout_2.spec_tokens)

    def _plot_row(ri, subtype, specs, col_offset, row_readout, side, ca_dsi, gt_dsi):
        wt = row_readout.traces
        for ci, token in enumerate(specs):
            ax = axes[ri, col_offset + ci]
            key = (subtype, token)
            if key not in wt.ca_mean:
                ax.axis('off')
                continue
            before_t = wt.before_t[token]
            cell_title = _moving_bar_cell_title(
                token, wt.ca_mean, row_readout.gt_mean,
                ca_dsi, gt_dsi, key,
                cell=subtype,
                cost_parts=cost_parts,
                cost_tasks=cost_tasks,
            )
            _plot_moving_bar_cell(
                ax, wt.ca_mean[key], wt.ca_std[key],
                cell_title, before_t,
                gt_trace=gt_trace_affine(
                    row_readout, subtype, row_readout.gt_mean.get(key),
                ),
                show_ylabel=(col_offset + ci == 0), show_std=not single_hex,
                mark_cost_window=True,
                v_th=row_readout.v_th_by_cell.get(subtype),
                e_leak=row_readout.e_leaks.get(subtype),
                linestyle=_moving_bar_spec_linestyle(side, subtype, token),
                pre_end=_moving_bar_pre_end(row_readout, subtype, token),
                show_pre=row_readout.show_pre,
                delta_ms=row_readout.session.delta_ms,
                ms_shown=row_readout.ms_shown,
            )

    for ri, subtype in enumerate(gt_cells):
        _plot_row(ri, subtype, row_specs[subtype], 0, readout, readout.side, ca_dsi_on, gt_dsi_on)
        if readout_2 is not None:
            _plot_row(
                ri, subtype, row_specs_2[subtype], ncols_half, readout_2, readout_2.side,
                ca_dsi_2, gt_dsi_2,
            )
        axes[ri, 0].set_ylabel(
            cell_ylabel(subtype, readout.traces.ca_n), fontsize=8, labelpad=12,
        )
    if title is None:
        title = 'Moving-bar ca-gt'
    scope = _moving_bar_scope_label(readout.session)
    fig.suptitle(
        title + f'  [{scope}, t_first_sti-aligned full trace]',
        fontsize=12,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.50, wspace=0.35)
    timer.end_draw()
    save_figure(fig, path, dpi=MOVING_BAR_DPI, rasterize=True, timer=timer)


@torch.no_grad()
def plot_moving_bar_all(path, *, readout, readout_2=None, title=None, right_only=True,
                        cost_parts=None):
    """Draw ca-all figure from a full-scope :class:`MovingBarTraceReadout`."""
    timer = ElapsedTimer(prior_prep=readout_prep_s(readout, readout_2))
    timer.end_prep()
    fig = _moving_bar_all_figure(
        readout, readout_2, title, right_only=right_only, cost_parts=cost_parts,
    )
    timer.end_draw()
    save_figure(fig, path, dpi=MOVING_BAR_DPI, rasterize=True, timer=timer)
