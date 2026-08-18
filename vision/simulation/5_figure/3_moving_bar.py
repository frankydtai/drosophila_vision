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
    fig1_trace_from_sti,
    load_fig1_trace,
    motion_preference,
    moving_bar_cell_title,
    dsi_from_trace_map,
)
from task.moving_bar.pack import (
    bar_specs_from_task,
    nodes_from_hexes,
    moving_bar_specs_by_cell,
    moving_bar_session_t0_grids,
)
from figure.gt import pack_cells
from network.construction import cells_in_order
from figure.panel import (
    GT_COLOR,
    TRACE_LW,
    ElapsedTimer,
    annotate_v_th,
    as_numpy,
    e_leak_from_z,
    format_moving_bar_cell_cost_lines,
    readout_prep_s,
    hex_scope_label,
    cell_ylabel,
    gt_affine_from_cell,
    gt_trace_affine,
    ms_shown_axis_xlim,
    overlay_reds,
    plot_pre_post_line,
    plot_std_band,
    plot_timecourse,
    pack_center_mask,
    save_figure,
    std_from_traces,
    overlay_axis,
    overlay_coords,
    suppress_cost_std,
    v_th_from_z,
)
import network.path  # noqa: F401  # ensure FAFBv783 modules are importable
from task.moving_bar.sti_geo import (
    filter_sti_hexes,
    moving_bar_cost_hexes,
    network_uv_np,
    sti_hexes,
)
from task.moving_bar.sti_spec import (
    COST_WINDOW_AFTER_MS,
    COST_WINDOW_BEFORE_MS,
)
from neuron.borst import t_from_ms

MOVING_BAR_DPI = 100


@dataclass
class MovingBarWindowTraces:
    ca_mean: dict
    ca_std: dict
    ca_n: dict
    before_t: dict
    after_t: dict
    t0_bn: np.ndarray
    cell_idxs: np.ndarray
    cells: list


@dataclass
class MovingBarTraceReadout:
    task: str
    contrast: str
    cells: list
    spec_tokens: list
    side: str
    single_hex: bool
    v_th_by_cell: dict
    e_leak_by_cell: dict
    gt_mean: dict
    n_t: int
    traces: MovingBarWindowTraces
    session: object
    at_x: int | None = None
    at_y: int | None = None
    n_filter_hex: int = 0
    overlay: dict | None = None
    overlay_axis: str | None = None
    overlay_xs: list | None = None
    overlay_ys: list | None = None
    align_at_x: int | None = None
    align_at_y: int | None = None
    prep_s: float = 0.0
    show_pre: bool = True
    t_onset: int | None = None
    gt_affine_by_cell: dict = field(default_factory=dict)
    ms_shown: tuple[float, float] | None = None

    @property
    def has_overlays(self):
        return bool(self.overlay)


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


def figure_cell_idx_from_node_cells(connectome_cells, node_cells, figure_cells):
    """Map node ``node_cells`` (connectome cell idx) to ``figure_cells`` idx.

    ``cells_in_order`` reorders cells; matching raw ``node_cells`` against
    ``enumerate(figure_cells)`` mislabels every cell whose plot idx ≠ connectome idx.
    """
    cell_idxs = np.asarray(as_numpy(node_cells), dtype=np.int64)
    cell_idx = dict(zip(
        [str(cell) for cell in figure_cells], range(len(figure_cells)),
    ))
    out = np.full(cell_idxs.shape, -1, dtype=np.int64)
    for ci, name in enumerate(connectome_cells):
        pi = cell_idx.get(str(name))
        if pi is None:
            continue
        out[cell_idxs == int(ci)] = int(pi)
    return out


def _cells_and_cell_idxs(session):
    """Plot-order cell order + remapped ``cell_idxs`` for bar."""
    connectome = session.connectome
    cells = cells_in_order(connectome.cells)
    cell_idxs = figure_cell_idx_from_node_cells(
        connectome.cells, connectome.node_cells, cells,
    )
    return cells, cell_idxs


def _filter_right_specs(spec_tokens, right_only):
    if right_only:
        return [token for token in spec_tokens if token.startswith('right_')]
    return list(spec_tokens)


def _windows_by_b(trace, t0_bn, n_t_by_b):
    """``n_t_by_b``: int (uniform) or length-``B`` sequence of trace lengths."""
    n_b = trace.shape[0]
    if isinstance(n_t_by_b, int):
        n_t_by_b = (n_t_by_b,) * n_b
    out = []
    for b in range(n_b):
        n_t_b = int(n_t_by_b[b])
        sl = trace[b:b + 1]
        t0 = t0_bn[b:b + 1]
        t_len = sl.shape[1]
        n_node = sl.shape[2]
        ts = np.arange(n_t_b, dtype=np.int64)
        t_abs = t0[..., None] + ts[None, None, :]
        t_safe = np.clip(t_abs, 0, t_len - 1)
        bs = np.zeros(1, dtype=np.int64)[:, None, None]
        nodes = np.arange(n_node, dtype=np.int64)[None, :, None]
        window = sl[bs, t_safe, nodes].astype(np.float64, copy=False)
        window[t_abs < 0] = 0.0
        out.append(window[0])
    return out


def _moving_bar_trace_means(
    windows_by_b, t0_bn, cell_idxs, cells, spec_tokens, single_hex, *,
    hex_mask=None,
):
    """``windows_by_b[b]`` shape ``(n_node, W_bi)``."""
    ca_mean, ca_std, ca_n = {}, {}, {}
    valid = t0_bn >= 0
    for ti, cell in enumerate(cells):
        cell_mask = cell_idxs == ti
        if not cell_mask.any():
            continue
        for b, token in enumerate(spec_tokens):
            node_mask = valid[b] & cell_mask
            if hex_mask is not None:
                node_mask = node_mask & hex_mask[b]
            if not node_mask.any():
                continue
            uids = np.nonzero(node_mask)[0]
            arr = windows_by_b[b][uids]
            key = (cell, token)
            ca_mean[key] = arr.mean(axis=0)
            ca_std[key] = std_from_traces(arr, single_hex=single_hex)
            ca_n[key] = int(np.unique(uids).size)
    return ca_mean, ca_std, ca_n


def _network_hex_node_mask(connectome, filt_hexes, n_b):
    node_u_np, node_v_np = network_uv_np(connectome)
    hex_uv = {(int(hex.u), int(hex.v)) for hex in filt_hexes}
    hex_mask = np.array(
        [(int(hex_u), int(hex_v)) in hex_uv for hex_u, hex_v in zip(node_u_np, node_v_np)],
        dtype=bool,
    )
    return np.broadcast_to(hex_mask, (n_b, connectome.n_node)).copy()


def _t0_from_align_hex(t0_bn, b, ref_hex, *, connectome):
    node_u_np, node_v_np = network_uv_np(connectome)
    ref_hex_mask = (node_u_np == int(ref_hex.u)) & (node_v_np == int(ref_hex.v))
    t0_ref = int(t0_bn[b, ref_hex_mask][0])
    if t0_ref < 0:
        loc = f'({int(ref_hex.u)},{int(ref_hex.v)})'
        raise SystemExit(f'--align-xy ref hex {loc} has no valid t0')
    return t0_ref


def t0_bn_overlay_from_ref(
    t0_bn, n_b, filt_hexes, align_at_x, align_at_y, *,
    session, cost_radius,
):
    """Copy ``t0_bn`` with overlay nodes forced to the ref hex ``t0`` (plot only)."""
    connectome = session.connectome
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
    for b in range(n_b):
        t0_ref = _t0_from_align_hex(out, b, ref_hex, connectome=connectome)
        for hex in filt_hexes:
            on_hex = (node_u_np == int(hex.u)) & (node_v_np == int(hex.v))
            out[b, on_hex] = t0_ref
    return out


def _moving_bar_overlay_traces(
    session, task, contrast, trace, base_wt, spec_tokens, *, at_x=None, at_y=None,
    align_at_x=None, align_at_y=None,
):
    """Per-hex overlay traces aligned to ``base_wt`` trace geometry."""
    if base_wt.t0_bn is None or base_wt.cell_idxs is None or base_wt.cells is None:
        raise ValueError("base_wt missing cached t0_bn/cells for overlay")
    pack = session.packs[task][contrast]
    cells = base_wt.cells
    cell_idxs = base_wt.cell_idxs
    t0_bn = base_wt.t0_bn
    n_b = len(spec_tokens)
    connectome = session.connectome
    hexes = moving_bar_cost_hexes(connectome, cost_radius=pack.cost_radius)
    filt_hexes = filter_sti_hexes(hexes, at_x=at_x, at_y=at_y)
    if not filt_hexes:
        return None
    hex_mask = _network_hex_node_mask(connectome, filt_hexes, n_b)
    n_t_by_b = [
        base_wt.before_t[token] + base_wt.after_t[token] + 1
        for token in spec_tokens
    ]
    t0_bn_aligned = t0_bn
    if align_at_x is not None and align_at_y is not None:
        t0_bn_aligned = t0_bn_overlay_from_ref(
            t0_bn, n_b, filt_hexes, align_at_x, align_at_y,
            session=session, cost_radius=pack.cost_radius,
        )
    windows_by_b = _windows_by_b(trace, t0_bn_aligned, n_t_by_b)
    ca_mean, ca_std, ca_n = _moving_bar_trace_means(
        windows_by_b, t0_bn_aligned, cell_idxs, cells, spec_tokens, True, hex_mask=hex_mask,
    )
    return MovingBarWindowTraces(
        ca_mean=ca_mean,
        ca_std=ca_std,
        ca_n=ca_n,
        before_t=base_wt.before_t,
        after_t=base_wt.after_t,
        t0_bn=t0_bn,
        cell_idxs=cell_idxs,
        cells=cells,
    )


def _fig1_trace_delta(trace: np.ndarray, delta_ms: float) -> np.ndarray:
    """ΔVm for fig1 cost-window traces (subtract pre-sti mean)."""
    trace = np.asarray(trace, dtype=np.float64)
    i_on = t_from_ms(COST_WINDOW_BEFORE_MS, delta_ms=delta_ms)
    if i_on > 0 and i_on < len(trace):
        return trace - float(np.mean(trace[:i_on]))
    return trace - float(trace[0])


def _load_moving_bar_gt_mean(session, task, contrast, cells, specs, side):
    gt_mean = {}
    row_cells = cells_in_order(pack_cells(session, task, contrast))
    for subtype in row_cells:
        if subtype not in cells:
            continue
        for spec in specs:
            trace_token = fig1_trace_from_sti(side, subtype, spec)
            if trace_token is None:
                continue
            trace = _fig1_trace_delta(load_fig1_trace(trace_token), session.delta_ms)
            gt_mean[(subtype, spec.token)] = trace
    return gt_mean


def _traces_from_forward(
    session, task, contrast, trace, specs, spec_tokens, *,
    at_x=None, at_y=None,
):
    pack = session.packs[task][contrast]
    cost_radius = pack.cost_radius
    n_t = int(session.n_t)
    _t_onset = train.pack_t_onset(pack)
    grids = moving_bar_session_t0_grids(
        session, specs, cost_radius, n_t, at_x=at_x, at_y=at_y,
        t_onset=_t_onset, delta_ms=session.delta_ms,
    )
    cells, cell_idxs = _cells_and_cell_idxs(session)
    t0_bn = grids.t0_bn
    before_t = grids.before_t
    after_t = grids.after_t
    side = grids.side
    n_filter_hex = grids.n_filter_hex
    single_hex = (
        suppress_cost_std(session, task, contrast) or n_filter_hex == 1
    )
    n_t_by_b = [
        before_t[token] + after_t[token] + 1
        for token in spec_tokens
    ]
    windows_by_b = _windows_by_b(trace, t0_bn, n_t_by_b)
    trace_mean, trace_std, trace_n = _moving_bar_trace_means(
        windows_by_b, t0_bn, cell_idxs, cells, spec_tokens, single_hex,
    )
    return MovingBarWindowTraces(
        ca_mean=trace_mean,
        ca_std=trace_std,
        ca_n=trace_n,
        before_t=before_t,
        after_t=after_t,
        t0_bn=t0_bn,
        cell_idxs=cell_idxs,
        cells=cells,
    ), cells, side, n_filter_hex, _t_onset, single_hex


@torch.no_grad()
def moving_bar_trace_readout(session, z, task, contrast, *, at_x=None, at_y=None,
                            at_xs=None, at_ys=None,
                            align_at_x=None, align_at_y=None,
                            show_pre=True, ms_shown=None):
    """Run one forward; t_first_sti-aligned v_readout traces."""
    t_prep0 = time.perf_counter()
    pack = session.packs[task][contrast]
    params = train.params_from_z(z, session)
    plot_t = train.forward_pack(session, params, pack.i_sti, pack)
    trace = plot_t.detach().cpu().numpy()
    specs = bar_specs_from_task(session, task)
    spec_tokens = [spec.token for spec in specs]
    n_t = int(session.n_t)
    connectome = session.connectome
    traces, cells, side, n_filter_hex, t_onset, single_hex = _traces_from_forward(
        session, task, contrast, trace, specs, spec_tokens,
    )
    v_th = v_th_from_z(z, session)
    if connectome is not None:
        hexes = moving_bar_cost_hexes(connectome, cost_radius=pack.cost_radius)
        if at_x is not None or at_y is not None:
            hexes = filter_sti_hexes(hexes, at_x=at_x, at_y=at_y)
        nodes_by_cell = {
            cell: nodes_from_hexes(connectome, cell, hexes) for cell in cells
        }
    else:
        cell_idxs = np.asarray(as_numpy(session.connectome.conn.node_cells), dtype=np.int64)
        entry_nodes = pack.entry_nodes.cpu().numpy()
        center = pack_center_mask(pack, session.connectome)
        node_cells = cell_idxs[entry_nodes]
        nodes_by_cell = {
            name: entry_nodes[center & (node_cells == cells.index(name))]
            for name in cells
        }
    e_leak = e_leak_from_z(z, session)
    v_th_by_cell = {cell: v_th.get(cell, np.nan) for cell in nodes_by_cell}
    e_leak_by_cell = {cell: e_leak.get(cell, np.nan) for cell in nodes_by_cell}
    gt_mean = _load_moving_bar_gt_mean(
        session, task, contrast, cells, specs, side,
    )
    gt_affine_by_cell = {
        str(name): gt_affine_from_cell(
            params, name, session.connectome, session=session,
        )
        for name in cells
    }
    overlay = None
    axis = None
    overlay_xs = None
    overlay_ys = None
    overlays = overlay_coords(at_xs, at_ys)
    if overlays:
        axis = overlay_axis(at_xs, at_ys)
        overlay_xs = list(at_xs) if at_xs is not None else None
        overlay_ys = list(at_ys) if at_ys is not None else None
        overlay = {}
        for label, xv, yv in overlays:
            wt = _moving_bar_overlay_traces(
                session, task, contrast, trace, traces, spec_tokens,
                at_x=xv, at_y=yv,
                align_at_x=align_at_x, align_at_y=align_at_y,
            )
            if wt is None:
                print(f'skip overlay {label}: no hex within cost_radius')
                continue
            overlay[label] = wt
        if not overlay:
            overlay = None
            axis = None
            overlay_xs = None
            overlay_ys = None
    return MovingBarTraceReadout(
        task=task,
        contrast=contrast,
        cells=cells,
        spec_tokens=spec_tokens,
        side=side,
        single_hex=single_hex,
        v_th_by_cell=v_th_by_cell,
        e_leak_by_cell=e_leak_by_cell,
        gt_mean=gt_mean,
        n_t=n_t,
        traces=traces,
        session=session,
        at_x=at_x,
        at_y=at_y,
        n_filter_hex=n_filter_hex,
        overlay=overlay,
        overlay_axis=axis,
        overlay_xs=overlay_xs,
        overlay_ys=overlay_ys,
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
        b = readout.spec_tokens.index(token)
        ti = wt.cells.index(subtype)
    except ValueError:
        return 0
    key = (subtype, token)
    if key not in wt.ca_mean:
        return 0
    n_t_figure = int(np.asarray(wt.ca_mean[key]).shape[0])
    cell_mask = wt.cell_idxs == ti
    valid = (wt.t0_bn[b] >= 0) & cell_mask
    if not bool(valid.any()):
        return 0
    pre = np.clip(int(t_onset) - wt.t0_bn[b, valid], 0, n_t_figure)
    return int(np.median(pre))


def _cost_window_xy(cost_trace, before_t, delta_ms):
    """Map cost_window GT onto trace x/y coordinates."""
    i0 = before_t - t_from_ms(COST_WINDOW_BEFORE_MS, delta_ms=delta_ms)
    trace = np.asarray(cost_trace, dtype=np.float64)
    if i0 < 0:
        trace = trace[-i0:]
        i0 = 0
    x = np.arange(i0, i0 + len(trace), dtype=np.int64)
    return x, trace


def _moving_bar_scope_label(session, *, at_x=None, at_y=None, n_filter_hex=None):
    pack = session.primary_pack
    cost_radius = pack.cost_radius
    connectome = session.connectome
    if at_x is not None or at_y is not None:
        ncol_part = f'{n_filter_hex} sti hex'
        if n_filter_hex != 1:
            ncol_part += 's'
        parts = [hex_scope_label(at_x, at_y), ncol_part]
        if cost_radius is not None:
            parts.insert(0, f'cost_radius={cost_radius}')
        return ', '.join(parts)
    if cost_radius is not None:
        n_sti_hex = len(moving_bar_cost_hexes(connectome, cost_radius=cost_radius))
        return f'cost_radius={cost_radius} ({n_sti_hex} sti hexes)'
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
        cw_before = t_from_ms(COST_WINDOW_BEFORE_MS, delta_ms=delta_ms)
        cw_after = t_from_ms(COST_WINDOW_AFTER_MS, delta_ms=delta_ms)
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
        gt_x, gt_y = _cost_window_xy(gt_trace, before_t, delta_ms)

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
    # Gray fig1 GT is already restricted to cost_window (no global pre).
    if gt_x is not None:
        ax.plot(gt_x, gt_y, color=GT_COLOR, linewidth=TRACE_LW, linestyle=linestyle)


def _plot_moving_bar_cell_overlays(
    ax,
    scope_trace,
    std_trace,
    overlay_traces,
    overlay_labels,
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
        gt_x, gt_y = _cost_window_xy(gt_trace, before_t, delta_ms)

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
    colors = overlay_reds(len(overlay_labels))
    for label, color in zip(overlay_labels, colors):
        plot_pre_post_line(
            ax, t, overlay_traces[label], pre_end=pre_end,
            show_pre=show_pre, plot_pre=True,
            color=color, linestyle='-', linewidth=TRACE_LW, label=label,
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
    if readout_on.has_overlays:
        pack = readout_on.session.primary_pack
        cost_radius = pack.cost_radius
        at_x = readout_on.overlay_xs if readout_on.overlay_axis in ('x', 'xy') else None
        at_y = readout_on.overlay_ys if readout_on.overlay_axis in ('y', 'xy') else None
        parts = [hex_scope_label(at_x, at_y), 'overlay + scope']
        if readout_on.align_at_x is not None and readout_on.align_at_y is not None:
            parts.append(
                'aligned to '
                + hex_scope_label(readout_on.align_at_x, readout_on.align_at_y),
            )
        if cost_radius is not None:
            parts.insert(0, f'cost_radius={cost_radius}')
        return ', '.join(parts)
    return _moving_bar_scope_label(readout_on.session)


def _moving_bar_cost_contrasts(readout_on, readout_2=None):
    contrasts = [readout_on.contrast]
    if readout_2 is not None and readout_2.contrast not in contrasts:
        contrasts.append(readout_2.contrast)
    return contrasts


def _moving_bar_all_figure(readout_on, readout_2, title, *, right_only=True, cost_parts=None):
    single_hex = readout_on.single_hex
    cells = readout_on.cells
    wt_on = readout_on.traces
    spec_tokens = _filter_right_specs(readout_on.spec_tokens, right_only)
    ncols_on = len(spec_tokens)
    ca_mean, ca_std, ca_n = wt_on.ca_mean, wt_on.ca_std, wt_on.ca_n
    gt_mean = readout_on.gt_mean
    v_th_by_cell = readout_on.v_th_by_cell
    e_leak_by_cell = readout_on.e_leak_by_cell
    v_th_by_cell_2 = None
    e_leak_by_cell_2 = None
    wt_2 = None
    overlay_labels = (
        list(readout_on.overlay.keys()) if readout_on.overlay else []
    )
    has_overlays = readout_on.has_overlays
    cost_contrasts = _moving_bar_cost_contrasts(readout_on, readout_2)
    if readout_2 is not None:
        wt_2 = readout_2.traces
        spec_2 = _filter_right_specs(readout_2.spec_tokens, right_only)
        spec_tokens = list(spec_tokens) + list(spec_2)
        ca_mean = {**ca_mean, **wt_2.ca_mean}
        ca_std = {**ca_std, **wt_2.ca_std}
        ca_n = {**ca_n, **wt_2.ca_n}
        gt_mean = {**gt_mean, **readout_2.gt_mean}
        v_th_by_cell_2 = readout_2.v_th_by_cell
        e_leak_by_cell_2 = readout_2.e_leak_by_cell
    show_std = not single_hex and not has_overlays
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
            el = e_leak_by_cell.get(cell)
            readout_src = readout_on if ci < ncols_on else readout_2
            wt = wt_on if ci < ncols_on else wt_2
            if v_th_by_cell_2 is not None and ci >= ncols_on:
                v_th = v_th_by_cell_2.get(cell)
                el = e_leak_by_cell_2.get(cell)
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
            if has_overlays and readout_src is not None and readout_src.overlay is not None:
                overlay_traces = {
                    label: readout_src.overlay[label].ca_mean[key]
                    for label in overlay_labels
                    if key in readout_src.overlay[label].ca_mean
                }
                if not overlay_traces:
                    ax.axis('off')
                    continue
                plot_labels = [label for label in overlay_labels if label in overlay_traces]
                _plot_moving_bar_cell_overlays(
                    ax, ca_mean[key], ca_std.get(key),
                    overlay_traces, plot_labels,
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
    fig.suptitle(title + f'  [{scope}, t_first_sti-aligned trace]', fontsize=12)
    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.50, wspace=0.35)
    return fig


@torch.no_grad()
def plot_moving_bar_gt(path, *, readout, readout_2=None, title=None, cost_parts=None):
    """Draw ca-gt figure from a scope :class:`MovingBarTraceReadout`."""
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
                e_leak=row_readout.e_leak_by_cell.get(subtype),
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
        title + f'  [{scope}, t_first_sti-aligned trace]',
        fontsize=12,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.50, wspace=0.35)
    timer.end_draw()
    save_figure(fig, path, dpi=MOVING_BAR_DPI, rasterize=True, timer=timer)


@torch.no_grad()
def plot_moving_bar_all(path, *, readout, readout_2=None, title=None, right_only=True,
                        cost_parts=None):
    """Draw ca-all figure from a scope :class:`MovingBarTraceReadout`."""
    timer = ElapsedTimer(prior_prep=readout_prep_s(readout, readout_2))
    timer.end_prep()
    fig = _moving_bar_all_figure(
        readout, readout_2, title, right_only=right_only, cost_parts=cost_parts,
    )
    timer.end_draw()
    save_figure(fig, path, dpi=MOVING_BAR_DPI, rasterize=True, timer=timer)
