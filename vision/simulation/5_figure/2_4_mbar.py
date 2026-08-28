"""Moving-bar plotting extracted from ``figure.plot``."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

import train
from task.mbar.gt import (
    GT_CELLS,
    fig1_trace_from_sti,
    load_fig1_trace,
    motion_preference,
)
from task.mbar.pack import (
    bar_specs_from_task,
    nodes_from_hexes,
    mbar_specs_by_cell,
    mbar_session_t0_grids,
)
from figure.spread import pack_cells, contrast_order
from network.construction import cells_in_order
from figure.plot import session_from_task
from figure.panel import (
    GT_COLOR,
    TRACE_LINE_W,
    ElapsedTimer,
    annotate_v_th,
    as_numpy,
    e_leak_from_z,
    readout_prep_s,
    at_xy_label,
    cell_ylabel,
    gt_trace_affine,
    ms_shown_axis_xlim,
    at_xy_reds,
    plot_trace,
    plot_sem_band,
    plot_timecourse,
    pack_center_mask,
    save_figure,
    sem_from_traces,
    expand_at_xy,
    is_single_hex_cost,
    v_th_from_z,
)
from task.mbar.sti_spec import PD_ND_LABELS
from task.spread.sti_spec import CONTRASTS
from task.spread.pack import cost_sti_hexes
import network.path  # noqa: F401  # ensure FAFBv783 modules are importable
from task.sbar.sti_geo import (
    sti_hexes_at_xy,
    node_us_vs,
    sti_hexes,
)
from task.mbar.sti_spec import (
    COST_WINDOW_AFTER_MS,
    COST_WINDOW_BEFORE_MS,
)
from neuron.borst import t_from_ms

MBAR_DPI = 100


def _mbar_cell_pd_nd_cost_sum(cost_parts, contrast, cell, pd_nd_label):
    suffix = f"_{cell}_{pd_nd_label}"
    total = 0.0
    found = False
    for part_key, val in cost_parts.items():
        if not part_key.startswith("mbar_") or not part_key.endswith(suffix):
            continue
        spec_token = part_key[5:-len(suffix)]
        spec_parts = spec_token.split("_")
        if len(spec_parts) >= 2 and spec_parts[1] == contrast:
            total += float(val)
            found = True
    return total if found else None


def _mbar_cell_cost_labels(cell, cost_parts, contrasts):
    """``ON: xx @PD yy @ND`` / ``OFF: …`` title labels for moving-bar panels."""
    label_map = {
        'bright': 'ON',
        'dark': 'OFF',
    }
    labels = []
    if not cost_parts:
        return labels
    for contrast in contrasts:
        bits = []
        for pd_nd_label in PD_ND_LABELS:
            part_cost = _mbar_cell_pd_nd_cost_sum(
                cost_parts, contrast, cell, pd_nd_label,
            )
            if part_cost is not None:
                bits.append(f'{part_cost:.1f} @{pd_nd_label}')
        if bits:
            labels.append(f'{label_map.get(contrast, contrast)}: {" ".join(bits)}')
    return labels


PLOT_AT_XY = True
PLOT_ALIGN_XY = True


@dataclass
class MbarWindowTraces:
    ca_mean_cell: dict
    ca_sem: dict
    ca_n: dict
    before_t: dict
    after_t: dict
    t0_bn: np.ndarray
    cell_idxs: np.ndarray
    cells: list


@dataclass
class MbarTraceReadout:
    task: str
    contrast: str
    cells: list
    spec_tokens: list
    side: str
    single_hex: bool
    v_th_by_cell: dict
    e_leak_by_cell: dict
    gt_traces: dict
    n_t: int
    traces: MbarWindowTraces
    session: object
    at_x: int | None = None
    at_y: int | None = None
    at_xs: list | None = None
    at_ys: list | None = None
    labels: list[str] | None = None
    ca_mean_cell_mean_hex_by_label: dict | None = None
    n_hex: int = 0
    align_at_x: int | None = None
    align_at_y: int | None = None
    prep_s: float = 0.0
    t_onset: int | None = None
    gt_affine_by_cell: dict = field(default_factory=dict)
    ms_shown: tuple[float, float] | None = None


def _mbar_figure(n_row, n_col, *, sharex='col'):
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(2.2 * n_col, 1.8 * n_row), sharex=sharex,
    )
    if n_row == 1:
        axes = np.asarray([axes])
    if n_col == 1:
        axes = axes[:, None]
    return fig, axes


def _mbar_cell_title(
    token,
    *,
    cell,
    cost_parts=None,
    cost_contrasts=None,
):
    head = token
    if cost_parts is not None and cost_contrasts:
        cost_labels = _mbar_cell_cost_labels(cell, cost_parts, cost_contrasts)
        if cost_labels:
            head = '\n'.join([f'{cell} Cost', *cost_labels, head])
    return head


def figure_cell_idx_from_node_cells(connectome_cells, node_cells, figure_cells):
    """Map node ``node_cells`` (connectome cell idx) to ``figure_cells`` idx.

    ``cells_in_order`` reorders cells; matching ``node_cells`` against
    ``enumerate(figure_cells)`` mislabels every cell whose plot idx ≠ connectome idx.
    """
    cell_idxs = np.asarray(as_numpy(node_cells), dtype=np.int64)
    cell_idx = dict(zip(
        [str(cell) for cell in figure_cells], range(len(figure_cells)),
    ))
    figure_cell_idx = np.full(cell_idxs.shape, -1, dtype=np.int64)
    for connectome_cell_idx, cell in enumerate(connectome_cells):
        figure_cell_idx_slot = cell_idx.get(str(cell))
        if figure_cell_idx_slot is None:
            continue
        figure_cell_idx[cell_idxs == int(connectome_cell_idx)] = int(figure_cell_idx_slot)
    return figure_cell_idx


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
    windows_by_b = []
    for b in range(n_b):
        n_t_b = int(n_t_by_b[b])
        window_slice = trace[b:b + 1]
        t0 = t0_bn[b:b + 1]
        t_len = window_slice.shape[1]
        n_node = window_slice.shape[2]
        ts = np.arange(n_t_b, dtype=np.int64)
        t_abs = t0[..., None] + ts[None, None, :]
        t_safe = np.clip(t_abs, 0, t_len - 1)
        bs = np.zeros(1, dtype=np.int64)[:, None, None]
        nodes = np.arange(n_node, dtype=np.int64)[None, :, None]
        window = window_slice[bs, t_safe, nodes].astype(np.float64, copy=False)
        window[t_abs < 0] = 0.0
        windows_by_b.append(window[0])
    return windows_by_b


def _mbar_ca_mean_cell(
    windows_by_b, t0_bn, cell_idxs, cells, spec_tokens, single_hex, *,
    hex_mask=None,
):
    """``windows_by_b[b]`` shape ``(n_node, n_t)``."""
    ca_mean_cell, ca_sem, ca_n = {}, {}, {}
    valid = t0_bn >= 0
    for figure_cell_idx, cell in enumerate(cells):
        cell_mask = cell_idxs == figure_cell_idx
        if not cell_mask.any():
            continue
        for b, token in enumerate(spec_tokens):
            node_mask = valid[b] & cell_mask
            if hex_mask is not None:
                node_mask = node_mask & hex_mask[b]
            if not node_mask.any():
                continue
            nodes = np.nonzero(node_mask)[0]
            traces = windows_by_b[b][nodes]
            key = (cell, token)
            ca_mean_cell[key] = traces.mean(axis=0)
            ca_sem[key] = sem_from_traces(traces, single_hex=single_hex)
            ca_n[key] = int(np.unique(nodes).size)
    return ca_mean_cell, ca_sem, ca_n


def _network_hex_node_mask(connectome, hexes, n_b):
    node_us, node_vs = node_us_vs(connectome)
    hex_uv = {(int(hex.u), int(hex.v)) for hex in hexes}
    hex_mask = np.array(
        [(int(u), int(v)) in hex_uv for u, v in zip(node_us, node_vs)],
        dtype=bool,
    )
    return np.broadcast_to(hex_mask, (n_b, connectome.n_node)).copy()


def _t0_from_align_hex(t0_bn, b, ref_hex, *, connectome):
    node_us, node_vs = node_us_vs(connectome)
    ref_hex_mask = (node_us == int(ref_hex.u)) & (node_vs == int(ref_hex.v))
    t0_ref = int(t0_bn[b, ref_hex_mask][0])
    if t0_ref < 0:
        loc = f'({int(ref_hex.u)},{int(ref_hex.v)})'
        raise SystemExit(f'--align-xy ref hex {loc} has no valid t0')
    return t0_ref


def t0_bn_from_align_at_xy(
    t0_bn, n_b, hexes, align_at_x, align_at_y, *,
    session, cost_radius,
):
    """Copy ``t0_bn`` with at_xy nodes forced to the ref hex ``t0`` (plot only)."""
    connectome = session.connectome
    cost_hexes = cost_sti_hexes(connectome, cost_radius=cost_radius)
    ref_hexes = sti_hexes_at_xy(cost_hexes, at_x=align_at_x, at_y=align_at_y)
    if len(ref_hexes) != 1:
        raise SystemExit(
            f'--align-xy must match exactly one sti hex within cost_radius, '
            f'got {len(ref_hexes)} for x={align_at_x!r} y={align_at_y!r}',
        )
    ref_hex = ref_hexes[0]
    node_us, node_vs = node_us_vs(connectome)
    t0_bn = t0_bn.copy()
    for b in range(n_b):
        t0_ref = _t0_from_align_hex(t0_bn, b, ref_hex, connectome=connectome)
        for hex in hexes:
            on_hex = (node_us == int(hex.u)) & (node_vs == int(hex.v))
            t0_bn[b, on_hex] = t0_ref
    return t0_bn


def _mbar_ca_mean_cell_mean_hex(
    session, task, contrast, trace, base_traces, spec_tokens, *, at_x=None, at_y=None,
    align_at_x=None, align_at_y=None,
):
    """Per at_xy mean_hex: cell mean over ``at_x``/``at_y`` hexes."""
    if base_traces.t0_bn is None or base_traces.cell_idxs is None or base_traces.cells is None:
        raise ValueError("base_traces missing cached t0_bn/cells for at_xy mean_hex")
    pack = session.packs[task][contrast]
    cells = base_traces.cells
    cell_idxs = base_traces.cell_idxs
    t0_bn = base_traces.t0_bn
    n_b = len(spec_tokens)
    connectome = session.connectome
    hexes = cost_sti_hexes(connectome, cost_radius=pack.cost_radius)
    hexes = sti_hexes_at_xy(hexes, at_x=at_x, at_y=at_y)
    if not hexes:
        return None
    hex_mask = _network_hex_node_mask(connectome, hexes, n_b)
    n_t_by_b = [
        base_traces.before_t[token] + base_traces.after_t[token] + 1
        for token in spec_tokens
    ]
    t0_bn_aligned = t0_bn
    if align_at_x is not None and align_at_y is not None:
        t0_bn_aligned = t0_bn_from_align_at_xy(
            t0_bn, n_b, hexes, align_at_x, align_at_y,
            session=session, cost_radius=pack.cost_radius,
        )
    windows_by_b = _windows_by_b(trace, t0_bn_aligned, n_t_by_b)
    ca_mean_cell_mean_hex, _, _ = _mbar_ca_mean_cell(
        windows_by_b, t0_bn_aligned, cell_idxs, cells, spec_tokens, True, hex_mask=hex_mask,
    )
    return ca_mean_cell_mean_hex


def _load_mbar_gt_traces(session, task, contrast, cells, specs, side):
    gt_traces = {}
    row_cells = cells_in_order(pack_cells(session, task, contrast))
    for cell in row_cells:
        if cell not in cells:
            continue
        for spec in specs:
            trace_token = fig1_trace_from_sti(side, cell, spec)
            if trace_token is None:
                continue
            trace = load_fig1_trace(
                trace_token, delta_ms=session.delta_ms,
                baseline_delta=True,
            )
            gt_traces[(cell, spec.token)] = trace
    return gt_traces


def _traces_from_forward(
    session, task, contrast, trace, specs, spec_tokens, *,
    at_x=None, at_y=None,
):
    pack = session.packs[task][contrast]
    cost_radius = pack.cost_radius
    n_t = int(session.n_t)
    _t_onset = train.pack_t_onset(pack)
    grids = mbar_session_t0_grids(
        session, specs, cost_radius, n_t, at_x=at_x, at_y=at_y,
        delta_ms=session.delta_ms,
    )
    cells, cell_idxs = _cells_and_cell_idxs(session)
    t0_bn = grids.t0_bn
    before_t = grids.before_t
    after_t = grids.after_t
    side = grids.side
    n_hex = grids.n_hex
    single_hex = (
        is_single_hex_cost(session, task, contrast) or n_hex == 1
    )
    n_t_by_b = [
        before_t[token] + after_t[token] + 1
        for token in spec_tokens
    ]
    windows_by_b = _windows_by_b(trace, t0_bn, n_t_by_b)
    ca_mean_cell, ca_sem, ca_n = _mbar_ca_mean_cell(
        windows_by_b, t0_bn, cell_idxs, cells, spec_tokens, single_hex,
    )
    return MbarWindowTraces(
        ca_mean_cell=ca_mean_cell,
        ca_sem=ca_sem,
        ca_n=ca_n,
        before_t=before_t,
        after_t=after_t,
        t0_bn=t0_bn,
        cell_idxs=cell_idxs,
        cells=cells,
    ), cells, side, n_hex, _t_onset, single_hex


@torch.no_grad()
def mbar_trace_readout(session, z, task, contrast, *, at_x=None, at_y=None,
                            at_xs=None, at_ys=None,
                            align_at_x=None, align_at_y=None,
                            ms_shown=None):
    """Run one forward; t_first_sti-aligned v_readout traces."""
    t_prep0 = time.perf_counter()
    pack = session.packs[task][contrast]
    params = train.params_from_z(z, session)
    trace = as_numpy(train.forward_pack(session, params, pack.i_sti, pack))
    specs = bar_specs_from_task(session, task, contrast)
    spec_tokens = [spec.token for spec in specs]
    n_t = int(session.n_t)
    connectome = session.connectome
    traces, cells, side, n_hex, t_onset, single_hex = _traces_from_forward(
        session, task, contrast, trace, specs, spec_tokens,
    )
    v_th = v_th_from_z(z, session)
    if connectome is not None:
        hexes = cost_sti_hexes(connectome, cost_radius=pack.cost_radius)
        if at_x is not None or at_y is not None:
            hexes = sti_hexes_at_xy(hexes, at_x=at_x, at_y=at_y)
        nodes_by_cell = {
            cell: nodes_from_hexes(connectome, cell, hexes) for cell in cells
        }
    else:
        cell_idxs = np.asarray(as_numpy(session.connectome.conn.node_cells), dtype=np.int64)
        entry_nodes = pack.entry_nodes.cpu().numpy()
        center = pack_center_mask(pack, session.connectome)
        node_cells = cell_idxs[entry_nodes]
        nodes_by_cell = {
            cell: entry_nodes[center & (node_cells == cells.index(cell))]
            for cell in cells
        }
    e_leak = e_leak_from_z(z, session)
    v_th_by_cell = {cell: v_th.get(cell, np.nan) for cell in nodes_by_cell}
    e_leak_by_cell = {cell: e_leak.get(cell, np.nan) for cell in nodes_by_cell}
    gt_traces = _load_mbar_gt_traces(
        session, task, contrast, cells, specs, side,
    )
    gt_affine_by_cell = {
        str(cell): train.gt_affine_from_cell(
            params, cell, session.connectome, session=session,
        )
        for cell in cells
    }
    mean_hex_by_label = None
    labels = None
    if at_xs is not None or at_ys is not None:
        mean_hex_by_label = {}
        labels = []
        for label, at_x, at_y in expand_at_xy(at_xs, at_ys)[0]:
            ca_mean_cell_mean_hex = _mbar_ca_mean_cell_mean_hex(
                session, task, contrast, trace, traces, spec_tokens,
                at_x=at_x, at_y=at_y,
                align_at_x=align_at_x, align_at_y=align_at_y,
            )
            if ca_mean_cell_mean_hex is None:
                print(f'skip at_xy {label}: no hex within cost_radius')
                continue
            mean_hex_by_label[label] = ca_mean_cell_mean_hex
            labels.append(label)
        if not mean_hex_by_label:
            mean_hex_by_label = None
            labels = None
    return MbarTraceReadout(
        task=task,
        contrast=contrast,
        cells=cells,
        spec_tokens=spec_tokens,
        side=side,
        single_hex=single_hex,
        v_th_by_cell=v_th_by_cell,
        e_leak_by_cell=e_leak_by_cell,
        gt_traces=gt_traces,
        n_t=n_t,
        traces=traces,
        session=session,
        at_x=at_x,
        at_y=at_y,
        at_xs=at_xs,
        at_ys=at_ys,
        labels=labels,
        ca_mean_cell_mean_hex_by_label=mean_hex_by_label,
        n_hex=n_hex,
        align_at_x=align_at_x,
        align_at_y=align_at_y,
        prep_s=time.perf_counter() - t_prep0,
        t_onset=int(t_onset),
        gt_affine_by_cell=gt_affine_by_cell,
        ms_shown=ms_shown,
    )


def _mbar_t_onset(readout, cell, token):
    """Median relative index of global ``t_onset`` within the plotted trace."""
    t_onset = readout.t_onset
    window_traces = readout.traces
    if t_onset is None or window_traces.t0_bn is None:
        return 0
    try:
        b = readout.spec_tokens.index(token)
        figure_cell_idx = window_traces.cells.index(cell)
    except ValueError:
        return 0
    key = (cell, token)
    if key not in window_traces.ca_mean_cell:
        return 0
    n_t_figure = int(np.asarray(window_traces.ca_mean_cell[key]).shape[0])
    cell_mask = window_traces.cell_idxs == figure_cell_idx
    valid = (window_traces.t0_bn[b] >= 0) & cell_mask
    if not bool(valid.any()):
        return 0
    pre = np.clip(int(t_onset) - window_traces.t0_bn[b, valid], 0, n_t_figure)
    return int(np.median(pre))


def _cost_window_xy(cost_trace, before_t, delta_ms):
    """Map cost_window GT onto trace hex-step x/y."""
    i0 = before_t - t_from_ms(COST_WINDOW_BEFORE_MS, delta_ms=delta_ms)
    trace = np.asarray(cost_trace, dtype=np.float64)
    if i0 < 0:
        trace = trace[-i0:]
        i0 = 0
    x = np.arange(i0, i0 + len(trace), dtype=np.int64)
    return x, trace


def _mbar_hexes_label(session, *, at_x=None, at_y=None, n_hex=None):
    pack = session.primary_pack
    cost_radius = pack.cost_radius
    connectome = session.connectome
    if at_x is not None or at_y is not None:
        ncol_part = f'{n_hex} sti hex'
        if n_hex != 1:
            ncol_part += 's'
        parts = [at_xy_label(at_x, at_y), ncol_part]
        if cost_radius is not None:
            parts.insert(0, f'cost_radius={cost_radius}')
        return ', '.join(parts)
    if cost_radius is not None:
        n_sti_hex = len(cost_sti_hexes(connectome, cost_radius=cost_radius))
        return f'cost_radius={cost_radius} ({n_sti_hex} sti hexes)'
    return f'avg over {len(sti_hexes(connectome))} sti hexes'


def _style_mbar_relative_axis(
    ax, before_t, n_t_figure, *,
    delta_ms,
    show_tick_labels=True, mark_cost_window=False,
    ms_shown=None,
):
    end = n_t_figure - 1
    xlim = ms_shown_axis_xlim(ms_shown, delta_ms=delta_ms, origin_t=before_t)
    if xlim is None:
        t_lo, t_hi = 0, end
    else:
        t_lo, t_hi = max(0, xlim[0]), min(end, xlim[1])
        if t_lo > t_hi:
            t_lo, t_hi = 0, end
    ax.set_xlim(t_lo, t_hi)
    ticks = [t for t in (t_lo, before_t, t_hi) if t_lo <= t <= t_hi]
    if len(ticks) < 2:
        ticks = [t_lo, t_hi]
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
            if t_lo <= x <= t_hi:
                ax.axvline(x, color='0.75', linewidth=0.6, linestyle='--', zorder=0)


def _mbar_spec_linestyle(side, cell, token):
    """Solid for PD stis, dashed for ND (Gruntman fig1 convention)."""
    if cell not in GT_CELLS:
        return '-'
    parts = str(token).split('_')
    if len(parts) < 3:
        return '-'
    direction, contrast = parts[0], parts[1]
    pref = motion_preference(side, cell, direction, contrast)
    if pref is None:
        return '-'
    return '--' if pref.pd_nd == 'ND' else '-'


def _plot_mbar_cell(
    ax,
    ca_trace,
    sem_trace,
    title,
    before_t,
    *,
    gt_trace=None,
    show_ylabel=False,
    show_sem=True,
    cell_ticks=True,
    show_tick_labels=True,
    mark_cost_window=False,
    v_th=None,
    e_leak=None,
    linestyle='-',
    t_onset=0,
    delta_ms=None,
    ms_shown=None,
):
    n_t_figure = len(ca_trace)
    gt_x, gt_y = None, None
    if gt_trace is not None:
        gt_x, gt_y = _cost_window_xy(gt_trace, before_t, delta_ms)

    def style_xaxis(ax):
        _style_mbar_relative_axis(
            ax, before_t, n_t_figure,
            delta_ms=delta_ms,
            show_tick_labels=show_tick_labels,
            mark_cost_window=mark_cost_window,
            ms_shown=ms_shown,
        )

    plot_timecourse(
        ax, np.arange(n_t_figure),
        [{
            "ca_mean_cell": ca_trace,
            "gt": None,
            "sem": sem_trace,
            "linestyle": linestyle,
        }],
        show_sem=show_sem and sem_trace is not None and np.any(sem_trace),
        title=title,
        v_th=v_th,
        e_leak=e_leak,
        show_ylabel=show_ylabel,
        ticksize=6 if cell_ticks else 5,
        style_xaxis=style_xaxis,
        t_onset=t_onset,
    )
    # Gray fig1 GT is already restricted to cost_window (no global pre).
    if gt_x is not None:
        ax.plot(gt_x, gt_y, color=GT_COLOR, linewidth=TRACE_LINE_W, linestyle=linestyle)


def _plot_mbar_cell_at_xy(
    ax,
    v_readout,
    sem_trace,
    ca_mean_cell_mean_hex_by_label,
    labels,
    title,
    before_t,
    *,
    gt_trace=None,
    show_ylabel=False,
    show_sem=True,
    show_legend=False,
    cell_ticks=True,
    show_tick_labels=True,
    mark_cost_window=False,
    v_th=None,
    e_leak=None,
    t_onset=0,
    delta_ms=None,
    ms_shown=None,
):
    n_t_figure = len(v_readout)
    gt_x, gt_y = None, None
    if gt_trace is not None:
        gt_x, gt_y = _cost_window_xy(gt_trace, before_t, delta_ms)

    def style_xaxis(ax):
        _style_mbar_relative_axis(
            ax, before_t, n_t_figure,
            delta_ms=delta_ms,
            show_tick_labels=show_tick_labels,
            mark_cost_window=mark_cost_window,
            ms_shown=ms_shown,
        )

    t = np.arange(n_t_figure)
    if gt_x is not None:
        ax.plot(gt_x, gt_y, color=GT_COLOR, linewidth=TRACE_LINE_W)
    colors = at_xy_reds(len(labels))
    for label, color in zip(labels, colors):
        plot_trace(
            ax, t, ca_mean_cell_mean_hex_by_label[label], t_onset=t_onset,
            color=color, linestyle='-', linewidth=TRACE_LINE_W, label=label,
        )
    if show_sem and sem_trace is not None and np.any(sem_trace):
        plot_sem_band(ax, t, v_readout, sem_trace)
    plot_trace(
        ax, t, v_readout, t_onset=t_onset,
        color=colors[-1], linestyle='-', linewidth=TRACE_LINE_W, label='hexes',
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


def _mbar_readout_hexes_label(readout):
    if readout.at_xs is not None or readout.at_ys is not None:
        pack = readout.session.primary_pack
        cost_radius = pack.cost_radius
        _, at_xy_mode = expand_at_xy(readout.at_xs, readout.at_ys)
        at_x = readout.at_xs if at_xy_mode in ('x', 'xy') else None
        at_y = readout.at_ys if at_xy_mode in ('y', 'xy') else None
        parts = [at_xy_label(at_x, at_y), 'at_xy + hexes']
        if readout.align_at_x is not None and readout.align_at_y is not None:
            parts.append(
                'aligned to '
                + at_xy_label(readout.align_at_x, readout.align_at_y),
            )
        if cost_radius is not None:
            parts.insert(0, f'cost_radius={cost_radius}')
        return ', '.join(parts)
    return _mbar_hexes_label(readout.session)


def _mbar_cost_contrasts(readouts):
    contrasts = []
    for readout in readouts.values():
        if readout.contrast not in contrasts:
            contrasts.append(readout.contrast)
    return contrasts


def _mbar_all_figure(readouts, title, *, right_only=True, cost_parts=None):
    order = contrast_order(readouts)
    readout = readouts[order[0]]
    paired_readout = readouts[order[1]] if len(order) > 1 else None
    single_hex = readout.single_hex
    cells = readout.cells
    window_traces = readout.traces
    spec_tokens = _filter_right_specs(readout.spec_tokens, right_only)
    n_col_readout = len(spec_tokens)
    ca_mean_cell, ca_sem, ca_n = window_traces.ca_mean_cell, window_traces.ca_sem, window_traces.ca_n
    gt_traces = readout.gt_traces
    v_th_by_cell = readout.v_th_by_cell
    e_leak_by_cell = readout.e_leak_by_cell
    paired_v_th_by_cell = None
    paired_e_leak_by_cell = None
    paired_window_traces = None
    labels = list(readout.labels or ())
    has_at_xy = readout.ca_mean_cell_mean_hex_by_label is not None
    cost_contrasts = _mbar_cost_contrasts(readouts)
    if paired_readout is not None:
        paired_window_traces = paired_readout.traces
        spec_tokens = list(spec_tokens) + list(
            _filter_right_specs(paired_readout.spec_tokens, right_only)
        )
        ca_mean_cell = {**ca_mean_cell, **paired_window_traces.ca_mean_cell}
        ca_sem = {**ca_sem, **paired_window_traces.ca_sem}
        ca_n = {**ca_n, **paired_window_traces.ca_n}
        gt_traces = {**gt_traces, **paired_readout.gt_traces}
        paired_v_th_by_cell = paired_readout.v_th_by_cell
        paired_e_leak_by_cell = paired_readout.e_leak_by_cell
    show_sem = not single_hex and not has_at_xy
    n_row = len(cells)
    n_col = len(spec_tokens)
    fig, axes = _mbar_figure(n_row, n_col)
    for row, cell in enumerate(cells):
        for col, token in enumerate(spec_tokens):
            ax = axes[row, col]
            key = (cell, token)
            if key not in ca_mean_cell:
                ax.axis('off')
                continue
            v_th = v_th_by_cell.get(cell)
            e_leak = e_leak_by_cell.get(cell)
            panel_readout = readout if col < n_col_readout else paired_readout
            panel_traces = window_traces if col < n_col_readout else paired_window_traces
            if paired_v_th_by_cell is not None and col >= n_col_readout:
                v_th = paired_v_th_by_cell.get(cell)
                e_leak = paired_e_leak_by_cell.get(cell)
            before_t = panel_traces.before_t[token]
            cell_title = _mbar_cell_title(
                token,
                cell=cell,
                cost_parts=cost_parts,
                cost_contrasts=cost_contrasts,
            )
            if has_at_xy and panel_readout is not None and panel_readout.ca_mean_cell_mean_hex_by_label is not None:
                ca_mean_cell_mean_hex_by_label = {
                    label: panel_readout.ca_mean_cell_mean_hex_by_label[label][key]
                    for label in labels
                    if key in panel_readout.ca_mean_cell_mean_hex_by_label[label]
                }
                if not ca_mean_cell_mean_hex_by_label:
                    ax.axis('off')
                    continue
                plot_labels = [
                    label for label in labels if label in ca_mean_cell_mean_hex_by_label
                ]
                _plot_mbar_cell_at_xy(
                    ax, ca_mean_cell[key], ca_sem.get(key),
                    ca_mean_cell_mean_hex_by_label, plot_labels,
                    cell_title, before_t,
                    gt_trace=gt_trace_affine(panel_readout, cell, gt_traces.get(key)),
                    show_ylabel=(col == 0),
                    show_sem=show_sem and key in ca_sem and np.any(ca_sem[key]),
                    show_legend=(row == 0 and col == 0),
                    cell_ticks=False,
                    show_tick_labels=(row == n_row - 1),
                    mark_cost_window=True,
                    v_th=v_th,
                    e_leak=e_leak,
                    t_onset=_mbar_t_onset(panel_readout, cell, token),
                    delta_ms=panel_readout.session.delta_ms,
                    ms_shown=panel_readout.ms_shown,
                )
            else:
                src = panel_readout or readout
                _plot_mbar_cell(
                    ax, ca_mean_cell[key], ca_sem.get(key),
                    cell_title, before_t,
                    gt_trace=gt_trace_affine(src, cell, gt_traces.get(key)),
                    show_ylabel=(col == 0),
                    show_sem=show_sem and key in ca_sem and np.any(ca_sem[key]),
                    cell_ticks=False,
                    show_tick_labels=(row == n_row - 1),
                    mark_cost_window=True,
                    v_th=v_th,
                    e_leak=e_leak,
                    t_onset=_mbar_t_onset(src, cell, token),
                    delta_ms=src.session.delta_ms,
                    ms_shown=src.ms_shown,
                )
        axes[row, 0].set_ylabel(cell_ylabel(cell, ca_n), fontsize=8, labelpad=12)
    if title is None:
        title = 'Moving-bar ca-all (right only)' if right_only else 'Moving-bar ca-all'
    hexes_label = _mbar_readout_hexes_label(readout)
    fig.suptitle(title + f'  [{hexes_label}, t_first_sti-aligned trace]', fontsize=12)
    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.50, wspace=0.35)
    return fig


@torch.no_grad()
def plot_gt(path, *, readouts, title, gts=None, cost_parts=None, right_only=True):
    """Plot ca-gt figure from contrast → :class:`MbarTraceReadout`."""
    order = contrast_order(readouts)
    readout = readouts[order[0]]
    paired_readout = readouts[order[1]] if len(order) > 1 else None
    timer = ElapsedTimer(prior_prep=readout_prep_s(*readouts.values()))
    timer.end_prep()
    single_hex = readout.single_hex
    row_specs = mbar_specs_by_cell(
        readout.session, readout.task, readout.contrast, readout.side,
    )
    gt_cells = list(row_specs.keys())
    n_col_half = max((len(specs) for specs in row_specs.values()), default=8)
    cost_contrasts = _mbar_cost_contrasts(readouts)
    if paired_readout is not None:
        paired_row_specs = mbar_specs_by_cell(
            paired_readout.session, paired_readout.task,
            paired_readout.contrast, paired_readout.side,
        )
        n_col_half = max(
            n_col_half, max((len(specs) for specs in paired_row_specs.values()), default=8),
        )
        n_col = n_col_half * 2
    else:
        paired_row_specs = None
        n_col = n_col_half
    n_row = len(gt_cells)
    fig, axes = _mbar_figure(n_row, n_col)

    def _plot_row(row, cell, specs, col_offset, row_readout, side):
        window_traces = row_readout.traces
        for col, token in enumerate(specs):
            ax = axes[row, col_offset + col]
            key = (cell, token)
            if key not in window_traces.ca_mean_cell:
                ax.axis('off')
                continue
            before_t = window_traces.before_t[token]
            cell_title = _mbar_cell_title(
                token,
                cell=cell,
                cost_parts=cost_parts,
                cost_contrasts=cost_contrasts,
            )
            _plot_mbar_cell(
                ax, window_traces.ca_mean_cell[key], window_traces.ca_sem[key],
                cell_title, before_t,
                gt_trace=gt_trace_affine(
                    row_readout, cell, row_readout.gt_traces.get(key),
                ),
                show_ylabel=(col_offset + col == 0), show_sem=not single_hex,
                mark_cost_window=True,
                v_th=row_readout.v_th_by_cell.get(cell),
                e_leak=row_readout.e_leak_by_cell.get(cell),
                linestyle=_mbar_spec_linestyle(side, cell, token),
                t_onset=_mbar_t_onset(row_readout, cell, token),
                delta_ms=row_readout.session.delta_ms,
                ms_shown=row_readout.ms_shown,
            )

    for row, cell in enumerate(gt_cells):
        _plot_row(row, cell, row_specs[cell], 0, readout, readout.side)
        if paired_readout is not None:
            _plot_row(
                row, cell, paired_row_specs[cell], n_col_half, paired_readout, paired_readout.side,
            )
        axes[row, 0].set_ylabel(
            cell_ylabel(cell, readout.traces.ca_n), fontsize=8, labelpad=12,
        )
    hexes_label = _mbar_hexes_label(readout.session)
    fig.suptitle(
        title + f'  [{hexes_label}, t_first_sti-aligned trace]',
        fontsize=12,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.50, wspace=0.35)
    timer.end_plot()
    save_figure(fig, path, dpi=MBAR_DPI, rasterize=True, timer=timer)


@torch.no_grad()
def plot_all(path, *, readouts, title, gts=None, cost_parts=None, right_only=True):
    """Plot ca-all figure from contrast → :class:`MbarTraceReadout`."""
    order = contrast_order(readouts)
    readout = readouts[order[0]]
    paired_readout = readouts[order[1]] if len(order) > 1 else None
    timer = ElapsedTimer(prior_prep=readout_prep_s(*readouts.values()))
    timer.end_prep()
    fig = _mbar_all_figure(
        readouts, title, right_only=right_only, cost_parts=cost_parts,
    )
    timer.end_plot()
    save_figure(fig, path, dpi=MBAR_DPI, rasterize=True, timer=timer)


_TASK = "mbar"


def build_readout(session, z, contrast, **readout_kwargs):
    return mbar_trace_readout(
        session_from_task(session, _TASK, contrast), z, _TASK, contrast, **readout_kwargs,
    )


def figure_titles(session, suffix, token, *, contrast=None):
    if contrast is None:
        return (
            f'Moving-bar {token}-gt ({suffix})',
            f'Moving-bar {token}-all ({suffix})',
        )
    return (
        f'mbar {contrast} {token}-gt ({suffix})',
        f'mbar {contrast} {token}-all ({suffix})',
    )
