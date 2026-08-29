"""Spread plotting and shared trace figures for spot / mbar."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

import train
from neuron.borst import t_abs_from_ms, ms_from_t
from figure.panel import (
    N_COL_ALL,
    N_COL_GT,
    PANEL_H,
    PANEL_W,
    ElapsedTimer,
    as_numpy,
    gt_trace_affine,
    e_leak_from_z,
    mark_sti_on,
    plot_timecourse,
    readout_prep_s,
    save_figure,
    traces_with_cost_ts,
    sd_from_traces,
    v_th_from_z,
)
from figure.plot import session_from_task
from network.construction import active_gt_cells, cell_rows, cells_in_order, gt_cells_from_opts
from network import path  # noqa: F401 -- FAFBv783 on sys.path
from task.spread.gt import (
    GT_CELLS,
    RF_SIGN,
    gt_sign,
    load_ir,
    spread_gt_active,
)
from task.spread.sti_spec import CONTRASTS, t_sti_end
from task.spread.pack import part_key as spread_part_key


def cells_from_nodes(session, nodes):
    nodes = np.asarray(as_numpy(nodes), dtype=np.int64)
    connectome = session.connectome
    node_cells = as_numpy(connectome.node_cells[nodes])
    cells = list(connectome.cells)
    return [str(cells[int(cell_idx)]) for cell_idx in node_cells]


def pack_cells(session, task=None, contrast=None):
    """Unique cells on pack.entry_nodes, pack order."""
    if task is None and contrast is None:
        pack = session.primary_pack
    elif task is None or contrast is None:
        raise ValueError("task and contrast must be passed together")
    else:
        pack = session.packs[task][contrast]
    return tuple(dict.fromkeys(cells_from_nodes(session, pack.entry_nodes)))


def active_spread_gt_cells(session, task=None, contrast=None):
    """Configured spread gt cells (sti opts), not cost-pack-only."""
    if task is None and contrast is None:
        pack = session.primary_pack
    elif task is None or contrast is None:
        raise ValueError("task and contrast must be passed together")
    else:
        pack = session.packs[task][contrast]
    opts = dict((session.train_opts or {}).get(f"{pack.task}_sti_opts") or {})
    connectome = session.connectome
    return tuple(
        active_gt_cells(
            gt_cells_from_opts(opts),
            GT_CELLS,
            connectome.cells,
            context="spread plot",
        )
    )


def contrast_from_pack(pack) -> str:
    """``bright`` / ``dark`` from a pack."""
    return str(pack.contrast)


def contrast_order(contrasts) -> tuple[str, ...]:
    """Stable plot order: bright, dark, then any extras."""
    preferred = ("bright", "dark")
    keys = [str(contrast) for contrast in contrasts]
    return tuple(contrast for contrast in preferred if contrast in keys) + tuple(
        contrast for contrast in keys if contrast not in preferred
    )


def contrast_linestyle(contrast: str) -> str:
    return {"bright": "-", "dark": "--"}.get(str(contrast), "-")


def spread_gts(
    session,
    task=None,
    *,
    contrasts=None,
    t_onset=None,
    n_t=None,
    ms_sti=None,
    delta_ms=None,
    filter="none",
):
    """Spread gts ``{contrast: {cell: gt}}`` (ir-only)."""
    task = task or session.primary_pack.task
    if contrasts is None:
        contrasts = tuple(session.contrasts)
    if filter is None:
        filter = str((session.train_opts or {}).get("filter", "none"))
    else:
        filter = str(filter)
    spread_gt_mode = str((session.train_opts or {})["spread_gt_mode"])
    delta_ms = float(session.delta_ms if delta_ms is None else delta_ms)
    gt_amp = float(session.gt_amp)
    cell_idx = {cell: index for index, cell in enumerate(GT_CELLS)}
    gts = {}
    for contrast in contrasts:
        contrast = str(contrast)
        if contrast not in CONTRASTS:
            raise ValueError(
                f"unknown contrast {contrast!r}; expected one of {CONTRASTS}"
            )
        irs = load_ir(
            t_onset=t_onset, n_t=n_t, ms_sti=ms_sti, delta_ms=delta_ms,
            filter=filter,
        )
        signs = np.array(
            [gt_sign(contrast, RF_SIGN[cell]) for cell in GT_CELLS],
            dtype=np.float64,
        )[:, None]
        gt_rows = irs * gt_amp * signs
        gts[contrast] = {
            str(cell): gt_rows[cell_idx[cell]]
            for cell in GT_CELLS
            if spread_gt_active(spread_gt_mode, contrast, int(RF_SIGN[cell]))
        }
    return gts


def _session_task_timing(session):
    """Extract onset ``t_onset`` / forward ``n_t``, ms_sti, and delta_ms from session."""
    pack = session.primary_pack
    opts = (session.train_opts or {}).get(f"{pack.task}_sti_opts") or {}
    t_onset = train.pack_t_onset(pack)
    n_t = int(pack.i_sti.shape[1])
    n_t_gt = int(pack.gts.shape[1])
    ms_sti = opts.get("ms_sti")
    delta_ms = float(opts["delta_ms"])
    return (
        int(t_onset),
        n_t,
        n_t_gt,
        None if ms_sti is None else float(ms_sti),
        delta_ms,
    )


def resolve_spread_gts(sessions, gts=None, *, filter=None):
    """``{contrast: {cell: gt}}`` for each entry in ``sessions``."""
    if gts is not None:
        return gts
    if not sessions:
        return {}
    gts = {}
    for contrast, session in sessions.items():
        t_onset, _n_t, n_t_gt, ms_sti, delta_ms = _session_task_timing(session)
        part = spread_gts(
            session, session.primary_pack.task, contrasts=(str(contrast),),
            t_onset=t_onset, n_t=n_t_gt, ms_sti=ms_sti, delta_ms=delta_ms,
            filter=filter,
        )
        gts.update(part)
    return gts


def _style_time_axis(
    ax, show_xlabel, n_t, *, delta_ms, delta_ms_pre, t_onset, ms_shown=None,
):
    dt = float(delta_ms)
    dt_pre = float(delta_ms_pre)
    t0 = int(t_onset or 0)
    t_last = max(int(n_t) - 1, 0)
    if ms_shown is None:
        t_lo, t_hi = 0, t_last
    else:
        start, stop = ms_shown
        t_lo = t_abs_from_ms(
            float(start), t_onset=t0, delta_ms_pre=dt_pre, delta_ms=dt,
        )
        t_hi = t_abs_from_ms(
            float(stop), t_onset=t0, delta_ms_pre=dt_pre, delta_ms=dt,
        )
        t_lo, t_hi = max(0, t_lo), min(t_last, t_hi)
        if t_lo > t_hi:
            t_lo, t_hi = 0, t_last
    t_lo_s = ms_from_t(t_lo, t_onset=t0, delta_ms_pre=dt_pre, delta_ms=dt) / 1000.0
    t_hi_s = ms_from_t(t_hi, t_onset=t0, delta_ms_pre=dt_pre, delta_ms=dt) / 1000.0
    t_mid_s = (t_lo_s + t_hi_s) / 2.0
    mid = (t_lo + t_hi) // 2
    ax.set_xlim(t_lo, t_hi)
    ax.set_xticks([t_lo, mid, t_hi])
    ax.set_xticklabels([f'{t_lo_s:g}', f'{t_mid_s:g}', f'{t_hi_s:g}'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('time [s]', fontsize=7)


def plot_cell_time(
    ax,
    traces,
    *,
    title=None,
    show_xlabels=False,
    show_ylabel=False,
    v_th=None,
    e_leak=None,
    n_t=None,
    t_onset=None,
    t_sti_end=None,
    delta_ms,
    delta_ms_pre,
    ms_shown=None,
):
    """Time-course panel for one cell across contrast ``traces`` (1D traces)."""
    t = np.arange(n_t)
    trace_t_onset = int(t_onset or 0)
    if title is not None:
        ax.set_title(title, fontsize=8, pad=2)
    mark_sti_on(ax, t_onset, t_sti_end)
    plot_timecourse(
        ax, t, traces,
        show_sd=any(trace.get("sd") is not None for trace in traces),
        v_th=v_th,
        e_leak=e_leak,
        show_ylabel=show_ylabel,
        style_xaxis=lambda ax_time: _style_time_axis(
            ax_time, show_xlabels, n_t,
            delta_ms=delta_ms, delta_ms_pre=delta_ms_pre, t_onset=t_onset,
            ms_shown=ms_shown,
        ),
        t_onset=trace_t_onset,
    )


def _rows_from_cell_rows(cell_rows, figure_cells):
    cell_idx = dict(zip(
        [str(cell) for cell in figure_cells], range(len(figure_cells)),
    ))
    rows = []
    for cells_in_row in cell_rows:
        cell_idxs = [cell_idx[str(cell)] for cell in cells_in_row if str(cell) in cell_idx]
        if cell_idxs:
            rows.append(cell_idxs)
    return rows


def _layout_cells_from_readouts(readouts, order):
    """Union of contrast cells in biological order."""
    seen = set()
    for contrast in order:
        seen.update(readouts[contrast].cells)
    cells = cells_in_order(list(seen))
    rows = [np.array(row) for row in cell_rows(cells)]
    return cells, _rows_from_cell_rows(rows, cells)


@dataclass
class TraceReadout:
    """One forward pass; per-cell mean time traces."""

    cells: list
    rows: list | None = None
    session: object = None
    n_t: int = 0
    prep_s: float = 0.0
    v_readout_mean_cell: dict = field(default_factory=dict)
    sd: dict = field(default_factory=dict)
    n_by_cell: dict = field(default_factory=dict)
    v_th_by_cell: dict = field(default_factory=dict)
    e_leak_by_cell: dict = field(default_factory=dict)
    gt_affine_by_cell: dict = field(default_factory=dict)
    t_onset: int | None = None
    t_sti_end: int | None = None
    ms_shown: tuple[float, float] | None = None


def _spread_gt_readout(readout):
    """Gt figure rows: configured active gt cells."""
    session = readout.session
    active = active_spread_gt_cells(
        session,
        session.primary_pack.task,
        session.primary_pack.contrast,
    )
    rows = [np.array(row) for row in cell_rows(active)]
    readout_cells = set(readout.cells)
    cells = [cell for cell in cells_in_order(active) if cell in readout_cells]
    return TraceReadout(
        cells=cells,
        rows=_rows_from_cell_rows(rows, cells),
        session=session,
        n_t=readout.n_t,
        v_readout_mean_cell={
            cell: readout.v_readout_mean_cell[cell] for cell in cells
            if cell in readout.v_readout_mean_cell
        },
        sd={
            cell: readout.sd[cell] for cell in cells
            if cell in readout.sd
        },
        n_by_cell={
            cell: readout.n_by_cell.get(cell) for cell in cells
        },
        v_th_by_cell={
            cell: readout.v_th_by_cell[cell] for cell in cells
            if cell in readout.v_th_by_cell
        },
        e_leak_by_cell={
            cell: readout.e_leak_by_cell[cell] for cell in cells
            if cell in readout.e_leak_by_cell
        },
        gt_affine_by_cell={
            cell: readout.gt_affine_by_cell[cell] for cell in cells
            if cell in readout.gt_affine_by_cell
        },
        t_onset=readout.t_onset,
        t_sti_end=readout.t_sti_end,
        ms_shown=readout.ms_shown,
    )


@torch.no_grad()
def _forward_spread_readout(session, z):
    """One forward; mean trace over all nodes per cell."""
    pack = session.primary_pack
    params = train.params_from_z(z, session)
    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    trace = as_numpy(train.forward_pack(session, params, i_sti, pack)[0])
    connectome = session.connectome
    cells = list(connectome.cells)
    n_t = int(i_sti.shape[1])
    node_cells = as_numpy(connectome.node_cells)

    v_readout_mean_cell = {}
    sd = {}
    n_by_cell = {}
    for cell_idx, cell in enumerate(cells):
        mask = node_cells == cell_idx
        if not mask.any():
            continue
        node_traces = trace[:, mask].T
        v_readout_mean_cell[cell] = node_traces.mean(axis=0)
        sd[cell] = sd_from_traces(node_traces, single_hex=False)
        n_by_cell[cell] = int(mask.sum())

    opts = dict((session.train_opts or {}).get(f"{pack.task}_sti_opts") or {})
    t_onset = int(train.pack_t_onset(pack))
    ms_sti = opts.get("ms_sti")
    delta_ms = float(opts["delta_ms"])
    v_th = v_th_from_z(z, session)
    e_leak = e_leak_from_z(z, session)
    gt_affine_by_cell = {
        cell: train.gt_affine_from_cell(params, cell, connectome, session=session)
        for cell in v_readout_mean_cell
    }
    active = cells_in_order(connectome.cells)
    rows = _rows_from_cell_rows(
        [np.array(row) for row in cell_rows(active)],
        cells_in_order(active),
    )
    return TraceReadout(
        cells=cells_in_order(list(v_readout_mean_cell)),
        rows=rows,
        session=session,
        n_t=n_t,
        v_readout_mean_cell=v_readout_mean_cell,
        sd=sd,
        n_by_cell=n_by_cell,
        v_th_by_cell={cell: v_th.get(cell, np.nan) for cell in v_readout_mean_cell},
        e_leak_by_cell={cell: e_leak.get(cell, np.nan) for cell in v_readout_mean_cell},
        gt_affine_by_cell=gt_affine_by_cell,
        t_onset=t_onset,
        t_sti_end=t_sti_end(t_onset, n_t, ms_sti, delta_ms=delta_ms),
    )


@torch.no_grad()
def network_spread_trace_readout(session, z, *, ms_shown=None):
    """Run one forward; spread mean traces over all network cells."""
    t_prep0 = time.perf_counter()
    readout = _forward_spread_readout(session, z)
    readout.prep_s = time.perf_counter() - t_prep0
    readout.ms_shown = ms_shown
    return readout


_TASK = "spread"
_GRID_KWARGS = dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98)


def _plot_figure(
    path, *,
    timer,
    readouts,
    title,
    gts=None,
    n_col,
    figure_size_from_grid,
    gridspec_kwargs,
    suptitle_fs=12,
    cost_parts=None,
):
    """Plot spread figure from ``readouts`` (contrast → TraceReadout)."""
    order = contrast_order(readouts)
    if not order:
        raise ValueError("_plot_figure requires at least one readout")
    primary = readouts[order[0]]
    cells, rows = _layout_cells_from_readouts(readouts, order)
    n_t = primary.n_t
    t_onset = primary.t_onset
    t_sti_end = primary.t_sti_end
    delta_ms = float(primary.session.delta_ms)
    delta_ms_pre = float(primary.session.delta_ms_pre)
    ms_shown = primary.ms_shown

    timer.end_prep()
    sessions = {contrast: readouts[contrast].session for contrast in order}
    gt_by_contrast = resolve_spread_gts(sessions, gts)
    n_row = len(rows)
    fig = plt.figure(figsize=figure_size_from_grid(n_col, n_row))
    grid_spec = fig.add_gridspec(n_row, n_col, **gridspec_kwargs)

    def _build_cell_traces(cell):
        traces = []
        for contrast in order:
            contrast_readout = readouts[contrast]
            if cell not in contrast_readout.v_readout_mean_cell:
                continue
            gt_by_cell = gt_by_contrast.get(contrast) or {}
            traces.append({
                "contrast": contrast,
                "v_readout_mean_cell": contrast_readout.v_readout_mean_cell[cell],
                "gt": gt_trace_affine(contrast_readout, cell, gt_by_cell.get(cell)),
                "sd": contrast_readout.sd.get(cell),
                "linestyle": contrast_linestyle(contrast),
            })
        return traces

    for row, cell_idxs in enumerate(rows):
        start = (n_col - len(cell_idxs)) // 2
        for col, cell_idx in enumerate(cell_idxs, start=start):
            cell = cells[cell_idx]
            ax = fig.add_subplot(grid_spec[row, col])
            traces = traces_with_cost_ts(_build_cell_traces(cell), readouts)
            if not traces:
                ax.axis("off")
                continue
            time_title = cell
            n_cell = primary.n_by_cell.get(cell)
            if n_cell is not None:
                time_title = f'{time_title} (n={int(n_cell)})'
            if cost_parts and order:
                for contrast in order:
                    key = spread_part_key(contrast, cell)
                    if key in cost_parts:
                        time_title = (
                            f'{time_title}\n{contrast}: {float(cost_parts[key]):.1f}'
                        )
            plot_cell_time(
                ax, traces,
                title=time_title,
                show_xlabels=(row == n_row - 1),
                show_ylabel=(col == start),
                v_th=primary.v_th_by_cell.get(cell),
                e_leak=primary.e_leak_by_cell.get(cell),
                n_t=n_t,
                t_onset=t_onset,
                t_sti_end=t_sti_end,
                delta_ms=delta_ms,
                delta_ms_pre=delta_ms_pre,
                ms_shown=ms_shown,
            )

    fig.suptitle(title, fontsize=suptitle_fs)
    timer.end_plot()
    save_figure(fig, path, dpi=150, timer=timer)


def plot_trace_figure(
    path, *, readouts, title, plot_figure, gts=None, cost_parts=None, **plot_figure_kwargs,
):
    """Trace figure dispatch with prep timer; task ``plot_figure`` draws and saves."""
    plot_figure(
        path,
        timer=ElapsedTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts=readouts,
        title=title,
        gts=gts,
        cost_parts=cost_parts,
        **plot_figure_kwargs,
    )


def plot_trace_gt(
    path, *, readouts, title, gt_readout, plot_figure, gts=None, cost_parts=None,
    n_col=N_COL_GT, **plot_figure_kwargs,
):
    """Gt figure dispatch: filter readouts, then task ``plot_figure``."""
    plot_figure(
        path,
        timer=ElapsedTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts={
            contrast: gt_readout(readout)
            for contrast, readout in readouts.items()
        },
        title=title,
        gts=gts,
        n_col=n_col,
        figure_size_from_grid=lambda n_col, n_row: (PANEL_W * n_col, PANEL_H * n_row),
        gridspec_kwargs=_GRID_KWARGS,
        cost_parts=cost_parts,
        **plot_figure_kwargs,
    )


def plot_trace_all(
    path, *, readouts, title, plot_figure, gts=None, cost_parts=None, n_col=N_COL_ALL,
    **plot_figure_kwargs,
):
    """All-cell figure dispatch: contrast → readout, then task ``plot_figure``."""
    plot_figure(
        path,
        timer=ElapsedTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts=readouts,
        title=title,
        gts=gts,
        n_col=n_col,
        figure_size_from_grid=lambda n_col, n_row: (PANEL_W * n_col, PANEL_H * n_row),
        gridspec_kwargs=_GRID_KWARGS,
        cost_parts=cost_parts,
        **plot_figure_kwargs,
    )


def plot_gt(path, *, readouts, title, gts=None, cost_parts=None):
    """Plot gt figure (active gt cells)."""
    plot_trace_gt(
        path, readouts=readouts, title=title, gts=gts, cost_parts=cost_parts,
        gt_readout=_spread_gt_readout, plot_figure=_plot_figure,
    )


def plot_all(path, *, readouts, title, gts=None, cost_parts=None):
    """Plot all-cell figure from contrast → readout."""
    plot_trace_all(
        path, readouts=readouts, title=title, gts=gts, cost_parts=cost_parts,
        plot_figure=_plot_figure,
    )


def build_readout(session, z, contrast, **readout_kwargs):
    return network_spread_trace_readout(
        session_from_task(session, _TASK, contrast), z, **readout_kwargs,
    )


def figure_titles(session, suffix, token, *, contrast=None):
    if contrast is None:
        return (
            f'Spread {token}-gt ({suffix})',
            f'Spread {token}-all ({suffix})',
        )
    return (
        f'spread {contrast} {token}-gt ({suffix})',
        f'spread {contrast} {token}-all ({suffix})',
    )
