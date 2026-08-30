"""Static-bar plotting (network static-bar task).

Each panel: one cell at one bar-position on the motion axis.
Columns are bar ``mid`` positions from the cost pack; rows are cells
(``plot_gt``: sbar ``gt_cells``; ``plot_all``: config ``sbar_figure_cells``).

GT traces are Gruntman Fig.2 width-1 flash responses (T4/T5) plus Mi1/Mi4
position contributions from ``task.sbar.gt``, using the same timing as the pack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

import train
from config import FIGURE_PLOT
from task.sbar.gt import GT_CELLS, gt_trace_key, load_gt_stats
from task.sbar.pack import SbarPack, part_key
from task.sbar.sti_geo import node_us_vs, sti_hexes_at_xy
from task.spread.pack import cost_sti_hexes
from figure.spread import (
    _session_task_timing,
    _style_time_axis,
    contrast_linestyle,
    contrast_order,
    plot_cell_time,
)
from network.construction import (
    active_gt_cells,
    cells_in_order,
    gt_cells_from_opts,
)
from figure.plot import session_from_task
from figure.panel import (
    ElapsedTimer,
    GT_COLOR,
    TRACE_LINE_W,
    annotate_v_th,
    as_numpy,
    at_xy_reds,
    cell_ylabel,
    e_leak_from_z,
    expand_at_xy,
    gt_trace_affine,
    gt_sd_affine,
    mark_sti_on,
    plot_trace,
    readout_prep_s,
    save_figure,
    sd_from_traces,
    traces_with_cost_ts,
    v_th_from_z,
)
from task.spread.sti_spec import t_sti_end


SBAR_DPI = 100
PLOT_AT_XY = True


def sbar_figure_cells() -> tuple[str, ...]:
    """``sbar_figure_cells`` from config (rows for ``plot_all``)."""
    cells = FIGURE_PLOT.get("sbar_figure_cells")
    if not cells:
        raise ValueError("config sbar_figure_cells must be a non-empty list")
    return tuple(str(cell) for cell in cells)


def active_sbar_gt_cells(session, task=None, contrast=None):
    """Configured sbar gt cells (sti opts), not cost-pack-only."""
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
            context="sbar plot",
        )
    )


@dataclass
class SbarTraceReadout:
    """One sbar forward pass: cell × mid position traces."""

    task: str
    contrast: str
    cells: list
    mids: list
    n_t: int
    t_onset: int | None = None
    t_sti_end: int | None = None
    ms_shown: tuple[float, float] | None = None
    v_readout: dict = field(default_factory=dict)
    sd: dict = field(default_factory=dict)
    n_by_cell_mid: dict = field(default_factory=dict)
    v_th_by_cell: dict = field(default_factory=dict)
    e_leak_by_cell: dict = field(default_factory=dict)
    gt_affine_by_cell: dict = field(default_factory=dict)
    a_sti_mid: dict[str, float] = field(default_factory=dict)
    a_sti_mid_sigma: float | None = None
    session: object = None
    prep_s: float = 0.0
    at_xs: list | None = None
    at_ys: list | None = None
    labels: list[str] | None = None
    v_readout_mean_hex_by_label: dict | None = None


def _sbar_mids_from_pack(pack: SbarPack) -> list:
    """Sorted unique ``mid`` values from ``pack.entry_part_keys``."""
    mids = set()
    for key in pack.entry_part_keys:
        mid_str = key.rsplit("_mid", 1)[1]
        mids.add(float(mid_str))
    return sorted(mids)


def resolve_sbar_gts(sessions, gts=None, gt_stds=None):
    """``(gts, gt_stds)``: contrast → ``{(cell, mid): (n_t,) trace}``.

    ``gts`` are mean traces on the simulation axis; ``gt_stds`` are T4/T5
    measured std only (Mi1/Mi4 source data have no cell-variation std).
    """
    if gts is not None:
        return gts, gt_stds or {}
    if not sessions:
        return {}, {}
    gts_out = {}
    gt_stds_out = {}
    for contrast, session in sessions.items():
        pack = session.primary_pack
        t_onset, n_t, n_t_gt, ms_sti, delta_ms = _session_task_timing(session)
        opts = dict((session.train_opts or {}).get(f"{pack.task}_sti_opts") or {})
        raw_gts, raw_gt_stds = load_gt_stats(
            t_onset=t_onset,
            ms_response=float(opts["ms_response"]),
            ms_sti=ms_sti,
            delta_ms=delta_ms,
            ms_post=float(opts.get("ms_post", 0.0)),
        )
        by_cell_mid = {}
        std_by_cell_mid = {}
        for cell in cells_in_order(GT_CELLS):
            if cell not in session.connectome.cells:
                continue
            for mid in _sbar_mids_from_pack(pack):
                key = gt_trace_key(cell, contrast, mid)
                if key not in raw_gts:
                    continue
                gt = np.full(n_t, np.nan, dtype=np.float64)
                gt[t_onset:t_onset + n_t_gt] = np.asarray(
                    raw_gts[key][t_onset:t_onset + n_t_gt], dtype=np.float64,
                )
                by_cell_mid[(cell, mid)] = gt
                if key in raw_gt_stds:
                    gt_std = np.full(n_t, np.nan, dtype=np.float64)
                    gt_std[t_onset:t_onset + n_t_gt] = np.asarray(
                        raw_gt_stds[key][t_onset:t_onset + n_t_gt],
                        dtype=np.float64,
                    )
                    std_by_cell_mid[(cell, mid)] = gt_std
        gts_out[str(contrast)] = by_cell_mid
        gt_stds_out[str(contrast)] = std_by_cell_mid
    return gts_out, gt_stds_out


def _sbar_entry_hex_mask(
    connectome, entry_nodes, cost_radius, *, cost_mid=None, at_x=None, at_y=None,
):
    """True for pack entries whose node sits on ``at_x``/``at_y`` cost hexes."""
    hexes = sti_hexes_at_xy(
        cost_sti_hexes(connectome, cost_radius=cost_radius, cost_mid=cost_mid),
        at_x=at_x,
        at_y=at_y,
    )
    if not hexes:
        return np.zeros(len(entry_nodes), dtype=bool)
    node_us, node_vs = node_us_vs(connectome)
    hex_uv = {(int(sti_hex.u), int(sti_hex.v)) for sti_hex in hexes}
    return np.array(
        [
            (int(node_us[node]), int(node_vs[node])) in hex_uv
            for node in np.asarray(entry_nodes, dtype=np.int64)
        ],
        dtype=bool,
    )


def _sbar_hex_uv_at_xy(connectome, cost_radius, *, cost_mid=None, at_x=None, at_y=None):
    if at_x is None and at_y is None:
        return None
    hexes = sti_hexes_at_xy(
        cost_sti_hexes(connectome, cost_radius=cost_radius, cost_mid=cost_mid),
        at_x=at_x,
        at_y=at_y,
    )
    return {(int(sti_hex.u), int(sti_hex.v)) for sti_hex in hexes}


def _sbar_build_v_readout(
    connectome, pack, trace, mids, *,
    entry_mask=None,
    hex_uv=None,
    include_figure_cells=True,
):
    """cell × mid mean traces from one forward ``trace``."""
    entry_bs = as_numpy(pack.entry_bs)
    entry_nodes = as_numpy(pack.entry_nodes)
    entry_part_keys = pack.entry_part_keys
    node_cells_arr = as_numpy(connectome.node_cells[entry_nodes])
    cells = list(connectome.cells)
    all_cells = cells_in_order(connectome.cells)
    if entry_mask is not None:
        entry_mask = np.asarray(entry_mask, dtype=bool)

    v_readout: dict = {}
    sd_out: dict = {}
    n_by_cell_mid: dict = {}

    for cell in all_cells:
        cell_entry_mask = node_cells_arr == cells.index(cell)
        if entry_mask is not None:
            cell_entry_mask = cell_entry_mask & entry_mask
        if not np.any(cell_entry_mask):
            continue
        v_readout[cell] = {}
        sd_out[cell] = {}
        n_by_cell_mid[cell] = {}
        for mid in mids:
            mid_entry_mask = cell_entry_mask.copy()
            for entry_idx, key in enumerate(entry_part_keys):
                mid_str = key.rsplit("_mid", 1)[1]
                if float(mid_str) != mid:
                    mid_entry_mask[entry_idx] = False
            if not np.any(mid_entry_mask):
                continue
            entry_traces = trace[
                entry_bs[mid_entry_mask], :, entry_nodes[mid_entry_mask]
            ]
            v_readout[cell][mid] = entry_traces.mean(axis=0)
            sd_out[cell][mid] = sd_from_traces(
                entry_traces, single_hex=(entry_traces.shape[0] == 1),
            )
            n_by_cell_mid[cell][mid] = int(entry_traces.shape[0])

    if include_figure_cells:
        us, vs = node_us_vs(connectome)
        for cell in sbar_figure_cells():
            if cell in v_readout or cell not in connectome.cells:
                continue
            cell_v: dict = {}
            cell_sd: dict = {}
            cell_n: dict = {}
            for mid in mids:
                mid_traces = []
                seen_b_uv: set[tuple[int, int, int]] = set()
                for entry_idx, key in enumerate(entry_part_keys):
                    if entry_mask is not None and not entry_mask[entry_idx]:
                        continue
                    mid_str = key.rsplit("_mid", 1)[1]
                    if float(mid_str) != mid:
                        continue
                    entry_node = int(entry_nodes[entry_idx])
                    b = int(entry_bs[entry_idx])
                    u, v = int(us[entry_node]), int(vs[entry_node])
                    if hex_uv is not None and (u, v) not in hex_uv:
                        continue
                    b_uv = (b, u, v)
                    if b_uv in seen_b_uv:
                        continue
                    seen_b_uv.add(b_uv)
                    for cell_node in connectome.nodes_at_uv(u, v, cell):
                        mid_traces.append(trace[b, :, int(cell_node)])
                if not mid_traces:
                    continue
                entry_traces = np.stack(mid_traces, axis=0)
                cell_v[mid] = entry_traces.mean(axis=0)
                cell_sd[mid] = sd_from_traces(
                    entry_traces, single_hex=(entry_traces.shape[0] == 1),
                )
                cell_n[mid] = int(entry_traces.shape[0])
            if not cell_v:
                continue
            v_readout[cell] = cell_v
            sd_out[cell] = cell_sd
            n_by_cell_mid[cell] = cell_n

    return v_readout, sd_out, n_by_cell_mid


@torch.no_grad()
def network_sbar_trace_readout(
    session, z, task, contrast, *, at_xs=None, at_ys=None, ms_shown=None,
):
    """Run one forward; return :class:`SbarTraceReadout`."""
    t_prep0 = time.perf_counter()
    pack: SbarPack = session.packs[task][contrast]
    params = train.params_from_z(z, session)
    a_sti_mid = {}
    a_sti_mid_sigma = None
    if "a_sti_mid" in params:
        spec = session.schema.get("a_sti_mid")
        if spec is not None:
            a_sti_mid_sigma = float(as_numpy(params["a_sti_mid"]).reshape(-1)[0])
            a_sti_mid = {
                str(mid): float(np.exp(-0.5 * (float(mid) / a_sti_mid_sigma) ** 2))
                for mid in (spec.get("mids") or ())
            }
    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    trace = train.forward_pack(session, params, i_sti, pack)
    trace = as_numpy(trace)

    connectome = session.connectome
    n_t = int(i_sti.shape[1])
    entry_nodes = as_numpy(pack.entry_nodes)
    mids = _sbar_mids_from_pack(pack)

    pairs = []
    if at_xs is not None or at_ys is not None:
        pairs, _ = expand_at_xy(at_xs, at_ys)
    filter_at_x = filter_at_y = None
    if len(pairs) == 1:
        _, filter_at_x, filter_at_y = pairs[0]
    entry_mask = None
    hex_uv = None
    if filter_at_x is not None or filter_at_y is not None:
        entry_mask = _sbar_entry_hex_mask(
            connectome, entry_nodes, pack.cost_radius,
            cost_mid=getattr(pack, "cost_mid", None),
            at_x=filter_at_x, at_y=filter_at_y,
        )
        hex_uv = _sbar_hex_uv_at_xy(
            connectome, pack.cost_radius,
            cost_mid=getattr(pack, "cost_mid", None),
            at_x=filter_at_x, at_y=filter_at_y,
        )

    v_readout, sd_out, n_by_cell_mid = _sbar_build_v_readout(
        connectome, pack, trace, mids,
        entry_mask=entry_mask,
        hex_uv=hex_uv,
    )

    v_readout_mean_hex_by_label = None
    labels = None
    if len(pairs) > 1:
        v_readout_mean_hex_by_label = {}
        labels = []
        for label, slice_at_x, slice_at_y in pairs:
            slice_mask = _sbar_entry_hex_mask(
                connectome, entry_nodes, pack.cost_radius,
                cost_mid=getattr(pack, "cost_mid", None),
                at_x=slice_at_x, at_y=slice_at_y,
            )
            if not np.any(slice_mask):
                print(f'skip at_xy {label}: no hex within cost_radius')
                continue
            slice_hex_uv = _sbar_hex_uv_at_xy(
                connectome, pack.cost_radius,
                cost_mid=getattr(pack, "cost_mid", None),
                at_x=slice_at_x, at_y=slice_at_y,
            )
            v_slice, _, _ = _sbar_build_v_readout(
                connectome, pack, trace, mids,
                entry_mask=slice_mask,
                hex_uv=slice_hex_uv,
                include_figure_cells=False,
            )
            if not any(
                np.isfinite(v_readout_trace).any()
                for cell_v in v_slice.values()
                for v_readout_trace in cell_v.values()
            ):
                print(f'skip at_xy {label}: no readouts')
                continue
            v_readout_mean_hex_by_label[label] = v_slice
            labels.append(label)
        if not v_readout_mean_hex_by_label:
            v_readout_mean_hex_by_label = None
            labels = None

    readout_cells = cells_in_order(list(v_readout))
    v_th = v_th_from_z(z, session)
    e_leak = e_leak_from_z(z, session)
    v_th_by_cell = {cell: v_th.get(cell, np.nan) for cell in readout_cells}
    e_leak_by_cell = {cell: e_leak.get(cell, np.nan) for cell in readout_cells}
    gt_affine_by_cell = {
        str(cell): train.gt_affine_from_cell(
            params, cell, session.connectome, session=session,
        )
        for cell in readout_cells
    }

    opts = dict((session.train_opts or {}).get(f"{task}_sti_opts") or {})
    t_onset = int(train.pack_t_onset(pack))
    ms_sti = opts.get("ms_sti")
    delta_ms = float(opts["delta_ms"])

    return SbarTraceReadout(
        task=task,
        contrast=contrast,
        cells=readout_cells,
        mids=mids,
        n_t=n_t,
        t_onset=t_onset,
        t_sti_end=t_sti_end(t_onset, n_t, ms_sti, delta_ms=delta_ms),
        ms_shown=ms_shown,
        v_readout=v_readout,
        sd=sd_out,
        n_by_cell_mid=n_by_cell_mid,
        v_th_by_cell=v_th_by_cell,
        e_leak_by_cell=e_leak_by_cell,
        gt_affine_by_cell=gt_affine_by_cell,
        a_sti_mid=a_sti_mid,
        a_sti_mid_sigma=a_sti_mid_sigma,
        session=session,
        prep_s=time.perf_counter() - t_prep0,
        at_xs=at_xs,
        at_ys=at_ys,
        labels=labels,
        v_readout_mean_hex_by_label=v_readout_mean_hex_by_label,
    )


def _sbar_filter_readout(readout, cells):
    """Keep only ``cells`` that have traces in ``readout``."""
    keep = [cell for cell in cells_in_order(cells) if cell in readout.v_readout]
    return SbarTraceReadout(
        task=readout.task,
        contrast=readout.contrast,
        cells=keep,
        mids=readout.mids,
        n_t=readout.n_t,
        t_onset=readout.t_onset,
        t_sti_end=readout.t_sti_end,
        ms_shown=readout.ms_shown,
        v_readout={cell: readout.v_readout[cell] for cell in keep},
        sd={cell: readout.sd[cell] for cell in keep if cell in readout.sd},
        n_by_cell_mid={
            cell: readout.n_by_cell_mid[cell]
            for cell in keep if cell in readout.n_by_cell_mid
        },
        v_th_by_cell={
            cell: readout.v_th_by_cell[cell]
            for cell in keep if cell in readout.v_th_by_cell
        },
        e_leak_by_cell={
            cell: readout.e_leak_by_cell[cell]
            for cell in keep if cell in readout.e_leak_by_cell
        },
        gt_affine_by_cell={
            cell: readout.gt_affine_by_cell[cell]
            for cell in keep if cell in readout.gt_affine_by_cell
        },
        a_sti_mid=dict(readout.a_sti_mid),
        a_sti_mid_sigma=readout.a_sti_mid_sigma,
        session=readout.session,
        prep_s=readout.prep_s,
        at_xs=readout.at_xs,
        at_ys=readout.at_ys,
        labels=readout.labels,
        v_readout_mean_hex_by_label=readout.v_readout_mean_hex_by_label,
    )


def _sbar_gt_readout(readout):
    """Gt figure rows: configured active gt cells (not cost-pack-only)."""
    session = readout.session
    return _sbar_filter_readout(
        readout,
        active_sbar_gt_cells(
            session,
            session.primary_pack.task,
            session.primary_pack.contrast,
        ),
    )


def _sbar_all_readout(readout):
    """All figure rows: config ``sbar_figure_cells``."""
    return _sbar_filter_readout(readout, sbar_figure_cells())


def _sbar_mid_label(mid: float) -> str:
    """Format mid as column label."""
    return f"{int(mid):+d}" if float(mid).is_integer() else f"{float(mid):+.1f}"


def _panel_a_sti_mid(readout: SbarTraceReadout, mid: float) -> str:
    """Effective symmetric stimulus amplitude shown in one sbar panel."""
    if np.isclose(float(mid), 0.0):
        return "1"
    key = str(float(abs(mid))).removesuffix(".0")
    value = readout.a_sti_mid.get(key)
    return "n/a" if value is None else f"{float(value):.4g}"


def _sbar_panel_cost_lines(cell, mid, cost_parts, order):
    if not cost_parts or not order:
        return []
    lines = []
    for contrast in order:
        key = part_key(contrast, cell, mid)
        if key in cost_parts:
            lines.append(f'{contrast}: {float(cost_parts[key]):.1f}')
    return lines


def _sbar_figure(n_row, n_col):
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(2.0 * n_col, 1.8 * n_row),
        sharex=True,
        sharey="row",
    )
    if n_row == 1:
        axes = np.asarray([axes])
    if n_col == 1:
        axes = axes[:, None]
    return fig, axes


def _plot_sbar_panel_at_xy(
    ax,
    readouts,
    order,
    *,
    cell,
    mid,
    labels,
    title,
    gt_by_contrast,
    show_xlabels,
    show_ylabel,
    v_th,
    e_leak,
    n_t,
    t_onset,
    t_sti_end,
    delta_ms,
    delta_ms_pre,
    ms_shown,
):
    t = np.arange(n_t)
    trace_t_onset = int(t_onset or 0)
    if title is not None:
        ax.set_title(title, fontsize=8, pad=2)
    mark_sti_on(ax, t_onset, t_sti_end)
    primary = readouts[order[0]]
    for contrast in order:
        contrast_readout = readouts[contrast]
        gt_by_cell_mid = gt_by_contrast.get(contrast) or {}
        gt_trace = gt_trace_affine(
            contrast_readout, cell, gt_by_cell_mid.get((cell, mid)),
        )
        if gt_trace is not None:
            plot_trace(
                ax, t, gt_trace, t_onset=trace_t_onset,
                color=GT_COLOR, linestyle=contrast_linestyle(contrast),
                linewidth=TRACE_LINE_W,
            )
    colors = at_xy_reds(len(labels))
    for label, color in zip(labels, colors):
        v_readout = (
            primary.v_readout_mean_hex_by_label.get(label, {})
            .get(cell, {})
            .get(mid)
        )
        if v_readout is None:
            continue
        plot_trace(
            ax, t, v_readout, t_onset=trace_t_onset,
            color=color, linestyle='-', linewidth=TRACE_LINE_W, label=label,
        )
    v_readout = primary.v_readout.get(cell, {}).get(mid)
    if v_readout is not None:
        plot_trace(
            ax, t, v_readout, t_onset=trace_t_onset,
            color=colors[-1], linestyle='-', linewidth=TRACE_LINE_W, label='hexes',
        )
    _style_time_axis(
        ax, show_xlabels, n_t,
        delta_ms=delta_ms, delta_ms_pre=delta_ms_pre, t_onset=t_onset,
        ms_shown=ms_shown,
    )
    if show_ylabel:
        ax.set_ylabel('mV', fontsize=8)
    ax.tick_params(labelsize=6)
    annotate_v_th(ax, v_th, e_leak=e_leak)


def _plot_figure(path, *, timer, readouts, title, gts=None, cost_parts=None):
    """Plot sbar figure from ``readouts`` (contrast → SbarTraceReadout)."""
    order = contrast_order(readouts)
    if not order:
        raise ValueError("_plot_figure requires at least one readout")
    primary = readouts[order[0]]
    cells = primary.cells
    n_row = len(cells)
    n_t = primary.n_t
    t_onset = primary.t_onset
    t_sti_end = primary.t_sti_end
    delta_ms = float(primary.session.delta_ms)
    delta_ms_pre = float(primary.session.delta_ms_pre)
    ms_shown = primary.ms_shown
    has_at_xy = primary.v_readout_mean_hex_by_label is not None
    labels = list(primary.labels or ())

    timer.end_prep()
    sessions = {contrast: readouts[contrast].session for contrast in order}
    gt_by_contrast, gt_std_by_contrast = resolve_sbar_gts(sessions, gts)
    mids_by_cell = {
        cell: sorted({
            float(mid)
            for contrast in order
            for mid in readouts[contrast].v_readout.get(cell, {})
        })
        for cell in cells
    }
    n_col = max((len(cell_mids) for cell_mids in mids_by_cell.values()), default=0)
    if n_col == 0:
        raise ValueError("_plot_figure requires at least one cell/mid trace")
    fig, axes = _sbar_figure(n_row, n_col)

    for row, cell in enumerate(cells):
        cell_mids = mids_by_cell[cell]
        start = (n_col - len(cell_mids)) // 2
        for col in range(n_col):
            if not (start <= col < start + len(cell_mids)):
                axes[row, col].axis("off")
        for col, mid in enumerate(cell_mids, start=start):
            ax = axes[row, col]
            n_entry = primary.n_by_cell_mid.get(cell, {}).get(mid)
            mid_label = _sbar_mid_label(mid)
            n_label = str(int(n_entry)) if n_entry is not None else "n/a"
            panel_title = (
                f"{mid_label}\n"
                f"n={n_label}\n"
                f"a_sti_mid={_panel_a_sti_mid(primary, mid)}"
            )
            for cost_line in _sbar_panel_cost_lines(cell, mid, cost_parts, order):
                panel_title += f'\n{cost_line}'
            traces = []
            for contrast in order:
                contrast_readout = readouts[contrast]
                v_readout = contrast_readout.v_readout.get(cell, {}).get(mid)
                if v_readout is None:
                    continue
                gt_by_cell_mid = gt_by_contrast.get(contrast) or {}
                gt_std_by_cell_mid = gt_std_by_contrast.get(contrast) or {}
                traces.append({
                    "contrast": contrast,
                    "v_readout_mean_cell": v_readout,
                    "gt": gt_trace_affine(
                        contrast_readout, cell, gt_by_cell_mid.get((cell, mid)),
                    ),
                    "gt_sd": gt_sd_affine(
                        contrast_readout, cell,
                        gt_std_by_cell_mid.get((cell, mid)),
                    ),
                    "sd": contrast_readout.sd.get(cell, {}).get(mid),
                    "linestyle": contrast_linestyle(contrast),
                })
            traces = traces_with_cost_ts(traces, readouts)
            if not traces:
                ax.axis("off")
                continue
            if has_at_xy:
                plot_labels = [
                    label for label in labels
                    if mid in primary.v_readout_mean_hex_by_label.get(label, {}).get(cell, {})
                ]
                if not plot_labels:
                    ax.axis("off")
                    continue
                _plot_sbar_panel_at_xy(
                    ax, readouts, order,
                    cell=cell,
                    mid=mid,
                    labels=plot_labels,
                    title=panel_title,
                    gt_by_contrast=gt_by_contrast,
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
            else:
                plot_cell_time(
                    ax, traces,
                    title=panel_title,
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
            ax.tick_params(labelsize=6)
        axes[row, start].set_ylabel(cell_ylabel(cell, None), fontsize=8, labelpad=12)
        for col in range(n_col):
            axes[row, col].tick_params(labelleft=(col == start))

    fig.subplots_adjust(top=0.98, bottom=0.10, hspace=0.85, wspace=0.35)
    timer.end_plot()
    save_figure(fig, path, dpi=SBAR_DPI, rasterize=True, timer=timer)


@torch.no_grad()
def plot_gt(path, *, readouts, title, gts=None, cost_parts=None):
    """Plot sbar gt figure from contrast → :class:`SbarTraceReadout`."""
    gt_readouts = {contrast: _sbar_gt_readout(readout) for contrast, readout in readouts.items()}
    _plot_figure(
        path,
        timer=ElapsedTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts=gt_readouts,
        title=title,
        gts=gts,
        cost_parts=cost_parts,
    )


@torch.no_grad()
def plot_all(path, *, readouts, title, gts=None, cost_parts=None):
    """Plot sbar all figure: config ``sbar_figure_cells`` rows."""
    all_readouts = {
        contrast: _sbar_all_readout(readout) for contrast, readout in readouts.items()
    }
    _plot_figure(
        path,
        timer=ElapsedTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts=all_readouts,
        title=title,
        gts=gts,
        cost_parts=cost_parts,
    )


_TASK = "sbar"


def build_readout(session, z, contrast, **readout_kwargs):
    return network_sbar_trace_readout(
        session_from_task(session, _TASK, contrast), z, _TASK, contrast, **readout_kwargs,
    )


def figure_titles(session, suffix, token, *, contrast=None):
    if contrast is None:
        return (
            f"Static-bar {token}-gt ({suffix})",
            f"Static-bar {token}-all ({suffix})",
        )
    return (
        f"sbar {contrast} {token}-gt ({suffix})",
        f"sbar {contrast} {token}-all ({suffix})",
    )
