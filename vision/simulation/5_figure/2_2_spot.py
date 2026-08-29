"""Spot plotting (network spot task).

Network RF profile axis is hex-lattice radius: ``v_readout_mean_cell_mean_radius[..., radius]``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

import train
from neuron.borst import t_from_ms

from figure.spread import (
    TraceReadout,
    _layout_cells_from_readouts,
    _rows_from_cell_rows,
    _session_task_timing,
    _style_time_axis,
    contrast_linestyle,
    contrast_order,
    plot_trace_all,
    plot_trace_gt,
)
from figure.panel import (
    GT_COLOR,
    V_READOUT_COLOR,
    TRACE_LINE_W,
    annotate_v_th,
    as_numpy,
    gt_trace_affine,
    e_leak_from_z,
    save_figure,
    at_xy_label,
    mark_sti_on,
    at_xy_reds,
    plot_trace,
    plot_timecourse,
    traces_with_cost_ts,
    sd_from_traces,
    expand_at_xy,
    is_single_hex_cost,
    v_th_from_z,
)
from figure.plot import (
    figure_subtitle_sti_geo,
    session_filter_figure_token,
    session_from_task,
)
from task.spot.pack import build_spot_center_readout, part_key as spot_part_key
from task.spread.pack import cost_sti_hexes
from network import path  # noqa: F401 -- FAFBv783 on sys.path
from network.construction import active_gt_cells, cell_rows, cells_in_order, gt_cells_from_opts
import build_hex
from task.sbar.sti_geo import (
    sti_hexes_at_xy,
    node_us_vs,
)
from task.spot.sti_geo import (
    resolve_spot,
    spot_sti_bs,
)
from task.spread.gt import RF_SIGN, contrast_sign, spread_gt_active
from task.spread.sti_spec import CONTRASTS, t_sti_end
from task.spot.gt import (
    GT_CELLS,
    RF_CENTER_RADIUS,
    RF_N_RADII,
    RF_RADIUS_DEG,
    load_gt,
    t_delay_from_ir,
)

RF_RADIUS_X = np.arange(RF_N_RADII) * RF_RADIUS_DEG


PLOT_AT_XY = True


def active_spot_gt_cells(session, task=None, contrast=None):
    """Configured spot gt cells (sti opts), not cost-pack-only."""
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
            context="spot plot",
        )
    )


def spot_gts(
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
    """Spot gts ``{contrast: {cell: gt}}``."""
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
        gt_rows = load_gt(
            contrast=contrast,
            t_onset=t_onset, n_t=n_t, ms_sti=ms_sti, delta_ms=delta_ms,
            filter=filter, spread_gt_mode=spread_gt_mode,
        ) * gt_amp
        gts[contrast] = {
            str(cell): gt_rows[cell_idx[cell]]
            for cell in GT_CELLS
            if spread_gt_active(spread_gt_mode, contrast, int(RF_SIGN[cell]))
        }
    return gts


def pack_spot_cost_radii(pack) -> tuple[int, ...]:
    """Active hex-lattice cost radii from ``pack.entry_radii``."""
    if pack.entry_radii is None:
        raise ValueError(f"{pack.task} pack missing entry_radii")
    return tuple(
        sorted({int(radius) for radius in pack.entry_radii.tolist()})
    )


def resolve_spot_gts(sessions, gts=None, *, filter=None):
    """``{contrast: {cell: gt}}`` for each entry in ``sessions``.

    Gt time length is response-only (no ``ms_post``).
    ``filter`` selects train GT kind (``none``→v ir, ``ca``→Arenz); default
    per session ``train_opts.filter``.
    """
    if gts is not None:
        return gts
    if not sessions:
        return {}
    gts = {}
    for contrast, session in sessions.items():
        t_onset, _n_t, n_t_gt, ms_sti, delta_ms = _session_task_timing(session)
        part = spot_gts(
            session, session.primary_pack.task, contrasts=(str(contrast),),
            t_onset=t_onset, n_t=n_t_gt, ms_sti=ms_sti, delta_ms=delta_ms,
            filter=filter,
        )
        gts.update(part)
    return gts


def center_trace_and_rf_profile(radius_time, center_radius, sd=None, *, t_onset=None, t_sti_end=None, t_delay=0):
    """Center-radius time course + RF profile from gt or v_readout.

    RF peak time ``peak_t`` is ``argmax |v - v_onset|`` inside the
    delay-shifted spot-on window ``[t_onset + t_delay, t_sti_end + t_delay]``
    (onset = first sample of that shifted window).
    Absolute ``|v|`` would pick onset when a large bias moves toward zero.
    """
    if t_onset is None:
        raise ValueError("center_trace_and_rf_profile requires t_onset")
    if t_sti_end is None:
        raise ValueError("center_trace_and_rf_profile requires t_sti_end")
    center_trace = radius_time[center_radius]
    t_delay = int(t_delay)
    t0 = max(0, int(t_onset) + t_delay)
    t1 = min(int(center_trace.shape[0]) - 1, int(t_sti_end) + t_delay)
    if t1 < t0:
        raise ValueError(
            "center_trace_and_rf_profile requires shifted t_sti_end >= shifted t_onset, "
            f"got [{t0}, {t1}] with t_delay={t_delay}"
        )
    resp = center_trace[t0:t1 + 1]
    if not np.isfinite(resp).any():
        return None, None, None
    ref = float(resp[0]) if np.isfinite(resp[0]) else float(resp[np.isfinite(resp)][0])
    peak_t = t0 + int(np.nanargmax(np.abs(resp - ref)))
    rf_profile = np.asarray(radius_time[:, peak_t], dtype=np.float64)
    center_sd = None if sd is None else sd[center_radius]
    return center_trace, rf_profile, center_sd


def _plot_rf_profile(ax, rf, *, color, label=None, linestyle='-', filled=False):
    """Plot finite RF radii only; skip NaN (no cost readout)."""
    if rf is None:
        return
    rf = np.asarray(rf, dtype=np.float64)
    mask = np.isfinite(rf)
    if not mask.any():
        return
    kwargs = dict(
        color=color,
        label=label,
        linestyle='none',
        marker='o',
    )
    if filled:
        kwargs.update(markersize=4, fillstyle='full', markeredgewidth=0.8)
    else:
        kwargs.update(markersize=6, fillstyle='none', markeredgewidth=1.2)
        if linestyle == '--':
            kwargs['markeredgewidth'] = 1.0
    ax.plot(RF_RADIUS_X[mask], rf[mask], **kwargs)


def _style_rf_profile_axis(ax, show_xlabel):
    """Style RF profile axis (degrees from center; index = radius)."""
    x_max = float((RF_N_RADII - 1) * RF_RADIUS_DEG)
    ax.set_xlim(-2, x_max + 2)
    ax.set_xticks([0, x_max / 2.0, x_max])
    ax.set_xticklabels(['0', f'{x_max / 2.0:g}', f'{x_max:g}'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('RF (°)', fontsize=7)


def _rf_profile_contrast_traces(
    traces, *, t_onset, t_sti_end, center_radius=RF_CENTER_RADIUS, t_delay=0,
):
    """RF profile + center traces for each contrast.

    ``traces`` items may include ``v_readout_mean_cell_mean_radius``, ``gt``, ``sd``.
    Returns a list of dicts with ``v_readout_center``, ``v_readout_rf_profile``,
    ``v_readout_sd``, ``gt_center``, ``gt_rf_profile`` plus passthrough keys.
    """
    rf_profile_kwargs = dict(t_onset=t_onset, t_sti_end=t_sti_end, t_delay=t_delay)
    for i, trace in enumerate(traces):
        v_readout = trace.get("v_readout_mean_cell_mean_radius")
        gt = trace.get("gt")
        if v_readout is not None:
            v_readout_center, v_readout_rf_profile, v_readout_sd = center_trace_and_rf_profile(
                v_readout, center_radius, trace.get("sd"), **rf_profile_kwargs,
            )
        else:
            v_readout_center, v_readout_rf_profile, v_readout_sd = None, None, None
        if gt is not None:
            gt_center, gt_rf_profile, _ = center_trace_and_rf_profile(
                gt, center_radius, **rf_profile_kwargs,
            )
        else:
            gt_center, gt_rf_profile = None, None
        traces[i] = {
            **trace,
            "v_readout_center": v_readout_center,
            "v_readout_rf_profile": v_readout_rf_profile,
            "v_readout_sd": v_readout_sd,
            "gt_center": gt_center,
            "gt_rf_profile": gt_rf_profile,
        }
    return traces


def plot_cell_rf(
    ax,
    title,
    traces,
    *,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    t_onset=None,
    t_sti_end=None,
    t_delay=0,
):
    """RF-profile panel for one cell across contrast ``traces``."""
    scaled = _rf_profile_contrast_traces(
        traces, t_onset=t_onset, t_sti_end=t_sti_end, t_delay=t_delay,
    )
    for trace in scaled:
        linestyle = trace.get("linestyle", "-")
        _plot_rf_profile(
            ax, trace["gt_rf_profile"], color=GT_COLOR,
            label=trace.get("gt_label"), linestyle=linestyle,
        )
        _plot_rf_profile(
            ax, trace["v_readout_rf_profile"], color=V_READOUT_COLOR,
            label=trace.get("v_readout_label"), linestyle=linestyle, filled=True,
        )
    ax.set_title(title, fontsize=8, pad=2)
    _style_rf_profile_axis(ax, show_xlabels)
    if show_ylabel:
        ax.set_ylabel('mV', fontsize=7)
    ax.tick_params(labelsize=6)
    if show_legend:
        ax.legend(loc='upper right', fontsize=6, frameon=False)


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
    gt_from_t=None,
    t_sti_end=None,
    t_delay=0,
    delta_ms,
    delta_ms_pre,
    ms_shown=None,
    center_radius=RF_CENTER_RADIUS,
):
    """Time-course panel for one cell across contrast ``traces``.

    Gray gt omits ``[0, t_onset)`` unless ``gt_from_t=0``.
    ``t_sti_end``: white sti-on band ``[t_onset, t_sti_end]``.
    """
    scaled = _rf_profile_contrast_traces(
        traces,
        t_onset=t_onset,
        t_sti_end=t_sti_end,
        t_delay=t_delay,
        center_radius=int(center_radius),
    )
    t = np.arange(n_t)
    trace_t_onset = int(t_onset or 0)
    if title is not None:
        ax.set_title(title, fontsize=8, pad=2)
    mark_sti_on(ax, t_onset, t_sti_end)
    plot_timecourse(
        ax, t,
        [
            {
                "v_readout_mean_cell_mean_radius": trace["v_readout_center"],
                "gt": trace["gt_center"],
                "sd": trace["v_readout_sd"],
                "linestyle": trace.get("linestyle", "-"),
                "ts": trace.get("ts"),
            }
            for trace in scaled
        ],
        show_sd=any(trace["v_readout_sd"] is not None for trace in scaled),
        v_th=v_th,
        e_leak=e_leak,
        show_ylabel=show_ylabel,
        style_xaxis=lambda ax_time: _style_time_axis(
            ax_time, show_xlabels, n_t,
            delta_ms=delta_ms, delta_ms_pre=delta_ms_pre, t_onset=t_onset,
            ms_shown=ms_shown,
        ),
        t_onset=trace_t_onset,
        gt_from_t=gt_from_t,
    )


def plot_cell_rf_time(
    ax_rf,
    ax_time,
    title,
    traces,
    *,
    time_title=None,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    v_th=None,
    e_leak=None,
    n_t=None,
    t_onset=None,
    t_sti_end=None,
    t_delay=0,
    delta_ms,
    delta_ms_pre,
    ms_shown=None,
):
    """RF + time panels for one cell (composes ``plot_cell_rf`` + ``plot_cell_time``)."""
    plot_cell_rf(
        ax_rf, title, traces,
        show_legend=show_legend,
        show_xlabels=show_xlabels,
        show_ylabel=show_ylabel,
        t_onset=t_onset,
        t_sti_end=t_sti_end,
        t_delay=t_delay,
    )
    plot_cell_time(
        ax_time, traces,
        title=time_title,
        show_xlabels=show_xlabels,
        show_ylabel=show_ylabel,
        v_th=v_th,
        e_leak=e_leak,
        n_t=n_t,
        t_onset=t_onset,
        t_sti_end=t_sti_end,
        t_delay=t_delay,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        ms_shown=ms_shown,
    )


def plot_cell_rf_time_at_xy(
    ax_rf,
    ax_time,
    title,
    traces,
    labels,
    *,
    time_title=None,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    v_th=None,
    e_leak=None,
    n_t=None,
    t_onset=None,
    t_sti_end=None,
    t_delay=0,
    delta_ms,
    delta_ms_pre,
    ms_shown=None,
):
    """RF + time panels with per at_xy traces across contrast ``traces``."""
    center_radius = RF_CENTER_RADIUS
    rf_profile_kwargs = dict(t_onset=t_onset, t_sti_end=t_sti_end, t_delay=t_delay)
    colors = at_xy_reds(len(labels))
    t = np.arange(n_t)
    trace_t_onset = int(t_onset or 0)
    mark_sti_on(ax_time, t_onset, t_sti_end)

    for trace_idx, trace in enumerate(traces):
        linestyle = trace.get("linestyle", "-")
        v_readout = trace.get("v_readout_mean_cell_mean_radius")
        gt = trace.get("gt")
        mean_hex_by_label = trace.get("mean_hex_by_label") or {}
        v_readout_center = v_readout_rf_profile = None
        if v_readout is not None:
            v_readout_center, v_readout_rf_profile, _ = center_trace_and_rf_profile(
                v_readout, center_radius, **rf_profile_kwargs,
            )
        gt_center = gt_rf_profile = None
        if gt is not None:
            gt_center, gt_rf_profile, _ = center_trace_and_rf_profile(
                gt, center_radius, **rf_profile_kwargs,
            )
        centers_by_label = {}
        rf_profiles_by_label = {}
        for label in labels:
            if label in mean_hex_by_label:
                center, rf_profile, _ = center_trace_and_rf_profile(
                    mean_hex_by_label[label], center_radius, **rf_profile_kwargs,
                )
                centers_by_label[label] = center
                rf_profiles_by_label[label] = rf_profile

        _plot_rf_profile(
            ax_rf, gt_rf_profile, color=GT_COLOR,
            label=trace.get("gt_label"), linestyle=linestyle,
        )
        for label, color in zip(labels, colors):
            _plot_rf_profile(
                ax_rf, rf_profiles_by_label.get(label), color=color,
                label=label if trace_idx == 0 else None, linestyle=linestyle,
            )
        _plot_rf_profile(
            ax_rf, v_readout_rf_profile, color=colors[-1],
            label=trace.get("hexes_label"), linestyle=linestyle, filled=True,
        )

        if gt_center is not None:
            t_gt = np.asarray(t)
            gt_center = np.asarray(gt_center, dtype=np.float64)
            if t_gt.shape[0] > gt_center.shape[0]:
                t_gt = t_gt[: gt_center.shape[0]]
            n_gt = int(gt_center.shape[0])
            gt_start = max(0, min(trace_t_onset, n_gt))
            if gt_start < n_gt:
                ax_time.plot(
                    t_gt[gt_start:], gt_center[gt_start:],
                    color=GT_COLOR, linestyle=linestyle, linewidth=TRACE_LINE_W,
                )
        for label, color in zip(labels, colors):
            plot_trace(
                ax_time, t, centers_by_label.get(label), t_onset=trace_t_onset,
                color=color, linestyle=linestyle, linewidth=TRACE_LINE_W,
                label=label if trace_idx == 0 else None,
            )
        plot_trace(
            ax_time, t, v_readout_center, t_onset=trace_t_onset,
            color=colors[-1], linestyle=linestyle, linewidth=TRACE_LINE_W,
            label=trace.get("hexes_label"),
        )

    ax_rf.set_title(title, fontsize=8, pad=2)
    _style_rf_profile_axis(ax_rf, show_xlabels)
    if show_ylabel:
        ax_rf.set_ylabel('mV', fontsize=7)
    ax_rf.tick_params(labelsize=6)
    if show_legend:
        ax_rf.legend(loc='upper right', fontsize=6, frameon=False)

    if time_title is not None:
        ax_time.set_title(time_title, fontsize=8, pad=2)
    _style_time_axis(
        ax_time, show_xlabels, n_t,
        delta_ms=delta_ms, delta_ms_pre=delta_ms_pre, t_onset=t_onset,
        ms_shown=ms_shown,
    )
    if show_ylabel:
        ax_time.set_ylabel('mV', fontsize=7)
    ax_time.tick_params(labelsize=6)
    annotate_v_th(ax_time, v_th, e_leak=e_leak)


def _spot_v_readout_mean_cell_mean_radius(readout, *, mask=None):
    """``cell → (n_radius, n_t)`` mean; also ``sd`` and ``n_by_cell``."""
    figure_cells = list(readout['figure_cells'])
    cells = readout['cells']
    n_t = readout['n_t']
    node_cell_idx = readout['node_cell_idx']
    du = readout['du']
    dv = readout['dv']
    figure_traces = readout['figure_traces']
    if mask is not None:
        node_cell_idx = node_cell_idx[mask]
        du = du[mask]
        dv = dv[mask]
        figure_traces = figure_traces[mask]
    v_stack = np.full((len(figure_cells), RF_N_RADII, n_t), np.nan)
    sd_stack = np.full((len(figure_cells), RF_N_RADII, n_t), np.nan)
    n_by_cell = {}
    for figure_cell_idx, figure_cell in enumerate(figure_cells):
        cell_entry_mask = node_cell_idx == cells.index(figure_cell)
        if not cell_entry_mask.any():
            n_by_cell[figure_cell] = {}
            continue
        entries_by_radius: dict[int, list] = {}
        for entry in np.where(cell_entry_mask)[0]:
            radius = int(build_hex.hex_radius(int(du[entry]), int(dv[entry])))
            entries_by_radius.setdefault(radius, []).append(int(entry))
        n_by_radius = {}
        for radius, grouped_entries in entries_by_radius.items():
            if radius < 0 or radius >= RF_N_RADII:
                continue
            traces = figure_traces[grouped_entries]
            v_stack[figure_cell_idx, radius] = traces.mean(axis=0)
            sd_stack[figure_cell_idx, radius] = sd_from_traces(
                traces, single_hex=False,
            )
            n_by_radius[radius] = len(grouped_entries)
        n_by_cell[figure_cell] = n_by_radius
    v_readout_mean_cell_mean_radius = {
        figure_cell: v_stack[figure_cell_idx]
        for figure_cell_idx, figure_cell in enumerate(figure_cells)
    }
    sd = {
        figure_cell: sd_stack[figure_cell_idx]
        for figure_cell_idx, figure_cell in enumerate(figure_cells)
    }
    return v_readout_mean_cell_mean_radius, sd, n_by_cell


@dataclass
class SpotTraceReadout(TraceReadout):
    """One forward pass; full cost-radius readout over all types."""

    v_readout_mean_cell_mean_radius: dict = field(default_factory=dict)
    at_xs: list | None = None
    at_ys: list | None = None
    labels: list[str] | None = None
    v_readout_mean_cell_mean_radius_mean_hex_by_label: dict[str, dict[str, np.ndarray]] | None = None
    a_sti_radius: dict[str, float] = field(default_factory=dict)


def _spot_gt_readout(readout):
    """Gt figure rows: configured active gt cells (not cost-pack-only)."""
    session = readout.session
    active = active_spot_gt_cells(
        session,
        session.primary_pack.task,
        session.primary_pack.contrast,
    )
    rows = [np.array(row) for row in cell_rows(active)]
    readout_cells = set(readout.cells)
    cells = [cell for cell in cells_in_order(active) if cell in readout_cells]
    return SpotTraceReadout(
        cells=cells,
        rows=_rows_from_cell_rows(rows, cells),
        session=session,
        n_t=readout.n_t,
        v_readout_mean_cell_mean_radius={
            cell: readout.v_readout_mean_cell_mean_radius[cell] for cell in cells
            if cell in readout.v_readout_mean_cell_mean_radius
        },
        sd={
            cell: readout.sd[cell] for cell in cells
            if cell in readout.sd
        },
        n_by_cell={
            cell: dict(readout.n_by_cell.get(cell) or {})
            for cell in cells
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
        a_sti_radius=dict(readout.a_sti_radius),
    )


def _spot_readout_hex_mask(connectome, nodes, cost_radius, *, at_x=None, at_y=None):
    """True for cost entries whose node sits on ``at_x``/``at_y`` cost-radius hexes."""
    hexes = sti_hexes_at_xy(
        cost_sti_hexes(connectome, cost_radius=cost_radius),
        at_x=at_x,
        at_y=at_y,
    )
    if not hexes:
        return np.zeros(len(nodes), dtype=bool)
    node_us, node_vs = node_us_vs(connectome)
    hex_uv = {(int(hex.u), int(hex.v)) for hex in hexes}
    return np.array(
        [(int(node_us[node]), int(node_vs[node])) in hex_uv for node in nodes],
        dtype=bool,
    )


def _spot_v_readout_mean_cell_mean_radius_mean_hex(readout, connectome, at_xs, at_ys):
    """Per at_xy mean_hex: cell × radius mean over ``at_x``/``at_y`` hexes."""
    mean_hex_by_label = {}
    labels = []
    pack = readout['pack']
    nodes = readout['nodes']
    for label, at_x, at_y in expand_at_xy(at_xs, at_ys)[0]:
        mask = _spot_readout_hex_mask(
            connectome, nodes, pack.cost_radius, at_x=at_x, at_y=at_y,
        )
        if not np.any(mask):
            print(f'skip at_xy {label}: no hex within cost_radius')
            continue
        v_readout_mean_cell_mean_radius_mean_hex, _, _ = (
            _spot_v_readout_mean_cell_mean_radius(readout, mask=mask)
        )
        if not any(
            np.isfinite(v_readout).any()
            for v_readout in v_readout_mean_cell_mean_radius_mean_hex.values()
        ):
            print(f'skip at_xy {label}: no readouts')
            continue
        mean_hex_by_label[label] = v_readout_mean_cell_mean_radius_mean_hex
        labels.append(label)
    if not mean_hex_by_label:
        return None, None
    return mean_hex_by_label, labels



@torch.no_grad()
def _forward_spot_readout(
    session, z, *,
    at_x=None, at_y=None,
):
    """One forward; cost-radius node readout over all network types."""
    pack = session.primary_pack
    params = train.params_from_z(z, session)
    a_sti_radius = {}
    if "a_sti_radius" in params:
        spec = session.schema.get("a_sti_radius")
        if spec is not None:
            a_sti_radius = {
                str(radius): float(val)
                for radius, val in zip(
                    spec.get("radii") or (),
                    as_numpy(params["a_sti_radius"]).reshape(-1),
                )
            }
    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    trace = train.forward_pack(session, params, i_sti, pack)
    connectome = session.connectome
    cells = list(connectome.cells)
    n_t = int(i_sti.shape[1])

    opts = dict((session.train_opts or {}).get(f"{pack.task}_sti_opts") or {})
    spot = resolve_spot(connectome, sti_opts=opts)
    spot_bs = spot_sti_bs(spot)
    active = cells_in_order(connectome.cells)
    rows = [np.array(row) for row in cell_rows(active)]
    figure_cells = cells_in_order(active)

    (
        bs, nodes, _radius, node_cell_idx, _sti_u, _sti_v, du, dv, center_entry_mask,
    ) = build_spot_center_readout(
        connectome, spot_bs, pack_spot_cost_radii(pack), pack.cost_radius,
    )

    figure_traces = as_numpy(trace[bs, :, nodes])

    sti_ms_pre = opts.get("ms_pre")
    delta_ms = float(opts["delta_ms"])
    delta_ms_pre = float(opts["delta_ms_pre"])
    sti_t_onset = (
        t_from_ms(float(sti_ms_pre), delta_ms=delta_ms_pre)
        if sti_ms_pre is not None else None
    )
    ms_sti = opts.get("ms_sti")
    readout = dict(
        figure_cells=figure_cells,
        cells=cells,
        node_cell_idx=node_cell_idx,
        nodes=nodes,
        du=du,
        dv=dv,
        center_entry_mask=center_entry_mask,
        figure_traces=figure_traces,
        t_onset=int(sti_t_onset) if sti_t_onset is not None else None,
        t_sti_end=(
            None if sti_t_onset is None else t_sti_end(
                sti_t_onset, n_t, ms_sti,
                delta_ms=delta_ms,
            )
        ),
        bs=bs,
        spot_bs=spot_bs,
        n_t=n_t,
        pack=pack,
        a_sti_radius=a_sti_radius,
    )
    mask = np.asarray(center_entry_mask, dtype=bool)
    if at_x is not None or at_y is not None:
        mask = mask & _spot_readout_hex_mask(
            connectome, nodes, pack.cost_radius, at_x=at_x, at_y=at_y,
        )
    nodes_by_cell = {
        cell: np.unique(nodes[mask & (node_cell_idx == cells.index(cell))])
        for cell in figure_cells
    }
    v_th = v_th_from_z(z, session)
    e_leak = e_leak_from_z(z, session)
    readout['v_th_by_cell'] = {
        cell: v_th.get(cell, np.nan) for cell in nodes_by_cell
    }
    readout['e_leak_by_cell'] = {
        cell: e_leak.get(cell, np.nan) for cell in nodes_by_cell
    }
    readout['gt_affine_by_cell'] = {
        cell: train.gt_affine_from_cell(
            params, cell, session.connectome, session=session,
        )
        for cell in figure_cells
    }
    readout['cell_rows'] = rows
    return readout


@torch.no_grad()
def network_spot_trace_readout(
    session, z, *,
    at_xs=None, at_ys=None,
    ms_shown=None,
):
    """Run one forward; full cost-radius spot traces over all types."""
    t_prep0 = time.perf_counter()
    readout = _forward_spot_readout(session, z)
    v_readout_mean_cell_mean_radius, sd, n_by_cell = (
        _spot_v_readout_mean_cell_mean_radius(readout)
    )
    figure_cells = list(readout['figure_cells'])
    n_t = readout['n_t']
    if is_single_hex_cost(
        session, task=readout['pack'].task, contrast=readout['pack'].contrast,
    ):
        sd = {}
    rows = _rows_from_cell_rows(readout['cell_rows'], figure_cells)
    mean_hex_by_label, labels = (None, None)
    if at_xs is not None or at_ys is not None:
        connectome = session.connectome
        mean_hex_by_label, labels = _spot_v_readout_mean_cell_mean_radius_mean_hex(
            readout, connectome, at_xs, at_ys,
        )
    return SpotTraceReadout(
        cells=figure_cells,
        rows=rows,
        session=session,
        at_xs=at_xs,
        at_ys=at_ys,
        labels=labels,
        v_readout_mean_cell_mean_radius_mean_hex_by_label=mean_hex_by_label,
        n_t=n_t,
        prep_s=time.perf_counter() - t_prep0,
        v_readout_mean_cell_mean_radius=v_readout_mean_cell_mean_radius,
        sd=sd,
        n_by_cell=n_by_cell,
        v_th_by_cell=dict(readout.get('v_th_by_cell') or {}),
        e_leak_by_cell=dict(readout.get('e_leak_by_cell') or {}),
        gt_affine_by_cell=dict(readout.get('gt_affine_by_cell') or {}),
        t_onset=readout.get('t_onset'),
        t_sti_end=readout.get('t_sti_end'),
        ms_shown=ms_shown,
        a_sti_radius=dict(readout.get('a_sti_radius') or {}),
    )


def _spot_suptitle(title, readout):
    head = title
    if readout is not None:
        a_sti_radius = readout.a_sti_radius
        if a_sti_radius and "1" in a_sti_radius:
            head = f"a_sti_radius 1 = {float(a_sti_radius['1']):.4g}"
        if readout.at_xs is not None or readout.at_ys is not None:
            at_xy_subtitle = at_xy_label(readout.at_xs, readout.at_ys)
            return f'{head}  [{at_xy_subtitle}, at_xy + hexes]'
    return head


_TASK = "spot"


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
    """Plot spot figure from ``readouts`` (contrast → SpotTraceReadout)."""
    order = contrast_order(readouts)
    if not order:
        raise ValueError("_plot_figure requires at least one readout")
    primary = readouts[order[0]]
    cells, rows = _layout_cells_from_readouts(readouts, order)
    has_at_xy = (
        primary.at_xs is not None or primary.at_ys is not None
    ) and primary.v_readout_mean_cell_mean_radius_mean_hex_by_label is not None
    labels = primary.labels or []
    n_t = primary.n_t
    t_onset = primary.t_onset
    t_sti_end = primary.t_sti_end
    delta_ms = float(primary.session.delta_ms)
    delta_ms_pre = float(primary.session.delta_ms_pre)
    ms_shown = primary.ms_shown
    figure_filter = session_filter_figure_token(primary.session)
    t_delay = t_delay_from_ir(
        delta_ms=delta_ms,
        filter=figure_filter,
    )
    cell_idx = {cell: index for index, cell in enumerate(GT_CELLS)}

    def _t_delay_from_cell(cell):
        index = cell_idx.get(cell)
        return 0 if index is None else int(t_delay[index])

    timer.end_prep()

    sessions = {contrast: readouts[contrast].session for contrast in order}
    gt_by_contrast = resolve_spot_gts(sessions, gts)

    part_keys = list(cost_parts or ()) or list(
        train.session_cost_part_keys(primary.session)
    )
    radii = list(train.cost_radii_from_packs(primary.session.iter_packs(), contrasts=order))
    center_radius = int(RF_CENTER_RADIUS)
    order_hs = [
        1 + len(radii) for _ in rows
    ]
    n_row = int(sum(order_hs))
    fig = plt.figure(figsize=figure_size_from_grid(n_col, n_row))
    grid_spec = fig.add_gridspec(n_row, n_col, **gridspec_kwargs)
    legend_done = False

    def _build_cell_traces(cell):
        traces = []
        for contrast in order:
            contrast_readout = readouts[contrast]
            if cell not in contrast_readout.v_readout_mean_cell_mean_radius:
                continue
            gt_by_cell = gt_by_contrast.get(contrast) or {}
            trace = {
                "contrast": contrast,
                "v_readout_mean_cell_mean_radius": contrast_readout.v_readout_mean_cell_mean_radius[cell],
                "gt": gt_trace_affine(contrast_readout, cell, gt_by_cell.get(cell)),
                "sd": contrast_readout.sd.get(cell),
                "v_th": contrast_readout.v_th_by_cell.get(cell),
                "linestyle": contrast_linestyle(contrast),
                "gt_label": f"{contrast} gt",
                "v_readout_label": f"{contrast} {figure_filter}",
                "hexes_label": f"{contrast} hexes",
            }
            by_label = contrast_readout.v_readout_mean_cell_mean_radius_mean_hex_by_label
            if by_label is not None:
                trace["mean_hex_by_label"] = {
                    label: mean_hex[cell]
                    for label, mean_hex in by_label.items()
                    if cell in mean_hex
                }
            else:
                trace["mean_hex_by_label"] = {}
            traces.append(trace)
        return traces

    def _plot_cell(cell, ax_rf, ax_time, show_ylabel, show_xlabels):
        nonlocal legend_done
        n_by_radius = primary.n_by_cell.get(cell) or {}
        radius = float(center_radius)
        radius_label = str(int(radius)) if radius == int(radius) else str(radius)
        time_title = f'radius={radius_label}'
        n_center = n_by_radius.get(center_radius)
        if n_center is not None:
            time_title = f'{time_title} (n={int(n_center)})'
        if cost_parts and order:
            for contrast in order:
                key = spot_part_key(contrast, cell, center_radius)
                if key in cost_parts:
                    time_title = (
                        f'{time_title}\n{contrast}: {float(cost_parts[key]):.1f}'
                    )
        v_th = primary.v_th_by_cell.get(cell)
        e_leak = primary.e_leak_by_cell.get(cell)
        if has_at_xy:
            traces = traces_with_cost_ts(
                _build_cell_traces(cell),
                readouts,
                entry_radius=float(center_radius),
            )
            mean_hex_by_label = (traces[0].get("mean_hex_by_label") or {}) if traces else {}
            if not mean_hex_by_label:
                ax_rf.axis("off")
                ax_time.axis("off")
                return
            plot_cell_rf_time_at_xy(
                ax_rf, ax_time, cell, traces, labels,
                time_title=time_title,
                show_legend=not legend_done,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                n_t=n_t,
                v_th=v_th,
                e_leak=e_leak,
                t_onset=t_onset,
                t_sti_end=t_sti_end,
                t_delay=_t_delay_from_cell(cell),
                delta_ms=delta_ms,
                delta_ms_pre=delta_ms_pre,
                ms_shown=ms_shown,
            )
        else:
            traces = traces_with_cost_ts(
                _build_cell_traces(cell),
                readouts,
                entry_radius=float(center_radius),
            )
            plot_cell_rf_time(
                ax_rf, ax_time, cell, traces,
                time_title=time_title,
                show_legend=not legend_done,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                n_t=n_t,
                v_th=v_th,
                e_leak=e_leak,
                t_onset=t_onset,
                t_sti_end=t_sti_end,
                t_delay=_t_delay_from_cell(cell),
                delta_ms=delta_ms,
                delta_ms_pre=delta_ms_pre,
                ms_shown=ms_shown,
            )
        legend_done = True

    row_cursor = 0
    for row_group, cell_idxs in enumerate(rows):
        group_h = int(order_hs[row_group])
        rf_row = row_cursor
        time_row0 = row_cursor + 1
        start = (n_col - len(cell_idxs)) // 2
        for col, cell_idx in enumerate(cell_idxs, start=start):
            cell = cells[cell_idx]
            ax_rf = fig.add_subplot(grid_spec[rf_row, col])
            if has_at_xy:
                ax_time = fig.add_subplot(grid_spec[time_row0, col])
                _plot_cell(
                    cell, ax_rf, ax_time,
                    show_ylabel=(col == start), show_xlabels=True,
                )
                for time_row in range(time_row0 + 1, row_cursor + group_h):
                    ax_off = fig.add_subplot(grid_spec[time_row, col])
                    ax_off.axis("off")
                continue

            traces = _build_cell_traces(cell)
            show_legend = not legend_done
            plot_cell_rf(
                ax_rf, cell, traces,
                show_legend=show_legend,
                show_xlabels=False,
                show_ylabel=(col == start),
                t_onset=t_onset,
                t_sti_end=t_sti_end,
                t_delay=_t_delay_from_cell(cell),
            )
            legend_done = True
            ylim_shared = None
            n_by_radius = primary.n_by_cell.get(cell) or {}
            v_th = primary.v_th_by_cell.get(cell)
            e_leak = primary.e_leak_by_cell.get(cell)
            for radius_row, time_row in enumerate(range(time_row0, row_cursor + group_h)):
                ax_time_radius = fig.add_subplot(grid_spec[time_row, col])
                if radius_row >= len(radii):
                    ax_time_radius.axis("off")
                    continue
                radius = int(radii[radius_row])
                radius_f = float(radius)
                radius_label = str(int(radius_f)) if radius_f == int(radius_f) else str(radius_f)
                time_title = f'radius={radius_label}'
                n_radius = n_by_radius.get(radius)
                if n_radius is not None:
                    time_title = f'{time_title} (n={int(n_radius)})'
                if cost_parts and order:
                    for contrast in order:
                        key = spot_part_key(contrast, cell, radius)
                        if key in cost_parts:
                            time_title = (
                                f'{time_title}\n{contrast}: '
                                f'{float(cost_parts[key]):.1f}'
                            )
                plot_cell_time(
                    ax_time_radius,
                    traces_with_cost_ts(
                        traces, readouts, entry_radius=float(radius),
                    ),
                    title=time_title,
                    show_xlabels=(radius_row == len(radii) - 1),
                    show_ylabel=(col == start),
                    v_th=v_th,
                    e_leak=e_leak,
                    n_t=n_t,
                    t_onset=t_onset,
                    t_sti_end=t_sti_end,
                    t_delay=_t_delay_from_cell(cell),
                    delta_ms=delta_ms,
                    delta_ms_pre=delta_ms_pre,
                    ms_shown=ms_shown,
                    center_radius=radius,
                )
                if radius_row == 0:
                    ylim_shared = ax_time_radius.get_ylim()
                elif ylim_shared is not None:
                    ax_time_radius.set_ylim(*ylim_shared)
        row_cursor += group_h
    fig.suptitle(_spot_suptitle(title, primary), fontsize=suptitle_fs)
    timer.end_plot()
    save_figure(fig, path, dpi=150, timer=timer)


def plot_gt(path, *, readouts, title, gts=None, cost_parts=None):
    """Plot gt figure (active gt cells; model ca for all active)."""
    plot_trace_gt(
        path, readouts=readouts, title=title, gts=gts, cost_parts=cost_parts,
        gt_readout=_spot_gt_readout, plot_figure=_plot_figure,
    )


def plot_all(path, *, readouts, title, gts=None, cost_parts=None):
    """Plot ca-all figure (all types) from contrast → readout."""
    plot_trace_all(
        path, readouts=readouts, title=title, gts=gts, cost_parts=cost_parts,
        plot_figure=_plot_figure,
    )


def build_readout(session, z, contrast, **readout_kwargs):
    return network_spot_trace_readout(
        session_from_task(session, _TASK, contrast), z, **readout_kwargs,
    )


def figure_titles(session, suffix, token, *, contrast=None):
    net_label = figure_subtitle_sti_geo(session, _TASK)
    if contrast is None:
        return (
            f'Spot {token}-gt ({suffix}){net_label}',
            f'Spot {token}-all ({suffix}){net_label}',
        )
    return (
        f'spot {contrast} {token}-gt ({suffix}){net_label}',
        f'spot {contrast} {token}-all ({suffix}){net_label}',
    )

