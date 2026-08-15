"""Spot plotting (network spot task).

Network RF profile axis is hex-lattice radius: v_readout[..., radius] = mean at that radius.
"""

from __future__ import annotations

from const_default import (
    NEURON_CONST,
)

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

from const_default import NEURON_CONST
import train
from neuron.param import t_from_ms, t_abs_from_ms, ms_from_t
from figure.gt import (
    contrast_linestyle,
    contrast_order,
    active_spot_gt_cells,
    spot_gts,
)
from figure.panel import (
    N_COL_ALL,
    N_COL_GT,
    PANEL_W,
    GT_COLOR,
    V_READOUT_COLOR,
    TRACE_LW,
    ElapsedTimer,
    annotate_v_th,
    readout_prep_s,
    format_spot_radius_time_title,
    gt_affine_from_cell,
    gt_trace_affine,
    e_leak_from_z,
    session_filter_figure_token,
    hex_scope_tag,
    mark_spot,
    overlay_reds,
    plot_pre_post_line,
    plot_timecourse,
    save_figure,
    std_from_traces,
    overlay_coords,
    suppress_cost_std,
    v_th_from_z,
)
from network.construction import cell_rows, cells_in_order
from network import path  # noqa: F401 -- FAFBv783 on sys.path
import build_hex
from task.moving_bar.sti_geo import (
    filter_sti_hexes,
    moving_bar_cost_hexes,
    network_uv_np,
)
from task.spot.sti_geo import (
    resolve_spot,
    spot_sti_bs,
)
from task.spot.sti_spec import (
    resolve_sti_timing,
    t_sti_end,
)
from task.spot.gt import (
    GT_CELLS,
    RF_CENTER_RADIUS,
    RF_N_RADII,
    RF_RADIUS_DEG,
    t_delay_from_ir,
)
from task.spot.pack import build_spot_center_readout

RF_RADIUS_X = np.arange(RF_N_RADII) * RF_RADIUS_DEG


def pack_spot_cost_radii(pack) -> tuple[int, ...]:
    """Active hex-lattice cost radii from ``pack.entry_radii``."""
    if pack.entry_radii is None:
        raise ValueError(f"{pack.task} pack missing entry_radii")
    return tuple(
        sorted({int(radius) for radius in pack.entry_radii.tolist()})
    )


def _session_spot_timing(session):
    """Extract onset ``t_onset`` / forward ``n_t``, ms_sti, and delta_ms from session."""
    opts = (session.train_opts or {}).get(
        f"{session.primary_pack.task}_sti_opts",
    ) or {}
    filter = str((session.train_opts or {}).get("filter", "none"))
    timing = resolve_sti_timing(opts)
    return (
        int(timing.t_onset),
        int(timing.n_t),
        int(timing.n_t_gt),
        None if timing.ms_sti is None else float(timing.ms_sti),
        float(timing.delta_ms),
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
    out = {}
    for contrast, session in sessions.items():
        t_onset, _n_t, n_t_gt, ms_sti, delta_ms = _session_spot_timing(session)
        part = spot_gts(
            session, session.primary_pack.task, contrasts=(str(contrast),),
            t_onset=t_onset, n_t=n_t_gt, ms_sti=ms_sti, delta_ms=delta_ms,
            filter=filter,
        )
        out.update(part)
    return out


def _session_cost_ts(session, t_onset, *, entry_radius=None):
    """Absolute cost ``ts`` for sparse spot cost (pack contract; or ``None``)."""
    if session is None:
        return None
    return train.pack_cost_abs_ts(
        session.primary_pack, t_onset, entry_radius=entry_radius,
    )


def _series_with_cost_points(series, readouts, entry_radius):
    """Copy ``series`` with ``ts`` for one hex-lattice ``entry_radius``."""
    out = []
    for entry in series:
        item = dict(entry)
        contrast = entry["contrast"]
        item["ts"] = _session_cost_ts(
            readouts[contrast].session,
            readouts[contrast].t_onset,
            entry_radius=entry_radius,
        )
        out.append(item)
    return out


def scale_curve(radius_time, center_radius, std=None, *, t_onset=None, t_sti_end=None, t_delay=0):
    """Center-radius time course + spatial profile from gt or v_readout.

    RF peak time ``t_v_max`` is ``argmax |v - v_onset|`` inside the
    delay-shifted spot-on window ``[t_onset + t_delay, t_sti_end + t_delay]``
    (onset = first sample of that shifted window).
    Absolute ``|v|`` would pick onset when a large bias moves toward zero.
    """
    if t_onset is None:
        raise ValueError("scale_curve requires t_onset")
    if t_sti_end is None:
        raise ValueError("scale_curve requires t_sti_end")
    center_t = radius_time[center_radius]
    t_delay = int(t_delay)
    t0 = max(0, int(t_onset) + t_delay)
    t1 = min(int(center_t.shape[0]) - 1, int(t_sti_end) + t_delay)
    if t1 < t0:
        raise ValueError(
            "scale_curve requires shifted t_sti_end >= shifted t_onset, "
            f"got [{t0}, {t1}] with t_delay={t_delay}"
        )
    resp = center_t[t0:t1 + 1]
    if not np.isfinite(resp).any():
        return None, None, None
    ref = float(resp[0]) if np.isfinite(resp[0]) else float(resp[np.isfinite(resp)][0])
    t_v_max = t0 + int(np.nanargmax(np.abs(resp - ref)))
    spatial = np.asarray(radius_time[:, t_v_max], dtype=np.float64)
    std_center = None if std is None else std[center_radius]
    return center_t, spatial, std_center


def _plot_rf_profile(ax, rf, *, color, label=None, linestyle='-', filled=False):
    """Plot finite RF radii only; skip NaN (no cost readout)."""
    if rf is None:
        return
    rf = np.asarray(rf, dtype=np.float64)
    mask = np.isfinite(rf)
    if not mask.any():
        return
    kw = dict(
        color=color,
        label=label,
        linestyle='none',
        marker='o',
    )
    if filled:
        kw.update(markersize=4, fillstyle='full', markeredgewidth=0.8)
    else:
        kw.update(markersize=6, fillstyle='none', markeredgewidth=1.2)
        if linestyle == '--':
            kw['markeredgewidth'] = 1.0
    ax.plot(RF_RADIUS_X[mask], rf[mask], **kw)


def _style_time_axis(
    ax, show_xlabel, n_t, *, delta_ms, delta_ms_pre, t_onset, ms_shown=None,
):
    dt = float(delta_ms)
    dt_pre = float(delta_ms_pre)
    t0 = int(t_onset or 0)
    t_last = max(int(n_t) - 1, 0)
    if ms_shown is None:
        lo, hi = 0, t_last
    else:
        start, stop = ms_shown
        lo = t_abs_from_ms(
            float(start), t_onset=t0, delta_ms_pre=dt_pre, delta_ms=dt,
        )
        hi = t_abs_from_ms(
            float(stop), t_onset=t0, delta_ms_pre=dt_pre, delta_ms=dt,
        )
        lo, hi = max(0, lo), min(t_last, hi)
        if lo > hi:
            lo, hi = 0, t_last
    t_lo_s = ms_from_t(lo, t_onset=t0, delta_ms_pre=dt_pre, delta_ms=dt) / 1000.0
    t_hi_s = ms_from_t(hi, t_onset=t0, delta_ms_pre=dt_pre, delta_ms=dt) / 1000.0
    t_mid_s = (t_lo_s + t_hi_s) / 2.0
    mid = (lo + hi) // 2
    ax.set_xlim(lo, hi)
    ax.set_xticks([lo, mid, hi])
    ax.set_xticklabels([f'{t_lo_s:g}', f'{t_mid_s:g}', f'{t_hi_s:g}'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('time [s]', fontsize=7)


def _style_rf_profile_axis(ax, show_xlabel):
    """Style RF profile axis (degrees from center; index = radius)."""
    x_max = float((RF_N_RADII - 1) * RF_RADIUS_DEG)
    ax.set_xlim(-2, x_max + 2)
    ax.set_xticks([0, x_max / 2.0, x_max])
    ax.set_xticklabels(['0', f'{x_max / 2.0:g}', f'{x_max:g}'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('RF (°)', fontsize=7)


def _scale_contrast_series(
    series, *, t_onset, t_sti_end, center_radius=RF_CENTER_RADIUS, t_delay=0,
):
    """Scale each contrast series entry to gt / v_readout plot curves.

    ``series`` items may include ``v_readout``, ``gt``, ``std``.
    Returns a list of dicts with ``v_readout_center``, ``v_readout_spatial``,
    ``v_readout_std``, ``gt_center``, ``gt_spatial`` plus passthrough keys.
    """
    sc_kw = dict(t_onset=t_onset, t_sti_end=t_sti_end, t_delay=t_delay)
    out = []
    for entry in series:
        item = dict(entry)
        v_readout = entry.get("v_readout")
        gt = entry.get("gt")
        if v_readout is not None:
            v_readout_center, v_readout_spatial, v_readout_std = scale_curve(
                v_readout, center_radius, entry.get("std"), **sc_kw,
            )
        else:
            v_readout_center, v_readout_spatial, v_readout_std = None, None, None
        if gt is not None:
            gt_center, gt_spatial, _ = scale_curve(gt, center_radius, **sc_kw)
        else:
            gt_center, gt_spatial = None, None
        item.update(
            v_readout_center=v_readout_center,
            v_readout_spatial=v_readout_spatial,
            v_readout_std=v_readout_std,
            gt_center=gt_center,
            gt_spatial=gt_spatial,
        )
        out.append(item)
    return out


def plot_cell_rf(
    ax,
    title,
    series,
    *,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    t_onset=None,
    t_sti_end=None,
    t_delay=0,
):
    """RF-profile panel for one cell across contrast ``series``."""
    scaled = _scale_contrast_series(
        series, t_onset=t_onset, t_sti_end=t_sti_end, t_delay=t_delay,
    )
    for item in scaled:
        ls = item.get("linestyle", "-")
        _plot_rf_profile(
            ax, item["gt_spatial"], color=GT_COLOR,
            label=item.get("label_gt"), linestyle=ls,
        )
        _plot_rf_profile(
            ax, item["v_readout_spatial"], color=V_READOUT_COLOR,
            label=item.get("label_v_readout"), linestyle=ls, filled=True,
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
    series,
    *,
    title=None,
    show_xlabels=False,
    show_ylabel=False,
    v_th=None,
    e_leak=None,
    n_t=None,
    t_onset=None,
    pre_end=None,
    show_pre=False,
    t_sti_end=None,
    t_delay=0,
    delta_ms,
    delta_ms_pre,
    ms_shown=None,
    center_radius=RF_CENTER_RADIUS,
):
    """Time-course panel for one cell across contrast ``series``.

    ``pre_end`` defaults to ``t_onset`` (gray gt omits ``[0, pre_end)``).
    Pass ``pre_end=0`` to draw the full gt trace including pre-onset.
    ``t_sti_end``: white sti-on band ``[t_onset, t_sti_end]``.
    """
    scaled = _scale_contrast_series(
        series,
        t_onset=t_onset,
        t_sti_end=t_sti_end,
        t_delay=t_delay,
        center_radius=int(center_radius),
    )
    t = np.arange(n_t)
    split = int(t_onset or 0) if pre_end is None else int(pre_end)
    if title is not None:
        ax.set_title(title, fontsize=8, pad=2)
    traces = [
        {
            "v_readout": item["v_readout_center"],
            "gt": item["gt_center"],
            "std": item["v_readout_std"],
            "linestyle": item.get("linestyle", "-"),
            "ts": item.get("ts"),
        }
        for item in scaled
    ]
    plot_timecourse(
        ax, t, traces,
        show_std=any(item["v_readout_std"] is not None for item in scaled),
        v_th=v_th,
        e_leak=e_leak,
        show_ylabel=show_ylabel,
        style_xaxis=lambda a: _style_time_axis(
            a, show_xlabels, n_t,
            delta_ms=delta_ms, delta_ms_pre=delta_ms_pre, t_onset=t_onset,
            ms_shown=ms_shown,
        ),
        pre_end=split,
        show_pre=show_pre,
        t_onset=t_onset,
        t_sti_end=t_sti_end,
    )


def plot_cell_rf_time(
    ax_rf,
    ax_time,
    title,
    series,
    *,
    time_title=None,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    v_th=None,
    e_leak=None,
    n_t=None,
    t_onset=None,
    show_pre=False,
    t_sti_end=None,
    t_delay=0,
    delta_ms,
    delta_ms_pre,
    ms_shown=None,
):
    """RF + time panels for one cell (composes ``plot_cell_rf`` + ``plot_cell_time``)."""
    plot_cell_rf(
        ax_rf, title, series,
        show_legend=show_legend,
        show_xlabels=show_xlabels,
        show_ylabel=show_ylabel,
        t_onset=t_onset,
        t_sti_end=t_sti_end,
        t_delay=t_delay,
    )
    plot_cell_time(
        ax_time, series,
        title=time_title,
        show_xlabels=show_xlabels,
        show_ylabel=show_ylabel,
        v_th=v_th,
        e_leak=e_leak,
        n_t=n_t,
        t_onset=t_onset,
        show_pre=show_pre,
        t_sti_end=t_sti_end,
        t_delay=t_delay,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        ms_shown=ms_shown,
    )


def plot_cell_rf_time_overlays(
    ax_rf,
    ax_time,
    title,
    series,
    overlay_labels,
    *,
    time_title=None,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    v_th=None,
    e_leak=None,
    n_t=None,
    t_onset=None,
    show_pre=False,
    t_sti_end=None,
    t_delay=0,
    delta_ms,
    delta_ms_pre,
    ms_shown=None,
):
    """RF + time panels with per-overlay curves across contrast ``series``."""
    center_radius = RF_CENTER_RADIUS
    sc_kw = dict(t_onset=t_onset, t_sti_end=t_sti_end, t_delay=t_delay)
    colors = overlay_reds(len(overlay_labels))
    t = np.arange(n_t)
    pre_end = int(t_onset or 0)
    mark_spot(ax_time, t_onset, t_sti_end)

    for si, item in enumerate(series):
        ls = item.get("linestyle", "-")
        v_readout = item.get("v_readout")
        gt = item.get("gt")
        overlay = item.get("overlay") or {}
        v_readout_center = v_readout_spatial = None
        if v_readout is not None:
            v_readout_center, v_readout_spatial, _ = scale_curve(
                v_readout, center_radius, **sc_kw,
            )
        gt_center = gt_spatial = None
        if gt is not None:
            gt_center, gt_spatial, _ = scale_curve(gt, center_radius, **sc_kw)
        overlay_centers = {}
        overlay_spatials = {}
        for label in overlay_labels:
            if label in overlay:
                overlay_center, overlay_spatial, _ = scale_curve(overlay[label], center_radius, **sc_kw)
                overlay_centers[label] = overlay_center
                overlay_spatials[label] = overlay_spatial

        _plot_rf_profile(
            ax_rf, gt_spatial, color=GT_COLOR,
            label=item.get("label_gt"), linestyle=ls,
        )
        for i, label in enumerate(overlay_labels):
            _plot_rf_profile(
                ax_rf, overlay_spatials.get(label), color=colors[i],
                label=label if si == 0 else None, linestyle=ls,
            )
        _plot_rf_profile(
            ax_rf, v_readout_spatial, color=colors[-1],
            label=item.get("label_scope"), linestyle=ls, filled=True,
        )

        plot_pre_post_line(
            ax_time, t, gt_center, pre_end=pre_end, show_pre=False, plot_pre=False,
            color=GT_COLOR, linestyle=ls, linewidth=TRACE_LW,
        )
        for i, label in enumerate(overlay_labels):
            plot_pre_post_line(
                ax_time, t, overlay_centers.get(label), pre_end=pre_end,
                show_pre=show_pre, plot_pre=True,
                color=colors[i], linestyle=ls, linewidth=TRACE_LW,
                label=label if si == 0 else None,
            )
        plot_pre_post_line(
            ax_time, t, v_readout_center, pre_end=pre_end, show_pre=show_pre, plot_pre=True,
            color=colors[-1], linestyle=ls, linewidth=TRACE_LW,
            label=item.get("label_scope"),
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


def _fill_member_v_readout(v_readout, std, ti, ft_global, type_idx, du, dv, figure_traces):
    """Fill profile radii from hex-lattice radius means; return ``{radius: n_entry}``."""
    mask = type_idx == ft_global
    if not mask.any():
        return {}
    entries = np.where(mask)[0]
    by_radius: dict[int, list] = {}
    for entry in entries:
        radius = int(build_hex.hex_radius(int(du[entry]), int(dv[entry])))
        by_radius.setdefault(radius, []).append(int(entry))
    n_by_radius = {}
    for radius, grouped_entries in by_radius.items():
        if radius < 0 or radius >= RF_N_RADII:
            continue
        traces = figure_traces[grouped_entries]
        v_readout[ti, radius] = traces.mean(axis=0)
        std[ti, radius] = std_from_traces(traces, single_hex=False)
        n_by_radius[radius] = len(grouped_entries)
    return n_by_radius


@dataclass
class SpotTraceReadout:
    """One forward pass; full cost-radius readout over all types."""

    cells: list
    rows: list | None = None
    session: object = None
    overlay: dict[str, dict[str, np.ndarray]] | None = None
    overlay_labels: list[str] | None = None
    overlay_xs: list | None = None
    overlay_ys: list | None = None
    n_t: int = 0
    prep_s: float = 0.0
    v_readout_by_cell: dict = field(default_factory=dict)
    std_by_cell: dict = field(default_factory=dict)
    n_by_cell: dict = field(default_factory=dict)
    v_th_by_cell: dict = field(default_factory=dict)
    e_leak_by_cell: dict = field(default_factory=dict)
    gt_affine_by_cell: dict = field(default_factory=dict)
    t_onset: int | None = None
    show_pre: bool = True
    t_sti_end: int | None = None
    ms_shown: tuple[float, float] | None = None
    center_only: bool = False
    a_sti_radius: dict[str, float] = field(default_factory=dict)

    @property
    def has_overlays(self):
        return bool(self.overlay)


def _rows_from_cell_rows(cell_rows, figure_cells):
    cell_idx = {str(n): i for i, n in enumerate(figure_cells)}
    rows = []
    for cells_in_row in cell_rows:
        cell_idxs = [cell_idx[str(n)] for n in cells_in_row if str(n) in cell_idx]
        if cell_idxs:
            rows.append(cell_idxs)
    return rows


def _spot_readout_gt_view(readout):
    """Gt figure rows: configured active gt cells (not cost-pack-only)."""
    session = readout.session
    active = active_spot_gt_cells(session, session.primary_pack.task)
    rows = [np.array(row) for row in cell_rows(active)]
    present = set(readout.cells)
    cells = [cell for cell in cells_in_order(active) if cell in present]
    return SpotTraceReadout(
        cells=cells,
        rows=_rows_from_cell_rows(rows, cells),
        session=session,
        n_t=readout.n_t,
        v_readout_by_cell={
            cell: readout.v_readout_by_cell[cell] for cell in cells
            if cell in readout.v_readout_by_cell
        },
        std_by_cell={
            cell: readout.std_by_cell[cell] for cell in cells
            if cell in readout.std_by_cell
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
        show_pre=readout.show_pre,
        t_sti_end=readout.t_sti_end,
        ms_shown=readout.ms_shown,
        center_only=bool(readout.center_only),
        a_sti_radius=dict(readout.a_sti_radius),
    )


def _layout_cells_from_readouts(readouts, order):
    """Union of contrast cells in biological order."""
    seen = set()
    for contrast in order:
        seen.update(readouts[contrast].cells)
    cells = cells_in_order(list(seen))
    rows = [np.array(row) for row in cell_rows(cells)]
    return cells, _rows_from_cell_rows(rows, cells)


def _spot_v_readout_from_readout_mask(readout, mask):
    figure_cells = readout['figure_cells']
    cells = readout['cells']
    mt = readout['mt']
    type_idx = readout['type_idx'][mask]
    du = readout['du'][mask]
    dv = readout['dv'][mask]
    figure_traces = readout['figure_traces'][mask]
    out = {}
    std_dummy = np.full((1, RF_N_RADII, mt), np.nan)
    for ft in figure_cells:
        v_readout = np.full((1, RF_N_RADII, mt), np.nan)
        ft_global = cells.index(ft)
        _fill_member_v_readout(v_readout, std_dummy, 0, ft_global, type_idx, du, dv, figure_traces)
        out[ft] = v_readout[0]
    return out


def _spot_readout_hex_mask(connectome, nodes, cost_radius, *, at_x=None, at_y=None):
    """True for cost entries whose node sits on matching cost-radius hexes."""
    filtered_hexes = filter_sti_hexes(
        moving_bar_cost_hexes(connectome, cost_radius=cost_radius),
        at_x=at_x,
        at_y=at_y,
    )
    if not filtered_hexes:
        return np.zeros(len(nodes), dtype=bool)
    node_u_np, node_v_np = network_uv_np(connectome)
    hex_uv = {(int(hex.u), int(hex.v)) for hex in filtered_hexes}
    return np.array(
        [(int(node_u_np[n]), int(node_v_np[n])) in hex_uv for n in nodes],
        dtype=bool,
    )


def _spot_overlay(readout, connectome, at_xs, at_ys):
    """Per-hex readout overlays (same ``filter_sti_hexes`` scope as moving_bar)."""
    overlay = {}
    labels = []
    pack = readout['pack']
    nodes = readout['nodes']
    for label, at_x, at_y in overlay_coords(at_xs, at_ys):
        mask = _spot_readout_hex_mask(
            connectome, nodes, pack.cost_radius, at_x=at_x, at_y=at_y,
        )
        if not np.any(mask):
            print(f'skip overlay {label}: no hex within cost_radius')
            continue
        v_readout_by_cell = _spot_v_readout_from_readout_mask(readout, mask)
        if not any(np.isfinite(v_readout).any() for v_readout in v_readout_by_cell.values()):
            print(f'skip overlay {label}: no readouts')
            continue
        overlay[label] = v_readout_by_cell
        labels.append(label)
    if not overlay:
        return None, None
    return overlay, labels



@torch.no_grad()
def _forward_spot_readout(
    session, z, *,
    at_x=None, at_y=None,
):
    """One forward; cost-radius node readout over all network types."""
    pack = session.primary_pack
    schema = train.schema_copy(session.schema)
    params = train.override_val_from(
        train.assign_params(z, schema, session.backend), session,
    )
    a_sti_radius = {}
    if "a_sti_radius" in params:
        spec = schema.get("a_sti_radius")
        if spec is not None:
            radii = [str(n) for n in (spec.get("radii") or ())]
            raw = params["a_sti_radius"].detach().cpu().numpy().reshape(-1)
            a_sti_radius = {
                radii[i]: float(raw[i]) for i in range(min(len(radii), raw.size))
            }
    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    v = train.forward_v(session, params, i_sti, pack=pack)
    t0 = train.pack_t_onset(pack)
    if str((session.train_opts or {}).get("filter", "none")) == "ca":
        v_ca = train.v_ca_from_v(v, params, session)
        trace_full = train.ca_from_v_ca(v_ca, params, session, t_onset=t0)
    else:
        trace_full = v
    train.override_val_from(params, session, onset_trace=trace_full, t_onset=t0)
    connectome = session.backend.network
    cells = list(connectome.cells)
    mt = int(i_sti.shape[1])

    opts = dict((session.train_opts or {}).get(f"{pack.task}_sti_opts") or {})
    spot = resolve_spot(connectome, sti_opts=opts)
    spot_bs = spot_sti_bs(spot)
    active = cells_in_order(connectome.cells)
    rows = [np.array(row) for row in cell_rows(active)]
    figure_cells = cells_in_order(active)

    (
        bs, nodes, _radius, type_idx, _sti_u, _sti_v, du, dv, center_entry_mask,
    ) = build_spot_center_readout(
        connectome, spot_bs, pack_spot_cost_radii(pack), pack.cost_radius,
    )

    figure_traces = trace_full[bs, :, nodes].cpu().numpy()

    sti_ms_pre = opts.get("ms_pre")
    dt = float(opts["delta_ms"])
    dt_pre = float(opts["delta_ms_pre"])
    sti_t_onset = (
        t_from_ms(float(sti_ms_pre), delta_ms=dt_pre)
        if sti_ms_pre is not None else None
    )
    filter = str((session.train_opts or {}).get("filter", "none"))
    timing = resolve_sti_timing(opts)
    readout = dict(
        figure_cells=figure_cells,
        cells=cells,
        type_idx=type_idx,
        nodes=nodes,
        du=du,
        dv=dv,
        center_entry_mask=center_entry_mask,
        figure_traces=figure_traces,
        t_onset=int(sti_t_onset) if sti_t_onset is not None else None,
        t_sti_end=(
            None if sti_t_onset is None else t_sti_end(
                sti_t_onset, mt, timing.ms_sti,
                delta_ms=timing.delta_ms,
            )
        ),
        bs=bs,
        spot_bs=spot_bs,
        mt=mt,
        pack=pack,
        a_sti_radius=a_sti_radius,
    )
    mask = np.asarray(center_entry_mask, dtype=bool)
    if at_x is not None or at_y is not None:
        mask = mask & _spot_readout_hex_mask(
            connectome, nodes, pack.cost_radius, at_x=at_x, at_y=at_y,
        )
    nodes_by_cell = {
        cell: np.unique(nodes[mask & (type_idx == cells.index(cell))])
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
        cell: gt_affine_from_cell(
            params, cell, session.backend, session=session,
        )
        for cell in figure_cells
    }
    readout['cell_rows'] = rows
    return readout


def _spot_v_readout_from_readout(readout, session):
    figure_cells = list(readout['figure_cells'])
    mt = readout['mt']
    v_stack = np.full((len(figure_cells), RF_N_RADII, mt), np.nan)
    std_stack = np.full((len(figure_cells), RF_N_RADII, mt), np.nan)
    n_by_cell = {}
    for ti, ft in enumerate(figure_cells):
        ft_global = readout['cells'].index(ft)
        n_by_cell[ft] = _fill_member_v_readout(
            v_stack, std_stack, ti, ft_global,
            readout['type_idx'], readout['du'], readout['dv'], readout['figure_traces'],
        )
    single_hex = suppress_cost_std(session, task=readout['pack'].task)
    v_readout_by_cell = {
        ft: v_stack[ti] for ti, ft in enumerate(figure_cells)
    }
    std_by_cell = (
        {} if single_hex else {
            ft: std_stack[ti] for ti, ft in enumerate(figure_cells)
        }
    )
    rows = _rows_from_cell_rows(readout['cell_rows'], figure_cells)
    return (
        figure_cells, rows, mt,
        v_readout_by_cell, std_by_cell, n_by_cell,
    )


@torch.no_grad()
def network_spot_trace_readout(
    session, z, *,
    at_xs=None, at_ys=None,
    show_pre=True,
    ms_shown=None,
    center_only=False,
):
    """Run one forward; full cost-radius spot traces over all types."""
    t_prep0 = time.perf_counter()
    at_x = at_xs[0] if at_xs else None
    at_y = at_ys[0] if at_ys else None
    readout = _forward_spot_readout(
        session, z,
        at_x=at_x, at_y=at_y,
    )
    (
        cells, rows, n_t,
        v_readout_by_cell, std_by_cell, n_by_cell,
    ) = _spot_v_readout_from_readout(readout, session)
    overlay, overlay_labels = (None, None)
    if at_xs is not None or at_ys is not None:
        connectome = session.backend.network
        if connectome is None:
            raise ValueError("spot overlay requires a network backend")
        overlay, overlay_labels = _spot_overlay(
            readout, connectome, at_xs, at_ys,
        )
    return SpotTraceReadout(
        cells=cells,
        rows=rows,
        session=session,
        overlay=overlay,
        overlay_labels=overlay_labels,
        overlay_xs=at_xs,
        overlay_ys=at_ys,
        n_t=n_t,
        prep_s=time.perf_counter() - t_prep0,
        v_readout_by_cell=v_readout_by_cell,
        std_by_cell=std_by_cell,
        n_by_cell=n_by_cell,
        v_th_by_cell=dict(readout.get('v_th_by_cell') or {}),
        e_leak_by_cell=dict(readout.get('e_leak_by_cell') or {}),
        gt_affine_by_cell=dict(readout.get('gt_affine_by_cell') or {}),
        t_onset=readout.get('t_onset'),
        show_pre=bool(show_pre),
        t_sti_end=readout.get('t_sti_end'),
        ms_shown=ms_shown,
        center_only=bool(center_only),
        a_sti_radius=dict(readout.get('a_sti_radius') or {}),
    )


def _spot_suptitle(title, readout):
    head = title
    if readout is not None:
        a_sti_radius = readout.a_sti_radius
        if a_sti_radius and "1" in a_sti_radius:
            head = f"a_sti_radius 1 = {float(a_sti_radius['1']):.4g}"
        if readout.has_overlays:
            scope = hex_scope_tag(readout.overlay_xs, readout.overlay_ys)
            return f'{head}  [{scope}, overlay + scope]'
    return head


def _trained_radii(cost_parts, contrasts, *, center_only=False):
    """Sorted integer radii from spot cost parts (``spot_{contrast}_{cell}_r*``)."""
    if center_only or not cost_parts:
        return [int(RF_CENTER_RADIUS)]
    out = set()
    for contrast in contrasts:
        prefix = f"spot_{contrast}_"
        for part_key in cost_parts:
            if not part_key.startswith(prefix):
                continue
            pos = part_key.rfind("_r")
            if pos < 0:
                continue
            radius_token = part_key[pos + 2:]
            try:
                r_f = float(radius_token)
            except ValueError:
                continue
            r_i = int(round(r_f))
            if abs(r_f - r_i) > 1e-6:
                continue
            if 0 <= r_i < RF_N_RADII:
                out.add(r_i)
    if not out:
        return [int(RF_CENTER_RADIUS)]
    return sorted(out)


def _plot_spot_figure(
    path, *,
    timer,
    readouts,
    title,
    gts=None,
    n_col,
    figsize_fn,
    gridspec_kw,
    suptitle_fs=12,
    cost_parts=None,
):
    """Draw spot figure from ``readouts`` (contrast → SpotTraceReadout)."""
    order = contrast_order(readouts)
    if not order:
        raise ValueError("_plot_spot_figure requires at least one readout")
    primary = readouts[order[0]]
    cells, rows = _layout_cells_from_readouts(readouts, order)
    has_overlays = primary.has_overlays
    overlay_labels = primary.overlay_labels or []
    n_t = primary.n_t
    t_onset = primary.t_onset
    t_sti_end = primary.t_sti_end
    delta_ms = float(primary.session.delta_ms)
    delta_ms_pre = float(primary.session.delta_ms_pre)
    show_pre = primary.show_pre
    ms_shown = primary.ms_shown
    t_delay = t_delay_from_ir(
        delta_ms=delta_ms,
        filter=session_filter_figure_token(primary.session),
    )
    gt_cell_idx = {name: i for i, name in enumerate(GT_CELLS)}

    def _t_delay_from_cell(name):
        idx = gt_cell_idx.get(name)
        return 0 if idx is None else int(t_delay[idx])

    timer.end_prep()

    sessions = {contrast: readouts[contrast].session for contrast in order}
    gt_by_contrast = resolve_spot_gts(sessions, gts)

    center_only = bool(primary.center_only)
    radii = _trained_radii(cost_parts, order, center_only=center_only)
    center_radius = int(RF_CENTER_RADIUS)
    order_hs = [
        1 + len(radii) for _ in rows
    ]
    n_row = int(sum(order_hs))
    fig = plt.figure(figsize=figsize_fn(n_col, n_row))
    gs = fig.add_gridspec(n_row, n_col, **gridspec_kw)
    legend_done = False

    def _series_from_cell(cell, *, with_overlays):
        series = []
        for contrast in order:
            ro = readouts[contrast]
            if cell not in ro.v_readout_by_cell:
                continue
            gt_by_cell = gt_by_contrast.get(contrast) or {}
            entry = {
                "contrast": contrast,
                "v_readout": ro.v_readout_by_cell[cell],
                "gt": gt_trace_affine(ro, cell, gt_by_cell.get(cell)),
                "std": ro.std_by_cell.get(cell),
                "v_th": ro.v_th_by_cell.get(cell),
                "linestyle": contrast_linestyle(contrast),
                "label_gt": f"{contrast} gt",
                "label_v_readout": (
                    f"{contrast} {session_filter_figure_token(primary.session)}"
                ),
                "label_scope": f"{contrast} scope",
            }
            if with_overlays:
                overlay = ro.overlay
                if overlay is not None:
                    entry["overlay"] = {
                        label: v_readout_by_cell[cell]
                        for label, v_readout_by_cell in overlay.items()
                        if cell in v_readout_by_cell
                    }
                else:
                    entry["overlay"] = {}
            series.append(entry)
        return series

    def _plot_cell(cell, ax_rf, ax_time, show_ylabel, show_xlabels):
        nonlocal legend_done
        n_by_radius = primary.n_by_cell.get(cell) or {}
        time_title = format_spot_radius_time_title(
            center_radius,
            n_by_radius.get(center_radius),
            cell,
            cost_parts,
            order,
        )
        v_th = primary.v_th_by_cell.get(cell)
        e_leak = primary.e_leak_by_cell.get(cell)
        if has_overlays and primary.overlay is not None:
            series = _series_with_cost_points(
                _series_from_cell(cell, with_overlays=True),
                readouts,
                float(center_radius),
            )
            overlay = (series[0].get("overlay") or {}) if series else {}
            if not overlay:
                ax_rf.axis("off")
                ax_time.axis("off")
                return
            plot_cell_rf_time_overlays(
                ax_rf, ax_time, cell, series, overlay_labels,
                time_title=time_title,
                show_legend=not legend_done,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                n_t=n_t,
                v_th=v_th,
                e_leak=e_leak,
                t_onset=t_onset,
                show_pre=show_pre,
                t_sti_end=t_sti_end,
                t_delay=_t_delay_from_cell(cell),
                delta_ms=delta_ms,
                delta_ms_pre=delta_ms_pre,
                ms_shown=ms_shown,
            )
        else:
            series = _series_with_cost_points(
                _series_from_cell(cell, with_overlays=False),
                readouts,
                float(center_radius),
            )
            plot_cell_rf_time(
                ax_rf, ax_time, cell, series,
                time_title=time_title,
                show_legend=not legend_done,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                n_t=n_t,
                v_th=v_th,
                e_leak=e_leak,
                t_onset=t_onset,
                show_pre=show_pre,
                t_sti_end=t_sti_end,
                t_delay=_t_delay_from_cell(cell),
                delta_ms=delta_ms,
                delta_ms_pre=delta_ms_pre,
                ms_shown=ms_shown,
            )
        legend_done = True

    row_cursor = 0
    for gi, cell_idxs in enumerate(rows):
        group_h = int(order_hs[gi])
        rf_row = row_cursor
        time_row0 = row_cursor + 1
        start = (n_col - len(cell_idxs)) // 2
        for j, ci in enumerate(cell_idxs):
            col = start + j
            cell = cells[ci]
            ax_rf = fig.add_subplot(gs[rf_row, col])
            if has_overlays:
                ax_time = fig.add_subplot(gs[time_row0, col])
                _plot_cell(
                    cell, ax_rf, ax_time,
                    show_ylabel=(j == 0), show_xlabels=True,
                )
                for rr in range(time_row0 + 1, row_cursor + group_h):
                    ax_off = fig.add_subplot(gs[rr, col])
                    ax_off.axis("off")
                continue

            series = _series_from_cell(cell, with_overlays=False)
            show_legend = not legend_done
            plot_cell_rf(
                ax_rf, cell, series,
                show_legend=show_legend,
                show_xlabels=False,
                show_ylabel=(j == 0),
                t_onset=t_onset,
                t_sti_end=t_sti_end,
                t_delay=_t_delay_from_cell(cell),
            )
            legend_done = True
            ylim0 = None
            n_by_radius = primary.n_by_cell.get(cell) or {}
            v_th = primary.v_th_by_cell.get(cell)
            e_leak = primary.e_leak_by_cell.get(cell)
            for local_i, rr in enumerate(range(time_row0, row_cursor + group_h)):
                ax_t = fig.add_subplot(gs[rr, col])
                if local_i >= len(radii):
                    ax_t.axis("off")
                    continue
                radius = int(radii[local_i])
                time_title = format_spot_radius_time_title(
                    radius, n_by_radius.get(radius), cell, cost_parts, order,
                )
                plot_cell_time(
                    ax_t,
                    _series_with_cost_points(series, readouts, float(radius)),
                    title=time_title,
                    show_xlabels=(local_i == len(radii) - 1),
                    show_ylabel=(j == 0),
                    v_th=v_th,
                    e_leak=e_leak,
                    n_t=n_t,
                    t_onset=t_onset,
                    show_pre=show_pre,
                    t_sti_end=t_sti_end,
                    t_delay=_t_delay_from_cell(cell),
                    delta_ms=delta_ms,
                    delta_ms_pre=delta_ms_pre,
                    ms_shown=ms_shown,
                    center_radius=radius,
                )
                if local_i == 0:
                    ylim0 = ax_t.get_ylim()
                elif ylim0 is not None:
                    ax_t.set_ylim(*ylim0)
        row_cursor += group_h
    fig.suptitle(_spot_suptitle(title, primary), fontsize=suptitle_fs)
    timer.end_draw()
    save_figure(fig, path, dpi=150, timer=timer)


def plot_network_spot_gt(path, *, readouts, title, gts=None, cost_parts=None):
    """Draw gt figure (active gt cells; model ca for all active)."""
    views = {
        contrast: _spot_readout_gt_view(readout)
        for contrast, readout in readouts.items()
    }
    _plot_spot_figure(
        path,
        timer=ElapsedTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts=views,
        title=title,
        gts=gts,
        n_col=N_COL_GT,
        figsize_fn=lambda n_col, n_row: (PANEL_W * n_col, 2.5 * n_row),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
        cost_parts=cost_parts,
    )


def plot_network_spot_all(path, *, readouts, title, gts=None, cost_parts=None):
    """Draw ca-all figure (all types) from contrast → readout."""
    _plot_spot_figure(
        path,
        timer=ElapsedTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts=readouts,
        title=title,
        gts=gts,
        n_col=N_COL_ALL,
        figsize_fn=lambda n_col, n_row: (PANEL_W * n_col, 2.5 * n_row),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
        cost_parts=cost_parts,
    )
