"""Spot plotting (network spot task).

Network RF profile axis is Euclidean radius: v_readout[..., radius] = mean at that radius.
"""

from __future__ import annotations

from default_params import (
    NEURON_PARAM,
)

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

from default_params import NEURON_PARAM
import train
from neuron.param import t_from_ms, t_abs_from_ms, ms_from_t
from figure.gt import (
    contrast_linestyle,
    contrast_order,
    active_spot_gt_cells,
    plot_cells_in_order,
    spot_gts,
)
from figure.util import (
    N_COL_ALL,
    N_COL_GT,
    PANEL_W,
    GT_COLOR,
    V_READOUT_COLOR,
    TRACE_LW,
    PlotTimer,
    annotate_v_th,
    params_for_types,
    readout_prep_s,
    format_spot_radius_time_title,
    gt_affine_scalars_for_cell,
    e_leak_by_type_name,
    session_filter_plot_token,
    hex_at_scope_tag,
    mark_spot,
    overlay_v_readout_reds,
    plot_pre_post_line,
    plot_timecourse,
    save_figure,
    std_from_traces,
    slice_coord_specs,
    suppress_cost_std,
    v_th_by_type_name,
)
from network.construction import cell_plot_rows, cells_in_order
from task.moving_bar.sti_geo import (
    filter_sti_hexes,
    moving_bar_cost_hexes,
    network_uv_np,
)
from task.spot.sti_geo import (
    euclid_hex_dist,
    resolve_spot,
    spot_sti_batches,
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


def pack_spot_cost_radii(pack) -> tuple[float, ...]:
    """Active Euclidean cost radii from ``pack.entry_radii``."""
    if pack.entry_radii is None:
        raise ValueError(f"{pack.name} pack missing entry_radii")
    return tuple(
        sorted({round(float(radius), 6) for radius in pack.entry_radii.tolist()})
    )


def _session_spot_timing(session):
    """Extract onset ``t_onset`` / forward ``n_t``, ms_sti, and delta_ms from session."""
    opts = (session.train_opts or {}).get(
        f"{session.primary_pack.name}_sti_opts",
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
            session, session.primary_pack.name, contrasts=(str(contrast),),
            t_onset=t_onset, n_t=n_t_gt, ms_sti=ms_sti, delta_ms=delta_ms,
            filter=filter,
        )
        out.update(part)
    return out


def _session_cost_time_idx(session, t_onset, *, entry_radius=None):
    """Absolute time indices for sparse spot cost (pack contract; or ``None``)."""
    if session is None:
        return None
    return train.pack_cost_abs_time_idx(
        session.primary_pack, t_onset, entry_radius=entry_radius,
    )


def _series_with_cost_points(series, readouts, entry_radius):
    """Copy ``series`` with ``point_idx`` for one Euclidean ``entry_radius``."""
    out = []
    for entry in series:
        item = dict(entry)
        c = entry["contrast"]
        item["point_idx"] = _session_cost_time_idx(
            readouts[c].session, readouts[c].t_onset, entry_radius=entry_radius,
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
    """Scale each contrast series entry to gt / v_readout plot slices.

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
            "point_idx": item.get("point_idx"),
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


def plot_cell_rf_time_slices(
    ax_rf,
    ax_time,
    title,
    series,
    slice_labels,
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
    """RF + time panels with per-slice overlays across contrast ``series``."""
    center_radius = RF_CENTER_RADIUS
    sc_kw = dict(t_onset=t_onset, t_sti_end=t_sti_end, t_delay=t_delay)
    colors = overlay_v_readout_reds(len(slice_labels))
    t = np.arange(n_t)
    pre_end = int(t_onset or 0)
    mark_spot(ax_time, t_onset, t_sti_end)

    for si, item in enumerate(series):
        ls = item.get("linestyle", "-")
        v_readout = item.get("v_readout")
        gt = item.get("gt")
        slice_overlay = item.get("slice_overlay") or {}
        v_readout_center = v_readout_spatial = None
        if v_readout is not None:
            v_readout_center, v_readout_spatial, _ = scale_curve(
                v_readout, center_radius, **sc_kw,
            )
        gt_center = gt_spatial = None
        if gt is not None:
            gt_center, gt_spatial, _ = scale_curve(gt, center_radius, **sc_kw)
        slice_centers = {}
        slice_spatials = {}
        for label in slice_labels:
            if label in slice_overlay:
                center_s, spatial_s, _ = scale_curve(slice_overlay[label], center_radius, **sc_kw)
                slice_centers[label] = center_s
                slice_spatials[label] = spatial_s

        _plot_rf_profile(
            ax_rf, gt_spatial, color=GT_COLOR,
            label=item.get("label_gt"), linestyle=ls,
        )
        for i, label in enumerate(slice_labels):
            _plot_rf_profile(
                ax_rf, slice_spatials.get(label), color=colors[i],
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
        for i, label in enumerate(slice_labels):
            plot_pre_post_line(
                ax_time, t, slice_centers.get(label), pre_end=pre_end,
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


def _fill_member_v_readout(v_readout, std, ti, ft_global, type_idx, du, dv, plot_traces):
    """Fill profile radii from Euclidean radius means; return ``{r: n_entry}``."""
    mask = type_idx == ft_global
    if not mask.any():
        return {}
    entry_idxs = np.where(mask)[0]
    by_radius: dict[float, list] = {}
    for entry_idx in entry_idxs:
        radius = round(float(euclid_hex_dist(int(du[entry_idx]), int(dv[entry_idx]))), 6)
        by_radius.setdefault(radius, []).append(int(entry_idx))
    n_by_radius = {}
    for radius, grouped_entry_idxs in by_radius.items():
        k = int(round(float(radius)))
        if abs(float(radius) - k) > 1e-6 or k < 0 or k >= RF_N_RADII:
            continue
        traces = plot_traces[grouped_entry_idxs]
        v_readout[ti, k] = traces.mean(axis=0)
        std[ti, k] = std_from_traces(traces, single_hex=False)
        n_by_radius[k] = len(grouped_entry_idxs)
    return n_by_radius


@dataclass
class SpotTraceReadout:
    """One forward pass; full cost-radius readout over all types."""

    cells: list
    plot_row_idxs: list | None = None
    session: object = None
    slice_overlay: dict[str, dict[str, np.ndarray]] | None = None
    slice_labels: list[str] | None = None
    slice_xs: list | None = None
    slice_ys: list | None = None
    n_t: int = 0
    prep_s: float = 0.0
    v_th_by_name: dict = field(default_factory=dict)
    t_onset: int | None = None
    show_pre: bool = True
    t_sti_end: int | None = None
    ms_shown: tuple[float, float] | None = None
    center_only: bool = False
    a_sti_radius: dict[str, float] = field(default_factory=dict)

    @property
    def has_slices(self):
        return bool(self.slice_overlay)


def _plot_row_idxs_from_cell_plot_rows(cell_plot_rows_list, names):
    i_from_name = {str(n): i for i, n in enumerate(names)}
    plot_row_idxs = []
    for names_in_row in cell_plot_rows_list:
        cell_idxs = [i_from_name[str(n)] for n in names_in_row if str(n) in i_from_name]
        if cell_idxs:
            plot_row_idxs.append(cell_idxs)
    return plot_row_idxs


def _spot_readout_gt_view(readout):
    """Gt figure rows: configured active gt cells (not cost-pack-only)."""
    session = readout.session
    active = active_spot_gt_cells(session, session.primary_pack.name)
    cell_plot_rows_list = [np.array(plot_row) for plot_row in cell_plot_rows(active)]
    names = cells_in_order(active)
    by_name = {c['name']: c for c in readout.cells}
    cells = [by_name[n] for n in names if n in by_name]
    return SpotTraceReadout(
        cells=cells,
        plot_row_idxs=_plot_row_idxs_from_cell_plot_rows(
            cell_plot_rows_list, [c['name'] for c in cells],
        ),
        session=session,
        n_t=readout.n_t,
        v_th_by_name=readout.v_th_by_name,
        t_onset=readout.t_onset,
        show_pre=readout.show_pre,
        t_sti_end=readout.t_sti_end,
        ms_shown=readout.ms_shown,
        center_only=bool(readout.center_only),
        a_sti_radius=dict(readout.a_sti_radius),
    )


def _layout_cells_from_readouts(readouts, order):
    """Union of contrast cells in biological order; metadata from first hit."""
    by_name = {}
    for contrast in order:
        for cell in readouts[contrast].cells:
            name = cell["name"]
            if name not in by_name:
                by_name[name] = cell
    names = cells_in_order(list(by_name))
    cells = [by_name[n] for n in names]
    cell_plot_rows_list = [np.array(plot_row) for plot_row in cell_plot_rows(names)]
    plot_row_idxs = _plot_row_idxs_from_cell_plot_rows(
        cell_plot_rows_list, [c["name"] for c in cells],
    )
    return cells, plot_row_idxs


def _spot_v_readouts_from_readout_mask(readout, mask):
    names = readout['names']
    cells = readout['cells']
    mt = readout['mt']
    type_idx = readout['type_idx'][mask]
    du = readout['du'][mask]
    dv = readout['dv'][mask]
    plot_traces = readout['plot_traces'][mask]
    out = {}
    std_dummy = np.full((1, RF_N_RADII, mt), np.nan)
    for ft in names:
        v_readout = np.full((1, RF_N_RADII, mt), np.nan)
        ft_global = cells.index(ft)
        _fill_member_v_readout(v_readout, std_dummy, 0, ft_global, type_idx, du, dv, plot_traces)
        out[ft] = v_readout[0]
    return out


def _spot_readout_hex_mask(connectome, node_idx, cost_radius, *, at_x=None, at_y=None):
    """True for cost entries whose node sits on matching cost-radius hexes."""
    filtered_hexes = filter_sti_hexes(
        moving_bar_cost_hexes(connectome, cost_radius=cost_radius),
        at_x=at_x,
        at_y=at_y,
    )
    if not filtered_hexes:
        return np.zeros(len(node_idx), dtype=bool)
    node_u_np, node_v_np = network_uv_np(connectome)
    hex_uv = {(int(c.u), int(c.v)) for c in filtered_hexes}
    return np.array(
        [(int(node_u_np[n]), int(node_v_np[n])) in hex_uv for n in node_idx],
        dtype=bool,
    )


def _spot_slice_overlay(readout, connectome, at_xs, at_ys):
    """Per-hex readout overlays (same ``filter_sti_hexes`` scope as moving_bar)."""
    overlay = {}
    labels = []
    pack = readout['pack']
    node_idx = readout['node_idx']
    for label, at_x, at_y in slice_coord_specs(at_xs, at_ys):
        mask = _spot_readout_hex_mask(
            connectome, node_idx, pack.cost_radius, at_x=at_x, at_y=at_y,
        )
        if not np.any(mask):
            print(f'skip slice overlay {label}: no hex within cost_radius')
            continue
        v_readouts = _spot_v_readouts_from_readout_mask(readout, mask)
        if not any(np.isfinite(c).any() for c in v_readouts.values()):
            print(f'skip slice overlay {label}: no readouts')
            continue
        overlay[label] = v_readouts
        labels.append(label)
    if not overlay:
        return None, None
    return overlay, labels



def _cells_from_v_readout(
    names, v_readout, std, v_th_by_name, *, single_hex,
    e_leaks=None,
    n_by_radius_by_name=None, gt_affine_by_name=None,
):
    n_by_radius_by_name = n_by_radius_by_name or {}
    gt_affine_by_name = gt_affine_by_name or {}
    e_leaks = e_leaks or {}
    out = []
    for i, n in enumerate(names):
        a_gt, bias = gt_affine_by_name.get(n, (1.0, 0.0))
        out.append(dict(
            name=n,
            v_readout=v_readout[i],
            std=None if single_hex else std[i],
            v_th=v_th_by_name.get(n),
            e_leak=e_leaks.get(n),
            n_by_radius=dict(n_by_radius_by_name.get(n) or {}),
            a_gt=float(a_gt),
            bias_gt=float(bias),
        ))
    return out


@torch.no_grad()
def _forward_spot_readout(
    session, z, *,
    at_x=None, at_y=None,
):
    """One forward; cost-radius node readout over all network types."""
    pack = session.primary_pack
    schema = list(session.schema)
    params = train.params_from_opts(
        train.assign_params(z, schema, session.backend), session,
    )
    a_sti_radius_by_name = {}
    if "a_sti_radius" in params:
        for segment in schema:
            if segment.get("name") != "a_sti_radius":
                continue
            names = [str(n) for n in (segment.get("node_names") or ())]
            raw = params["a_sti_radius"].detach().cpu().numpy().reshape(-1)
            a_sti_radius_by_name = {
                names[i]: float(raw[i]) for i in range(min(len(names), raw.size))
            }
            break
    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    v = train.forward_v(session, params, i_sti, pack=pack)
    t0 = train.pack_t_onset(pack)
    if str((session.train_opts or {}).get("filter", "none")) == "ca":
        v_ca = train.v_ca_from_v(v, params, session)
        plot_full = train.ca_from_v_ca(v_ca, params, session, t_onset=t0)
    else:
        plot_full = v
    train.params_from_opts(params, session, onset_trace=plot_full, t_onset=t0)
    connectome = session.backend.network
    cells = list(connectome.cells)
    mt = int(i_sti.shape[1])

    opts = dict((session.train_opts or {}).get(f"{pack.name}_sti_opts") or {})
    spot = resolve_spot(connectome, sti_opts=opts)
    batches = spot_sti_batches(spot)
    active = plot_cells_in_order(connectome.cells)
    cell_plot_rows_list = [np.array(plot_row) for plot_row in cell_plot_rows(active)]
    names = cells_in_order(active)

    (
        batch_idx, node_idx, _radius, type_idx, _sti_u, _sti_v, du, dv, center_entry_mask,
    ) = build_spot_center_readout(
        connectome, batches, pack_spot_cost_radii(pack), pack.cost_radius,
    )

    plot_traces = plot_full[batch_idx, :, node_idx].cpu().numpy()

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
        names=names,
        cells=cells,
        type_idx=type_idx,
        node_idx=node_idx,
        du=du,
        dv=dv,
        center_entry_mask=center_entry_mask,
        plot_traces=plot_traces,
        t_onset=int(sti_t_onset) if sti_t_onset is not None else None,
        t_sti_end=(
            None if sti_t_onset is None else t_sti_end(
                sti_t_onset, mt, timing.ms_sti,
                delta_ms=timing.delta_ms,
            )
        ),
        batch_idx=batch_idx,
        batches=batches,
        mt=mt,
        pack=pack,
        a_sti_radius=a_sti_radius_by_name,
    )
    mask = np.asarray(center_entry_mask, dtype=bool)
    if at_x is not None or at_y is not None:
        mask = mask & _spot_readout_hex_mask(
            connectome, node_idx, pack.cost_radius, at_x=at_x, at_y=at_y,
        )
    nodes_by_name = {
        name: np.unique(node_idx[mask & (type_idx == cells.index(name))])
        for name in names
    }
    readout['v_th_by_name'] = params_for_types(
        nodes_by_name, v_th_by_type_name(z, session),
    )
    readout['e_leaks'] = params_for_types(
        nodes_by_name, e_leak_by_type_name(z, session),
    )
    readout['gt_affine_by_name'] = {
        name: gt_affine_scalars_for_cell(
            params, name, session.backend, session=session,
        )
        for name in names
    }
    readout['cell_plot_rows'] = cell_plot_rows_list
    return readout


def _spot_v_readout_from_readout(readout, session):
    names = readout['names']
    mt = readout['mt']
    v_readout = np.full((len(names), RF_N_RADII, mt), np.nan)
    std = np.full((len(names), RF_N_RADII, mt), np.nan)
    n_by_radius_by_name = {}
    for ti, ft in enumerate(names):
        ft_global = readout['cells'].index(ft)
        n_by_radius_by_name[ft] = _fill_member_v_readout(
            v_readout, std, ti, ft_global,
            readout['type_idx'], readout['du'], readout['dv'], readout['plot_traces'],
        )
    single_hex = suppress_cost_std(session, task=readout['pack'].name)
    cells = _cells_from_v_readout(
        names, v_readout, std, readout['v_th_by_name'],
        e_leaks=readout.get('e_leaks'),
        single_hex=single_hex, n_by_radius_by_name=n_by_radius_by_name,
        gt_affine_by_name=readout.get('gt_affine_by_name'),
    )
    plot_row_idxs = _plot_row_idxs_from_cell_plot_rows(readout['cell_plot_rows'], names)
    return cells, plot_row_idxs, mt


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
    cells, plot_row_idxs, n_t = _spot_v_readout_from_readout(readout, session)
    slice_overlay, slice_labels = (None, None)
    if at_xs is not None or at_ys is not None:
        connectome = session.backend.network
        if connectome is None:
            raise ValueError("spot slice overlay requires a network backend")
        slice_overlay, slice_labels = _spot_slice_overlay(
            readout, connectome, at_xs, at_ys,
        )
    return SpotTraceReadout(
        cells=cells,
        plot_row_idxs=plot_row_idxs,
        session=session,
        slice_overlay=slice_overlay,
        slice_labels=slice_labels,
        slice_xs=at_xs,
        slice_ys=at_ys,
        n_t=n_t,
        prep_s=time.perf_counter() - t_prep0,
        v_th_by_name=v_th_by_type_name(z, session),
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
        if readout.has_slices:
            scope = hex_at_scope_tag(readout.slice_xs, readout.slice_ys)
            return f'{head}  [{scope}, overlay + scope]'
    return head


def _trained_radii(cost_parts, contrasts, *, center_only=False):
    """Sorted integer radii from spot cost parts (``spot_{contrast}_{cell}_r*``)."""
    if center_only or not cost_parts:
        return [int(RF_CENTER_RADIUS)]
    out = set()
    for contrast in contrasts:
        prefix = f"spot_{contrast}_"
        for key in cost_parts:
            if not key.startswith(prefix):
                continue
            pos = key.rfind("_r")
            if pos < 0:
                continue
            r_s = key[pos + 2:]
            try:
                r_f = float(r_s)
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
    cells, plot_row_idxs = _layout_cells_from_readouts(readouts, order)
    has_slices = primary.has_slices
    slice_labels = primary.slice_labels or []
    n_t = primary.n_t
    t_onset = primary.t_onset
    t_sti_end = primary.t_sti_end
    delta_ms = float(primary.session.delta_ms)
    delta_ms_pre = float(primary.session.delta_ms_pre)
    show_pre = primary.show_pre
    ms_shown = primary.ms_shown
    t_delay = t_delay_from_ir(
        delta_ms=delta_ms,
        filter=session_filter_plot_token(primary.session),
    )
    gt_cell_idx = {name: i for i, name in enumerate(GT_CELLS)}

    def _t_delay_for_cell(name):
        idx = gt_cell_idx.get(name)
        return 0 if idx is None else int(t_delay[idx])

    timer.end_prep()

    sessions = {c: readouts[c].session for c in order}
    gt_by_contrast = resolve_spot_gts(sessions, gts)

    cells_by_contrast = {}
    for c in order:
        cells_by_contrast[c] = {cell["name"]: cell for cell in readouts[c].cells}

    center_only = bool(primary.center_only)
    radii = _trained_radii(cost_parts, order, center_only=center_only)
    center_radius = int(RF_CENTER_RADIUS)
    order_heights = [
        1 + len(radii) for _ in plot_row_idxs
    ]
    n_row = int(sum(order_heights))
    fig = plt.figure(figsize=figsize_fn(n_col, n_row))
    gs = fig.add_gridspec(n_row, n_col, **gridspec_kw)
    legend_done = False

    def _series_for_cell(name, *, with_slices):
        series = []
        for c in order:
            cell = cells_by_contrast[c].get(name)
            if cell is None:
                continue
            gt_by_cell = gt_by_contrast.get(c) or {}
            gt_raw = gt_by_cell.get(name)
            if gt_raw is not None:
                gt = (
                    float(cell.get("a_gt", 1.0)) * np.asarray(gt_raw, dtype=float)
                    + float(cell.get("bias_gt", 0.0))
                )
            else:
                gt = None
            entry = {
                "contrast": c,
                "v_readout": cell["v_readout"],
                "gt": gt,
                "std": cell.get("std"),
                "v_th": cell.get("v_th"),
                "linestyle": contrast_linestyle(c),
                "label_gt": f"{c} gt",
                "label_v_readout": f"{c} {session_filter_plot_token(primary.session)}",
                "label_scope": f"{c} scope",
            }
            if with_slices:
                overlay = readouts[c].slice_overlay
                if overlay is not None:
                    entry["slice_overlay"] = {
                        label: v_readouts[name]
                        for label, v_readouts in overlay.items()
                        if name in v_readouts
                    }
                else:
                    entry["slice_overlay"] = {}
            series.append(entry)
        return series

    def _plot_cell(name, cell_primary, ax_rf, ax_time, show_ylabel, show_xlabels):
        nonlocal legend_done
        time_title = format_spot_radius_time_title(
            center_radius,
            (cell_primary.get("n_by_radius") or {}).get(center_radius),
            name,
            cost_parts,
            order,
        )
        if has_slices and primary.slice_overlay is not None:
            series = _series_with_cost_points(
                _series_for_cell(name, with_slices=True),
                readouts,
                float(center_radius),
            )
            slice_overlay = (series[0].get("slice_overlay") or {}) if series else {}
            if not slice_overlay:
                ax_rf.axis("off")
                ax_time.axis("off")
                return
            plot_cell_rf_time_slices(
                ax_rf, ax_time, name, series, slice_labels,
                time_title=time_title,
                show_legend=not legend_done,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                n_t=n_t,
                v_th=cell_primary.get("v_th"),
                e_leak=cell_primary.get("e_leak"),
                t_onset=t_onset,
                show_pre=show_pre,
                t_sti_end=t_sti_end,
                t_delay=_t_delay_for_cell(name),
                delta_ms=delta_ms,
                delta_ms_pre=delta_ms_pre,
                ms_shown=ms_shown,
            )
        else:
            series = _series_with_cost_points(
                _series_for_cell(name, with_slices=False),
                readouts,
                float(center_radius),
            )
            plot_cell_rf_time(
                ax_rf, ax_time, name, series,
                time_title=time_title,
                show_legend=not legend_done,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                n_t=n_t,
                v_th=cell_primary.get("v_th"),
                e_leak=cell_primary.get("e_leak"),
                t_onset=t_onset,
                show_pre=show_pre,
                t_sti_end=t_sti_end,
                t_delay=_t_delay_for_cell(name),
                delta_ms=delta_ms,
                delta_ms_pre=delta_ms_pre,
                ms_shown=ms_shown,
            )
        legend_done = True

    row_cursor = 0
    for gi, cell_idxs in enumerate(plot_row_idxs):
        group_h = int(order_heights[gi])
        rf_row = row_cursor
        time_row0 = row_cursor + 1
        start = (n_col - len(cell_idxs)) // 2
        for j, ci in enumerate(cell_idxs):
            col = start + j
            cell_on = cells[ci]
            ax_rf = fig.add_subplot(gs[rf_row, col])
            if has_slices:
                ax_time = fig.add_subplot(gs[time_row0, col])
                _plot_cell(
                    cell_on["name"], cell_on, ax_rf, ax_time,
                    show_ylabel=(j == 0), show_xlabels=True,
                )
                for rr in range(time_row0 + 1, row_cursor + group_h):
                    ax_off = fig.add_subplot(gs[rr, col])
                    ax_off.axis("off")
                continue

            name = cell_on["name"]
            series = _series_for_cell(name, with_slices=False)
            show_legend = not legend_done
            plot_cell_rf(
                ax_rf, name, series,
                show_legend=show_legend,
                show_xlabels=False,
                show_ylabel=(j == 0),
                t_onset=t_onset,
                t_sti_end=t_sti_end,
                t_delay=_t_delay_for_cell(name),
            )
            legend_done = True
            ylim0 = None
            n_by_radius = cell_on.get("n_by_radius") or {}
            for local_i, rr in enumerate(range(time_row0, row_cursor + group_h)):
                ax_t = fig.add_subplot(gs[rr, col])
                if local_i >= len(radii):
                    ax_t.axis("off")
                    continue
                radius = int(radii[local_i])
                time_title = format_spot_radius_time_title(
                    radius, n_by_radius.get(radius), name, cost_parts, order,
                )
                plot_cell_time(
                    ax_t,
                    _series_with_cost_points(series, readouts, float(radius)),
                    title=time_title,
                    show_xlabels=(local_i == len(radii) - 1),
                    show_ylabel=(j == 0),
                    v_th=cell_on.get("v_th"),
                    e_leak=cell_on.get("e_leak"),
                    n_t=n_t,
                    t_onset=t_onset,
                    show_pre=show_pre,
                    t_sti_end=t_sti_end,
                    t_delay=_t_delay_for_cell(name),
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
    views = {c: _spot_readout_gt_view(b) for c, b in readouts.items()}
    _plot_spot_figure(
        path,
        timer=PlotTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts=views,
        title=title,
        gts=gts,
        n_col=N_COL_GT,
        figsize_fn=lambda c, r: (PANEL_W * c, 2.5 * r),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
        cost_parts=cost_parts,
    )


def plot_network_spot_all(path, *, readouts, title, gts=None, cost_parts=None):
    """Draw ca-all figure (all types) from contrast → readout."""
    _plot_spot_figure(
        path,
        timer=PlotTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts=readouts,
        title=title,
        gts=gts,
        n_col=N_COL_ALL,
        figsize_fn=lambda c, r: (PANEL_W * c, 2.5 * r),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
        cost_parts=cost_parts,
    )
