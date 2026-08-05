"""Spot plotting (network spot task).

Network RF profile axis is Euclidean radius: cube[..., radius] = mean at that radius.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

from param_defaults import DELTA_MS
import training
from figure.readout import (
    contrast_for_task,
    contrast_linestyle,
    contrast_order,
    pack_readout_cells,
    plot_cells_in_order,
    spot_gt_cubes,
)
from figure.util import (
    GT_COLOR,
    MODEL_COLOR,
    TRACE_LW,
    PlotTimer,
    annotate_baseline,
    baselines_for_types,
    bundle_prep_s,
    format_spot_radius_time_title,
    gt_affine_scalars_for_cell,
    bias_gt_from_v_onset_enabled,
    mean_v_onset_by_cell_name,
    hex_at_scope_tag,
    mark_pulse,
    ms_shown_axis_xlim,
    overlay_model_reds,
    plot_pre_post_line,
    plot_timecourse,
    save_figure,
    sem_from_traces,
    slice_coord_specs,
    suppress_cost_sem,
    v_ref_by_type_name,
    v_ref_schema_name,
)
from network.construction import cell_order_rows, cell_names_in_order
from task.moving_bar.input import (
    filter_sti_hexes,
    moving_bar_cost_hexes,
    network_uv_np,
)
from task.spot.input import (
    euclid_hex_dist,
    spot_from_opts,
    spot_stimulus_batches,
)
from task.spot.gt import (
    RF_CENTER_RADIUS,
    RF_N_RADII,
    RF_RADIUS_DEG,
    build_spot_center_readout,
)
from neuron.params import ms_to_t

RF_RADIUS_X = np.arange(RF_N_RADII) * RF_RADIUS_DEG


def _pulse_end_from_opts(opts, t_onset, n_t):
    """Exclusive end index of stimulus-on ``u[t]`` (matches ``spot_input_waveform``)."""
    if t_onset is None:
        return None
    t0 = int(t_onset)
    mt = int(n_t)
    ms_pulse = opts.get("ms_pulse")
    if ms_pulse is None:
        return mt
    dt = float(opts.get("delta_ms", DELTA_MS))
    width = max(1, ms_to_t(float(ms_pulse), delta_ms=dt))
    return min(mt, t0 + width)


def pack_spot_cost_radii(pack) -> tuple[float, ...]:
    """Active cost radii from ``pack.cost_radius`` (already resolved at pack build)."""
    if pack.cost_radius is None:
        raise ValueError(f"{pack.name} pack missing cost_radius")
    return tuple(
        sorted({round(float(radius), 6) for radius in pack.cost_radius.tolist()})
    )


def _integer_profile_radius(radius):
    """Return profile axis index for integer Euclidean radius, else ``None``."""
    k = int(round(float(radius)))
    if abs(float(radius) - k) > 1e-6:
        return None
    if k < 0 or k >= RF_N_RADII:
        return None
    return k


def _session_spot_timing(session):
    """Extract onset ``t_onset`` / forward ``n_t``, ms_pulse, and delta_ms from session."""
    from task.spot.input import spot_gt_n_t_from_opts, spot_timing_t_from_opts

    opts = (session.train_opts or {}).get(
        f"{session.primary_readout.name}_stimulus_opts",
    ) or {}
    t_onset, n_t = spot_timing_t_from_opts(opts)
    ms_pulse = opts.get("ms_pulse")
    return (
        int(t_onset),
        int(n_t),
        int(spot_gt_n_t_from_opts(opts)),
        float(ms_pulse) if ms_pulse is not None else None,
        float(opts.get("delta_ms", DELTA_MS)),
    )


def resolve_spot_gt_cubes(sessions, gt_cubes=None):
    """``{contrast: {cell: (RF_N_RADII, T)}}`` for each entry in ``sessions``.

    Gt time length is response-only (no ``ms_post``).
    """
    if gt_cubes is not None:
        return gt_cubes
    if not sessions:
        return {}
    out = {}
    for contrast, session in sessions.items():
        t_onset, _n_t, n_t_gt, ms_pulse, delta_ms = _session_spot_timing(session)
        part = spot_gt_cubes(
            session, session.primary_readout.name, contrasts=(str(contrast),),
            t_onset=t_onset, n_t=n_t_gt, ms_pulse=ms_pulse, delta_ms=delta_ms,
        )
        out.update(part)
    return out


def _session_cost_time_ix(session, response_start):
    """Absolute time indices used for sparse spot cost (or ``None``)."""
    if session is None:
        return None
    ix = getattr(session.primary_readout, "cost_time_ix", None)
    if ix is None:
        return None
    base = int(response_start or 0)
    ix_np = ix.detach().cpu().numpy().astype(np.int64, copy=False)
    return base + ix_np


def scale_curve(xt, center_radius, sem_xt=None, *, response_start=None, pulse_end=None):
    """Center-radius time course + RF profile from one ``(RF_N_RADII, T)`` cube.

    RF peak time ``maxt`` is ``argmax |v|`` inside pulse ``[response_start, pulse_end)``.
    """
    if response_start is None:
        raise ValueError("scale_curve requires response_start (t_onset)")
    if pulse_end is None:
        raise ValueError("scale_curve requires pulse_end")
    t0 = int(response_start)
    t1 = int(pulse_end)
    if t1 <= t0:
        raise ValueError(f"scale_curve requires pulse_end > response_start, got [{t0}, {t1})")
    imp = xt[center_radius]
    resp = imp[t0:t1]
    if not np.isfinite(resp).any():
        if sem_xt is not None:
            return None, None, None
        return None, None
    maxt = t0 + int(np.argmax(np.abs(resp)))
    spatial = np.asarray(xt[:, maxt], dtype=np.float64)
    if sem_xt is not None:
        return imp, spatial, sem_xt[center_radius]
    return imp, spatial


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


def _style_time_axis(ax, show_xlabel, n_t, delta_ms=None, ms_shown=None):
    dt = float(DELTA_MS if delta_ms is None else delta_ms)
    t_last = max(int(n_t) - 1, 0)
    xlim = ms_shown_axis_xlim(ms_shown, delta_ms=dt, origin_t=0)
    if xlim is None:
        lo, hi = 0, t_last
    else:
        lo, hi = max(0, xlim[0]), min(t_last, xlim[1])
        if lo > hi:
            lo, hi = 0, t_last
    t_lo_s = lo * dt / 1000.0
    t_hi_s = hi * dt / 1000.0
    t_mid_s = (t_lo_s + t_hi_s) / 2.0
    mid = (lo + hi) // 2
    ax.set_xlim(lo, hi)
    ax.set_xticks([lo, mid, hi])
    ax.set_xticklabels([f'{t_lo_s:g}', f'{t_mid_s:g}', f'{t_hi_s:g}'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('time [s]', fontsize=7)


def _style_recf_profile_axis(ax, show_xlabel):
    """Style RF profile axis (degrees from center; index = radius)."""
    x_max = float((RF_N_RADII - 1) * RF_RADIUS_DEG)
    ax.set_xlim(-2, x_max + 2)
    ax.set_xticks([0, x_max / 2.0, x_max])
    ax.set_xticklabels(['0', f'{x_max / 2.0:g}', f'{x_max:g}'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('RF (°)', fontsize=7)


def _scale_contrast_series(
    series, *, response_start, pulse_end, center_radius=RF_CENTER_RADIUS,
):
    """Scale each contrast series entry to ImpR + RF radii.

    ``series`` items may include ``model_xt``, ``gt_xt``, ``sem_xt``.
    Returns a list of dicts with ``imp_model``, ``rf_model``, ``imp_sem``,
    ``imp_gt``, ``rf_gt`` plus passthrough keys.
    """
    sc_kw = dict(response_start=response_start, pulse_end=pulse_end)
    out = []
    for entry in series:
        item = dict(entry)
        model_xt = entry.get("model_xt")
        gt_xt = entry.get("gt_xt")
        sem_xt = entry.get("sem_xt")
        imp_sem = None
        if model_xt is not None:
            if sem_xt is not None:
                imp_model, rf_model, imp_sem = scale_curve(
                    model_xt, center_radius, sem_xt, **sc_kw,
                )
            else:
                imp_model, rf_model = scale_curve(model_xt, center_radius, **sc_kw)
        else:
            imp_model, rf_model = None, None
        if gt_xt is not None:
            imp_gt, rf_gt = scale_curve(gt_xt, center_radius, **sc_kw)
        else:
            imp_gt, rf_gt = None, None
        item.update(
            imp_model=imp_model,
            rf_model=rf_model,
            imp_sem=imp_sem,
            imp_gt=imp_gt,
            rf_gt=rf_gt,
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
    baseline=None,
    response_start=None,
    pulse_end=None,
):
    """RF-profile panel for one cell across contrast ``series``."""
    scaled = _scale_contrast_series(
        series, response_start=response_start, pulse_end=pulse_end,
    )
    for item in scaled:
        ls = item.get("linestyle", "-")
        _plot_rf_profile(
            ax, item["rf_gt"], color=GT_COLOR,
            label=item.get("label_gt"), linestyle=ls,
        )
        _plot_rf_profile(
            ax, item["rf_model"], color=MODEL_COLOR,
            label=item.get("label_model"), linestyle=ls, filled=True,
        )
    ax.set_title(title, fontsize=8, pad=2)
    _style_recf_profile_axis(ax, show_xlabels)
    if show_ylabel:
        ax.set_ylabel('mV', fontsize=7)
    ax.tick_params(labelsize=6)
    annotate_baseline(ax, baseline)
    if show_legend:
        ax.legend(loc='upper right', fontsize=6, frameon=False)


def plot_cell_time(
    ax,
    series,
    *,
    title=None,
    show_xlabels=False,
    show_ylabel=False,
    baseline=None,
    n_t=None,
    response_start=None,
    pre_end=None,
    show_pre=False,
    pulse_end=None,
    delta_ms=None,
    ms_shown=None,
    center_radius=RF_CENTER_RADIUS,
):
    """Time-course panel for one cell across contrast ``series``.

    ``pre_end`` defaults to ``response_start`` (gray gt omits ``[0, pre_end)``).
    Pass ``pre_end=0`` to draw the full gt trace including pre-onset.
    ``pulse_end``: white stimulus-on band ``[response_start, pulse_end)``.
    """
    scaled = _scale_contrast_series(
        series,
        response_start=response_start,
        pulse_end=pulse_end,
        center_radius=int(center_radius),
    )
    t = np.arange(n_t)
    split = int(response_start or 0) if pre_end is None else int(pre_end)
    if title is not None:
        ax.set_title(title, fontsize=8, pad=2)
    traces = [
        {
            "model": item["imp_model"],
            "gt": item["imp_gt"],
            "sem": item["imp_sem"],
            "linestyle": item.get("linestyle", "-"),
            "point_ix": item.get("point_ix"),
        }
        for item in scaled
    ]
    plot_timecourse(
        ax, t, traces,
        show_sem=any(item["imp_sem"] is not None for item in scaled),
        baseline=baseline,
        show_ylabel=show_ylabel,
        style_xaxis=lambda a: _style_time_axis(
            a, show_xlabels, n_t, delta_ms, ms_shown=ms_shown,
        ),
        pre_end=split,
        show_pre=show_pre,
        pulse_start=response_start,
        pulse_end=pulse_end,
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
    baseline=None,
    n_t=None,
    response_start=None,
    show_pre=False,
    pulse_end=None,
    delta_ms=None,
    ms_shown=None,
):
    """RF + time panels for one cell (composes ``plot_cell_rf`` + ``plot_cell_time``)."""
    plot_cell_rf(
        ax_rf, title, series,
        show_legend=show_legend,
        show_xlabels=show_xlabels,
        show_ylabel=show_ylabel,
        baseline=baseline,
        response_start=response_start,
        pulse_end=pulse_end,
    )
    plot_cell_time(
        ax_time, series,
        title=time_title,
        show_xlabels=show_xlabels,
        show_ylabel=show_ylabel,
        baseline=baseline,
        n_t=n_t,
        response_start=response_start,
        show_pre=show_pre,
        pulse_end=pulse_end,
        delta_ms=delta_ms,
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
    baseline=None,
    n_t=None,
    response_start=None,
    show_pre=False,
    pulse_end=None,
    delta_ms=None,
    ms_shown=None,
):
    """RF + time panels with per-slice overlays across contrast ``series``."""
    center_radius = RF_CENTER_RADIUS
    sc_kw = dict(response_start=response_start, pulse_end=pulse_end)
    colors = overlay_model_reds(len(slice_labels))
    t = np.arange(n_t)
    pre_end = int(response_start or 0)
    mark_pulse(ax_time, response_start, pulse_end)

    for si, item in enumerate(series):
        ls = item.get("linestyle", "-")
        model_xt = item.get("model_xt")
        gt_xt = item.get("gt_xt")
        slice_overlay = item.get("slice_overlay") or {}
        imp_model = rf_model = None
        if model_xt is not None:
            imp_model, rf_model = scale_curve(model_xt, center_radius, **sc_kw)
        imp_gt = rf_gt = None
        if gt_xt is not None:
            imp_gt, rf_gt = scale_curve(gt_xt, center_radius, **sc_kw)
        slice_imps = {}
        slice_rfs = {}
        for label in slice_labels:
            if label in slice_overlay:
                imp_s, rf_s = scale_curve(slice_overlay[label], center_radius, **sc_kw)
                slice_imps[label] = imp_s
                slice_rfs[label] = rf_s

        _plot_rf_profile(
            ax_rf, rf_gt, color=GT_COLOR,
            label=item.get("label_gt"), linestyle=ls,
        )
        for i, label in enumerate(slice_labels):
            _plot_rf_profile(
                ax_rf, slice_rfs.get(label), color=colors[i],
                label=label if si == 0 else None, linestyle=ls,
            )
        _plot_rf_profile(
            ax_rf, rf_model, color=colors[-1],
            label=item.get("label_total"), linestyle=ls, filled=True,
        )

        plot_pre_post_line(
            ax_time, t, imp_gt, pre_end=pre_end, show_pre=False, draw_pre=False,
            color=GT_COLOR, linestyle=ls, linewidth=TRACE_LW,
        )
        for i, label in enumerate(slice_labels):
            plot_pre_post_line(
                ax_time, t, slice_imps.get(label), pre_end=pre_end,
                show_pre=show_pre, draw_pre=True,
                color=colors[i], linestyle=ls, linewidth=TRACE_LW,
                label=label if si == 0 else None,
            )
        plot_pre_post_line(
            ax_time, t, imp_model, pre_end=pre_end, show_pre=show_pre, draw_pre=True,
            color=colors[-1], linestyle=ls, linewidth=TRACE_LW,
            label=item.get("label_total"),
        )

    ax_rf.set_title(title, fontsize=8, pad=2)
    _style_recf_profile_axis(ax_rf, show_xlabels)
    if show_ylabel:
        ax_rf.set_ylabel('mV', fontsize=7)
    ax_rf.tick_params(labelsize=6)
    annotate_baseline(ax_rf, baseline)
    if show_legend:
        ax_rf.legend(loc='upper right', fontsize=6, frameon=False)

    if time_title is not None:
        ax_time.set_title(time_title, fontsize=8, pad=2)
    _style_time_axis(ax_time, show_xlabels, n_t, delta_ms, ms_shown=ms_shown)
    if show_ylabel:
        ax_time.set_ylabel('mV', fontsize=7)
    ax_time.tick_params(labelsize=6)
    annotate_baseline(ax_time, baseline)



def _as_index(node_index, device):
    if not torch.is_tensor(node_index):
        return torch.tensor(node_index, dtype=torch.long, device=device)
    return node_index.to(device)


@torch.no_grad()
def _simulate(session, z, node_index):
    node_index = _as_index(node_index, z.device)
    schema = list(session.schema)
    backend = session.backend
    p = training.assign_params(z, schema, backend)
    stacked = training.forward_nodes(session, p, node_index=node_index)
    # Match prior layout: (n_nodes, T) for 2-D forward_nodes output.
    return stacked.transpose(0, 1) if stacked.dim() == 2 else stacked


def calc_ca_full_all(session, z):
    n_cells = session.backend.n_cells
    mt = session.n_t
    ca_full = np.zeros((n_cells, RF_N_RADII, mt))
    for radius in range(RF_N_RADII):
        col_index = torch.arange(
            radius * n_cells,
            (radius + 1) * n_cells,
            dtype=torch.long,
            device=z.device,
        )
        ca_full[:, radius] = _simulate(session, z, col_index).cpu().numpy()
    return ca_full


def _fill_member_cube(cube, sem, ti, ft_global, type_idx, du, dv, plot_traces):
    """Fill profile radii from Euclidean radius means; return ``{r: n_rows}``."""
    mask = type_idx == ft_global
    if not mask.any():
        return {}
    rows = np.where(mask)[0]
    by_radius: dict[float, list] = {}
    for row in rows:
        radius = round(float(euclid_hex_dist(int(du[row]), int(dv[row]))), 6)
        by_radius.setdefault(radius, []).append(int(row))
    n_by_radius = {}
    for radius, row_ix in by_radius.items():
        profile_radius = _integer_profile_radius(radius)
        if profile_radius is None:
            continue
        traces = plot_traces[row_ix]
        mean_trace = traces.mean(axis=0)
        sem_trace = sem_from_traces(traces, single_hex=False)
        cube[ti, profile_radius] = mean_trace
        sem[ti, profile_radius] = sem_trace
        n_by_radius[profile_radius] = len(row_ix)
    return n_by_radius


@dataclass
class SpotTraceBundle:
    """One forward pass; full cost-extent readout over all types."""

    cells: list
    order_row_ixs: list | None = None
    session: object = None
    slice_overlay: dict[str, dict[str, np.ndarray]] | None = None
    slice_labels: list[str] | None = None
    slice_xs: list | None = None
    slice_ys: list | None = None
    n_t: int = 0
    prep_s: float = 0.0
    v_ref_by_name: dict = field(default_factory=dict)
    v_ref_name: str | None = None
    response_start: int | None = None
    show_pre: bool = True
    pulse_end: int | None = None
    ms_shown: tuple[float, float] | None = None
    r0_only: bool = False
    a_sti_r: dict[str, float] = field(default_factory=dict)

    @property
    def has_slices(self):
        return bool(self.slice_overlay)


def _order_row_ixs_from_rows(order_rows, names):
    name_to_i = {str(n): i for i, n in enumerate(names)}
    order_row_ixs = []
    for names_row in order_rows:
        row_idx = [name_to_i[str(n)] for n in names_row if str(n) in name_to_i]
        if row_idx:
            order_row_ixs.append(row_idx)
    return order_row_ixs


def _spot_all_cell_names(session):
    if session.backend.network is None:
        raise ValueError("_spot_all_cell_names requires session.backend.network")
    return plot_cells_in_order(session.backend.network.cell_names)


def _spot_readout_bundle_view(bundle):
    """ca-gt view: same traces, rows restricted to ``pack_readout_cells``."""
    session = bundle.session
    present = pack_readout_cells(session, session.primary_readout.name)
    order_rows = [np.array(row) for row in cell_order_rows(present)]
    names = cell_names_in_order(present)
    by_name = {c['name']: c for c in bundle.cells}
    cells = [by_name[n] for n in names if n in by_name]
    cell_names = [c['name'] for c in cells]
    return SpotTraceBundle(
        cells=cells,
        order_row_ixs=_order_row_ixs_from_rows(order_rows, cell_names),
        session=session,
        n_t=bundle.n_t,
        v_ref_by_name=bundle.v_ref_by_name,
        v_ref_name=bundle.v_ref_name,
        response_start=bundle.response_start,
        show_pre=bundle.show_pre,
        pulse_end=bundle.pulse_end,
        ms_shown=bundle.ms_shown,
        r0_only=bool(getattr(bundle, "r0_only", False)),
        a_sti_r=dict(getattr(bundle, "a_sti_r", None) or {}),
    )


def _cells_with_order_row_ixs(order_rows, names, build_cell):
    cells = [build_cell(str(name)) for name in names]
    return cells, _order_row_ixs_from_rows(order_rows, names)


def _spot_cubes_from_row_mask(rows, mask):
    names = rows['names']
    cell_names = rows['cell_names']
    mt = rows['mt']
    type_idx = rows['type_idx'][mask]
    du = rows['du'][mask]
    dv = rows['dv'][mask]
    plot_traces = rows['plot_traces'][mask]
    out = {}
    sem_dummy = np.full((1, RF_N_RADII, mt), np.nan)
    for ft in names:
        cube = np.full((1, RF_N_RADII, mt), np.nan)
        ft_global = cell_names.index(ft)
        _fill_member_cube(cube, sem_dummy, 0, ft_global, type_idx, du, dv, plot_traces)
        out[ft] = cube[0]
    return out


def _spot_readout_hex_mask(C, node_idx, cost_extent, *, at_x=None, at_y=None):
    """True for readout rows whose node sits on matching cost-extent hexes."""
    filt = filter_sti_hexes(
        moving_bar_cost_hexes(C, cost_extent=cost_extent),
        at_x=at_x,
        at_y=at_y,
    )
    if not filt:
        return np.zeros(len(node_idx), dtype=bool)
    u_np, v_np = network_uv_np(C)
    col_uv = {(int(c.u), int(c.v)) for c in filt}
    return np.array(
        [(int(u_np[n]), int(v_np[n])) in col_uv for n in node_idx],
        dtype=bool,
    )


def _spot_slice_overlay(rows, C, at_xs, at_ys):
    """Per-hex readout overlays (same ``filter_sti_hexes`` scope as moving_bar)."""
    overlay = {}
    labels = []
    pack = rows['pack']
    node_idx = rows['node_idx']
    for label, at_x, at_y in slice_coord_specs(at_xs, at_ys):
        mask = _spot_readout_hex_mask(
            C, node_idx, pack.cost_extent, at_x=at_x, at_y=at_y,
        )
        if not np.any(mask):
            print(f'skip slice overlay {label}: no column within cost_extent')
            continue
        cubes = _spot_cubes_from_row_mask(rows, mask)
        if not any(np.isfinite(c).any() for c in cubes.values()):
            print(f'skip slice overlay {label}: no readouts')
            continue
        overlay[label] = cubes
        labels.append(label)
    if not overlay:
        return None, None
    return overlay, labels



def _cells_from_cube(
    names, cube, sem, baselines, *, single_hex,
    n_by_radius_by_name=None, gt_affine_by_name=None,
):
    n_by_radius_by_name = n_by_radius_by_name or {}
    gt_affine_by_name = gt_affine_by_name or {}
    out = []
    for i, n in enumerate(names):
        scale, bias = gt_affine_by_name.get(n, (1.0, 0.0))
        out.append(dict(
            name=n,
            cube=cube[i],
            sem=None if single_hex else sem[i],
            baseline=baselines.get(n),
            n_by_radius=dict(n_by_radius_by_name.get(n) or {}),
            a_gt=float(scale),
            bias_gt=float(bias),
        ))
    return out


@torch.no_grad()
def _spot_forward_rows(
    session, z, *,
    at_x=None, at_y=None,
):
    """One forward; cost-extent node readout over all network types."""
    pack = session.primary_readout
    schema = list(session.schema)
    p = training.assign_params(z, schema, session.backend)
    a_sti_r_by_name = {}
    if "a_sti_r" in p:
        for seg in schema:
            if seg.get("name") != "a_sti_r":
                continue
            names = [str(n) for n in (seg.get("node_names") or ())]
            raw = p["a_sti_r"].detach().cpu().numpy().reshape(-1)
            a_sti_r_by_name = {
                names[i]: float(raw[i]) for i in range(min(len(names), raw.size))
            }
            break
    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    trace_full = training.forward_full(session, p, i_sti, pack=pack)
    C = session.backend.network
    cell_names = list(C.cell_names)
    mt = int(i_sti.shape[1])

    opts = dict((session.train_opts or {}).get(f"{pack.name}_stimulus_opts") or {})
    spot = spot_from_opts(C, stimulus_opts=opts)
    batches = spot_stimulus_batches(spot)
    present = _spot_all_cell_names(session)
    order_rows = [np.array(row) for row in cell_order_rows(present)]
    names = cell_names_in_order(present)

    (
        batch_idx, node_idx, _radius, type_idx, _stim_u, _stim_v, du, dv, center_row,
    ) = build_spot_center_readout(
        C, batches, pack_spot_cost_radii(pack), pack.cost_extent,
    )

    plot_traces = trace_full[batch_idx, :, node_idx].cpu().numpy()

    stim_ms_pre = opts.get("ms_pre")
    dt = float(opts.get("delta_ms", DELTA_MS))
    stim_t_onset = (
        ms_to_t(float(stim_ms_pre), delta_ms=dt) if stim_ms_pre is not None else None
    )
    rows = dict(
        names=names,
        cell_names=cell_names,
        type_idx=type_idx,
        node_idx=node_idx,
        du=du,
        dv=dv,
        center_row=center_row,
        plot_traces=plot_traces,
        t_onset=int(stim_t_onset) if stim_t_onset is not None else None,
        pulse_end=_pulse_end_from_opts(opts, stim_t_onset, mt),
        batch_idx=batch_idx,
        batches=batches,
        mt=mt,
        pack=pack,
        a_sti_r=a_sti_r_by_name,
    )
    mask = np.asarray(center_row, dtype=bool)
    if at_x is not None or at_y is not None:
        mask = mask & _spot_readout_hex_mask(
            C, node_idx, pack.cost_extent, at_x=at_x, at_y=at_y,
        )
    nodes_by_name = {
        name: np.unique(node_idx[mask & (type_idx == cell_names.index(name))])
        for name in names
    }
    rows['baselines'] = baselines_for_types(
        nodes_by_name, v_ref_by_type_name(z, session),
    )
    onset_bias = None
    if bias_gt_from_v_onset_enabled(session):
        onset_bias = mean_v_onset_by_cell_name(
            plot_traces, type_idx, cell_names, names, rows.get('t_onset'),
        )
    rows['gt_affine_by_name'] = {
        name: gt_affine_scalars_for_cell(
            p, name, session.backend,
            bias_gt=None if onset_bias is None else onset_bias.get(name),
        )
        for name in names
    }
    rows['order_rows'] = order_rows
    return rows


def _spot_cube_from_rows(rows, session):
    names = rows['names']
    mt = rows['mt']
    cube = np.full((len(names), RF_N_RADII, mt), np.nan)
    sem = np.full((len(names), RF_N_RADII, mt), np.nan)
    n_by_radius_by_name = {}
    for ti, ft in enumerate(names):
        ft_global = rows['cell_names'].index(ft)
        n_by_radius_by_name[ft] = _fill_member_cube(
            cube, sem, ti, ft_global,
            rows['type_idx'], rows['du'], rows['dv'], rows['plot_traces'],
        )
    single_hex = suppress_cost_sem(session, task=rows['pack'].name)
    cells = _cells_from_cube(
        names, cube, sem, rows['baselines'],
        single_hex=single_hex, n_by_radius_by_name=n_by_radius_by_name,
        gt_affine_by_name=rows.get('gt_affine_by_name'),
    )
    order_row_ixs = _order_row_ixs_from_rows(rows['order_rows'], names)
    return cells, order_row_ixs, mt


@torch.no_grad()
def network_spot_trace_bundle(
    session, z, *,
    at_xs=None, at_ys=None,
    show_pre=True,
    ms_shown=None,
    r0_only=False,
):
    """Run one forward; full cost-extent spot traces over all types."""
    t_prep0 = time.perf_counter()
    at_x = at_xs[0] if at_xs else None
    at_y = at_ys[0] if at_ys else None
    rows = _spot_forward_rows(
        session, z,
        at_x=at_x, at_y=at_y,
    )
    cells, order_row_ixs, n_t = _spot_cube_from_rows(rows, session)
    slice_overlay, slice_labels = (None, None)
    if at_xs is not None or at_ys is not None:
        C = session.backend.network
        if C is None:
            raise ValueError("spot slice overlay requires a network backend")
        slice_overlay, slice_labels = _spot_slice_overlay(
            rows, C, at_xs, at_ys,
        )
    return SpotTraceBundle(
        cells=cells,
        order_row_ixs=order_row_ixs,
        session=session,
        slice_overlay=slice_overlay,
        slice_labels=slice_labels,
        slice_xs=at_xs,
        slice_ys=at_ys,
        n_t=n_t,
        prep_s=time.perf_counter() - t_prep0,
        v_ref_by_name=v_ref_by_type_name(z, session),
        v_ref_name=v_ref_schema_name(session.schema),
        response_start=rows.get('t_onset'),
        show_pre=bool(show_pre),
        pulse_end=rows.get('pulse_end'),
        ms_shown=ms_shown,
        r0_only=bool(r0_only),
        a_sti_r=dict(rows.get('a_sti_r') or {}),
    )


def _format_a_sti_r_suptitle(a_sti_r):
    """Figure title ``a_sti_r 1 = <value>`` from named slot values."""
    if not a_sti_r or "1" not in a_sti_r:
        return None
    return f"a_sti_r 1 = {float(a_sti_r['1']):.4g}"


def _spot_suptitle(title, bundle):
    head = title
    if bundle is not None:
        formatted = _format_a_sti_r_suptitle(getattr(bundle, "a_sti_r", None) or {})
        if formatted is not None:
            head = formatted
    if bundle is not None and bundle.has_slices:
        scope = hex_at_scope_tag(bundle.slice_xs, bundle.slice_ys)
        return f'{head}  [{scope}, overlay + total]'
    return head


def _trained_radii(cost_parts, contrasts, *, r0_only=False):
    """Sorted integer radii from spot cost parts (``spot_{contrast}_{cell}_r*``)."""
    if r0_only:
        return [int(RF_CENTER_RADIUS)]
    if not cost_parts:
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
    bundles,
    title,
    gt_cubes=None,
    ncols,
    figsize_fn,
    gridspec_kw,
    suptitle_fs=12,
    cost_parts=None,
):
    """Draw spot figure from ``bundles`` (contrast → SpotTraceBundle)."""
    order = contrast_order(bundles)
    if not order:
        raise ValueError("_plot_spot_figure requires at least one bundle")
    primary = bundles[order[0]]
    cells = primary.cells
    order_row_ixs = primary.order_row_ixs
    has_slices = primary.has_slices
    slice_labels = primary.slice_labels or []
    n_t = primary.n_t
    response_start = primary.response_start
    pulse_end = getattr(primary, "pulse_end", None)
    delta_ms = float(primary.session.delta_ms)
    show_pre = getattr(primary, "show_pre", True)
    ms_shown = getattr(primary, "ms_shown", None)
    timer.end_prep()

    sessions = {c: bundles[c].session for c in order}
    gt_by_contrast = resolve_spot_gt_cubes(sessions, gt_cubes)

    cells_by_contrast = {}
    for c in order:
        cells_by_contrast[c] = {cell["name"]: cell for cell in bundles[c].cells}

    r0_only = bool(getattr(primary, "r0_only", False))
    radii = _trained_radii(cost_parts, order, r0_only=r0_only)
    order_heights = [
        1 + len(radii) for _ in order_row_ixs
    ]
    nrows = int(sum(order_heights))
    fig = plt.figure(figsize=figsize_fn(ncols, nrows))
    gs = fig.add_gridspec(nrows, ncols, **gridspec_kw)
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
                gt_xt = (
                    float(cell.get("a_gt", 1.0)) * np.asarray(gt_raw, dtype=float)
                    + float(cell.get("bias_gt", 0.0))
                )
            else:
                gt_xt = None
            entry = {
                "contrast": c,
                "model_xt": cell["cube"],
                "gt_xt": gt_xt,
                "sem_xt": cell.get("sem"),
                "baseline": cell.get("baseline"),
                "linestyle": contrast_linestyle(c),
                "label_gt": f"{c} gt",
                "label_model": f"{c} model",
                "label_total": f"{c} total",
                "point_ix": _session_cost_time_ix(
                    bundles[c].session, bundles[c].response_start,
                ),
            }
            if with_slices:
                overlay = bundles[c].slice_overlay
                if overlay is not None:
                    entry["slice_overlay"] = {
                        label: cubes[name]
                        for label, cubes in overlay.items()
                        if name in cubes
                    }
                else:
                    entry["slice_overlay"] = {}
            series.append(entry)
        return series

    def _plot_cell(name, cell_primary, ax_rf, ax_time, show_ylabel, show_xlabels):
        nonlocal legend_done
        r0 = int(RF_CENTER_RADIUS)
        time_title = format_spot_radius_time_title(
            r0,
            (cell_primary.get("n_by_radius") or {}).get(r0),
            name,
            cost_parts,
            order,
        )
        if has_slices and primary.slice_overlay is not None:
            series = _series_for_cell(name, with_slices=True)
            slice_xt = (series[0].get("slice_overlay") or {}) if series else {}
            if not slice_xt:
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
                baseline=cell_primary.get("baseline"),
                response_start=response_start,
                show_pre=show_pre,
                pulse_end=pulse_end,
                delta_ms=delta_ms,
                ms_shown=ms_shown,
            )
        else:
            series = _series_for_cell(name, with_slices=False)
            plot_cell_rf_time(
                ax_rf, ax_time, name, series,
                time_title=time_title,
                show_legend=not legend_done,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                n_t=n_t,
                baseline=cell_primary.get("baseline"),
                response_start=response_start,
                show_pre=show_pre,
                pulse_end=pulse_end,
                delta_ms=delta_ms,
                ms_shown=ms_shown,
            )
        legend_done = True

    row_cursor = 0
    for gi, row_idx in enumerate(order_row_ixs):
        group_h = int(order_heights[gi])
        rf_row = row_cursor
        time_row0 = row_cursor + 1
        start = (ncols - len(row_idx)) // 2
        for j, ci in enumerate(row_idx):
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
                baseline=cell_on.get("baseline"),
                response_start=response_start,
                pulse_end=pulse_end,
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
                    ax_t, series,
                    title=time_title,
                    show_xlabels=(local_i == len(radii) - 1),
                    show_ylabel=(j == 0),
                    baseline=cell_on.get("baseline"),
                    n_t=n_t,
                    response_start=response_start,
                    show_pre=show_pre,
                    pulse_end=pulse_end,
                    delta_ms=delta_ms,
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
    save_figure(fig, path, dpi=150)
    timer.log(path)


def plot_network_spot_gt(path, *, bundles, title, gt_cubes=None, cost_parts=None):
    """Draw ca-gt figure (pack readout types) from contrast → bundle."""
    timer = PlotTimer(prior_prep=bundle_prep_s(*bundles.values()))
    views = {
        c: _spot_readout_bundle_view(b) for c, b in bundles.items()
    }
    _plot_spot_figure(
        path,
        timer=timer,
        bundles=views,
        title=title,
        gt_cubes=gt_cubes,
        ncols=5,
        figsize_fn=lambda c, r: (3.0 * c, 2.5 * r),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
        cost_parts=cost_parts,
    )


def plot_network_spot_all(path, *, bundles, title, gt_cubes=None, cost_parts=None):
    """Draw ca-all figure (all types) from contrast → bundle."""
    timer = PlotTimer(prior_prep=bundle_prep_s(*bundles.values()))
    _plot_spot_figure(
        path,
        timer=timer,
        bundles=bundles,
        title=title,
        gt_cubes=gt_cubes,
        ncols=8,
        figsize_fn=lambda c, r: (2.2 * c, 2.5 * r),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
        cost_parts=cost_parts,
    )
