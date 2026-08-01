"""Spot plotting (network spot task).

Network RF bins are ring means: r=0 -> j4, r=1 -> j3/j5, r=2 -> j2/j6.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.defaults import DELTA_MS
import training
from figure.readout import (
    contrast_for_task,
    contrast_linestyle,
    contrast_order,
    pack_readout_cells,
    plot_present_layout,
    plot_cells_in_order,
    spot_data_cubes,
)
from figure.util import (
    DATA_COLOR,
    MODEL_COLOR,
    TRACE_LW,
    TRACE_YLIM,
    PlotTimer,
    annotate_baseline,
    apply_out_scale,
    baselines_for_types,
    batches_at_stim_xy,
    bundle_cell_title,
    bundle_prep_s,
    hex_at_scope_tag,
    mark_pulse,
    overlay_model_reds,
    plot_pre_post_line,
    plot_timecourse,
    readout_n_by_name,
    save_figure,
    save_forward_trace_csvs,
    sem_from_traces,
    slice_coord_specs,
    suppress_cost_sem,
    v_ref_by_type_name,
    v_ref_schema_name,
)
from task.spot.input import (
    euclid_hex_dist,
    spot_from_opts,
    spot_stimulus_batches,
)
from task.spot.data import spot_center_bin_layout
from neuron.params import ms_to_t

CENTER_BIN = 4  # RecF spatial center bin (j=4 in 0..8)
RF_N_BINS = 9
RF_BIN_X = np.arange(RF_N_BINS) * 5  # j=0..8 on mirrored RF axis (-20..20)


def _pulse_end_from_opts(opts, t_onset, n_t):
    """Exclusive end index of stimulus-on ``u[t]`` (matches ``spot_input_waveform``)."""
    if t_onset is None:
        return None
    t0 = int(t_onset)
    mt = int(n_t)
    pulse_ms = opts.get("pulse_ms")
    if pulse_ms is None:
        return mt
    dt = float(opts.get("delta_ms", DELTA_MS))
    width = max(1, ms_to_t(float(pulse_ms), delta_ms=dt))
    return min(mt, t0 + width)


def pack_spot_cost_radii(pack) -> tuple[float, ...]:
    """Active cost rings from ``pack.cost_radius`` (already resolved at pack build)."""
    if pack.cost_radius is None:
        raise ValueError(f"{pack.name} pack missing cost_radius")
    return tuple(sorted({round(float(r), 6) for r in pack.cost_radius.tolist()}))


def _radius_to_profile_bins(radius):
    """Map integer radius ring to mirrored profile bins."""
    k = int(round(float(radius)))
    if abs(float(radius) - k) > 1e-6:
        return ()
    if k < 0 or k > CENTER_BIN:
        return ()
    if k == 0:
        return (CENTER_BIN,)
    return (CENTER_BIN - k, CENTER_BIN + k)


def _session_spot_timing(session):
    """Extract onset ``t_onset`` / ``n_t``, pulse_ms, and delta_ms from session stimulus opts."""
    from task.spot.input import spot_timing_t_from_opts

    opts = (session.train_opts or {}).get(
        f"{session.primary_readout.name}_stimulus_opts",
    ) or {}
    t_onset, n_t = spot_timing_t_from_opts(opts)
    pulse_ms = opts.get("pulse_ms")
    return (
        int(t_onset),
        int(n_t),
        float(pulse_ms) if pulse_ms is not None else None,
        float(opts.get("delta_ms", DELTA_MS)),
    )


def resolve_spot_data_cubes(sessions, data_cubes=None):
    """``{contrast: {cell: (9, T)}}`` for each entry in ``sessions`` (contrast → session)."""
    if data_cubes is not None:
        return data_cubes
    if not sessions:
        return {}
    out = {}
    for contrast, session in sessions.items():
        t_onset, n_t, pulse_ms, delta_ms = _session_spot_timing(session)
        part = spot_data_cubes(
            session, session.primary_readout.name, contrasts=(str(contrast),),
            t_onset=t_onset, n_t=n_t, pulse_ms=pulse_ms, delta_ms=delta_ms,
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


def scale_curve(xt, center, sem_xt=None, *, response_start=None, pulse_end=None):
    """Center time course + discrete RF bins from one ``(9, T)`` cube.

    RF peak time ``maxt`` is ``argmax |v_delta|`` inside pulse ``[response_start, pulse_end)``.
    """
    if response_start is None:
        raise ValueError("scale_curve requires response_start (t_onset)")
    if pulse_end is None:
        raise ValueError("scale_curve requires pulse_end")
    t0 = int(response_start)
    t1 = int(pulse_end)
    if t1 <= t0:
        raise ValueError(f"scale_curve requires pulse_end > response_start, got [{t0}, {t1})")
    imp = xt[center]
    resp = imp[t0:t1]
    if not np.isfinite(resp).any():
        if sem_xt is not None:
            return None, None, None
        return None, None
    maxt = t0 + int(np.argmax(np.abs(resp)))
    spatial = np.asarray(xt[:, maxt], dtype=np.float64)
    if sem_xt is not None:
        return imp, spatial, sem_xt[center]
    return imp, spatial


def _plot_rf_profile(ax, rf, *, color, label=None, linestyle='-', filled=False):
    """Plot finite RF bins only (j=0..8); skip NaN (no cost readout)."""
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
    ax.plot(RF_BIN_X[mask], rf[mask], **kw)


def _style_time_axis(ax, show_xlabel, n_t, delta_ms=None):
    dt = float(DELTA_MS if delta_ms is None else delta_ms)
    t_last = max(int(n_t) - 1, 0)
    t_end = t_last * dt / 1000.0
    t_mid = t_end / 2.0
    ax.set_xlim(0, t_last)
    ax.set_xticks([0, t_last // 2, t_last])
    ax.set_xticklabels(['0', f'{t_mid:g}', f'{t_end:g}'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('time [s]', fontsize=7)


def _style_recf_profile_axis(ax, show_xlabel):
    """Style mirrored RF-profile axis."""
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 20, 40])
    ax.set_xticklabels(['-20', '0', '20'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('RF (°)', fontsize=7)


def _scale_contrast_series(series, *, response_start, pulse_end):
    """Scale each contrast series entry to ImpR + RF bins.

    ``series`` items may include ``model_xt``, ``data_xt``, ``sem_xt``.
    Returns a list of dicts with ``imp_model``, ``rf_model``, ``imp_sem``,
    ``imp_data``, ``rf_data`` plus passthrough keys.
    """
    center = CENTER_BIN
    sc_kw = dict(response_start=response_start, pulse_end=pulse_end)
    out = []
    for entry in series:
        item = dict(entry)
        model_xt = entry.get("model_xt")
        data_xt = entry.get("data_xt")
        sem_xt = entry.get("sem_xt")
        imp_sem = None
        if model_xt is not None:
            if sem_xt is not None:
                imp_model, rf_model, imp_sem = scale_curve(
                    model_xt, center, sem_xt, **sc_kw,
                )
            else:
                imp_model, rf_model = scale_curve(model_xt, center, **sc_kw)
        else:
            imp_model, rf_model = None, None
        if data_xt is not None:
            imp_data, rf_data = scale_curve(data_xt, center, **sc_kw)
        else:
            imp_data, rf_data = None, None
        item.update(
            imp_model=imp_model,
            rf_model=rf_model,
            imp_sem=imp_sem,
            imp_data=imp_data,
            rf_data=rf_data,
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
    ylo, yhi = TRACE_YLIM
    for item in scaled:
        ls = item.get("linestyle", "-")
        _plot_rf_profile(
            ax, item["rf_data"], color=DATA_COLOR,
            label=item.get("label_data"), linestyle=ls,
        )
        _plot_rf_profile(
            ax, item["rf_model"], color=MODEL_COLOR,
            label=item.get("label_model"), linestyle=ls, filled=True,
        )
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_ylim(ylo, yhi)
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
):
    """Time-course panel for one cell across contrast ``series``.

    ``pre_end`` defaults to ``response_start`` (gray data omits ``[0, pre_end)``).
    Pass ``pre_end=0`` to draw the full data trace including pre-onset.
    ``pulse_end``: white stimulus-on band ``[response_start, pulse_end)``.
    """
    scaled = _scale_contrast_series(
        series, response_start=response_start, pulse_end=pulse_end,
    )
    ylo, yhi = TRACE_YLIM
    t = np.arange(n_t)
    split = int(response_start or 0) if pre_end is None else int(pre_end)
    if title is not None:
        ax.set_title(title, fontsize=8, pad=2)
    traces = [
        {
            "model": item["imp_model"],
            "data": item["imp_data"],
            "sem": item["imp_sem"],
            "linestyle": item.get("linestyle", "-"),
            "point_ix": item.get("point_ix"),
        }
        for item in scaled
    ]
    plot_timecourse(
        ax, t, traces,
        show_sem=any(item["imp_sem"] is not None for item in scaled),
        ylim=(ylo, yhi),
        baseline=baseline,
        show_ylabel=show_ylabel,
        style_xaxis=lambda a: _style_time_axis(a, show_xlabels, n_t, delta_ms),
        pre_end=split,
        show_pre=show_pre,
        pulse_start=response_start,
        pulse_end=pulse_end,
    )


def plot_cell_pair(
    ax_rf,
    ax_time,
    title,
    series,
    *,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    baseline=None,
    n_t=None,
    response_start=None,
    show_pre=False,
    pulse_end=None,
    delta_ms=None,
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
        show_xlabels=show_xlabels,
        show_ylabel=show_ylabel,
        baseline=baseline,
        n_t=n_t,
        response_start=response_start,
        show_pre=show_pre,
        pulse_end=pulse_end,
        delta_ms=delta_ms,
    )


def plot_cell_pair_slices(
    ax_rf,
    ax_time,
    title,
    series,
    slice_labels,
    *,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    baseline=None,
    n_t=None,
    response_start=None,
    show_pre=False,
    pulse_end=None,
    delta_ms=None,
):
    """RF + time panels with per-slice overlays across contrast ``series``."""
    center = CENTER_BIN
    sc_kw = dict(response_start=response_start, pulse_end=pulse_end)
    ylo, yhi = TRACE_YLIM
    colors = overlay_model_reds(len(slice_labels))
    t = np.arange(n_t)
    pre_end = int(response_start or 0)
    mark_pulse(ax_time, response_start, pulse_end)

    for si, item in enumerate(series):
        ls = item.get("linestyle", "-")
        model_xt = item.get("model_xt")
        data_xt = item.get("data_xt")
        slice_overlay = item.get("slice_overlay") or {}
        imp_model = rf_model = None
        if model_xt is not None:
            imp_model, rf_model = scale_curve(model_xt, center, **sc_kw)
        imp_data = rf_data = None
        if data_xt is not None:
            imp_data, rf_data = scale_curve(data_xt, center, **sc_kw)
        slice_imps = {}
        slice_rfs = {}
        for label in slice_labels:
            if label in slice_overlay:
                imp_s, rf_s = scale_curve(slice_overlay[label], center, **sc_kw)
                slice_imps[label] = imp_s
                slice_rfs[label] = rf_s

        _plot_rf_profile(
            ax_rf, rf_data, color=DATA_COLOR,
            label=item.get("label_data"), linestyle=ls,
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
            ax_time, t, imp_data, pre_end=pre_end, show_pre=False, draw_pre=False,
            color=DATA_COLOR, linestyle=ls, linewidth=TRACE_LW,
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
    ax_rf.set_ylim(ylo, yhi)
    _style_recf_profile_axis(ax_rf, show_xlabels)
    if show_ylabel:
        ax_rf.set_ylabel('mV', fontsize=7)
    ax_rf.tick_params(labelsize=6)
    annotate_baseline(ax_rf, baseline)
    if show_legend:
        ax_rf.legend(loc='upper right', fontsize=6, frameon=False)

    ax_time.set_ylim(ylo, yhi)
    _style_time_axis(ax_time, show_xlabels, n_t, delta_ms)
    if show_ylabel:
        ax_time.set_ylabel('mV', fontsize=7)
    ax_time.tick_params(labelsize=6)
    annotate_baseline(ax_time, baseline)



def _as_index(node_index, device):
    if not torch.is_tensor(node_index):
        return torch.tensor(node_index, dtype=torch.long, device=device)
    return node_index.to(device)


@torch.no_grad()
def _simulate(session, z, node_index, return_v_onset=False):
    node_index = _as_index(node_index, z.device)
    schema = list(session.schema)
    backend = session.backend
    p = training.assign_params(z, schema, backend)
    stacked, ref = training.run_nodes(
        session, p, node_index=node_index, return_v_onset=True,
    )
    trace = apply_out_scale(
        p, stacked.transpose(0, 1), node_index, backend,
    )
    if return_v_onset:
        return trace, ref
    return trace


def calc_ca_full_all(session, z, return_v_onset=False):
    n_cells = session.backend.n_cells
    mt = session.n_t
    ca_full = np.zeros((n_cells, 9, mt))
    ref_full = np.full((n_cells, 9), np.nan)
    for col in range(5):
        col_index = torch.arange(
            col * n_cells,
            (col + 1) * n_cells,
            dtype=torch.long,
            device=z.device,
        )
        if return_v_onset:
            trace, ref = _simulate(session, z, col_index, return_v_onset=True)
            ca_full[:, col + 2] = trace.cpu().numpy()
            ref_full[:, col + 2] = ref.cpu().numpy()
        else:
            ca_full[:, col + 2] = _simulate(session, z, col_index).cpu().numpy()
    if return_v_onset:
        return ca_full, ref_full
    return ca_full


def _fill_member_cube(cube, sem, ti, ft_global, type_idx, du, dv, plot_traces):
    """Fill mirrored profile bins from radius-ring means."""
    mask = type_idx == ft_global
    if not mask.any():
        return
    rows = np.where(mask)[0]
    by_radius: dict[float, list] = {}
    for row in rows:
        radius = round(float(euclid_hex_dist(int(du[row]), int(dv[row]))), 6)
        by_radius.setdefault(radius, []).append(int(row))
    for radius, row_ix in by_radius.items():
        bins = _radius_to_profile_bins(radius)
        if not bins:
            continue
        traces = plot_traces[row_ix]
        mean_trace = traces.mean(axis=0)
        sem_trace = sem_from_traces(traces, single_hex=False)
        for bin_j in bins:
            cube[ti, bin_j] = mean_trace
            sem[ti, bin_j] = sem_trace


@dataclass
class SpotTraceBundle:
    """One forward pass; full cost-extent readout over all types."""

    cells: list
    group_rows: list | None = None
    session: object = None
    slice_overlay: dict[str, dict[str, np.ndarray]] | None = None
    slice_labels: list[str] | None = None
    slice_x_list: list | None = None
    slice_y_list: list | None = None
    n_t: int = 0
    prep_s: float = 0.0
    v_ref_by_name: dict = field(default_factory=dict)
    v_ref_name: str | None = None
    response_start: int | None = None
    show_pre: bool = True
    pulse_end: int | None = None

    @property
    def has_slices(self):
        return bool(self.slice_overlay)


def _group_rows_from_groups(groups, names):
    name_to_i = {str(n): i for i, n in enumerate(names)}
    group_rows = []
    for names_row in groups:
        row_idx = [name_to_i[str(n)] for n in names_row if str(n) in name_to_i]
        if row_idx:
            group_rows.append(row_idx)
    return group_rows


def _spot_all_cell_names(session):
    if session.backend.network is None:
        raise ValueError("_spot_all_cell_names requires session.backend.network")
    return plot_cells_in_order(session.backend.network.cell_names)


def _spot_readout_bundle_view(bundle):
    """ca-data view: same traces, rows restricted to ``pack_readout_cells``."""
    session = bundle.session
    present = pack_readout_cells(session, session.primary_readout.name)
    groups, names = plot_present_layout(present)
    by_name = {c['name']: c for c in bundle.cells}
    cells = [by_name[n] for n in names if n in by_name]
    cell_names = [c['name'] for c in cells]
    return SpotTraceBundle(
        cells=cells,
        group_rows=_group_rows_from_groups(groups, cell_names),
        session=session,
        n_t=bundle.n_t,
        v_ref_by_name=bundle.v_ref_by_name,
        v_ref_name=bundle.v_ref_name,
        response_start=bundle.response_start,
        show_pre=bundle.show_pre,
        pulse_end=bundle.pulse_end,
    )


def _cells_with_group_rows(groups, names, build_cell):
    cells = [build_cell(str(name)) for name in names]
    return cells, _group_rows_from_groups(groups, names)


def _spot_cubes_from_row_mask(rows, mask):
    names = rows['names']
    cell_names = rows['cell_names']
    mt = rows['mt']
    type_idx = rows['type_idx'][mask]
    du = rows['du'][mask]
    dv = rows['dv'][mask]
    plot_traces = rows['plot_traces'][mask]
    out = {}
    sem_dummy = np.full((1, 9, mt), np.nan)
    for ft in names:
        cube = np.full((1, 9, mt), np.nan)
        ft_global = cell_names.index(ft)
        _fill_member_cube(cube, sem_dummy, 0, ft_global, type_idx, du, dv, plot_traces)
        out[ft] = cube[0]
    return out


def _spot_slice_overlay(rows, batches, at_x_list, at_y_list):
    overlay = {}
    labels = []
    batch_idx = rows['batch_idx']
    for label, at_x, at_y in slice_coord_specs(at_x_list, at_y_list):
        match_b = batches_at_stim_xy(batches, at_x=at_x, at_y=at_y)
        if not match_b:
            print(f'skip slice overlay {label}: no stimulus batch')
            continue
        mask = np.isin(batch_idx, match_b)
        cubes = _spot_cubes_from_row_mask(rows, mask)
        if not any(np.isfinite(c).any() for c in cubes.values()):
            print(f'skip slice overlay {label}: no readouts')
            continue
        overlay[label] = cubes
        labels.append(label)
    if not overlay:
        return None, None
    return overlay, labels



def _cells_from_cube(names, cube, sem, baselines, *, single_hex, n_by_name=None):
    n_by_name = n_by_name or {}
    return [
        dict(
            name=n,
            cube=cube[i],
            sem=None if single_hex else sem[i],
            baseline=baselines.get(n),
            n=n_by_name.get(n),
        )
        for i, n in enumerate(names)
    ]


@torch.no_grad()
def _spot_forward_rows(
    session, z, *,
    save_trace_csv_dir=None, at_x=None, at_y=None,
):
    """One forward; cost-extent node layout over all network types."""
    pack = session.primary_readout
    schema = list(session.schema)
    p = training.assign_params(z, schema, session.backend)
    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    trace_full, v_onset, _v_full = training.run_full(
        session, p, i_sti, return_v_onset=True, pack=pack,
    )
    v_onset_np = v_onset[0].cpu().numpy()
    save_forward_trace_csvs(
        save_trace_csv_dir, pack.name,
        ref=v_onset_np, trace_full=trace_full,
        ref_stem='spot_v_onset',
    )
    C = session.backend.network
    cell_names = list(C.cell_names)
    mt = int(i_sti.shape[1])

    opts = dict((session.train_opts or {}).get(f"{pack.name}_stimulus_opts") or {})
    spot = spot_from_opts(C, stimulus_opts=opts)
    batches = spot_stimulus_batches(spot)
    groups, names = plot_present_layout(_spot_all_cell_names(session))

    (
        batch_idx, node_idx, _radius, type_idx, _stim_u, _stim_v, du, dv, center_row,
    ) = spot_center_bin_layout(
        C, batches, pack_spot_cost_radii(pack), pack.cost_extent,
    )

    raw = trace_full[batch_idx, :, node_idx]
    plot_traces = apply_out_scale(
        p, raw, node_idx, session.backend,
    ).cpu().numpy()

    stim_pre_ms = opts.get("pre_ms")
    dt = float(opts.get("delta_ms", DELTA_MS))
    stim_t_onset = (
        ms_to_t(float(stim_pre_ms), delta_ms=dt) if stim_pre_ms is not None else None
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
    )
    mask = np.asarray(center_row, dtype=bool)
    if at_x is not None or at_y is not None:
        match_b = batches_at_stim_xy(batches, at_x=at_x, at_y=at_y)
        mask = mask & np.isin(batch_idx, match_b)
    nodes_by_name = {
        name: np.unique(node_idx[mask & (type_idx == cell_names.index(name))])
        for name in names
    }
    rows['baselines'] = baselines_for_types(
        v_onset_np, nodes_by_name, v_ref_by_type_name(z, session),
    )
    rows['groups'] = groups
    return rows


def _spot_cube_from_rows(rows, session):
    names = rows['names']
    mt = rows['mt']
    cube = np.full((len(names), 9, mt), np.nan)
    sem = np.full((len(names), 9, mt), np.nan)
    for ti, ft in enumerate(names):
        ft_global = rows['cell_names'].index(ft)
        _fill_member_cube(
            cube, sem, ti, ft_global,
            rows['type_idx'], rows['du'], rows['dv'], rows['plot_traces'],
        )
    single_hex = suppress_cost_sem(session, task=rows['pack'].name)
    n_by_name = readout_n_by_name(
        rows['type_idx'], rows['cell_names'], names, rows['node_idx'],
    )
    cells = _cells_from_cube(
        names, cube, sem, rows['baselines'],
        single_hex=single_hex, n_by_name=n_by_name,
    )
    group_rows = _group_rows_from_groups(rows['groups'], names)
    return cells, group_rows, mt


@torch.no_grad()
def network_spot_trace_bundle(
    session, z, *,
    at_x_list=None, at_y_list=None,
    save_trace_csv_dir: str | None = None, show_pre=True,
):
    """Run one forward; full cost-extent spot traces over all types."""
    t_prep0 = time.perf_counter()
    at_x = at_x_list[0] if at_x_list else None
    at_y = at_y_list[0] if at_y_list else None
    rows = _spot_forward_rows(
        session, z,
        save_trace_csv_dir=save_trace_csv_dir,
        at_x=at_x, at_y=at_y,
    )
    cells, group_rows, n_t = _spot_cube_from_rows(rows, session)
    slice_overlay, slice_labels = (None, None)
    if at_x_list is not None or at_y_list is not None:
        slice_overlay, slice_labels = _spot_slice_overlay(
            rows, rows['batches'], at_x_list, at_y_list,
        )
    return SpotTraceBundle(
        cells=cells,
        group_rows=group_rows,
        session=session,
        slice_overlay=slice_overlay,
        slice_labels=slice_labels,
        slice_x_list=at_x_list,
        slice_y_list=at_y_list,
        n_t=n_t,
        prep_s=time.perf_counter() - t_prep0,
        v_ref_by_name=v_ref_by_type_name(z, session),
        v_ref_name=v_ref_schema_name(session.schema),
        response_start=rows.get('t_onset'),
        show_pre=bool(show_pre),
        pulse_end=rows.get('pulse_end'),
    )


def _spot_suptitle(title, bundle):
    if bundle is not None and bundle.has_slices:
        scope = hex_at_scope_tag(bundle.slice_x_list, bundle.slice_y_list)
        return f'{title}  [{scope}, overlay + total]'
    return title


def _plot_spot_figure(
    path, *,
    timer,
    bundles,
    title,
    data_cubes=None,
    ncols,
    figsize_fn,
    gridspec_kw,
    suptitle_fs=12,
):
    """Draw spot figure from ``bundles`` (contrast → SpotTraceBundle)."""
    order = contrast_order(bundles)
    if not order:
        raise ValueError("_plot_spot_figure requires at least one bundle")
    primary = bundles[order[0]]
    cells = primary.cells
    group_rows = primary.group_rows
    has_slices = primary.has_slices
    slice_labels = primary.slice_labels or []
    n_t = primary.n_t
    response_start = primary.response_start
    pulse_end = getattr(primary, "pulse_end", None)
    delta_ms = float(primary.session.delta_ms)
    show_pre = getattr(primary, "show_pre", True)
    timer.end_prep()

    sessions = {c: bundles[c].session for c in order}
    data_by_contrast = resolve_spot_data_cubes(sessions, data_cubes)

    cells_by_contrast = {}
    for c in order:
        cells_by_contrast[c] = {cell["name"]: cell for cell in bundles[c].cells}

    nrows = 2 * len(group_rows)
    fig = plt.figure(figsize=figsize_fn(ncols, nrows))
    gs = fig.add_gridspec(nrows, ncols, **gridspec_kw)
    legend_done = False

    def _series_for_cell(name, *, with_slices):
        series = []
        for c in order:
            cell = cells_by_contrast[c].get(name)
            if cell is None:
                continue
            data_cells = data_by_contrast.get(c) or {}
            entry = {
                "contrast": c,
                "model_xt": cell["cube"],
                "data_xt": data_cells.get(name),
                "sem_xt": cell.get("sem"),
                "baseline": cell.get("baseline"),
                "linestyle": contrast_linestyle(c),
                "label_data": f"{c} data",
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
        cell_title = bundle_cell_title(primary, name, cell_primary.get("n"))
        if has_slices and primary.slice_overlay is not None:
            series = _series_for_cell(name, with_slices=True)
            slice_xt = (series[0].get("slice_overlay") or {}) if series else {}
            if not slice_xt:
                ax_rf.axis("off")
                ax_time.axis("off")
                return
            plot_cell_pair_slices(
                ax_rf, ax_time, cell_title, series, slice_labels,
                show_legend=not legend_done,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                n_t=n_t,
                baseline=cell_primary.get("baseline"),
                response_start=response_start,
                show_pre=show_pre,
                pulse_end=pulse_end,
                delta_ms=delta_ms,
            )
        else:
            series = _series_for_cell(name, with_slices=False)
            plot_cell_pair(
                ax_rf, ax_time, cell_title, series,
                show_legend=not legend_done,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                n_t=n_t,
                baseline=cell_primary.get("baseline"),
                response_start=response_start,
                show_pre=show_pre,
                pulse_end=pulse_end,
                delta_ms=delta_ms,
            )
        legend_done = True

    for gi, row_idx in enumerate(group_rows):
        rf_row = 2 * gi
        start = (ncols - len(row_idx)) // 2
        for j, ci in enumerate(row_idx):
            col = start + j
            cell_on = cells[ci]
            ax_rf = fig.add_subplot(gs[rf_row, col])
            ax_time = fig.add_subplot(gs[rf_row + 1, col])
            _plot_cell(
                cell_on["name"], cell_on, ax_rf, ax_time,
                show_ylabel=(j == 0), show_xlabels=True,
            )
    fig.suptitle(_spot_suptitle(title, primary), fontsize=suptitle_fs)
    timer.end_draw()
    save_figure(fig, path, dpi=150)
    timer.log(path)


def plot_network_spot_data(path, *, bundles, title, data_cubes=None):
    """Draw ca-data figure (pack readout types) from contrast → bundle."""
    timer = PlotTimer(prior_prep=bundle_prep_s(*bundles.values()))
    views = {
        c: _spot_readout_bundle_view(b) for c, b in bundles.items()
    }
    _plot_spot_figure(
        path,
        timer=timer,
        bundles=views,
        title=title,
        data_cubes=data_cubes,
        ncols=5,
        figsize_fn=lambda c, r: (3.0 * c, 2.5 * r),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
    )


def plot_network_spot_all(path, *, bundles, title, data_cubes=None):
    """Draw ca-all figure (all types) from contrast → bundle."""
    timer = PlotTimer(prior_prep=bundle_prep_s(*bundles.values()))
    _plot_spot_figure(
        path,
        timer=timer,
        bundles=bundles,
        title=title,
        data_cubes=data_cubes,
        ncols=8,
        figsize_fn=lambda c, r: (2.2 * c, 2.5 * r),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
    )
