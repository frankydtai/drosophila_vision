"""Spot plotting (network spot target).

Network RF bins are ring means: r=0 -> j4, r=1 -> j3/j5, r=2 -> j2/j6.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuron_model.param import DELTAT_MS
import training as fc
from figure.readout import (
    pack_readout_types,
    plot_present_layout,
    plot_types_in_order,
    spot_ref_cubes,
)
from figure.util import (
    DATA_COLOR,
    MODEL_COLOR,
    TRACE_LW,
    TRACE_YLIM,
    PlotTimer,
    annotate_baseline,
    batches_at_stim_xy,
    bundle_cell_title,
    bundle_prep_s,
    column_at_scope_tag,
    overlay_model_reds,
    plot_timecourse,
    readout_n_by_name,
    save_figure,
    save_forward_trace_csvs,
    sem_from_traces,
    slice_coord_specs,
    suppress_cost_sem,
    v_th_by_type_name,
)
from task.spot.input import (
    euclid_hex_dist,
    spotting_from_opts,
    spot_stimulus_batches,
)
from task.spot.data import (
    resolve_spot_cost_radii,
    spot_center_bin_layout,
)

CENTER_BIN = 4  # RecF spatial centre bin (j=4 in 0..8)
RF_N_BINS = 9
RF_BIN_X = np.arange(RF_N_BINS) * 5  # j=0..8 on mirrored RF axis (-20..20)


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


def _baseline_from_ref_grid(ref_grid, row_i):
    """Resting potential at stimulus onset for one cell type (center column)."""
    return ref_grid[row_i, CENTER_BIN]


def _session_spot_timing(session):
    """Extract t_on / maxtime from session stimulus opts (or None for defaults)."""
    opts = (session.train_opts or {}).get(
        f"{session.primary_pack.name}_stimulus_opts",
    ) or {}
    t_on = opts.get("t_on")
    maxtime = opts.get("maxtime")
    return (int(t_on) if t_on is not None else None,
            int(maxtime) if maxtime is not None else None)


def resolve_spot_ref_cubes(session_1, session_2=None, ref_cubes=None, ref_cubes_2=None,
                           *, v_delta=False):
    dual = session_2 is not None
    t_on_1, mt_1 = _session_spot_timing(session_1)
    if ref_cubes is not None:
        ref_1 = ref_cubes
    elif dual:
        ref_1 = spot_ref_cubes(session_1, 'spot_bright', dark=False,
                               t_on=t_on_1, maxtime=mt_1, v_delta=v_delta)
    else:
        ref_1 = spot_ref_cubes(
            session_1, session_1.primary_pack.name, dark=_session_dark(session_1),
            t_on=t_on_1, maxtime=mt_1, v_delta=v_delta,
        )
    if ref_cubes_2 is not None:
        ref_2 = ref_cubes_2
    elif dual:
        t_on_2, mt_2 = _session_spot_timing(session_2)
        ref_2 = spot_ref_cubes(session_2, 'spot_dark', dark=True,
                               t_on=t_on_2, maxtime=mt_2, v_delta=v_delta)
    else:
        ref_2 = None
    return ref_1, ref_2


def _session_dark(session):
    return session.primary_pack.name == "spot_dark"


def scale_curve(xt, center, sem_xt=None, *, response_start=None):
    """Center time course + discrete RF bins from one ``(9, T)`` cube."""
    if response_start is None:
        raise ValueError("scale_curve requires response_start (t_on step)")
    t0 = int(response_start)
    imp = xt[center]
    resp = imp[t0:]
    if not np.isfinite(resp).any():
        if sem_xt is not None:
            return None, None, None
        return None, None
    maxt = t0 + int(np.argmax(np.abs(resp)))
    spatial = np.asarray(xt[:, maxt], dtype=np.float64)
    if sem_xt is not None:
        return imp, spatial, sem_xt[center]
    return imp, spatial


def _plot_rf_profile(ax, rf, *, color, label=None, linestyle='-'):
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
        markersize=6,
        fillstyle='none',
        markeredgewidth=1.2,
    )
    if linestyle == '--':
        kw['markeredgewidth'] = 1.0
    ax.plot(RF_BIN_X[mask], rf[mask], **kw)


def _style_time_axis(ax, show_xlabel, maxtime):
    t_end = maxtime * DELTAT_MS / 1000.0
    t_mid = t_end / 2.0
    ax.set_xlim(0, maxtime)
    ax.set_xticks([0, maxtime // 2, maxtime])
    ax.set_xticklabels(['0', f'{t_mid:g}', f'{t_end:g}'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('time [s]', fontsize=7)


def _style_recf_profile_axis(ax, show_xlabel):
    """Style mirrored RF-profile axis."""
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 20, 40])
    ax.set_xticklabels(['-20', '0', '20'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('RF profile', fontsize=7)


def plot_cell_pair(
    ax_rf,
    ax_time,
    model_xt,
    ref_xt,
    title,
    *,
    sem_xt=None,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    baseline=None,
    maxtime=None,
    model_2_xt=None,
    ref_2_xt=None,
    baseline_2=None,
    linestyle_1='-',
    linestyle_2='--',
    label_1_data='bright data',
    label_1_model='bright model',
    label_2_data='dark data',
    label_2_model='dark model',
    response_start=None,
):
    center = CENTER_BIN
    sc_kw = dict(response_start=response_start)
    if sem_xt is not None:
        imp_model, rf_model, imp_sem = scale_curve(model_xt, center, sem_xt, **sc_kw)
    else:
        imp_model, rf_model = scale_curve(model_xt, center, **sc_kw)
        imp_sem = None
    if ref_xt is not None:
        imp_data, rf_data = scale_curve(ref_xt, center, **sc_kw)
    else:
        imp_data, rf_data = None, None
    model_2_imp_model = model_2_rf_model = model_2_imp_data = model_2_rf_data = None
    if model_2_xt is not None:
        model_2_imp_model, model_2_rf_model = scale_curve(model_2_xt, center, **sc_kw)
    if ref_2_xt is not None:
        model_2_imp_data, model_2_rf_data = scale_curve(ref_2_xt, center, **sc_kw)
    ylo, yhi = TRACE_YLIM

    _plot_rf_profile(ax_rf, rf_data, color=DATA_COLOR, label=label_1_data, linestyle=linestyle_1)
    _plot_rf_profile(ax_rf, rf_model, color=MODEL_COLOR, label=label_1_model, linestyle=linestyle_1)
    _plot_rf_profile(
        ax_rf, model_2_rf_data, color=DATA_COLOR, label=label_2_data, linestyle=linestyle_2,
    )
    _plot_rf_profile(
        ax_rf, model_2_rf_model, color=MODEL_COLOR, label=label_2_model, linestyle=linestyle_2,
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

    t = np.arange(maxtime)
    plot_timecourse(
        ax_time, t, imp_model,
        data=imp_data,
        sem=imp_sem,
        show_sem=imp_sem is not None,
        model_2=model_2_imp_model,
        data_2=model_2_imp_data,
        linestyle_2=linestyle_2,
        ylim=(ylo, yhi),
        baseline=baseline_2 if baseline_2 is not None else baseline,
        show_ylabel=show_ylabel,
        linestyle=linestyle_1,
        style_xaxis=lambda ax: _style_time_axis(ax, show_xlabels, maxtime),
    )


def plot_cell_pair_slices(
    ax_rf,
    ax_time,
    model_xt,
    ref_xt,
    title,
    slice_overlay_xt,
    slice_labels,
    *,
    model_2_xt=None,
    ref_2_xt=None,
    slice_overlay_2_xt=None,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    baseline=None,
    maxtime=None,
    baseline_2=None,
    linestyle_1='-',
    linestyle_2='--',
    label_1_data='bright data',
    label_1_total='bright total',
    label_2_data='dark data',
    label_2_total='dark total',
    response_start=None,
):
    center = CENTER_BIN
    sc_kw = dict(response_start=response_start)
    imp_model, rf_model = scale_curve(model_xt, center, **sc_kw)
    imp_data, rf_data = (None, None)
    if ref_xt is not None:
        imp_data, rf_data = scale_curve(ref_xt, center, **sc_kw)
    model_2_imp_model = model_2_rf_model = model_2_imp_data = model_2_rf_data = None
    if model_2_xt is not None:
        model_2_imp_model, model_2_rf_model = scale_curve(model_2_xt, center, **sc_kw)
    if ref_2_xt is not None:
        model_2_imp_data, model_2_rf_data = scale_curve(ref_2_xt, center, **sc_kw)

    slice_imps = {}
    slice_rfs = {}
    slice_2_imps = {}
    slice_2_rfs = {}
    for label in slice_labels:
        if label in slice_overlay_xt:
            imp_s, rf_s = scale_curve(slice_overlay_xt[label], center, **sc_kw)
            slice_imps[label] = imp_s
            slice_rfs[label] = rf_s
        if slice_overlay_2_xt and label in slice_overlay_2_xt:
            imp_s, rf_s = scale_curve(slice_overlay_2_xt[label], center, **sc_kw)
            slice_2_imps[label] = imp_s
            slice_2_rfs[label] = rf_s

    ylo, yhi = TRACE_YLIM

    colors = overlay_model_reds(len(slice_labels))
    _plot_rf_profile(ax_rf, rf_data, color=DATA_COLOR, label=label_1_data, linestyle=linestyle_1)
    for i, label in enumerate(slice_labels):
        _plot_rf_profile(
            ax_rf, slice_rfs.get(label), color=colors[i], label=label, linestyle=linestyle_1,
        )
    _plot_rf_profile(
        ax_rf, rf_model, color=colors[-1], label=label_1_total, linestyle=linestyle_1,
    )
    _plot_rf_profile(
        ax_rf, model_2_rf_data, color=DATA_COLOR, label=label_2_data, linestyle=linestyle_2,
    )
    for i, label in enumerate(slice_labels):
        _plot_rf_profile(
            ax_rf, slice_2_rfs.get(label), color=colors[i], linestyle=linestyle_2,
        )
    _plot_rf_profile(
        ax_rf, model_2_rf_model, color=colors[-1], label=label_2_total, linestyle=linestyle_2,
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

    t = np.arange(maxtime)
    if imp_data is not None:
        ax_time.plot(t, imp_data, color=DATA_COLOR, linewidth=TRACE_LW, linestyle=linestyle_1)
    for i, label in enumerate(slice_labels):
        imp_s = slice_imps.get(label)
        if imp_s is not None:
            ax_time.plot(
                t, imp_s, color=colors[i], linewidth=TRACE_LW,
                label=label, linestyle=linestyle_1,
            )
    if imp_model is not None:
        ax_time.plot(
            t, imp_model, color=colors[-1], linewidth=TRACE_LW,
            label=label_1_total, linestyle=linestyle_1,
        )
    if model_2_imp_data is not None:
        ax_time.plot(t, model_2_imp_data, color=DATA_COLOR, linewidth=TRACE_LW, linestyle=linestyle_2)
    for i, label in enumerate(slice_labels):
        imp_s = slice_2_imps.get(label)
        if imp_s is not None:
            ax_time.plot(
                t, imp_s, color=colors[i], linewidth=TRACE_LW,
                linestyle=linestyle_2,
            )
    if model_2_imp_model is not None:
        ax_time.plot(t, model_2_imp_model, color=colors[-1], linewidth=TRACE_LW, linestyle=linestyle_2)
    ax_time.set_ylim(ylo, yhi)
    _style_time_axis(ax_time, show_xlabels, maxtime)
    if show_ylabel:
        ax_time.set_ylabel('mV', fontsize=7)
    ax_time.tick_params(labelsize=6)
    annotate_baseline(ax_time, baseline_2 if baseline_2 is not None else baseline)


def _as_index(neuron_index, device):
    if not torch.is_tensor(neuron_index):
        return torch.tensor(neuron_index, dtype=torch.long, device=device)
    return neuron_index.to(device)


def _scale_plot_traces(raw, scale):
    """``(N, maxtime)`` readout -> scaled traces for spot timecourse plots."""
    return scale[:, None] * raw


def _mask_pre_ton_plot_traces(traces, *, show_pre=True, t_on_step=None):
    """Zero absolute steps ``[0, t_on)`` when ``show_pre`` is false."""
    if show_pre:
        return traces
    if t_on_step is None:
        raise ValueError("_mask_pre_ton_plot_traces requires t_on_step")
    t_on_step = int(t_on_step)
    if torch.is_tensor(traces):
        out = traces.clone()
        out[..., :t_on_step] = 0
        return out
    out = np.asarray(traces, dtype=np.float64).copy()
    out[..., :t_on_step] = 0.0
    return out


@torch.no_grad()
def _simulate(session, z, neuron_index, return_ref=False, *, trace_kind='ca', show_pre=True):
    neuron_index = _as_index(neuron_index, z.device)
    schema = list(session.schema)
    backend = session.backend
    p = fc.assign_params(z, schema, backend)
    stacked, ref = fc.run_units(
        session, p, neuron_index=neuron_index, return_ref=True,
        return_v_delta=(trace_kind == 'v'),
    )
    if trace_kind == 'v':
        scale = torch.ones((int(neuron_index.shape[0]),), dtype=stacked.dtype, device=stacked.device)
    else:
        scale = fc.out_scale_for_units(p, neuron_index, backend)
    trace = _scale_plot_traces(stacked.transpose(0, 1), scale)
    trace = _mask_pre_ton_plot_traces(trace, show_pre=show_pre)
    if return_ref:
        return trace, ref
    return trace


def calc_ca_full_all(session, z, return_ref=False, *, trace_kind='ca', show_pre=True):
    n_types = session.backend.n_types
    mt = session.maxtime
    ca_full = np.zeros((n_types, 9, mt))
    ref_full = np.full((n_types, 9), np.nan)
    for col in range(5):
        col_index = torch.arange(
            col * n_types,
            (col + 1) * n_types,
            dtype=torch.long,
            device=z.device,
        )
        if return_ref:
            trace, ref = _simulate(
                session, z, col_index, return_ref=True, trace_kind=trace_kind, show_pre=show_pre,
            )
            ca_full[:, col + 2] = trace.cpu().numpy()
            ref_full[:, col + 2] = ref.cpu().numpy()
        else:
            ca_full[:, col + 2] = _simulate(
                session, z, col_index, trace_kind=trace_kind, show_pre=show_pre,
            ).cpu().numpy()
    if return_ref:
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
        sem_trace = sem_from_traces(traces, single_column=False)
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
    maxtime: int = 0
    prep_s: float = 0.0
    v_th_by_name: dict = field(default_factory=dict)
    response_start: int | None = None
    trace_kind: str = 'ca'

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


def _spot_all_type_names(session):
    if session.backend.network is None:
        raise ValueError("_spot_all_type_names requires session.backend.network")
    return plot_types_in_order(session.backend.network.type_names)


def _spot_readout_bundle_view(bundle):
    """ca-data view: same traces, rows restricted to ``pack_readout_types``."""
    session = bundle.session
    present = pack_readout_types(session, session.primary_pack.name)
    groups, names = plot_present_layout(present)
    by_name = {c['name']: c for c in bundle.cells}
    cells = [by_name[n] for n in names if n in by_name]
    cell_names = [c['name'] for c in cells]
    return SpotTraceBundle(
        cells=cells,
        group_rows=_group_rows_from_groups(groups, cell_names),
        session=session,
        maxtime=bundle.maxtime,
        v_th_by_name=bundle.v_th_by_name,
        response_start=bundle.response_start,
    )


def _cells_with_group_rows(groups, names, build_cell):
    cells = [build_cell(str(name)) for name in names]
    return cells, _group_rows_from_groups(groups, names)


def _spot_cubes_from_row_mask(rows, mask):
    names = rows['names']
    type_names = rows['type_names']
    mt = rows['mt']
    type_idx = rows['type_idx'][mask]
    du = rows['du'][mask]
    dv = rows['dv'][mask]
    plot_traces = rows['plot_traces'][mask]
    out = {}
    sem_dummy = np.full((1, 9, mt), np.nan)
    for ft in names:
        cube = np.full((1, 9, mt), np.nan)
        ft_global = type_names.index(ft)
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



def _cells_from_cube(names, cube, sem, baselines, *, single_column, n_by_name=None):
    n_by_name = n_by_name or {}
    return [
        dict(
            name=n,
            cube=cube[i],
            sem=None if single_column else sem[i],
            baseline=baselines.get(n),
            n=n_by_name.get(n),
        )
        for i, n in enumerate(names)
    ]


def _spot_baselines(rows, v_ref, names, *, at_x=None, at_y=None):
    """Mean ``v_ref`` per type over stim-centred ``(0, 0)`` units (matches trace scope)."""
    v_ref = np.asarray(v_ref, dtype=np.float64)
    batch_idx = rows['batch_idx']
    unit_idx = rows['unit_idx']
    type_idx = rows['type_idx']
    type_names = rows['type_names']
    center_row = rows['center_row']

    mask = np.asarray(center_row, dtype=bool)
    if at_x is not None or at_y is not None:
        match_b = batches_at_stim_xy(rows['batches'], at_x=at_x, at_y=at_y)
        mask = mask & np.isin(batch_idx, match_b)

    out = {}
    for name in names:
        ti = type_names.index(name)
        units = np.unique(unit_idx[mask & (type_idx == ti)])
        out[name] = float(v_ref[units].mean()) if units.size else np.nan
    return out


@torch.no_grad()
def _spot_forward_rows(
    session, z, *, trace_kind='ca',
    save_trace_csv_dir=None, at_x=None, at_y=None, show_pre=True,
):
    """One forward; cost-extent unit layout over all network types."""
    pack = session.primary_pack
    schema = list(session.schema)
    p = fc.assign_params(z, schema, session.backend)
    sig = pack.signal if pack.signal.dim() == 3 else pack.signal.unsqueeze(0)
    if trace_kind == 'v':
        trace_full, v_ref, _v_full = fc.run_full(
            session, p, sig, return_ref=True, return_v_delta=True, pack=pack,
        )
    else:
        trace_full, v_ref = fc.run_full(
            session, p, sig, return_ref=True, pack=pack,
        )
    v_ref_np = v_ref[0].cpu().numpy()
    save_forward_trace_csvs(
        save_trace_csv_dir, pack.name,
        trace_kind=trace_kind, ref=v_ref_np, trace_full=trace_full,
        ref_stem='spot_v_ref' if trace_kind == 'v' else None,
    )
    C = session.backend.network
    type_names = list(C.type_names)
    mt = int(sig.shape[1])

    opts = dict((session.train_opts or {}).get(f"{pack.name}_stimulus_opts") or {})
    spotting = spotting_from_opts(C, stimulus_opts=opts)
    batches = spot_stimulus_batches(spotting)
    groups, names = plot_present_layout(_spot_all_type_names(session))

    cost_radii = resolve_spot_cost_radii(stimulus_opts=opts)
    (
        batch_idx, unit_idx, _radius, type_idx, _stim_u, _stim_v, du, dv, center_row,
    ) = spot_center_bin_layout(C, batches, cost_radii, pack.cost_extent)

    raw = trace_full[batch_idx, :, unit_idx]
    if trace_kind == 'v':
        scale = torch.ones((int(raw.shape[0]),), dtype=raw.dtype, device=raw.device)
    else:
        scale = fc.out_scale_for_units(
            p, torch.as_tensor(unit_idx, dtype=torch.long, device=z.device), session.backend,
        )
    stim_t_on = opts.get("t_on")
    plot_traces = _mask_pre_ton_plot_traces(
        _scale_plot_traces(raw, scale), show_pre=show_pre,
        t_on_step=int(stim_t_on) if stim_t_on is not None else None,
    ).cpu().numpy()
    rows = dict(
        names=names,
        type_names=type_names,
        type_idx=type_idx,
        unit_idx=unit_idx,
        du=du,
        dv=dv,
        center_row=center_row,
        plot_traces=plot_traces,
        t_on=int(stim_t_on) if stim_t_on is not None else None,
        batch_idx=batch_idx,
        batches=batches,
        mt=mt,
        pack=pack,
    )
    rows['baselines'] = _spot_baselines(rows, v_ref_np, names, at_x=at_x, at_y=at_y)
    rows['groups'] = groups
    return rows


def _spot_cube_from_rows(rows, session):
    names = rows['names']
    mt = rows['mt']
    cube = np.full((len(names), 9, mt), np.nan)
    sem = np.full((len(names), 9, mt), np.nan)
    for ti, ft in enumerate(names):
        ft_global = rows['type_names'].index(ft)
        _fill_member_cube(
            cube, sem, ti, ft_global,
            rows['type_idx'], rows['du'], rows['dv'], rows['plot_traces'],
        )
    single_column = suppress_cost_sem(session, target=rows['pack'].name)
    n_by_name = readout_n_by_name(
        rows['type_idx'], rows['type_names'], names, rows['unit_idx'],
    )
    cells = _cells_from_cube(
        names, cube, sem, rows['baselines'],
        single_column=single_column, n_by_name=n_by_name,
    )
    group_rows = _group_rows_from_groups(rows['groups'], names)
    return cells, group_rows, mt


@torch.no_grad()
def network_spot_trace_bundle(
    session, z, *,
    at_x_list=None, at_y_list=None, trace_kind='ca',
    save_trace_csv_dir: str | None = None, show_pre=True,
):
    """Run one forward; full cost-extent spot traces over all types."""
    t_prep0 = time.perf_counter()
    at_x = at_x_list[0] if at_x_list else None
    at_y = at_y_list[0] if at_y_list else None
    rows = _spot_forward_rows(
        session, z, trace_kind=trace_kind,
        save_trace_csv_dir=save_trace_csv_dir,
        at_x=at_x, at_y=at_y, show_pre=show_pre,
    )
    cells, group_rows, maxtime = _spot_cube_from_rows(rows, session)
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
        maxtime=maxtime,
        prep_s=time.perf_counter() - t_prep0,
        v_th_by_name=v_th_by_type_name(z, session),
        response_start=rows.get('t_on'),
        trace_kind=trace_kind,
    )


def _spot_suptitle(title, bundle):
    if bundle is not None and bundle.has_slices:
        scope = column_at_scope_tag(bundle.slice_x_list, bundle.slice_y_list)
        return f'{title}  [{scope}, overlay + total]'
    return title


def _plot_spot_figure(
    path, *,
    timer,
    bundle_on,
    bundle_2=None,
    title,
    ref_cubes=None,
    ref_cubes_2=None,
    ncols,
    figsize_fn,
    gridspec_kw,
    suptitle_fs=12,
):
    session_1 = bundle_on.session
    session_2 = bundle_2.session if bundle_2 is not None else None
    cells = bundle_on.cells
    group_rows = bundle_on.group_rows
    cells_2 = bundle_2.cells if bundle_2 is not None else None
    has_slices = bundle_on.has_slices
    slice_labels = bundle_on.slice_labels or []
    slice_overlay_on = bundle_on.slice_overlay
    slice_overlay_2 = bundle_2.slice_overlay if bundle_2 is not None else None
    maxtime = bundle_on.maxtime
    response_start = bundle_on.response_start
    timer.end_prep()
    dual = session_2 is not None
    # #5: when the model traces are delta-Vm, invert the Ca filter on the gray
    # reference data so both curves share the same units.
    ref_1, ref_2 = resolve_spot_ref_cubes(
        session_1, session_2, ref_cubes, ref_cubes_2,
        v_delta=(getattr(bundle_on, "trace_kind", "ca") == "v"),
    )
    nrows = 2 * len(group_rows)
    fig = plt.figure(figsize=figsize_fn(ncols, nrows))
    gs = fig.add_gridspec(nrows, ncols, **gridspec_kw)
    legend_done = False
    contrast_1 = 'dark' if _session_dark(session_1) else 'bright'
    linestyle_1 = '--' if contrast_1 == 'dark' else '-'
    label_1_data = f'{contrast_1} data'
    label_1_model = f'{contrast_1} model'
    label_1_total = f'{contrast_1} total'
    contrast_2 = 'dark' if (session_2 is not None and _session_dark(session_2)) else 'bright'
    linestyle_2 = '--' if contrast_2 == 'dark' else '-'
    label_2_data = f'{contrast_2} data'
    label_2_model = f'{contrast_2} model'
    label_2_total = f'{contrast_2} total'

    def _plot_cell(cell_on, cell_2, ax_rf, ax_time, show_ylabel, show_xlabels):
        nonlocal legend_done
        name = cell_on['name']
        cell_title = bundle_cell_title(bundle_on, name, cell_on.get('n'))
        if has_slices and slice_overlay_on is not None:
            kw_2 = {}
            if dual and cell_2 is not None:
                slice_2_xt = None
                if slice_overlay_2 is not None:
                    slice_2_xt = {
                        label: cubes[name]
                        for label, cubes in slice_overlay_2.items()
                        if name in cubes
                    }
                kw_2 = {
                    'model_2_xt': cell_2['cube'],
                    'ref_2_xt': ref_2.get(name) if ref_2 else None,
                    'baseline_2': cell_2.get('baseline'),
                    'slice_overlay_2_xt': slice_2_xt,
                }
            slice_xt = {
                label: cubes[name]
                for label, cubes in slice_overlay_on.items()
                if name in cubes
            }
            if not slice_xt:
                ax_rf.axis('off')
                ax_time.axis('off')
                return
            plot_cell_pair_slices(
                ax_rf, ax_time, cell_on['cube'], ref_1.get(name), cell_title,
                slice_xt, slice_labels,
                show_legend=not legend_done,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                maxtime=maxtime,
                baseline=cell_on.get('baseline'),
                **kw_2,
                linestyle_1=linestyle_1,
                linestyle_2=linestyle_2,
                label_1_data=label_1_data,
                label_1_total=label_1_total,
                label_2_data=label_2_data,
                label_2_total=label_2_total,
                response_start=response_start,
            )
        else:
            kw_2 = {}
            if dual and cell_2 is not None:
                kw_2 = {
                    'model_2_xt': cell_2['cube'],
                    'ref_2_xt': ref_2.get(name) if ref_2 else None,
                    'baseline_2': cell_2.get('baseline'),
                }
            plot_cell_pair(
                ax_rf, ax_time, cell_on['cube'], ref_1.get(name), cell_title,
                sem_xt=cell_on.get('sem'),
                show_legend=not legend_done,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                maxtime=maxtime,
                baseline=cell_on.get('baseline'),
                **kw_2,
                linestyle_1=linestyle_1,
                linestyle_2=linestyle_2,
                label_1_data=label_1_data,
                label_1_model=label_1_model,
                label_2_data=label_2_data,
                label_2_model=label_2_model,
                response_start=response_start,
            )
        legend_done = True

    cells_2_by_name = (
        {c['name']: c for c in cells_2} if dual and cells_2 is not None else {}
    )
    for gi, row_idx in enumerate(group_rows):
        rf_row = 2 * gi
        start = (ncols - len(row_idx)) // 2
        for j, ci in enumerate(row_idx):
            col = start + j
            cell_on = cells[ci]
            cell_2 = cells_2_by_name.get(cell_on['name']) if dual else None
            ax_rf = fig.add_subplot(gs[rf_row, col])
            ax_time = fig.add_subplot(gs[rf_row + 1, col])
            _plot_cell(cell_on, cell_2, ax_rf, ax_time, show_ylabel=(j == 0), show_xlabels=True)
    fig.suptitle(_spot_suptitle(title, bundle_on), fontsize=suptitle_fs)
    timer.end_draw()
    save_figure(fig, path, dpi=150)
    timer.log(path)


def plot_network_spot_data(path, *, bundle, bundle_2=None, title,
                           ref_cubes=None, ref_cubes_2=None):
    """Draw ca-data figure (pack readout types) from a full-scope bundle."""
    timer = PlotTimer(prior_prep=bundle_prep_s(bundle, bundle_2))
    view = _spot_readout_bundle_view(bundle)
    view_2 = _spot_readout_bundle_view(bundle_2) if bundle_2 is not None else None
    _plot_spot_figure(
        path,
        timer=timer,
        bundle_on=view,
        bundle_2=view_2,
        title=title,
        ref_cubes=ref_cubes,
        ref_cubes_2=ref_cubes_2,
        ncols=5,
        figsize_fn=lambda c, r: (3.0 * c, 2.5 * r),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
    )


def plot_network_spot_all(path, *, bundle, bundle_2=None, title,
                          ref_cubes=None, ref_cubes_2=None):
    """Draw ca-all figure (all types) from a full-scope bundle."""
    timer = PlotTimer(prior_prep=bundle_prep_s(bundle, bundle_2))
    _plot_spot_figure(
        path,
        timer=timer,
        bundle_on=bundle,
        bundle_2=bundle_2,
        title=title,
        ref_cubes=ref_cubes,
        ref_cubes_2=ref_cubes_2,
        ncols=8,
        figsize_fn=lambda c, r: (2.2 * c, 2.5 * r),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
    )


