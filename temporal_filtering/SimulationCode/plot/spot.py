"""Spot plotting (Borst + network spot target).

Network RF bins are ring means: r=0 -> j4, r=1 -> j3/j5, r=2 -> j2/j6.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

import Medulla_Library as ml
from training_config import DELTAT_MS, IMPULSE_MAXTIME
import blindschleiche_py3 as bs
import FiveCol_MedSim_Pytorch as fc
from plot.readout import (
    spot_model_data_groups,
    spot_model_data_names,
    spot_ref_cubes,
)
from plot.utils import (
    DATA_COLOR,
    MODEL_COLOR,
    TRACE_LW,
    annotate_baseline,
    baselines_for_types,
    batches_at_stim_xy,
    cell_title_with_n,
    column_at_scope_tag,
    filter_borst_sti_columns,
    log_plot_elapsed,
    nice_ylim,
    overlay_model_reds,
    plot_timecourse,
    readout_n_by_name,
    save_figure,
    save_forward_trace_csvs,
    sem_from_traces,
    slice_axis_label,
    slice_xy_label,
    suppress_cost_sem,
)
from column_mapper import borst_sti_columns
from network.spot_target import (
    euclid_hex_dist,
    spot_cost_unit_radius_layout,
    spot_stimulus_batches,
    spotting_from_opts,
    resolve_spot_cost_radii,
)

CENTER_BIN = ml.CENTER_COL + 2
CTYPE = ml.ctype
_SPOT_RECF_PROFILE_BINS = 9


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


def _readout_duv_from_batches(C, batch_idx, unit_idx, *, stim_u, stim_v):
    """Stim-centred axial ``(du, dv)`` per readout row (per-row stim anchor)."""
    u_all = C.u.detach().cpu().numpy() if hasattr(C.u, "detach") else np.asarray(C.u)
    v_all = C.v.detach().cpu().numpy() if hasattr(C.v, "detach") else np.asarray(C.v)
    stim_u = np.asarray(stim_u, dtype=np.int64)
    stim_v = np.asarray(stim_v, dtype=np.int64)
    mu = u_all[unit_idx]
    mv = v_all[unit_idx]
    return mu - stim_u, mv - stim_v


def _baseline_from_ref_grid(ref_grid, row_i):
    """Resting potential at stimulus onset for one cell type (center column)."""
    return ref_grid[row_i, CENTER_BIN]


def _resolve_spot_ref_cubes(session_1, session_2=None, ref_cubes=None, ref_cubes_2=None):
    dual = session_2 is not None
    if ref_cubes is not None:
        ref_1 = ref_cubes
    elif dual:
        ref_1 = spot_ref_cubes(session_1, 'spot_bright', dark=False)
    else:
        ref_1 = spot_ref_cubes(
            session_1, session_1.primary_pack.name, dark=_session_dark(session_1),
        )
    if ref_cubes_2 is not None:
        ref_2 = ref_cubes_2
    elif dual:
        ref_2 = spot_ref_cubes(session_2, 'spot_dark', dark=True)
    else:
        ref_2 = None
    return ref_1, ref_2


def _session_dark(session):
    return session.primary_pack.name == "spot_dark"


def _scale_curve(xt, center, sem_xt=None):
    """Center time course + spatial profile from one ``(9, T)`` cube."""
    imp = xt[center]
    if not np.isfinite(imp).any():
        if sem_xt is not None:
            return None, None, None
        return None, None
    maxt = int(np.argmax(np.abs(imp)))
    spatial = np.nan_to_num(xt[:, maxt], nan=0.0)
    rf = bs.blurr(bs.rebin(spatial, 45), 5)
    amp = float(np.max(np.abs(imp)))
    rf = rf / (np.max(np.abs(rf)) + 1e-12) * amp
    if sem_xt is not None:
        return imp, np.roll(rf, -2), sem_xt[center]
    return imp, np.roll(rf, -2)


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
    maxtime=IMPULSE_MAXTIME,
    model_2_xt=None,
    ref_2_xt=None,
    baseline_2=None,
    linestyle_1='-',
    linestyle_2='--',
    label_1_data='bright data',
    label_1_model='bright model',
    label_2_data='dark data',
    label_2_model='dark model',
):
    center = CENTER_BIN
    if sem_xt is not None:
        imp_model, rf_model, imp_sem = _scale_curve(model_xt, center, sem_xt)
    else:
        imp_model, rf_model = _scale_curve(model_xt, center)
        imp_sem = None
    if ref_xt is not None:
        imp_data, rf_data = _scale_curve(ref_xt, center)
    else:
        imp_data, rf_data = None, None
    model_2_imp_model = model_2_rf_model = model_2_imp_data = model_2_rf_data = None
    if model_2_xt is not None:
        model_2_imp_model, model_2_rf_model = _scale_curve(model_2_xt, center)
    if ref_2_xt is not None:
        model_2_imp_data, model_2_rf_data = _scale_curve(ref_2_xt, center)
    curves = [c for c in (
        imp_model, imp_data, rf_model, rf_data,
        model_2_imp_model, model_2_imp_data, model_2_rf_model, model_2_rf_data,
    ) if c is not None]
    if imp_sem is not None and imp_model is not None:
        curves.extend([imp_model + imp_sem, imp_model - imp_sem])
    ylo, yhi = nice_ylim(*curves)

    if rf_data is not None:
        ax_rf.plot(
            rf_data, color=DATA_COLOR, linewidth=TRACE_LW,
            label=label_1_data, linestyle=linestyle_1,
        )
    if rf_model is not None:
        ax_rf.plot(
            rf_model, color=MODEL_COLOR, linewidth=TRACE_LW,
            label=label_1_model, linestyle=linestyle_1,
        )
    if model_2_rf_data is not None:
        ax_rf.plot(
            model_2_rf_data, color=DATA_COLOR, linewidth=TRACE_LW,
            linestyle=linestyle_2, label=label_2_data,
        )
    if model_2_rf_model is not None:
        ax_rf.plot(
            model_2_rf_model, color=MODEL_COLOR, linewidth=TRACE_LW,
            linestyle=linestyle_2, label=label_2_model,
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
    maxtime=IMPULSE_MAXTIME,
    baseline_2=None,
    linestyle_1='-',
    linestyle_2='--',
    label_1_data='bright data',
    label_1_total='bright total',
    label_2_data='dark data',
    label_2_total='dark total',
):
    center = CENTER_BIN
    imp_model, rf_model = _scale_curve(model_xt, center)
    imp_data, rf_data = (None, None)
    if ref_xt is not None:
        imp_data, rf_data = _scale_curve(ref_xt, center)
    model_2_imp_model = model_2_rf_model = model_2_imp_data = model_2_rf_data = None
    if model_2_xt is not None:
        model_2_imp_model, model_2_rf_model = _scale_curve(model_2_xt, center)
    if ref_2_xt is not None:
        model_2_imp_data, model_2_rf_data = _scale_curve(ref_2_xt, center)

    slice_imps = {}
    slice_rfs = {}
    slice_2_imps = {}
    slice_2_rfs = {}
    for label in slice_labels:
        if label in slice_overlay_xt:
            imp_s, rf_s = _scale_curve(slice_overlay_xt[label], center)
            slice_imps[label] = imp_s
            slice_rfs[label] = rf_s
        if slice_overlay_2_xt and label in slice_overlay_2_xt:
            imp_s, rf_s = _scale_curve(slice_overlay_2_xt[label], center)
            slice_2_imps[label] = imp_s
            slice_2_rfs[label] = rf_s

    curves = [
        imp_model, imp_data, rf_model, rf_data,
        model_2_imp_model, model_2_imp_data, model_2_rf_model, model_2_rf_data,
    ]
    for label in slice_labels:
        curves.extend([slice_imps.get(label), slice_rfs.get(label)])
        curves.extend([slice_2_imps.get(label), slice_2_rfs.get(label)])
    ylo, yhi = nice_ylim(*[c for c in curves if c is not None])

    colors = overlay_model_reds(len(slice_labels))
    if rf_data is not None:
        ax_rf.plot(
            rf_data, color=DATA_COLOR, linewidth=TRACE_LW,
            label=label_1_data, linestyle=linestyle_1,
        )
    for i, label in enumerate(slice_labels):
        rf_s = slice_rfs.get(label)
        if rf_s is not None:
            ax_rf.plot(
                rf_s, color=colors[i], linewidth=TRACE_LW,
                label=label, linestyle=linestyle_1,
            )
    if rf_model is not None:
        ax_rf.plot(
            rf_model, color=colors[-1], linewidth=TRACE_LW,
            label=label_1_total, linestyle=linestyle_1,
        )
    if model_2_rf_data is not None:
        ax_rf.plot(
            model_2_rf_data, color=DATA_COLOR, linewidth=TRACE_LW,
            linestyle=linestyle_2, label=label_2_data,
        )
    for i, label in enumerate(slice_labels):
        rf_s = slice_2_rfs.get(label)
        if rf_s is not None:
            ax_rf.plot(
                rf_s, color=colors[i], linewidth=TRACE_LW,
                linestyle=linestyle_2,
            )
    if model_2_rf_model is not None:
        ax_rf.plot(
            model_2_rf_model, color=colors[-1], linewidth=TRACE_LW,
            linestyle=linestyle_2, label=label_2_total,
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


@torch.no_grad()
def _simulate(session, z, neuron_index, return_ref=False, *, trace_kind='model'):
    neuron_index = _as_index(neuron_index, z.device)
    schema = list(session.schema)
    backend = session.backend
    if session.model == 'adaptive':
        p = fc.assign_params_adaptive(z, schema, backend)
        stacked, ref = fc._run_adaptive(p, session, neuron_index=neuron_index, return_ref=True)
    else:
        p = fc.assign_params(z, schema, backend)
        if trace_kind == 'vm':
            stacked, ref = fc._run_conductance(
                session, p, neuron_index=neuron_index, return_ref=True, return_vm=True,
            )
        else:
            stacked, ref = fc._run_conductance(session, p, neuron_index=neuron_index, return_ref=True)
    mt = session.maxtime
    if trace_kind == 'vm':
        scale = torch.ones((int(neuron_index.shape[0]),), dtype=stacked.dtype, device=stacked.device)
    else:
        scale = fc.out_scale_for_units(p, neuron_index, backend)
    trace = fc.pad_plot_traces(stacked.transpose(0, 1), scale, mt)
    if return_ref:
        return trace, ref
    return trace


def calc_model_full_all(session, z, return_ref=False, *, trace_kind='model'):
    n_types = session.backend.n_types
    mt = session.maxtime
    model_full = np.zeros((n_types, 9, mt))
    ref_full = np.full((n_types, 9), np.nan)
    for col in range(5):
        col_index = torch.arange(
            col * n_types,
            (col + 1) * n_types,
            dtype=torch.long,
            device=z.device,
        )
        if return_ref:
            trace, ref = _simulate(session, z, col_index, return_ref=True, trace_kind=trace_kind)
            model_full[:, col + 2] = trace.cpu().numpy()
            ref_full[:, col + 2] = ref.cpu().numpy()
        else:
            model_full[:, col + 2] = _simulate(session, z, col_index, trace_kind=trace_kind).cpu().numpy()
    if return_ref:
        return model_full, ref_full
    return model_full


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
    cells_on: list
    cells_2: list | None = None
    group_rows: list | None = None
    slice_overlay: dict[str, dict[str, np.ndarray]] | None = None
    slice_overlay_2: dict[str, dict[str, np.ndarray]] | None = None
    slice_labels: list[str] | None = None
    slice_x_list: list | None = None
    slice_y_list: list | None = None
    maxtime: int = IMPULSE_MAXTIME

    @property
    def has_slices(self):
        return bool(self.slice_overlay)


def _spot_slice_specs(at_x_list, at_y_list):
    if at_x_list is not None and at_y_list is not None:
        return [
            (slice_xy_label(xv, yv), xv, yv)
            for xv in at_x_list for yv in at_y_list
        ]
    if at_x_list is not None:
        return [(slice_axis_label(xv), xv, None) for xv in at_x_list]
    if at_y_list is not None:
        return [(slice_axis_label(yv), None, yv) for yv in at_y_list]
    return []


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
    for label, at_x, at_y in _spot_slice_specs(at_x_list, at_y_list):
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


def _borst_slice_overlay(model, names, at_x_list, at_y_list):
    overlay = {}
    labels = []
    cols_all = list(borst_sti_columns())
    mt = model.shape[2]
    for label, at_x, at_y in _spot_slice_specs(at_x_list, at_y_list):
        try:
            filt = filter_borst_sti_columns(
                cols_all, at_x=at_x, at_y=at_y if at_y is not None else 0.0,
            )
        except ValueError as exc:
            print(f'skip slice overlay {label}: {exc}')
            continue
        if not filt:
            print(f'skip slice overlay {label}: no Borst columns')
            continue
        cubes = {}
        for i, name in enumerate(names):
            cube = np.full((9, mt), np.nan)
            for col in filt:
                bin_j = col.col + 2
                cube[bin_j] = model[i, bin_j, :]
            cubes[name] = cube
        if any(np.isfinite(c).any() for c in cubes.values()):
            overlay[label] = cubes
            labels.append(label)
    if not overlay:
        return None, None
    return overlay, labels


def _cells_from_cube(names, cube, sem, baselines, *, single_column, n_by_name=None):
    return [
        dict(
            name=n,
            cube=cube[i],
            sem=None if single_column else sem[i],
            baseline=baselines.get(n),
            n=None if n_by_name is None else n_by_name.get(n),
        )
        for i, n in enumerate(names)
    ]


def _spot_baselines(rows, vm_ref, names, *, at_x=None, at_y=None):
    """Mean ``Vm_ref`` per type over stim-centred ``(0, 0)`` units (matches trace scope)."""
    vm_ref = np.asarray(vm_ref, dtype=np.float64)
    batch_idx = rows['batch_idx']
    unit_idx = rows['unit_idx']
    type_idx = rows['type_idx']
    du, dv = rows['du'], rows['dv']
    type_names = rows['type_names']

    mask = np.ones(len(batch_idx), dtype=bool)
    if at_x is not None or at_y is not None:
        match_b = batches_at_stim_xy(rows['batches'], at_x=at_x, at_y=at_y)
        mask &= np.isin(batch_idx, match_b)
    center = mask & (du == 0) & (dv == 0)

    out = {}
    for name in names:
        ti = type_names.index(name)
        units = np.unique(unit_idx[center & (type_idx == ti)])
        out[name] = float(vm_ref[units].mean()) if units.size else np.nan
    return out


@torch.no_grad()
def _spot_forward_rows(
    session, z, all_cells=False, group_list=None, *, trace_kind='model',
    save_trace_csv_dir=None, at_x=None, at_y=None,
):
    pack = session.primary_pack
    schema = list(session.schema)
    p = fc.assign_params(z, schema, session.backend)
    sig = pack.signal if pack.signal.dim() == 3 else pack.signal.unsqueeze(0)
    if trace_kind == 'vm':
        model_full, vm_ref, vm_full = fc._run_conductance_full(
            session, p, sig, return_ref=True, return_vm=True,
        )
        trace_full = vm_full - vm_ref[:, None, :]
    else:
        model_full, vm_ref = fc._run_conductance_full(session, p, sig, return_ref=True)
        trace_full = model_full
    vm_ref_np = vm_ref[0].cpu().numpy()
    save_forward_trace_csvs(
        save_trace_csv_dir, pack.name,
        trace_kind=trace_kind, ref=vm_ref_np, trace_full=trace_full,
        ref_stem='spot_ref_vm' if trace_kind == 'vm' else None,
    )
    C = session.backend.network
    type_names = list(C.type_names)
    type_ids = C.node_type.cpu().numpy()
    mt = session.maxtime

    opts = dict((session.train_opts or {}).get(f"{pack.name}_stimulus_opts") or {})
    spotting = spotting_from_opts(C, stimulus_opts=opts)
    batches = spot_stimulus_batches(spotting)

    if all_cells:
        cost_radii = resolve_spot_cost_radii(stimulus_opts=opts)
        batch_idx, unit_idx, _radius, type_idx, stim_u, stim_v = spot_cost_unit_radius_layout(
            C, batches, cost_radii, pack.cost_extent,
        )
        names = [str(n) for n in type_names]
    else:
        batch_idx = pack.readout_batch.cpu().numpy()
        unit_idx = pack.readout_unit.cpu().numpy()
        type_idx = type_ids[unit_idx]
        names = spot_model_data_names(session, pack.name, group_list)
        stim_u = pack.readout_stim_u.cpu().numpy()
        stim_v = pack.readout_stim_v.cpu().numpy()

    du, dv = _readout_duv_from_batches(C, batch_idx, unit_idx, stim_u=stim_u, stim_v=stim_v)

    raw = trace_full[batch_idx, :, unit_idx]
    if trace_kind == 'vm':
        scale = torch.ones((int(raw.shape[0]),), dtype=raw.dtype, device=raw.device)
    else:
        scale = fc.out_scale_for_units(
            p, torch.as_tensor(unit_idx, dtype=torch.long, device=z.device), session.backend,
        )
    plot_traces = fc.pad_plot_traces(raw, scale, mt).cpu().numpy()
    rows = dict(
        names=names,
        type_names=type_names,
        type_idx=type_idx,
        unit_idx=unit_idx,
        du=du,
        dv=dv,
        plot_traces=plot_traces,
        batch_idx=batch_idx,
        batches=batches,
        mt=mt,
        pack=pack,
    )
    if all_cells:
        baselines = _spot_baselines(rows, vm_ref_np, names, at_x=at_x, at_y=at_y)
    else:
        baselines = baselines_for_types(
            pack, session.backend, vm_ref_np, names, type_ids, type_names,
        )
    rows['baselines'] = baselines
    return rows


@torch.no_grad()
def multicol_cube(
    session, z, all_cells=False, group_list=None, *, trace_kind='model',
    save_trace_csv_dir=None,
):
    rows = _spot_forward_rows(
        session, z, all_cells=all_cells, group_list=group_list,
        trace_kind=trace_kind, save_trace_csv_dir=save_trace_csv_dir,
    )
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
    n_by_name = readout_n_by_name(
        rows['type_idx'], rows['type_names'], names, rows['unit_idx'],
    )
    return names, cube, sem, rows['baselines'], n_by_name


def _prepare_borst_spot(session, z, all_cells, group_list, *, trace_kind='model', save_trace_csv_dir=None):
    model, ref = calc_model_full_all(session, z, return_ref=True, trace_kind=trace_kind)
    if all_cells:
        names = [str(CTYPE[i]) for i in range(session.backend.n_types)]
        cells = [
            dict(name=names[i], cube=model[i], sem=None, baseline=_baseline_from_ref_grid(ref, i), n=1)
            for i in range(len(names))
        ]
        return cells, None
    groups = spot_model_data_groups(session, session.primary_pack.name, group_list)
    cells = []
    group_rows = []
    for names_row in groups:
        row_idx = []
        for name in names_row:
            name = str(name)
            ctype_i = int(np.where(CTYPE == name)[0][0])
            row_idx.append(len(cells))
            cells.append(dict(
                name=name, cube=model[ctype_i], sem=None,
                baseline=_baseline_from_ref_grid(ref, ctype_i),
                n=1,
            ))
        group_rows.append(row_idx)
    return cells, group_rows


def _prepare_network_spot(
    session, z, all_cells, group_list, *, trace_kind='model',
    save_trace_csv_dir=None,
):
    pack = session.primary_pack
    names, cube, sem, baselines, n_by_name = multicol_cube(
        session, z, all_cells=all_cells, group_list=group_list,
        trace_kind=trace_kind, save_trace_csv_dir=save_trace_csv_dir,
    )
    single_column = suppress_cost_sem(session, target=pack.name)
    cells = _cells_from_cube(
        names, cube, sem, baselines, single_column=single_column, n_by_name=n_by_name,
    )
    return cells, None


def _prepare_network_spot_bundle(
    session, z, all_cells, group_list, *,
    at_x_list=None, at_y_list=None, trace_kind='model',
    save_trace_csv_dir: str | None = None,
):
    pack = session.primary_pack
    rows = _spot_forward_rows(
        session, z, all_cells=all_cells, group_list=group_list, trace_kind=trace_kind,
        save_trace_csv_dir=save_trace_csv_dir,
    )
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
    single_column = suppress_cost_sem(session, target=pack.name)
    n_by_name = readout_n_by_name(
        rows['type_idx'], rows['type_names'], names, rows['unit_idx'],
    )
    cells = _cells_from_cube(
        names, cube, sem, rows['baselines'], single_column=single_column, n_by_name=n_by_name,
    )
    slice_overlay, slice_labels = (None, None)
    if all_cells and (at_x_list is not None or at_y_list is not None):
        slice_overlay, slice_labels = _spot_slice_overlay(
            rows, rows['batches'], at_x_list, at_y_list,
        )
    return SpotTraceBundle(
        cells_on=cells,
        slice_overlay=slice_overlay,
        slice_labels=slice_labels,
        slice_x_list=at_x_list,
        slice_y_list=at_y_list,
        maxtime=mt,
    )


def _prepare_borst_spot_bundle(
    session, z, all_cells, group_list, *,
    at_x_list=None, at_y_list=None, trace_kind='model', save_trace_csv_dir: str | None = None,
):
    model, ref = calc_model_full_all(session, z, return_ref=True, trace_kind=trace_kind)
    if all_cells:
        names = [str(CTYPE[i]) for i in range(session.backend.n_types)]
        cells = [
            dict(name=names[i], cube=model[i], sem=None, baseline=_baseline_from_ref_grid(ref, i), n=1)
            for i in range(len(names))
        ]
        slice_overlay, slice_labels = (None, None)
        if at_x_list is not None or at_y_list is not None:
            slice_overlay, slice_labels = _borst_slice_overlay(
                model, names, at_x_list, at_y_list,
            )
        return SpotTraceBundle(
            cells_on=cells,
            slice_overlay=slice_overlay,
            slice_labels=slice_labels,
            slice_x_list=at_x_list,
            slice_y_list=at_y_list,
            maxtime=session.maxtime,
        )
    groups = spot_model_data_groups(session, session.primary_pack.name, group_list)
    cells = []
    group_rows = []
    for names_row in groups:
        row_idx = []
        for name in names_row:
            name = str(name)
            ctype_i = int(np.where(CTYPE == name)[0][0])
            row_idx.append(len(cells))
            cells.append(dict(
                name=name, cube=model[ctype_i], sem=None,
                baseline=_baseline_from_ref_grid(ref, ctype_i),
                n=1,
            ))
        group_rows.append(row_idx)
    return SpotTraceBundle(cells_on=cells, group_rows=group_rows, maxtime=session.maxtime)


def _spot_suptitle(title, bundle):
    if bundle is not None and bundle.has_slices:
        scope = column_at_scope_tag(bundle.slice_x_list, bundle.slice_y_list)
        return f'{title}  [{scope}, overlay + total]'
    return title


def _plot_spot_figure(
    session_1,
    z,
    path,
    *,
    prepare_fn=None,
    prepare_bundle_fn=None,
    bundle_on=None,
    bundle_2=None,
    session_2=None,
    all_cells=False,
    title,
    ref_cubes=None,
    ref_cubes_2=None,
    group_list=None,
    ncols,
    figsize_fn,
    gridspec_kw,
    suptitle_fs=12,
    trace_kind='model',
    save_trace_csv_dir: str | None = None,
    borst_all_cells=False,
):
    t0 = time.perf_counter()
    if prepare_bundle_fn is not None:
        bundle_on = prepare_bundle_fn(session_1)
        bundle_2 = prepare_bundle_fn(session_2) if session_2 is not None else None
        cells_on = bundle_on.cells_on
        group_rows = bundle_on.group_rows
        cells_2 = bundle_2.cells_on if bundle_2 is not None else None
        has_slices = bundle_on.has_slices
        slice_labels = bundle_on.slice_labels or []
        slice_overlay_on = bundle_on.slice_overlay
        slice_overlay_2 = bundle_2.slice_overlay if bundle_2 is not None else None
        maxtime = bundle_on.maxtime
    elif bundle_on is not None:
        cells_on = bundle_on.cells_on
        group_rows = bundle_on.group_rows
        cells_2 = bundle_2.cells_on if bundle_2 is not None else None
        has_slices = bundle_on.has_slices
        slice_labels = bundle_on.slice_labels or []
        slice_overlay_on = bundle_on.slice_overlay
        slice_overlay_2 = bundle_2.slice_overlay if bundle_2 is not None else None
        maxtime = bundle_on.maxtime
    else:
        cells_on, group_rows = prepare_fn(
            session_1, z, all_cells, group_list, trace_kind=trace_kind,
            save_trace_csv_dir=save_trace_csv_dir,
        )
        cells_2 = None
        if session_2 is not None:
            cells_2, _ = prepare_fn(
                session_2, z, all_cells, group_list, trace_kind=trace_kind,
                save_trace_csv_dir=save_trace_csv_dir,
            )
        has_slices = False
        slice_labels = []
        slice_overlay_on = None
        slice_overlay_2 = None
        maxtime = session_1.maxtime
    t_prep = time.perf_counter()
    dual = session_2 is not None
    ref_1, ref_2 = _resolve_spot_ref_cubes(
        session_1, session_2, ref_cubes, ref_cubes_2,
    )
    if group_rows is not None:
        nrows = 2 * len(group_rows)
    else:
        nrows = 2 * ((len(cells_on) + ncols - 1) // ncols)
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
        cell_title = cell_title_with_n(name, cell_on.get('n'))
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
            )
        legend_done = True

    if group_rows is not None:
        for gi, row_idx in enumerate(group_rows):
            rf_row = 2 * gi
            start = (ncols - len(row_idx)) // 2
            for j, ci in enumerate(row_idx):
                col = start + j
                cell_2 = cells_2[ci] if dual else None
                ax_rf = fig.add_subplot(gs[rf_row, col])
                ax_time = fig.add_subplot(gs[rf_row + 1, col])
                _plot_cell(cells_on[ci], cell_2, ax_rf, ax_time, show_ylabel=(j == 0), show_xlabels=True)
    else:
        borst_all = all_cells and prepare_fn is _prepare_borst_spot
        for i, cell_on in enumerate(cells_on):
            blk, col = divmod(i, ncols)
            cell_2 = cells_2[i] if dual else None
            ax_rf = fig.add_subplot(gs[2 * blk, col])
            ax_time = fig.add_subplot(gs[2 * blk + 1, col])
            show_xlabels = True
            if borst_all:
                show_xlabels = (blk == (len(cells_on) + ncols - 1) // ncols - 1)
            _plot_cell(cell_on, cell_2, ax_rf, ax_time, show_ylabel=(col == 0), show_xlabels=show_xlabels)
    fig.suptitle(_spot_suptitle(title, bundle_on), fontsize=suptitle_fs)
    t_draw = time.perf_counter()
    save_figure(fig, path, dpi=150)
    log_plot_elapsed(
        path, t0,
        prep=t_prep - t0,
        draw=t_draw - t_prep,
        save=time.perf_counter() - t_draw,
    )


def plot_network_spot(session_1, z, path, *, session_2=None, all_cells=False,
                      title, ref_cubes=None, ref_cubes_2=None, group_list=None,
                      trace_kind='model', at_x_list=None, at_y_list=None,
                      save_trace_csv_dir: str | None = None):
    ncols = 5 if not all_cells else 8
    use_slices = all_cells and (at_x_list is not None or at_y_list is not None)
    if use_slices:
        def _bundle(session):
            return _prepare_network_spot_bundle(
                session, z, all_cells, group_list,
                at_x_list=at_x_list, at_y_list=at_y_list, trace_kind=trace_kind,
                save_trace_csv_dir=save_trace_csv_dir,
            )

        _plot_spot_figure(
            session_1, z, path,
            prepare_bundle_fn=_bundle,
            session_2=session_2,
            all_cells=all_cells,
            title=title,
            ref_cubes=ref_cubes,
            ref_cubes_2=ref_cubes_2,
            group_list=group_list,
            ncols=ncols,
            figsize_fn=lambda c, r: (3.0 * c if not all_cells else 2.2 * c, 2.5 * r),
            gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
            trace_kind=trace_kind,
        )
        return
    _plot_spot_figure(
        session_1, z, path,
        prepare_fn=_prepare_network_spot,
        session_2=session_2,
        all_cells=all_cells,
        title=title,
        ref_cubes=ref_cubes,
        ref_cubes_2=ref_cubes_2,
        group_list=group_list,
        ncols=ncols,
        figsize_fn=lambda c, r: (3.0 * c if not all_cells else 2.2 * c, 2.5 * r),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.95, bottom=0.06, left=0.07, right=0.98),
        trace_kind=trace_kind,
        save_trace_csv_dir=save_trace_csv_dir,
    )


def plot_borst_spot(session_1, z, path, *, session_2=None, all_cells=False,
                    title, ref_cubes=None, ref_cubes_2=None, group_list=None,
                    trace_kind='model', at_x_list=None, at_y_list=None,
                    save_trace_csv_dir: str | None = None):
    ncols = 13
    if all_cells:
        gs_kw = dict(hspace=0.65, wspace=0.45, top=0.97, bottom=0.03, left=0.04, right=0.99)
        figsize_fn = lambda c, r: (26, 32)
        suptitle_fs = 14
    else:
        gs_kw = dict(hspace=0.5, wspace=0.55, top=0.95, bottom=0.05, left=0.06, right=0.98)
        figsize_fn = lambda c, r: (16, 2.5 * r)
        suptitle_fs = 12
    use_slices = all_cells and (at_x_list is not None or at_y_list is not None)
    if use_slices:
        def _bundle(session):
            return _prepare_borst_spot_bundle(
                session, z, all_cells, group_list,
                at_x_list=at_x_list, at_y_list=at_y_list, trace_kind=trace_kind,
            )

        _plot_spot_figure(
            session_1, z, path,
            prepare_bundle_fn=_bundle,
            session_2=session_2,
            all_cells=all_cells,
            title=title,
            ref_cubes=ref_cubes,
            ref_cubes_2=ref_cubes_2,
            group_list=group_list,
            ncols=ncols,
            figsize_fn=figsize_fn,
            gridspec_kw=gs_kw,
            suptitle_fs=suptitle_fs,
            trace_kind=trace_kind,
        )
        return
    _plot_spot_figure(
        session_1, z, path,
        prepare_fn=_prepare_borst_spot,
        session_2=session_2,
        all_cells=all_cells,
        title=title,
        ref_cubes=ref_cubes,
        ref_cubes_2=ref_cubes_2,
        group_list=group_list,
        ncols=ncols,
        figsize_fn=figsize_fn,
        gridspec_kw=gs_kw,
        suptitle_fs=suptitle_fs,
        trace_kind=trace_kind,
        save_trace_csv_dir=save_trace_csv_dir,
    )
