"""Shared plotting helpers (no target-specific logic)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import FiveCol_MedSim_Pytorch as fc

DATA_COLOR = 'gray'
MODEL_COLOR = 'red'
SEM_COLOR = 'pink'
TRACE_LW = 1.5


def nice_ylim(*curves, margin=1.25, step=5.0, floor=5.0, min_pad=3.0):
    """Symmetric y-limits that comfortably contain all provided curves."""
    vals = [np.asarray(c).ravel() for c in curves if c is not None]
    if not vals:
        return -floor, floor
    peak = float(np.max(np.abs(np.concatenate(vals))))
    ymax = max(peak * margin, peak + min_pad, floor)
    ymax = float(np.ceil(ymax / step) * step)
    return -ymax, ymax


def annotate_baseline(ax, baseline):
    """Middle y tick at 0 with resting-potential label (delta-mV plots)."""
    if baseline is None or not np.isfinite(baseline):
        return
    ylo, yhi = ax.get_ylim()
    ax.set_yticks([ylo, 0.0, yhi])
    ax.set_yticklabels([f'{ylo:+.0f}', f'{baseline:.1f}', f'{yhi:+.0f}'], fontsize=6)
    ax.axhline(0.0, color='0.4', linewidth=0.6, linestyle=':', zorder=0)


def center_column_only(session, target=None):
    """True when the pack uses centre-column readout only (no column-mean SEM band)."""
    pack = session.primary_pack if target is None else session.pack_for(target)
    return bool(pack.center_column)


def readout_center_mask(pack, backend):
    """Boolean mask over pack.readout rows: centre-column / ring-0 units."""
    readout = pack.readout_unit.cpu().numpy()
    if backend.network is not None:
        if pack.cost_radius is not None:
            return np.floor(pack.cost_radius.cpu().numpy()).astype(int) == 0
        return np.ones(readout.shape[0], dtype=bool)
    if pack.center_column:
        sl = fc._borst_tile_cost_row_slice(True)
        return np.array([sl.start <= int(u) < sl.stop for u in readout])
    return np.ones(readout.shape[0], dtype=bool)


def baselines_for_types(pack, backend, vm_ref, names, type_ids, global_type_names):
    """Mean Vm_ref at stimulus onset for centre readout units, keyed by type name."""
    readout = pack.readout_unit.cpu().numpy()
    vm_ref = np.asarray(vm_ref, dtype=np.float64)
    center = readout_center_mask(pack, backend)
    unit_types = type_ids[readout]
    out = {}
    for name in names:
        ti = global_type_names.index(name)
        mask = center & (unit_types == ti)
        out[name] = float(vm_ref[readout[mask]].mean()) if mask.any() else np.nan
    return out


def sem_from_traces(traces, center_only=False):
    """Per-time SEM across readout rows; zero when centre-only or a single row."""
    if center_only or traces.shape[0] == 1:
        return np.zeros(traces.shape[1], dtype=np.float64)
    return traces.std(axis=0) / np.sqrt(traces.shape[0])


def ylim_for_traces(
    model,
    *,
    data=None,
    sem=None,
    show_sem=False,
    off_model=None,
    off_data=None,
    extra=(),
):
    curves = [model]
    if data is not None:
        curves.append(data)
    if show_sem and sem is not None and np.any(sem):
        curves.extend([model + sem, model - sem])
    if off_model is not None:
        curves.append(off_model)
    if off_data is not None:
        curves.append(off_data)
    curves.extend(extra)
    return nice_ylim(*curves)


def ylim_for_keys(model_mean, model_sem, data_mean, keys, *, show_sem=False):
    """Shared y-limits for keyed trace dicts (moving-bar grids)."""
    curves = []
    for key in keys:
        m = model_mean[key]
        curves.append(m)
        if data_mean:
            d = data_mean.get(key)
            if d is not None:
                curves.append(d)
        if show_sem and key in model_sem:
            s = model_sem[key]
            if np.any(s):
                curves.extend([m + s, m - s])
    return nice_ylim(*curves)


def plot_sem_band(ax, t, model, sem, *, label=r'$\pm$SEM'):
    if sem is None or not np.any(sem):
        return
    ax.fill_between(
        t, model - sem, model + sem,
        color=SEM_COLOR, alpha=0.8, linewidth=0, label=label,
    )


def plot_timecourse(
    ax,
    t,
    model,
    *,
    data=None,
    sem=None,
    show_sem=True,
    off_model=None,
    off_data=None,
    title=None,
    title_fs=7,
    ylim=None,
    baseline=None,
    show_ylabel=False,
    ylabel='mV',
    ticksize=6,
    style_xaxis=None,
    linestyle='-',
):
    """Model (red) vs data (gray) time course with optional SEM and on/off overlay."""
    if ylim is None:
        ylo, yhi = ylim_for_traces(
            model, data=data, sem=sem, show_sem=show_sem,
            off_model=off_model, off_data=off_data,
        )
    else:
        ylo, yhi = ylim
    if data is not None:
        ax.plot(t, data, color=DATA_COLOR, linewidth=TRACE_LW, linestyle=linestyle)
    if show_sem:
        plot_sem_band(ax, t, model, sem)
    ax.plot(t, model, color=MODEL_COLOR, linewidth=TRACE_LW, linestyle=linestyle)
    if off_data is not None:
        ax.plot(t, off_data, color=DATA_COLOR, linewidth=TRACE_LW, linestyle='--')
    if off_model is not None:
        ax.plot(t, off_model, color=MODEL_COLOR, linewidth=TRACE_LW, linestyle='--')
    if title is not None:
        ax.set_title(title, fontsize=title_fs, pad=2)
    ax.set_ylim(ylo, yhi)
    if style_xaxis is not None:
        style_xaxis(ax)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=ticksize)
    annotate_baseline(ax, baseline)


def save_figure(fig, path, dpi=150, rasterize=False):
    if rasterize:
        for ax in fig.axes:
            ax.set_rasterized(True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _cost_curve_subplot_rows(names, costs_by_target, total_costs):
    """Build subplot specs; merge moving_bar ``_PD``/``_ND`` part keys per target."""
    rows = [{'title': 'total (weighted)', 'curves': [(None, np.asarray(total_costs), '-')]}]
    seen = set()
    for key in names:
        if key not in costs_by_target or key in seen:
            continue
        if key.endswith('_PD'):
            base = key[:-3]
            nd_key = f"{base}_ND"
            curves = [('PD', np.asarray(costs_by_target[key]), '-')]
            if nd_key in costs_by_target:
                curves.append(('ND', np.asarray(costs_by_target[nd_key]), '--'))
                seen.add(nd_key)
            rows.append({'title': base, 'curves': curves})
            seen.add(key)
        elif key.endswith('_ND'):
            base = key[:-3]
            rows.append({'title': base, 'curves': [('ND', np.asarray(costs_by_target[key]), '--')]})
            seen.add(key)
        else:
            rows.append({
                'title': key,
                'curves': [(None, np.asarray(costs_by_target[key]), '-')],
            })
            seen.add(key)
    return rows


def plot_cost(costs, path, *, costs_by_target=None, target_order=None):
    """Plot training cost; total + one subplot per target when ``costs_by_target`` is given."""
    if costs_by_target:
        names = list(target_order) if target_order else list(costs_by_target.keys())
        names = [n for n in names if n in costs_by_target and len(costs_by_target[n])]
        if names and costs is not None and len(costs):
            rows = _cost_curve_subplot_rows(names, costs_by_target, costs)
            n = len(rows)
            fig, axes = plt.subplots(n, 1, figsize=(8, 2.8 * n), sharex=True)
            if n == 1:
                axes = [axes]
            nsteps = len(costs)
            for ax, row in zip(axes, rows):
                for leg, curve, ls in row['curves']:
                    ax.plot(
                        curve, color='steelblue', linewidth=2, linestyle=ls,
                        label=leg,
                    )
                if len(row['curves']) > 1:
                    ax.legend(fontsize=8)
                ax.set_ylabel('cost [% data power]')
                ax.set_title(row['title'])
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel('step')
            fig.suptitle(f'Training cost ({nsteps} steps)', fontsize=12, y=1.01)
            fig.tight_layout()
            save_figure(fig, path, dpi=150)
            return
        if len(names) == 1:
            costs = costs_by_target[names[0]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(costs, color='steelblue', linewidth=2)
    ax.set_xlabel('step')
    ax.set_ylabel('cost [% data power]')
    ax.set_title(f'Training cost ({len(costs)} steps)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, path, dpi=150)
