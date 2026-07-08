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
    chunks = []
    for c in curves:
        if c is None:
            continue
        v = np.asarray(c).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            chunks.append(v)
    if not chunks:
        return -floor, floor
    peak = float(np.max(np.abs(np.concatenate(chunks))))
    ymax = max(peak * margin, peak + min_pad, floor)
    ymax = float(np.ceil(ymax / step) * step)
    return -ymax, ymax


def annotate_baseline(ax, baseline):
    """Middle y tick at 0 with resting-potential label (delta-mV plots)."""
    ylo, yhi = ax.get_ylim()
    ax.set_yticks([ylo, 0.0, yhi])
    if baseline is None or not np.isfinite(baseline):
        ax.set_yticklabels([f'{ylo:+.0f}', '', f'{yhi:+.0f}'], fontsize=6)
    else:
        ax.set_yticklabels([f'{ylo:+.0f}', f'{baseline:.1f}', f'{yhi:+.0f}'], fontsize=6)
    ax.axhline(0.0, color='0.4', linewidth=0.6, linestyle=':', zorder=0)


def suppress_cost_sem(session, target=None):
    """True when cost uses a single column (no column-mean SEM band)."""
    pack = session.primary_pack if target is None else session.pack_for(target)
    return pack.cost_extent == 0


def readout_center_mask(pack, backend):
    """Boolean mask over pack.readout rows included in the cost extent."""
    readout = pack.readout_unit.cpu().numpy()
    if backend.network is not None:
        if pack.cost_extent is not None:
            import column_mapper
            C = backend.network
            u_all = C.u.detach().cpu().numpy() if hasattr(C.u, "detach") else np.asarray(C.u)
            v_all = C.v.detach().cpu().numpy() if hasattr(C.v, "detach") else np.asarray(C.v)
            return column_mapper.inside_mask(u_all[readout], v_all[readout], int(pack.cost_extent))
        if pack.cost_radius is not None:
            return np.round(pack.cost_radius.cpu().numpy(), 6) == 0.0
        return np.ones(readout.shape[0], dtype=bool)
    return np.ones(readout.shape[0], dtype=bool)


def baselines_for_types(
    pack,
    backend,
    vm_ref,
    names,
    type_ids,
    global_type_names,
    *,
    readout_unit_idx=None,
    readout_type_idx=None,
    readout_du=None,
    readout_dv=None,
):
    """Mean Vm_ref at stimulus onset, keyed by type name.

    Default: centre cost-readout units (``readout_center_mask``).
    With ``readout_du`` / ``readout_dv``: stim-centred centre members ``(0, 0)``
    only (network ``model_all_spot``).
    """
    vm_ref = np.asarray(vm_ref, dtype=np.float64)
    out = {}
    if readout_du is not None and readout_dv is not None:
        if readout_unit_idx is None or readout_type_idx is None:
            raise ValueError("readout_du/dv require readout_unit_idx and readout_type_idx")
        center = (readout_du == 0) & (readout_dv == 0)
        for name in names:
            ti = global_type_names.index(name)
            units = np.unique(readout_unit_idx[center & (readout_type_idx == ti)])
            out[name] = float(vm_ref[units].mean()) if len(units) else np.nan
        return out
    readout = pack.readout_unit.cpu().numpy()
    center = readout_center_mask(pack, backend)
    unit_types = type_ids[readout]
    for name in names:
        ti = global_type_names.index(name)
        mask = center & (unit_types == ti)
        out[name] = float(vm_ref[readout[mask]].mean()) if mask.any() else np.nan
    return out


def sem_from_traces(traces, single_column=False):
    """Per-time SEM across readout rows; zero when single-column cost or one row."""
    if single_column or traces.shape[0] == 1:
        return np.zeros(traces.shape[1], dtype=np.float64)
    return traces.std(axis=0) / np.sqrt(traces.shape[0])


def _hex_coord_token(val):
    v = float(val)
    if np.isclose(v, round(v)):
        return str(int(round(v)))
    return str(v).replace('.', 'p').replace('-', 'm')


def _parse_comma_floats(text):
    return [float(t.strip()) for t in str(text).split(',') if t.strip()]


def parse_axis_slice_list(text):
    """Parse comma-separated ``--x`` / ``--y`` values (empty → ``None``)."""
    if not text:
        return None
    vals = _parse_comma_floats(text)
    if not vals:
        raise ValueError('empty comma-separated axis slice')
    return vals


def _coord_matches(val, axis_filter, tol=1e-6):
    if axis_filter is None:
        return True
    if isinstance(axis_filter, (list, tuple)):
        return any(np.isclose(val, float(v), atol=tol) for v in axis_filter)
    return np.isclose(val, float(axis_filter), atol=tol)


def filter_sti_columns(cols, *, at_x=None, at_y=None, tol=1e-6):
    """Keep network sti columns whose hex-step ``(x, y)`` matches ``at_x`` / ``at_y``."""
    if at_x is None and at_y is None:
        return list(cols)
    out = []
    for col in cols:
        if not _coord_matches(col.x, at_x, tol=tol):
            continue
        if not _coord_matches(col.y, at_y, tol=tol):
            continue
        out.append(col)
    return out


def filter_borst_sti_columns(cols, *, at_x=None, at_y=None, tol=1e-6):
    """Keep Borst sti columns for plot ``--x`` (``k`` in -2..+2) / ``--y=0``."""
    if at_x is None and at_y is None:
        return list(cols)
    if at_y is not None and not _coord_matches(0.0, at_y, tol=tol):
        raise ValueError('Borst moving-bar plot --y must be 0 (horizontal row)')
    if at_x is None:
        return list(cols)
    if isinstance(at_x, (list, tuple)):
        ks = []
        for v in at_x:
            k = int(round(float(v)))
            if not np.isclose(float(v), k, atol=tol):
                raise ValueError(
                    f'Borst --x={v!r} must be an integer column k in -2..+2',
                )
            ks.append(k)
        out = [col for col in cols if col.k in ks]
        if not out:
            raise ValueError(f'no Borst columns match x={list(at_x)!r}')
        return out
    k = int(round(float(at_x)))
    if not np.isclose(float(at_x), k, atol=tol):
        raise ValueError(f'Borst --x={at_x!r} must be an integer column k in -2..+2')
    out = [col for col in cols if col.k == k]
    if not out:
        raise ValueError(f'no Borst column with k={k!r}')
    return out


def column_at_scope_tag(at_x, at_y):
    """Subtitle fragment for plot column slice."""
    parts = []
    if at_x is not None:
        xs = at_x if isinstance(at_x, (list, tuple)) else [at_x]
        parts.append('x=' + ','.join(_hex_coord_token(v) for v in xs))
    if at_y is not None:
        ys = at_y if isinstance(at_y, (list, tuple)) else [at_y]
        parts.append('y=' + ','.join(_hex_coord_token(v) for v in ys))
    return ', '.join(parts)


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
    curves = []
    if model is not None:
        curves.append(model)
    if data is not None:
        curves.append(data)
    if show_sem and sem is not None and model is not None:
        curves.append(model + sem)
        curves.append(model - sem)
    if off_model is not None:
        curves.append(off_model)
    if off_data is not None:
        curves.append(off_data)
    curves.extend(c for c in extra if c is not None)
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
    if model is not None:
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
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
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
