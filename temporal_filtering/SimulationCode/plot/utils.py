"""Shared plotting helpers (no target-specific logic)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import network_bootstrap  # noqa: F401 — connectome_io on sys.path
import FiveCol_MedSim_Pytorch as fc
from connectome_io import parse_comma_list

DATA_COLOR = 'gray'
MODEL_COLOR = 'red'
SEM_COLOR = 'pink'
TRACE_LW = 1.5


def _as_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def save_forward_trace_csvs(
    save_dir,
    target,
    *,
    trace_kind,
    ref,
    trace_full,
    ref_stem: str | None = None,
    trace_stem: str | None = None,
):
    """Write per-target ref + trace CSVs under ``save_dir``.

    ``trace_kind=='vm'`` → ``<target>_ref_vm.csv``, ``<target>_vm.csv`` (unless
    overridden by ``ref_stem`` / ``trace_stem``).
    else → ``<target>_ref.csv``, ``<target>.csv``.

    Ref is one column (``ref``) with constant ``N``. Trace is ``(B*T', N)`` with
    constant ``B`` / ``Tprime`` columns, then one column per unit.
    """
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    ref_np = _as_numpy(ref).reshape(-1)
    trace_np = _as_numpy(trace_full)
    if trace_np.ndim != 3:
        raise ValueError(f'trace_full must be (B, T\', N), got shape {trace_np.shape}')
    bsz, tprime, n_units = (int(x) for x in trace_np.shape)
    if ref_np.size != n_units:
        raise ValueError(
            f'ref length {ref_np.size} != n_units {n_units} for target {target!r}'
        )
    if trace_kind == 'vm':
        ref_stem_default, trace_stem_default = f'{target}_ref_vm', f'{target}_vm'
    else:
        ref_stem_default, trace_stem_default = f'{target}_ref', f'{target}'
    ref_stem_final = ref_stem_default if ref_stem is None else ref_stem
    trace_stem_final = trace_stem_default if trace_stem is None else trace_stem
    ref_path = os.path.join(save_dir, f'{ref_stem_final}.csv')
    trace_path = os.path.join(save_dir, f'{trace_stem_final}.csv')
    ref_table = np.column_stack([
        np.full(ref_np.shape[0], ref_np.size, dtype=np.int64),
        ref_np.astype(np.float64, copy=False),
    ])
    # Some traces reuse Vm_ref across targets (e.g. bright/dark); avoid redundant writes.
    if not os.path.exists(ref_path):
        np.savetxt(
            ref_path, ref_table, delimiter=',', header='N,ref', comments='',
        )
    flat = trace_np.reshape(-1, n_units).astype(np.float64, copy=False)
    n_rows = flat.shape[0]
    trace_table = np.column_stack([
        np.full(n_rows, bsz, dtype=np.int64),
        np.full(n_rows, tprime, dtype=np.int64),
        flat,
    ])
    unit_header = ','.join(f'u{i}' for i in range(n_units))
    np.savetxt(
        trace_path, trace_table, delimiter=',',
        header=f'B,Tprime,{unit_header}', comments='',
    )


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


def baselines_for_types(pack, backend, vm_ref, names, type_ids, global_type_names):
    """Mean Vm_ref at stimulus onset over centre cost-readout units, keyed by type name."""
    vm_ref = np.asarray(vm_ref, dtype=np.float64)
    readout = pack.readout_unit.cpu().numpy()
    center = readout_center_mask(pack, backend)
    unit_types = type_ids[readout]
    out = {}
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


def parse_axis_slice_list(text):
    """Parse comma-separated ``--x`` / ``--y`` values (empty → ``None``)."""
    if not text:
        return None
    vals = [float(x) for x in parse_comma_list(text)]
    if not vals:
        raise ValueError('empty comma-separated axis slice')
    return vals


def parse_align_xy(text):
    """Parse ``--align-xy X,Y`` reference sti column (empty → ``None``)."""
    if not text:
        return None
    parts = parse_comma_list(text)
    if len(parts) != 2:
        raise ValueError('--align-xy requires exactly two comma-separated values X,Y')
    return float(parts[0]), float(parts[1])


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


def batches_at_stim_xy(batches, *, at_x=None, at_y=None, tol=1e-6):
    """Batch indices where any ``stim_uv`` column matches hex-step slice."""
    if at_x is None and at_y is None:
        return list(range(len(batches)))
    import column_mapper
    out = []
    for b, batch in enumerate(batches):
        for su, sv in batch.stim_uv:
            x, y = column_mapper.uv_to_xy(int(su), int(sv))
            if not _coord_matches(x, at_x, tol=tol):
                continue
            if not _coord_matches(y, at_y, tol=tol):
                continue
            out.append(b)
            break
    return out


def slice_axis_label(val):
    fv = float(val)
    if np.isclose(fv, round(fv)):
        return str(int(round(fv)))
    return str(fv)


def slice_xy_label(xv, yv):
    return f'({slice_axis_label(xv)},{slice_axis_label(yv)})'


def overlay_model_reds(n_slices):
    """Red shades for per-slice traces plus a darker ``total`` trace."""
    n = n_slices + 1
    return [plt.cm.Reds(v) for v in np.linspace(0.35, 0.95, n)]


def ylim_for_traces(
    model,
    *,
    data=None,
    sem=None,
    show_sem=False,
    model_2=None,
    data_2=None,
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
    if model_2 is not None:
        curves.append(model_2)
    if data_2 is not None:
        curves.append(data_2)
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
    model_2=None,
    data_2=None,
    linestyle_2='--',
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
    """Model (red) vs data (gray) time course with optional SEM and two-trace overlay."""
    if ylim is None:
        ylo, yhi = ylim_for_traces(
            model, data=data, sem=sem, show_sem=show_sem,
            model_2=model_2, data_2=data_2,
        )
    else:
        ylo, yhi = ylim
    if data is not None:
        ax.plot(t, data, color=DATA_COLOR, linewidth=TRACE_LW, linestyle=linestyle)
    if model is not None:
        if show_sem:
            plot_sem_band(ax, t, model, sem)
        ax.plot(t, model, color=MODEL_COLOR, linewidth=TRACE_LW, linestyle=linestyle)
    if data_2 is not None:
        ax.plot(t, data_2, color=DATA_COLOR, linewidth=TRACE_LW, linestyle=linestyle_2)
    if model_2 is not None:
        ax.plot(t, model_2, color=MODEL_COLOR, linewidth=TRACE_LW, linestyle=linestyle_2)
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
