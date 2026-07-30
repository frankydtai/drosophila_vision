"""Shared plotting helpers (no target-specific logic)."""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import network.path  # noqa: F401 — connectome_io on sys.path
import training as fc
from connectome_io import parse_comma_list

DATA_COLOR = 'gray'
MODEL_COLOR = 'red'
SEM_COLOR = 'pink'
TRACE_LW = 1.5
# Fixed symmetric ylim for spot / moving-bar model-data panels (delta-mV).
TRACE_YLIM = (-30.0, 30.0)


def as_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _as_numpy(arr):
    return as_numpy(arr)


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

    ``trace_kind=='v'`` → ``<target>_v_ref.csv``, ``<target>_v.csv`` (unless
    overridden by ``ref_stem`` / ``trace_stem``).
    else → ``<target>_ca_ref.csv``, ``<target>_ca.csv``.

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
    if trace_kind == 'v':
        ref_stem_default, trace_stem_default = f'{target}_v_ref', f'{target}_v'
    else:
        ref_stem_default, trace_stem_default = f'{target}_ca_ref', f'{target}_ca'
    ref_stem_final = ref_stem_default if ref_stem is None else ref_stem
    trace_stem_final = trace_stem_default if trace_stem is None else trace_stem
    ref_path = os.path.join(save_dir, f'{ref_stem_final}.csv')
    trace_path = os.path.join(save_dir, f'{trace_stem_final}.csv')
    ref_table = np.column_stack([
        np.full(ref_np.shape[0], ref_np.size, dtype=np.int64),
        ref_np.astype(np.float64, copy=False),
    ])
    # Some traces reuse v_ref across targets (e.g. bright/dark); avoid redundant writes.
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


def cost_ylim(*curves, pct=99.0, pad=1.1, floor=1.0):
    """Non-negative ylim from high percentile so cost spikes do not dominate."""
    chunks = []
    for c in curves:
        if c is None:
            continue
        v = np.asarray(c, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            chunks.append(v)
    if not chunks:
        return 0.0, floor
    hi = float(np.percentile(np.concatenate(chunks), pct))
    yhi = max(hi * pad, floor)
    return 0.0, yhi


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


def baselines_for_types(pack, backend, v_ref, names, type_ids, global_type_names):
    """Mean v_ref at stimulus onset over centre cost-readout units, keyed by type name."""
    v_ref = np.asarray(v_ref, dtype=np.float64)
    readout = pack.readout_unit.cpu().numpy()
    center = readout_center_mask(pack, backend)
    unit_types = type_ids[readout]
    out = {}
    for name in names:
        ti = global_type_names.index(name)
        mask = center & (unit_types == ti)
        out[name] = float(v_ref[readout[mask]].mean()) if mask.any() else np.nan
    return out


def sem_from_traces(traces, single_column=False):
    """Per-time SEM across readout rows; zero when single-column cost or one row."""
    if single_column or traces.shape[0] == 1:
        return np.zeros(traces.shape[1], dtype=np.float64)
    return traces.std(axis=0) / np.sqrt(traces.shape[0])


def readout_n_by_name(type_idx, type_names, names, unit_idx):
    """Unique readout unit count per plotted type name."""
    type_idx = np.asarray(type_idx)
    unit_idx = np.asarray(unit_idx)
    return {
        name: int(np.unique(unit_idx[type_idx == type_names.index(name)]).size)
        for name in names
    }


def v_th_by_type_name(z, session):
    """Per-type ``v_th`` (mV) keyed by type name; empty if schema has no ``v_th``."""
    schema = list(session.schema)
    if not any(s.get('name') == 'v_th' for s in schema):
        return {}
    arr = np.asarray(fc.z_to_unit_values(z, schema)['v_th'], dtype=np.float64).reshape(-1)
    names = fc.type_unit_names(session.backend)
    if arr.shape[0] != len(names):
        raise ValueError(f"v_th length {arr.shape[0]} != n_types {len(names)}")
    return {str(n): float(arr[i]) for i, n in enumerate(names)}


def label_with_n(label, n=None):
    """Cell / row label with optional sample count."""
    if n is None:
        return label
    return f'{label} (n={int(n)})'


def n_for_type(ca_n, tname):
    """Sample count for *tname* from keyed ``ca_n`` ``{(type, spec): n}``."""
    for (t, _s), n in ca_n.items():
        if t == tname:
            return n
    return None


def cell_ylabel(label, ca_n=None, n=None):
    """Row / cell axis label with ``n`` from *model_n* or explicit *n*."""
    if n is None and ca_n is not None:
        n = n_for_type(ca_n, label)
    return label_with_n(label, n)


def cell_title_with_n(label, n=None, v_th=None):
    """Spot-style panel title: cell name + optional ``n``, ``v_th`` on next line."""
    head = label_with_n(label, n)
    if v_th is None:
        return head
    return f'{head}\nv_th={float(v_th):.1f} mV'


def panel_title_with_v_th(label, v_th=None):
    """Stimulus / panel title with optional ``v_th`` line (no ``n``)."""
    if v_th is None:
        return label
    return f'{label}\nv_th={float(v_th):.1f} mV'


def bundle_cell_title(bundle, label, n=None, *, type_name=None):
    """Spot panel title from ``bundle.v_th_by_name`` (key = *type_name* or *label*)."""
    key = label if type_name is None else type_name
    return cell_title_with_n(label, n, bundle.v_th_by_name.get(key))


def bundle_panel_title(bundle, label, *, type_name=None):
    """Moving-bar panel title from ``bundle.v_th_by_name`` (no ``n``)."""
    key = label if type_name is None else type_name
    return panel_title_with_v_th(label, bundle.v_th_by_name.get(key))


def network_column_count(C):
    """Unique axial columns on connectome ``C``."""
    return len({(int(u), int(v)) for u, v in zip(C.u, C.v)})


def log_plot_elapsed(path, t0, **parts):
    """Print per-figure timing (seconds) after saving a plot."""
    total = time.perf_counter() - t0
    bits = [f'{name}={float(val):.1f}s' for name, val in parts.items()]
    bits.append(f'total={total:.1f}s')
    print(f'plot {path}: {"  ".join(bits)}')


def bundle_prep_s(*bundles):
    """Sum and clear ``prep_s`` on trace bundles (``None`` skipped).

    First figure that calls this receives the forward cost; later shared
    figures see ``0``.
    """
    total = 0.0
    for b in bundles:
        if b is None:
            continue
        total += float(getattr(b, 'prep_s', 0.0) or 0.0)
        b.prep_s = 0.0
    return total


class PlotTimer:
    """Shared prep / draw / save timing for bar and spot figures.

    ``prior_prep`` is forward time already stored on ``TraceBundle.prep_s``.
    """

    __slots__ = ('t0', '_t_prep', '_t_draw')

    def __init__(self, prior_prep=0.0):
        self.t0 = time.perf_counter() - float(prior_prep)
        self._t_prep = None
        self._t_draw = None

    def end_prep(self):
        self._t_prep = time.perf_counter()

    def end_draw(self):
        self._t_draw = time.perf_counter()

    def log(self, path):
        t_prep = self._t_prep if self._t_prep is not None else time.perf_counter()
        t_draw = self._t_draw if self._t_draw is not None else time.perf_counter()
        now = time.perf_counter()
        log_plot_elapsed(
            path, self.t0,
            prep=t_prep - self.t0,
            draw=t_draw - t_prep,
            save=now - t_draw,
        )


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


def slice_coord_specs(at_x_list, at_y_list):
    """Expand optional x/y lists to ``[(label, at_x, at_y), ...]`` (missing axis is None)."""
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


def slice_axis_name(at_x_list, at_y_list):
    """``'xy'`` / ``'x'`` / ``'y'`` / ``None`` matching :func:`slice_coord_specs`."""
    if at_x_list is not None and at_y_list is not None:
        return 'xy'
    if at_x_list is not None:
        return 'x'
    if at_y_list is not None:
        return 'y'
    return None


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


def ylim_for_keys(ca_mean, ca_sem, data_mean, keys, *, show_sem=False):
    """Shared y-limits for keyed trace dicts (moving-bar grids)."""
    curves = []
    for key in keys:
        m = ca_mean[key]
        curves.append(m)
        if data_mean:
            d = data_mean.get(key)
            if d is not None:
                curves.append(d)
        if show_sem and key in ca_sem:
            s = ca_sem[key]
            if np.any(s):
                curves.extend([m + s, m - s])
    return nice_ylim(*curves)


def plot_sem_band(ax, t, model, sem, *, color=None, alpha=None, label=r'$\pm$SEM'):
    if sem is None or not np.any(sem):
        return
    ax.fill_between(
        t, model - sem, model + sem,
        color=SEM_COLOR if color is None else color,
        alpha=0.8 if alpha is None else alpha,
        linewidth=0, label=label,
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
    t0 = time.perf_counter()
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
                curves = []
                for leg, curve, ls in row['curves']:
                    curves.append(curve)
                    ax.plot(
                        curve, color='steelblue', linewidth=2, linestyle=ls,
                        label=leg,
                    )
                ax.set_ylim(*cost_ylim(*curves))
                if len(row['curves']) > 1:
                    ax.legend(fontsize=8)
                ax.set_ylabel('cost [% data power]')
                ax.set_title(row['title'])
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel('step')
            fig.suptitle(f'Training cost ({nsteps} steps)', fontsize=12, y=1.01)
            fig.tight_layout()
            t_draw = time.perf_counter()
            save_figure(fig, path, dpi=150)
            log_plot_elapsed(path, t0, draw=t_draw - t0, save=time.perf_counter() - t_draw)
            return
        if len(names) == 1:
            costs = costs_by_target[names[0]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(costs, color='steelblue', linewidth=2)
    ax.set_ylim(*cost_ylim(costs))
    ax.set_xlabel('step')
    ax.set_ylabel('cost [% data power]')
    ax.set_title(f'Training cost ({len(costs)} steps)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    t_draw = time.perf_counter()
    save_figure(fig, path, dpi=150)
    log_plot_elapsed(path, t0, draw=t_draw - t0, save=time.perf_counter() - t_draw)
