"""Shared plotting helpers (no task-specific logic)."""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import network.path  # noqa: F401 — FAFB path on sys.path
import training
from import_bootstrap import parse_comma_list

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


def apply_out_scale(p, traces, node_index, backend):
    """Multiply traces by schema ``out_scale`` (matches cost)."""
    if node_index is None:
        node_index = torch.arange(traces.shape[-1], device=traces.device)
    else:
        node_index = torch.as_tensor(node_index, dtype=torch.long, device=traces.device)
    s = training.out_scale_for_nodes(p, node_index, backend, sim_dtype=traces.dtype)
    return traces * (s if traces.ndim == 3 else s[:, None])


def save_forward_trace_csvs(
    save_dir,
    task,
    *,
    ref,
    trace_full,
    ref_stem: str | None = None,
    trace_stem: str | None = None,
):
    """Write per-task ref + ``v`` trace CSVs under ``save_dir``.

    Default stems: ``<task>_v_onset.csv``, ``<task>_v.csv`` (unless
    overridden by ``ref_stem`` / ``trace_stem``).

    Ref is one column (``ref``) with constant ``N``. Trace is ``(B*T', N)`` with
    constant ``B`` / ``Tprime`` columns, then one column per node.
    """
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    ref_np = _as_numpy(ref).reshape(-1)
    trace_np = _as_numpy(trace_full)
    if trace_np.ndim != 3:
        raise ValueError(f'trace_full must be (B, T\', N), got shape {trace_np.shape}')
    bsz, tprime, n_nodes = (int(x) for x in trace_np.shape)
    if ref_np.size != n_nodes:
        raise ValueError(
            f'ref length {ref_np.size} != n_nodes {n_nodes} for task {task!r}'
        )
    ref_stem_default, trace_stem_default = f'{task}_v_onset', f'{task}_v'
    ref_stem_final = ref_stem_default if ref_stem is None else ref_stem
    trace_stem_final = trace_stem_default if trace_stem is None else trace_stem
    ref_path = os.path.join(save_dir, f'{ref_stem_final}.csv')
    trace_path = os.path.join(save_dir, f'{trace_stem_final}.csv')
    ref_table = np.column_stack([
        np.full(ref_np.shape[0], ref_np.size, dtype=np.int64),
        ref_np.astype(np.float64, copy=False),
    ])
    # Some traces reuse v_onset across tasks (e.g. bright/dark); avoid redundant writes.
    if not os.path.exists(ref_path):
        np.savetxt(
            ref_path, ref_table, delimiter=',', header='N,ref', comments='',
        )
    flat = trace_np.reshape(-1, n_nodes).astype(np.float64, copy=False)
    n_rows = flat.shape[0]
    trace_table = np.column_stack([
        np.full(n_rows, bsz, dtype=np.int64),
        np.full(n_rows, tprime, dtype=np.int64),
        flat,
    ])
    node_header = ','.join(f'u{i}' for i in range(n_nodes))
    np.savetxt(
        trace_path, trace_table, delimiter=',',
        header=f'B,Tprime,{node_header}', comments='',
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
    """Y-axis: mid tick at 0 labeled with ``v_onset``; dashed line at ``v_th``/``v_rest``.

    ``baseline`` is ``v_onset`` or ``(v_onset, v_ref)`` in absolute mV. Traces are
    delta-mV, so the line is at ``y = v_ref - v_onset`` (or ``y = 0`` if no ref).
    """
    ylo, yhi = ax.get_ylim()
    v_onset, v_ref = _unpack_baseline(baseline)
    y_line = 0.0
    if (
        v_onset is not None and v_ref is not None
        and np.isfinite(v_onset) and np.isfinite(v_ref)
    ):
        y_line = float(v_ref) - float(v_onset)
    mid = f'{float(v_onset):.1f}' if v_onset is not None and np.isfinite(v_onset) else ''
    ax.set_yticks([ylo, 0.0, yhi])
    ax.set_yticklabels([f'{ylo:+.0f}', mid, f'{yhi:+.0f}'], fontsize=6)
    ax.axhline(y_line, color='0.4', linewidth=0.6, linestyle=':', zorder=0)


def _unpack_baseline(baseline):
    """Return ``(v_onset, v_ref_or_none)`` from a scalar or ``(onset, ref)`` pair."""
    if baseline is None:
        return None, None
    if isinstance(baseline, (tuple, list)) and len(baseline) >= 2:
        return baseline[0], baseline[1]
    return baseline, None


def mark_pulse(ax, pulse_start, pulse_end):
    """White band for stimulus-on samples ``[pulse_start, pulse_end)`` (axes face is gray)."""
    if pulse_start is None or pulse_end is None:
        return
    t0 = int(pulse_start)
    t1 = int(pulse_end)
    if t1 <= t0:
        return
    ax.axvspan(t0, t1, facecolor='white', edgecolor='none', zorder=0)


def suppress_cost_sem(session, task=None):
    """True when cost uses a single column (no column-mean SEM band)."""
    pack = session.primary_readout if task is None else session.pack_for(task)
    return pack.cost_extent == 0


def readout_center_mask(pack, backend):
    """Boolean mask over pack.readout rows included in the cost extent."""
    readout = pack.readout_node.cpu().numpy()
    if backend.network is not None:
        if pack.cost_extent is not None:
            import build_hex
            C = backend.network
            u_all = C.u.detach().cpu().numpy() if hasattr(C.u, "detach") else np.asarray(C.u)
            v_all = C.v.detach().cpu().numpy() if hasattr(C.v, "detach") else np.asarray(C.v)
            return build_hex.inside_mask(u_all[readout], v_all[readout], int(pack.cost_extent))
        if pack.cost_radius is not None:
            return np.round(pack.cost_radius.cpu().numpy(), 6) == 0.0
        return np.ones(readout.shape[0], dtype=bool)
    return np.ones(readout.shape[0], dtype=bool)


def baselines_for_types(v_onset, nodes_by_name, v_ref_by_name=None):
    """``{name: (v_onset_mean, v_ref)}`` from per-cell node index arrays.

    Callers select nodes (spot center row, moving-bar cost hexes, pack
    readout mask); this only averages ``v_onset`` and pairs with ``v_ref``.
    """
    v_onset = np.asarray(v_onset, dtype=np.float64)
    v_ref_by_name = v_ref_by_name or {}
    out = {}
    for name, nodes in nodes_by_name.items():
        u = np.asarray(nodes, dtype=np.int64).reshape(-1)
        onset = float(v_onset[u].mean()) if u.size else np.nan
        ref = v_ref_by_name.get(name, np.nan)
        out[name] = (
            onset,
            float(ref) if ref is not None and np.isfinite(ref) else np.nan,
        )
    return out


def sem_from_traces(traces, single_hex=False):
    """Per-time SEM across readout rows; zero when single-column cost or one row."""
    if single_hex or traces.shape[0] == 1:
        return np.zeros(traces.shape[1], dtype=np.float64)
    return traces.std(axis=0) / np.sqrt(traces.shape[0])


def readout_n_by_name(type_idx, cell_names, names, node_idx):
    """Unique readout node count per plotted type name."""
    type_idx = np.asarray(type_idx)
    node_idx = np.asarray(node_idx)
    return {
        name: int(np.unique(node_idx[type_idx == cell_names.index(name)]).size)
        for name in names
    }


def v_ref_schema_name(schema):
    """``'v_th'`` (borst) or ``'v_rest'`` (hp_lp); ``None`` if neither in schema."""
    names = {s.get('name') for s in schema}
    if 'v_th' in names:
        return 'v_th'
    if 'v_rest' in names:
        return 'v_rest'
    return None


def v_ref_by_type_name(z, session):
    """Per-cell ``v_th`` / ``v_rest`` (mV) keyed by type name; empty if schema has neither."""
    schema = list(session.schema)
    key = v_ref_schema_name(schema)
    if key is None:
        return {}
    arr = np.asarray(training.z_to_node_values(z, schema)[key], dtype=np.float64).reshape(-1)
    names = training.cell_node_names(session.backend)
    if arr.shape[0] != len(names):
        raise ValueError(f"{key} length {arr.shape[0]} != n_cells {len(names)}")
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


def cell_title_with_n(label, n=None, v_ref=None, *, v_ref_name=None):
    """Spot-style panel title: cell name + optional ``n``, ``v_ref`` on next line."""
    head = label_with_n(label, n)
    if v_ref is None or v_ref_name is None:
        return head
    return f'{head}\n{v_ref_name}={float(v_ref):.1f} mV'


def panel_title_with_v_ref(label, v_ref=None, *, v_ref_name=None):
    """Stimulus / panel title with optional ``v_ref`` line (no ``n``)."""
    if v_ref is None or v_ref_name is None:
        return label
    return f'{label}\n{v_ref_name}={float(v_ref):.1f} mV'


def _bundle_v_ref_name(bundle):
    name = getattr(bundle, 'v_ref_name', None)
    if name is not None:
        return name
    session = getattr(bundle, 'session', None)
    if session is None:
        return None
    return v_ref_schema_name(session.schema)


def bundle_cell_title(bundle, label, n=None, *, type_name=None):
    """Spot panel title from ``bundle.v_ref_by_name`` (key = *type_name* or *label*)."""
    key = label if type_name is None else type_name
    return cell_title_with_n(
        label, n, bundle.v_ref_by_name.get(key),
        v_ref_name=_bundle_v_ref_name(bundle),
    )


def bundle_panel_title(bundle, label, *, type_name=None):
    """Moving-bar panel title from ``bundle.v_ref_by_name`` (no ``n``)."""
    key = label if type_name is None else type_name
    return panel_title_with_v_ref(
        label, bundle.v_ref_by_name.get(key),
        v_ref_name=_bundle_v_ref_name(bundle),
    )


def network_hex_count(C):
    """Unique axial hexes on connectome ``C``."""
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
    """Parse ``--align-xy X,Y`` reference sti hex (empty → ``None``)."""
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


def hex_at_scope_tag(at_x, at_y):
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
    import build_hex
    out = []
    for b, batch in enumerate(batches):
        for su, sv in batch.stim_uv:
            x, y = build_hex.uv_to_xy(int(su), int(sv))
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
    *curve_groups,
    show_sem=False,
    extra=(),
):
    """Y-limits from one or more ``(model, data, sem)`` groups plus ``extra`` curves."""
    curves = []
    for group in curve_groups:
        if group is None:
            continue
        if isinstance(group, dict):
            model = group.get("model")
            data = group.get("data")
            sem = group.get("sem")
        else:
            model, data, sem = (list(group) + [None, None, None])[:3]
        if model is not None:
            curves.append(model)
        if data is not None:
            curves.append(data)
        if show_sem and sem is not None and model is not None:
            curves.append(model + sem)
            curves.append(model - sem)
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
    """Shaded ±SEM for continuous line traces."""
    if sem is None or not np.any(sem):
        return
    t_arr = np.asarray(t)
    m_arr = np.asarray(model, dtype=np.float64)
    s_arr = np.asarray(sem, dtype=np.float64)
    mask = np.isfinite(m_arr) & np.isfinite(s_arr)
    if not np.any(mask):
        return
    ax.fill_between(
        t_arr[mask],
        m_arr[mask] - s_arr[mask],
        m_arr[mask] + s_arr[mask],
        color=SEM_COLOR if color is None else color,
        alpha=0.3 if alpha is None else alpha,
        linewidth=0,
        label=label,
        zorder=1,
    )


def plot_sem_errorbar(ax, t, model, sem, *, color=None, alpha=None, label=r'$\pm$SEM'):
    """Error bars for discrete (dot) traces."""
    if sem is None or not np.any(sem):
        return
    ax.errorbar(
        t, model, yerr=sem,
        fmt='none',
        ecolor=SEM_COLOR if color is None else color,
        alpha=0.8 if alpha is None else alpha,
        elinewidth=0.8,
        capsize=1.5,
        capthick=0.8,
        label=label,
    )


def _series_points(t, y, point_ix=None):
    """Return finite ``(x, y)`` points, optionally subsampled by integer indices."""
    if y is None:
        return None, None
    t_arr = np.asarray(t)
    y_arr = np.asarray(y, dtype=np.float64)
    if point_ix is not None:
        ix = np.asarray(point_ix, dtype=np.int64)
        ix = ix[(ix >= 0) & (ix < y_arr.shape[0])]
        t_arr = t_arr[ix]
        y_arr = y_arr[ix]
    mask = np.isfinite(y_arr)
    if not np.any(mask):
        return None, None
    return t_arr[mask], y_arr[mask]


def plot_pre_post_line(
    ax,
    t,
    y,
    *,
    pre_end=0,
    show_pre=False,
    color=MODEL_COLOR,
    linestyle='-',
    linewidth=TRACE_LW,
    label=None,
    draw_pre=False,
):
    """Plot a 1-D series with optional dashed pre-``pre_end`` segment.

    ``pre_end`` is the first post-onset index (samples ``[0, pre_end)`` are pre).
    Gray data uses ``draw_pre=False`` (never draws pre). Model uses
    ``draw_pre=show_pre`` (dashed pre when true; omit pre when false).
    """
    if y is None:
        return
    t_arr = np.asarray(t)
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1 or t_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f'plot_pre_post_line expects 1-D t/y of equal length, '
            f'got t={getattr(t_arr, "shape", None)} y={getattr(y_arr, "shape", None)}'
        )
    n = int(y_arr.shape[0])
    split = max(0, min(int(pre_end or 0), n))
    if draw_pre and show_pre and split > 0:
        # Include the onset sample so dashed and solid segments meet.
        end_pre = min(split + 1, n)
        ax.plot(
            t_arr[:end_pre], y_arr[:end_pre],
            color=color, linewidth=linewidth, linestyle='--',
        )
    if split >= n:
        return
    ax.plot(
        t_arr[split:], y_arr[split:],
        color=color, linewidth=linewidth, linestyle=linestyle, label=label,
    )


def plot_timecourse(
    ax,
    t,
    traces,
    *,
    show_sem=True,
    title=None,
    title_fs=7,
    ylim=None,
    baseline=None,
    show_ylabel=False,
    ylabel='mV',
    ticksize=6,
    style_xaxis=None,
    pre_end=0,
    show_pre=False,
    pulse_start=None,
    pulse_end=None,
):
    """Model (red) vs data (gray) time courses for one or more contrast traces.

    ``traces``: sequence of dicts with keys ``model``, ``data``, optional
    ``sem``, ``linestyle`` (default ``'-'``), ``point_ix``.
    Gray data never draws ``[0, pre_end)``; red model draws that pre segment
    dashed only when ``show_pre`` is true.
    ``pulse_start`` / ``pulse_end``: white stimulus-on band ``[pulse_start, pulse_end)``.
    """
    traces = list(traces or ())
    if ylim is None:
        ylo, yhi = ylim_for_traces(*traces, show_sem=show_sem)
    else:
        ylo, yhi = ylim
    mark_pulse(ax, pulse_start, pulse_end)
    split = max(0, int(pre_end or 0))
    for tr in traces:
        model = tr.get("model")
        data = tr.get("data")
        sem = tr.get("sem")
        linestyle = tr.get("linestyle", "-")
        point_ix = tr.get("point_ix")
        discrete = point_ix is not None
        if not discrete:
            if data is not None:
                plot_pre_post_line(
                    ax, t, data, pre_end=split, show_pre=False, draw_pre=False,
                    color=DATA_COLOR, linestyle=linestyle, linewidth=TRACE_LW,
                )
            if model is not None:
                if show_sem and sem is not None:
                    t_arr = np.asarray(t)
                    m_arr = np.asarray(model, dtype=np.float64)
                    s_arr = np.asarray(sem, dtype=np.float64)
                    if split < m_arr.shape[0]:
                        plot_sem_band(ax, t_arr[split:], m_arr[split:], s_arr[split:])
                plot_pre_post_line(
                    ax, t, model, pre_end=split, show_pre=show_pre, draw_pre=True,
                    color=MODEL_COLOR, linestyle=linestyle, linewidth=TRACE_LW,
                )
        else:
            x_data, y_data = _series_points(t, data, point_ix=point_ix)
            if x_data is not None:
                ax.plot(
                    x_data, y_data, linestyle='none', marker='o', markersize=4,
                    fillstyle='none', markeredgewidth=1.0, color=DATA_COLOR,
                )
            if model is not None:
                model_arr = np.asarray(model, dtype=np.float64)
                ix_model = np.asarray(point_ix, dtype=np.int64)
                ix_model = ix_model[(ix_model >= 0) & (ix_model < model_arr.shape[0])]
                x_model = np.asarray(t)[ix_model]
                y_model = model_arr[ix_model]
                mask_model = np.isfinite(y_model)
                if show_sem and sem is not None:
                    sem_arr = np.asarray(sem, dtype=np.float64)
                    sem_sub = sem_arr[ix_model]
                    mask_sem = mask_model & np.isfinite(sem_sub)
                    if np.any(mask_sem):
                        plot_sem_errorbar(
                            ax, x_model[mask_sem], y_model[mask_sem], sem_sub[mask_sem],
                        )
                if np.any(mask_model):
                    ax.plot(
                        x_model[mask_model], y_model[mask_model], linestyle='none',
                        marker='o', markersize=2.5, fillstyle='full',
                        markeredgewidth=0.8, color=MODEL_COLOR,
                    )
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


def _cost_curve_subplot_rows(names, costs_by_task, total_costs):
    """Build subplot specs; merge moving_bar ``_PD``/``_ND`` part keys per task."""
    rows = [{'title': 'total (weighted)', 'curves': [(None, np.asarray(total_costs), '-')]}]
    seen = set()
    for key in names:
        if key not in costs_by_task or key in seen:
            continue
        if key.endswith('_PD'):
            base = key[:-3]
            nd_key = f"{base}_ND"
            curves = [('PD', np.asarray(costs_by_task[key]), '-')]
            if nd_key in costs_by_task:
                curves.append(('ND', np.asarray(costs_by_task[nd_key]), '--'))
                seen.add(nd_key)
            rows.append({'title': base, 'curves': curves})
            seen.add(key)
        elif key.endswith('_ND'):
            base = key[:-3]
            rows.append({'title': base, 'curves': [('ND', np.asarray(costs_by_task[key]), '--')]})
            seen.add(key)
        else:
            rows.append({
                'title': key,
                'curves': [(None, np.asarray(costs_by_task[key]), '-')],
            })
            seen.add(key)
    return rows


def plot_cost(costs, path, *, costs_by_task=None, task_order=None):
    """Plot training cost; total + one subplot per task when ``costs_by_task`` is given."""
    t0 = time.perf_counter()
    if costs_by_task:
        names = list(task_order) if task_order else list(costs_by_task.keys())
        names = [n for n in names if n in costs_by_task and len(costs_by_task[n])]
        if names and costs is not None and len(costs):
            rows = _cost_curve_subplot_rows(names, costs_by_task, costs)
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
            costs = costs_by_task[names[0]]
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
