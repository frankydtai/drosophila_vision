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

GT_COLOR = 'gray'
MODEL_COLOR = 'red'
SEM_COLOR = 'pink'
TRACE_LW = 1.5


def as_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _as_numpy(arr):
    return as_numpy(arr)


def apply_gt_affine(p, gt, node_index, backend):
    """``a_gt * gt + bias_gt`` (``+ v_th`` if present; matches cost).

    ``gt`` leading axis = nodes.
    """
    node_index = torch.as_tensor(node_index, dtype=torch.long)
    scale, bias = training.gt_affine_for_nodes(
        p, node_index, backend, sim_dtype=torch.float32,
    )
    scale_np = scale.detach().cpu().numpy()
    bias_np = bias.detach().cpu().numpy()
    g = np.asarray(gt, dtype=np.float64)
    if g.ndim == 1:
        return scale_np * g + bias_np
    return (
        scale_np.reshape(-1, *([1] * (g.ndim - 1))) * g
        + bias_np.reshape(-1, *([1] * (g.ndim - 1)))
    )


def gt_affine_scalars_for_cell(p, cell_name, backend) -> tuple[float, float]:
    """``(a_gt, effective_bias)`` for one cell type name (matches cost)."""
    names = [str(n) for n in backend.network.cell_names]
    ci = names.index(str(cell_name))
    gs = p["a_gt"]
    gb = p["bias_gt"]
    scale = float(gs[ci] if torch.is_tensor(gs) and gs.dim() > 0 else gs)
    bias = float(gb[ci] if torch.is_tensor(gb) and gb.dim() > 0 else gb)
    if "v_th" in p:
        vt = p["v_th"]
        bias = bias + float(vt[ci] if torch.is_tensor(vt) and vt.dim() > 0 else vt)
    return scale, bias


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
    """Horizontal dashed line at ``v_th`` (borst) / ``-bias_out`` (hp_lp).

    Leaves matplotlib auto y-ticks / labels alone.
    """
    if baseline is None or not np.isfinite(baseline):
        return
    ax.axhline(float(baseline), color='0.4', linewidth=0.6, linestyle=':', zorder=0)


def baselines_for_types(nodes_by_name, v_ref_by_name=None):
    """``{name: baseline}`` from per-cell ``v_th`` / ``-bias_out`` (absolute mV)."""
    v_ref_by_name = v_ref_by_name or {}
    out = {}
    for name in nodes_by_name:
        ref = v_ref_by_name.get(name, np.nan)
        out[name] = float(ref) if ref is not None and np.isfinite(ref) else np.nan
    return out


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


def sem_from_traces(traces, single_hex=False):
    """Per-time SEM across readout rows; zero when single-column cost or one row."""
    if single_hex or traces.shape[0] == 1:
        return np.zeros(traces.shape[1], dtype=np.float64)
    return traces.std(axis=0) / np.sqrt(traces.shape[0])


def v_ref_schema_name(schema):
    """``'v_th'`` (borst) or ``'-bias_out'`` (hp_lp); ``None`` if neither applies."""
    names = {s.get('name') for s in schema}
    if 'v_th' in names:
        return 'v_th'
    if 'bias_out' in names:
        return '-bias_out'
    return None


def v_ref_by_type_name(z, session):
    """Per-cell baseline mV: ``v_th`` (borst) or ``-bias_out`` (hp_lp)."""
    schema = list(session.schema)
    key = v_ref_schema_name(schema)
    if key is None:
        return {}
    param = 'v_th' if key == 'v_th' else 'bias_out'
    arr = np.asarray(training.z_to_node_values(z, schema)[param], dtype=np.float64).reshape(-1)
    if key == '-bias_out':
        arr = -arr
    names = training.cell_node_names(session.backend)
    if arr.shape[0] != len(names):
        raise ValueError(f"{param} length {arr.shape[0]} != n_cells {len(names)}")
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


def format_spot_radius_time_title(radius, n, cell, cost_parts, contrasts):
    """Time-panel title: ``r=0 (n=252)`` + ``bright: 63.3`` / ``dark: …``."""
    from training.config import spot_cost_part_key

    r = float(radius)
    r_s = str(int(r)) if r == int(r) else str(r)
    head = f'r={r_s}'
    if n is not None:
        head = f'{head} (n={int(n)})'
    if not cost_parts or not contrasts:
        return head
    lines = [head]
    for contrast in contrasts:
        key = spot_cost_part_key(f'spot_{contrast}', cell, r)
        if key in cost_parts:
            lines.append(f'{contrast}: {float(cost_parts[key]):.1f}')
    return '\n'.join(lines)


def format_moving_bar_cell_cost_lines(cell, cost_parts, task_names):
    """Lines ``ON: xx @PD yy @ND`` / ``OFF: …`` for moving-bar titles."""
    tag = {
        'moving_bar_bright': 'ON',
        'moving_bar_dark': 'OFF',
    }
    lines = []
    if not cost_parts:
        return lines
    for task in task_names:
        bits = []
        for lab in ('PD', 'ND'):
            key = f'{task}_{cell}_{lab}'
            if key in cost_parts:
                bits.append(f'{float(cost_parts[key]):.1f} @{lab}')
        if bits:
            lines.append(f'{tag.get(task, task)}: {" ".join(bits)}')
    return lines


def bundle_panel_title(bundle, label, *, type_name=None):
    """Moving-bar panel base title (cell / stimulus label; no ``v_rest``)."""
    _ = (bundle, type_name)
    return label


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


def parse_axis_slices(text):
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


def add_ms_shown_argument(parser):
    """Register ``--ms-shown START,STOP`` display / analyze time window.

    Absolute aligned ms — not stimulus-length (``--ms-pre``) and not
    onset-relative for spot. Spot: ``0`` = trial start, pre = ``0,ms_pre``.
    Bar: ``0`` = bar t0 at the node. See ``analyze.cell_dynamics`` module doc.
    """
    parser.add_argument(
        '--ms-shown',
        default=None,
        metavar='START,STOP',
        help=(
            'absolute aligned ms START,STOP (not --ms-pre; not onset-relative). '
            'spot: 0=trial start, pre=0,ms_pre (e.g. 0,1000); '
            'bar: 0=t0 at node (neg START ok); omit = full trace'
        ),
    )


def parse_ms_shown_range(token, *, flag='--ms-shown'):
    """Parse ``START,STOP`` ms (comma; one token)."""
    parts = parse_comma_list(token)
    if len(parts) != 2:
        raise ValueError(f'{flag} must be START,STOP')
    start, stop = float(parts[0]), float(parts[1])
    if start > stop:
        raise ValueError(f'{flag} START={start} > STOP={stop}')
    return start, stop


def ms_shown_axis_xlim(ms_shown, *, delta_ms, origin_t=0):
    """Inclusive t-index xlim from ``--ms-shown``; ``origin_t`` is t0 on the axis."""
    if ms_shown is None:
        return None
    start, stop = ms_shown
    lo = int(origin_t) + int(training.ms_to_t(float(start), delta_ms=float(delta_ms)))
    hi = int(origin_t) + int(training.ms_to_t(float(stop), delta_ms=float(delta_ms)))
    if lo > hi:
        raise ValueError(f'ms-shown xlim START t={lo} > STOP t={hi}')
    return lo, hi


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


def slice_axis_label(val):
    fv = float(val)
    if np.isclose(fv, round(fv)):
        return str(int(round(fv)))
    return str(fv)


def slice_xy_label(xv, yv):
    return f'({slice_axis_label(xv)},{slice_axis_label(yv)})'


def slice_coord_specs(at_xs, at_ys):
    """Expand optional x/y lists to ``[(label, at_x, at_y), ...]`` (missing axis is None)."""
    if at_xs is not None and at_ys is not None:
        return [
            (slice_xy_label(xv, yv), xv, yv)
            for xv in at_xs for yv in at_ys
        ]
    if at_xs is not None:
        return [(slice_axis_label(xv), xv, None) for xv in at_xs]
    if at_ys is not None:
        return [(slice_axis_label(yv), None, yv) for yv in at_ys]
    return []


def slice_axis_name(at_xs, at_ys):
    """``'xy'`` / ``'x'`` / ``'y'`` / ``None`` matching :func:`slice_coord_specs`."""
    if at_xs is not None and at_ys is not None:
        return 'xy'
    if at_xs is not None:
        return 'x'
    if at_ys is not None:
        return 'y'
    return None


def overlay_model_reds(n_slices):
    """Red shades for per-slice traces plus a darker ``total`` trace."""
    n = n_slices + 1
    return [plt.cm.Reds(v) for v in np.linspace(0.35, 0.95, n)]


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
    Gray gt uses ``draw_pre=False`` (never draws pre). Model uses
    ``draw_pre=show_pre`` (dashed pre when true; omit pre when false).
    """
    if y is None:
        return
    t_arr = np.asarray(t)
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1:
        raise ValueError(
            f'plot_pre_post_line expects 1-D y, got y={getattr(y_arr, "shape", None)}'
        )
    if t_arr.shape[0] > y_arr.shape[0]:
        # Spot gt omits ms_post; model / axis may be longer.
        t_arr = t_arr[: y_arr.shape[0]]
    elif t_arr.shape[0] < y_arr.shape[0]:
        raise ValueError(
            f'plot_pre_post_line expects t length >= y length, '
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
    """Model (red) vs gt (gray) time courses for one or more contrast traces.

    ``traces``: sequence of dicts with keys ``model``, ``gt``, optional
    ``sem``, ``linestyle`` (default ``'-'``), ``point_ix``.
    Gray gt never draws ``[0, pre_end)``; red model draws that pre segment
    dashed only when ``show_pre`` is true.
    ``pulse_start`` / ``pulse_end``: white stimulus-on band ``[pulse_start, pulse_end)``.
    Y-limits / ticks: matplotlib autoscale.
    """
    traces = list(traces or ())
    mark_pulse(ax, pulse_start, pulse_end)
    split = max(0, int(pre_end or 0))
    for tr in traces:
        model = tr.get("model")
        gt = tr.get("gt")
        sem = tr.get("sem")
        linestyle = tr.get("linestyle", "-")
        point_ix = tr.get("point_ix")
        discrete = point_ix is not None
        if not discrete:
            if gt is not None:
                plot_pre_post_line(
                    ax, t, gt, pre_end=split, show_pre=False, draw_pre=False,
                    color=GT_COLOR, linestyle=linestyle, linewidth=TRACE_LW,
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
            x_gt, y_gt = _series_points(t, gt, point_ix=point_ix)
            if x_gt is not None:
                ax.plot(
                    x_gt, y_gt, linestyle='none', marker='o', markersize=4,
                    fillstyle='none', markeredgewidth=1.0, color=GT_COLOR,
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
    if style_xaxis is not None:
        style_xaxis(ax)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=ticksize)
    annotate_baseline(ax, baseline)


_MPL_DASH = {
    '-': 'solid',
    '--': 'dash',
    '-.': 'dashdot',
    ':': 'dot',
    'None': 'solid',
    'none': 'solid',
    '': 'solid',
}


def plot_file_ext(*, html=False):
    """Figure extension: ``.png`` (default) or ``.html`` when ``html``."""
    return '.html' if html else '.png'


def plot_html_path(path):
    """Rewrite path stem to ``.html``."""
    path = os.fspath(path)
    root, ext = os.path.splitext(path)
    if ext.lower() == '.html':
        return path
    return root + '.html'


def _mpl_color_hex(color):
    from matplotlib.colors import to_hex

    try:
        return to_hex(color, keep_alpha=False)
    except (ValueError, TypeError):
        return '#1f77b4'


def _mpl_color_rgba(color):
    from matplotlib.colors import to_rgba

    try:
        r, g, b, a = to_rgba(color)
    except (ValueError, TypeError):
        return 'rgba(31,119,180,1)'
    return f'rgba({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))},{a:g})'


def _write_interactive_html(fig, path):
    """Write standalone HTML: hover traces for x/y (plotly), matching PNG style."""
    import plotly.graph_objects as go
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    fig.canvas.draw()
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    plot_bg = (
        _mpl_color_hex(visible_axes[0].get_facecolor())
        if visible_axes else plt.rcParams['axes.facecolor']
    )
    traces = []
    shapes = []
    layout = {
        'autosize': False,
        'width': max(400, int(fig.get_figwidth() * 100)),
        'height': max(300, int(fig.get_figheight() * 100)),
        'margin': dict(l=40, r=20, t=60, b=40),
        'hovermode': 'closest',
        'showlegend': False,
        'plot_bgcolor': plot_bg,
        'paper_bgcolor': _mpl_color_hex(fig.get_facecolor()),
    }
    if fig._suptitle is not None:
        layout['title'] = dict(text=fig._suptitle.get_text(), x=0.5, xanchor='center')

    axis_i = 0
    for ax in visible_axes:
        axis_i += 1
        xaxis = 'x' if axis_i == 1 else f'x{axis_i}'
        yaxis = 'y' if axis_i == 1 else f'y{axis_i}'
        xkey = 'xaxis' if axis_i == 1 else f'xaxis{axis_i}'
        ykey = 'yaxis' if axis_i == 1 else f'yaxis{axis_i}'
        pos = ax.get_position()
        layout[xkey] = dict(
            domain=[float(pos.x0), float(pos.x1)],
            anchor=yaxis,
            title=ax.get_xlabel() or None,
            showgrid=True,
            zeroline=False,
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='#444',
            gridcolor='#ddd',
        )
        layout[ykey] = dict(
            domain=[float(pos.y0), float(pos.y1)],
            anchor=xaxis,
            title=ax.get_ylabel() or None,
            showgrid=True,
            zeroline=False,
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='#444',
            gridcolor='#ddd',
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        layout[xkey]['range'] = [float(xlim[0]), float(xlim[1])]
        layout[ykey]['range'] = [float(ylim[0]), float(ylim[1])]
        if ax.get_title():
            layout.setdefault('annotations', []).append(dict(
                text=ax.get_title(),
                xref='paper',
                yref='paper',
                x=(float(pos.x0) + float(pos.x1)) / 2,
                y=float(pos.y1),
                xanchor='center',
                yanchor='bottom',
                showarrow=False,
                font=dict(size=10),
            ))

        # Pulse / axvspan bands (data-x, axes-y) → shapes under traces.
        for patch in ax.patches:
            if not isinstance(patch, Rectangle):
                continue
            x0 = float(patch.get_x())
            x1 = x0 + float(patch.get_width())
            shapes.append(dict(
                type='rect',
                xref=xaxis,
                yref=f'{yaxis} domain',
                x0=x0,
                x1=x1,
                y0=0,
                y1=1,
                fillcolor=_mpl_color_rgba(patch.get_facecolor()),
                line=dict(width=0),
                layer='below',
            ))

        # SEM fill_between etc. before line traces (same stacking as PNG).
        for coll in ax.collections:
            if not isinstance(coll, PolyCollection):
                continue
            fcs = coll.get_facecolors()
            if fcs is None or len(fcs) == 0:
                continue
            for pi, coll_path in enumerate(coll.get_paths()):
                verts = np.asarray(coll_path.vertices, dtype=float)
                if verts.size == 0:
                    continue
                fc = fcs[min(pi, len(fcs) - 1)]
                traces.append(go.Scatter(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    mode='lines',
                    fill='toself',
                    fillcolor=_mpl_color_rgba(fc),
                    line=dict(width=0, color=_mpl_color_rgba(fc)),
                    hoverinfo='skip',
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                ))

        for li, line in enumerate(ax.get_lines()):
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd = np.asarray(line.get_ydata(), dtype=float)
            if xd.size == 0:
                continue
            label = line.get_label()
            if not label or str(label).startswith('_'):
                label = ax.get_title() or f'trace{li}'
            ls = line.get_linestyle()
            if isinstance(ls, (tuple, list)):
                dash = 'dash' if ls not in (None, 'None', 'none', '-', 'solid') else 'solid'
            else:
                dash = _MPL_DASH.get(str(ls), 'solid')
            marker = line.get_marker()
            mode = 'lines'
            if marker not in (None, 'None', 'none', ''):
                mode = 'lines+markers' if dash != 'solid' or line.get_linewidth() else 'markers'
                if line.get_linestyle() in ('None', 'none', ''):
                    mode = 'markers'
            traces.append(go.Scatter(
                x=xd,
                y=yd,
                mode=mode,
                name=str(label),
                line=dict(
                    color=_mpl_color_hex(line.get_color()),
                    width=float(line.get_linewidth() or 1.0),
                    dash=dash,
                ),
                marker=dict(size=max(4.0, float(line.get_markersize() or 4.0))),
                xaxis=xaxis,
                yaxis=yaxis,
                hovertemplate='x=%{x}<br>y=%{y}<extra>%{fullData.name}</extra>',
            ))

    if shapes:
        layout['shapes'] = shapes
    pfig = go.Figure(data=traces, layout=layout)
    pfig.write_html(path, include_plotlyjs=True, full_html=True, config={
        'displayModeBar': True,
        'scrollZoom': True,
    })


def save_figure(fig, path, dpi=150, rasterize=False):
    """Save figure: ``.png`` (default) or interactive ``.html`` (plotly hover x/y)."""
    path = os.fspath(path)
    if path.lower().endswith('.html'):
        _write_interactive_html(fig, path)
    else:
        if rasterize:
            for ax in fig.axes:
                ax.set_rasterized(True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _cost_curve_subplot_rows(names, costs_by_part, total_costs):
    """Build subplot specs; merge moving_bar ``_PD``/``_ND`` part keys per task/cell."""
    rows = [{'title': 'total (weighted)', 'curves': [(None, np.asarray(total_costs), '-')]}]
    seen = set()
    for key in names:
        if key not in costs_by_part or key in seen:
            continue
        if key.endswith('_PD'):
            base = key[:-3]
            nd_key = f"{base}_ND"
            curves = [('PD', np.asarray(costs_by_part[key]), '-')]
            if nd_key in costs_by_part:
                curves.append(('ND', np.asarray(costs_by_part[nd_key]), '--'))
                seen.add(nd_key)
            rows.append({'title': base, 'curves': curves})
            seen.add(key)
        elif key.endswith('_ND'):
            base = key[:-3]
            rows.append({'title': base, 'curves': [('ND', np.asarray(costs_by_part[key]), '--')]})
            seen.add(key)
        else:
            rows.append({
                'title': key,
                'curves': [(None, np.asarray(costs_by_part[key]), '-')],
            })
            seen.add(key)
    return rows


def plot_cost(costs, path, *, costs_by_part=None, part_order=None):
    """Plot training cost; total + one subplot per part when ``costs_by_part`` is given."""
    t0 = time.perf_counter()
    if costs_by_part:
        names = list(part_order) if part_order else list(costs_by_part.keys())
        names = [n for n in names if n in costs_by_part and len(costs_by_part[n])]
        if names and costs is not None and len(costs):
            rows = _cost_curve_subplot_rows(names, costs_by_part, costs)
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
                ax.set_ylabel('cost [% gt power]')
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
            costs = costs_by_part[names[0]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(costs, color='steelblue', linewidth=2)
    ax.set_ylim(*cost_ylim(costs))
    ax.set_xlabel('step')
    ax.set_ylabel('cost [% gt power]')
    ax.set_title(f'Training cost ({len(costs)} steps)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    t_draw = time.perf_counter()
    save_figure(fig, path, dpi=150)
    log_plot_elapsed(path, t0, draw=t_draw - t0, save=time.perf_counter() - t_draw)
