"""Shared plotting helpers (no task-specific logic)."""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import network.path  # noqa: F401 — FAFB path on sys.path
import training

GT_COLOR = 'gray'
V_READOUT_COLOR = 'red'
STD_COLOR = 'pink'
TRACE_LW = 1.5
NCOLS_GT = 5
NCOLS_ALL = 8
PANEL_W = 3.0
PANEL_H = 2.2


def as_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def gt_affine_scalars_for_cell(p, cell_name, backend, *, bias_gt=None) -> tuple[float, float]:
    """``(a_gt, effective_bias)`` for one cell type name (matches cost).

    Pass ``bias_gt`` to override schema bias (e.g. mean ``v`` at onset when
    ``bias_gt_from_v_onset``). Onset overrides are clipped to schema
    ``bias_gt`` lo/hi (same as cost).
    """
    names = [str(n) for n in backend.network.cell_names]
    ci = names.index(str(cell_name))
    gs = p["a_gt"]
    scale = float(gs[ci] if torch.is_tensor(gs) and gs.dim() > 0 else gs)
    if bias_gt is not None:
        from param_defaults import PARAM_BOXES
        box = PARAM_BOXES["bias_gt"]
        bias = float(bias_gt)
        if np.isfinite(bias):
            bias = float(np.clip(bias, float(box["lo"]), float(box["hi"])))
    else:
        gb = p["bias_gt"]
        bias = float(gb[ci] if torch.is_tensor(gb) and gb.dim() > 0 else gb)
        if "v_th" in p:
            vt = p["v_th"]
            bias = bias + float(vt[ci] if torch.is_tensor(vt) and vt.dim() > 0 else vt)
    return scale, bias


def bias_gt_from_v_onset_enabled(session) -> bool:
    from param_defaults import BIAS_GT_FROM_V_ONSET
    opts = session.train_opts or {}
    return bool(opts.get("bias_gt_from_v_onset", BIAS_GT_FROM_V_ONSET))


def filter_plot_token(filter=None) -> str:
    """Readout token for plot stems / labels: ``none`` → ``v``, ``ca`` → ``ca``."""
    if filter is None or str(filter) == "none":
        return "v"
    return str(filter)


def session_filter_plot_token(session) -> str:
    opts = (session.train_opts if session is not None else None) or {}
    return filter_plot_token(opts.get("filter"))


def mean_v_onset_by_cell_name(traces, type_idx, cell_names, names, t_onset):
    """Per-type mean of ``traces[row, t_onset]`` for names in ``names``."""
    if t_onset is None:
        raise ValueError("t_onset required for bias_gt_from_v_onset plot affine")
    t0 = int(t_onset)
    out = {}
    for name in names:
        ci = cell_names.index(name) if name in cell_names else None
        if ci is None:
            out[str(name)] = float("nan")
            continue
        m = np.asarray(type_idx) == int(ci)
        if not np.any(m):
            out[str(name)] = float("nan")
            continue
        out[str(name)] = float(np.nanmean(np.asarray(traces)[m, t0]))
    return out


def cost_ylim(*curves, pct=99.0, pad=1.1, floor=1.0, log=False):
    """Ylim from high percentile so cost spikes do not dominate.

    ``log=True`` returns a positive lower bound (for ``yscale('log')``).
    """
    chunks = []
    for c in curves:
        if c is None:
            continue
        v = np.asarray(c, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            chunks.append(v)
    if not chunks:
        return (floor, floor * 10.0) if log else (0.0, floor)
    all_v = np.concatenate(chunks)
    hi = float(np.percentile(all_v, pct))
    yhi = max(hi * pad, floor)
    if not log:
        return 0.0, yhi
    pos = all_v[all_v > 0]
    if not pos.size:
        return floor, max(yhi, floor * 10.0)
    ylo = max(float(pos.min()) / pad, floor)
    if ylo >= yhi:
        yhi = ylo * 10.0
    return ylo, yhi


def _apply_cost_yscale(ax, *curves):
    """Log y-scale with shared-style ylim from *curves*."""
    ax.set_yscale('log')
    ax.set_ylim(*cost_ylim(*curves, log=True))


def annotate_v_th(ax, v_th, *, e_leak=None):
    """Text annotation ``e_leak=…`` / ``v_th=…`` on time panels (no horizontal line)."""
    lines = []
    if v_th is not None and np.isfinite(v_th):
        lines.append(f"v_th={float(v_th):.1f}")
    if e_leak is not None and np.isfinite(e_leak):
        lines.append(f"e_leak={float(e_leak):.1f}")
    if not lines:
        return
    ax.text(
        0.98, 0.02,
        '\n'.join(lines),
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=6, color="k",
        clip_on=False,
    )


def params_for_types(nodes_by_name, param_by_name=None):
    """``{cell_name: value}`` per type from *param_by_name*."""
    param_by_name = param_by_name or {}
    out = {}
    for name in nodes_by_name:
        val = param_by_name.get(name, np.nan)
        out[name] = float(val) if val is not None and np.isfinite(val) else np.nan
    return out


def mark_spot(ax, t_onset, t_spot_end):
    """White band for stimulus-on samples ``[t_onset, t_spot_end]`` (axes face is gray)."""
    if t_onset is None or t_spot_end is None:
        return
    t0 = int(t_onset)
    t1 = int(t_spot_end)
    if t1 < t0:
        return
    # axvspan end is exclusive in continuous x; +1 covers inclusive last sample.
    ax.axvspan(t0, t1 + 1, facecolor='white', edgecolor='none', zorder=0)


def suppress_cost_std(session, task=None):
    """True when cost uses a single hex (no hex-mean STD band)."""
    pack = session.primary_readout if task is None else session.pack_for(task)
    return pack.cost_extent == 0


def readout_center_mask(pack, backend):
    """Boolean mask over pack cost entries included in the cost extent."""
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


def std_from_traces(traces, single_hex=False):
    """Per-time STD across cost entries; zero when single-hex cost or one entry."""
    if single_hex or traces.shape[0] == 1:
        return np.zeros(traces.shape[1], dtype=np.float64)
    return traces.std(axis=0)


def v_th_by_type_name(z, session):
    """Per-cell ``v_th`` (absolute mV)."""
    return _param_by_type_name(z, session, 'v_th')


def e_leak_by_type_name(z, session):
    """Per-cell leak reversal mV from ``e_leak``."""
    return _param_by_type_name(z, session, 'e_leak')


def _param_by_type_name(z, session, name):
    schema = list(session.schema)
    if name not in {s.get('name') for s in schema}:
        return {}
    arr = np.asarray(training.z_to_node_values(z, schema)[name], dtype=np.float64).reshape(-1)
    cell_names = training.cell_node_names(session.backend)
    if arr.shape[0] != len(cell_names):
        raise ValueError(f"{name} length {arr.shape[0]} != n_cells {len(cell_names)}")
    return {str(n): float(arr[i]) for i, n in enumerate(cell_names)}


def cell_ylabel(label, ca_n=None, n=None):
    """Row / cell axis label with ``n`` from *ca_n* or explicit *n*."""
    if n is None and ca_n is not None:
        for (t, _s), count in ca_n.items():
            if t == label:
                n = count
                break
    if n is None:
        return label
    return f'{label} (n={int(n)})'


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
    """Moving-bar panel base title (cell / stimulus label; no ``e_leak``)."""
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
    """Subtitle fragment for plot hex slice."""

    def token(val):
        v = float(val)
        if np.isclose(v, round(v)):
            return str(int(round(v)))
        return str(v).replace('.', 'p').replace('-', 'm')

    parts = []
    if at_x is not None:
        xs = at_x if isinstance(at_x, (list, tuple)) else [at_x]
        parts.append('x=' + ','.join(token(v) for v in xs))
    if at_y is not None:
        ys = at_y if isinstance(at_y, (list, tuple)) else [at_y]
        parts.append('y=' + ','.join(token(v) for v in ys))
    return ', '.join(parts)


def slice_coord_specs(at_xs, at_ys):
    """Expand optional x/y lists to ``[(label, at_x, at_y), ...]`` (missing axis is None)."""

    def axis_label(val):
        fv = float(val)
        if np.isclose(fv, round(fv)):
            return str(int(round(fv)))
        return str(fv)

    if at_xs is not None and at_ys is not None:
        return [
            (f'({axis_label(xv)},{axis_label(yv)})', xv, yv)
            for xv in at_xs for yv in at_ys
        ]
    if at_xs is not None:
        return [(axis_label(xv), xv, None) for xv in at_xs]
    if at_ys is not None:
        return [(axis_label(yv), None, yv) for yv in at_ys]
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


def overlay_v_readout_reds(n_slices):
    """Red shades for per-slice traces plus a darker ``total`` trace."""
    n = n_slices + 1
    return [plt.cm.Reds(v) for v in np.linspace(0.35, 0.95, n)]


def plot_std_band(ax, t, v_readout, std, *, color=None, alpha=None, label=r'$\pm$STD'):
    """Shaded ±STD for continuous line traces."""
    if std is None or not np.any(std):
        return
    t_arr = np.asarray(t)
    m_arr = np.asarray(v_readout, dtype=np.float64)
    s_arr = np.asarray(std, dtype=np.float64)
    mask = np.isfinite(m_arr) & np.isfinite(s_arr)
    if not np.any(mask):
        return
    ax.fill_between(
        t_arr[mask],
        m_arr[mask] - s_arr[mask],
        m_arr[mask] + s_arr[mask],
        color=STD_COLOR if color is None else color,
        alpha=0.3 if alpha is None else alpha,
        linewidth=0,
        label=label,
        zorder=1,
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
    color=V_READOUT_COLOR,
    linestyle='-',
    linewidth=TRACE_LW,
    label=None,
    draw_pre=False,
):
    """Plot a 1-D series with optional dashed pre-``pre_end`` segment.

    ``pre_end`` is the first post-onset index (samples ``[0, pre_end)`` are pre).
    Gray gt uses ``draw_pre=False`` (never draws pre). v_readout uses
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
        # Spot gt omits ms_post; v_readout / axis may be longer.
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
    show_std=True,
    title=None,
    title_fs=7,
    v_th=None,
    e_leak=None,
    show_ylabel=False,
    ylabel='mV',
    ticksize=6,
    style_xaxis=None,
    pre_end=0,
    show_pre=False,
    t_onset=None,
    t_spot_end=None,
):
    """v_readout (red) vs gt (gray) time courses for one or more contrast traces.

    ``traces``: sequence of dicts with keys ``v_readout``, ``gt``, optional
    ``std``, ``linestyle`` (default ``'-'``), ``point_ix``.
    When ``point_ix`` is set, gray gt is drawn as open dots at those indices
    (still never draws ``[0, pre_end)`` via line); otherwise gt is a solid
    post-onset line. Red v_readout always uses continuous pre/post lines: dashed
    pre when ``show_pre`` is true, solid after.
    ``t_onset`` / ``t_spot_end``: white stimulus-on band ``[t_onset, t_spot_end]``.
    Y-limits / ticks: matplotlib autoscale.
    """
    traces = list(traces or ())
    mark_spot(ax, t_onset, t_spot_end)
    split = max(0, int(pre_end or 0))
    for tr in traces:
        v_readout = tr.get("v_readout")
        gt = tr.get("gt")
        std = tr.get("std")
        linestyle = tr.get("linestyle", "-")
        point_ix = tr.get("point_ix")
        if point_ix is not None:
            x_gt, y_gt = _series_points(t, gt, point_ix=point_ix)
            if x_gt is not None:
                ax.plot(
                    x_gt, y_gt, linestyle='none', marker='o', markersize=4,
                    fillstyle='none', markeredgewidth=1.0, color=GT_COLOR,
                )
        elif gt is not None:
            plot_pre_post_line(
                ax, t, gt, pre_end=split, show_pre=False, draw_pre=False,
                color=GT_COLOR, linestyle=linestyle, linewidth=TRACE_LW,
            )
        if v_readout is not None:
            if show_std and std is not None:
                t_arr = np.asarray(t)
                m_arr = np.asarray(v_readout, dtype=np.float64)
                s_arr = np.asarray(std, dtype=np.float64)
                if split < m_arr.shape[0]:
                    plot_std_band(ax, t_arr[split:], m_arr[split:], s_arr[split:])
            plot_pre_post_line(
                ax, t, v_readout, pre_end=split, show_pre=show_pre, draw_pre=True,
                color=V_READOUT_COLOR, linestyle=linestyle, linewidth=TRACE_LW,
            )
    if title is not None:
        ax.set_title(title, fontsize=title_fs, pad=2)
    if style_xaxis is not None:
        style_xaxis(ax)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=ticksize)
    annotate_v_th(ax, v_th, e_leak=e_leak)


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

        # Spot-on / axvspan bands (data-x, axes-y) → shapes under traces.
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

        # STD fill_between etc. before line traces (same stacking as PNG).
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


def save_figure(fig, path, dpi=150, rasterize=False, *, timer=None):
    """Save figure: ``.png`` (default) or interactive ``.html`` (plotly hover x/y).

    Always prints prep/draw/save via :class:`PlotTimer`. Pass *timer* when the
    caller already marked prep/draw; otherwise only *save* is timed.
    """
    path = os.fspath(path)
    if timer is None:
        timer = PlotTimer()
        timer.end_prep()
        timer.end_draw()
    if path.lower().endswith('.html'):
        _write_interactive_html(fig, path)
    else:
        if rasterize:
            for ax in fig.axes:
                ax.set_rasterized(True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    timer.log(path)


def plot_cost(costs, path, *, costs_by_part=None, part_order=None):
    """Plot training cost; total + grouped subplots with role colors.

    Rules (applies to spot / moving_bar / future tasks):
    - All curves are solid (``linestyle='-'``).
    - Colors are assigned by *role order* (R0, R1, R2, ... for spot radii; PD/ND/DSI for moving bar).
    - Cell layout follows canonical order rows: first block log + shared ylim, then
      linear total + linear per-panel ylim (original).
    """
    from network.construction import CELL_ORDER_ROWS, cell_order_rows

    timer = PlotTimer()
    timer.end_prep()
    if costs is None or not hasattr(costs, "__len__") or len(costs) == 0:
        raise ValueError("plot_cost requires non-empty `costs` array")

    def _save_total_only():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(costs, color='steelblue', linewidth=2, linestyle='-')
        _apply_cost_yscale(ax, costs)
        ax.set_xlabel('step')
        ax.set_ylabel('cost [% gt power]')
        ax.set_title(f'Training cost ({len(costs)} steps)')
        ax.grid(True, alpha=0.3, which='both')
        fig.tight_layout()
        timer.end_draw()
        save_figure(fig, path, dpi=150, timer=timer)

    if not costs_by_part:
        _save_total_only()
        return

    part_keys = list(part_order) if part_order else list(costs_by_part.keys())
    part_keys = [k for k in part_keys if k in costs_by_part and len(costs_by_part[k])]

    if not part_keys:
        _save_total_only()
        return

    def _normalize_radius(r_f: float):
        r_i = int(round(r_f))
        if abs(r_f - r_i) < 1e-6:
            return r_i
        return float(r_f)

    def _spot_parse(key: str):
        if key.startswith("spot_bright_"):
            contrast = "bright"
            task = "spot_bright"
        elif key.startswith("spot_dark_"):
            contrast = "dark"
            task = "spot_dark"
        else:
            return None
        pos = key.rfind("_r")
        if pos < 0:
            return None
        r_s = key[pos + 2:]
        try:
            r_f = float(r_s)
        except ValueError:
            return None
        r = _normalize_radius(r_f)
        # prefix is "{task}_{cell}"
        cell = key[len(task) + 1:pos]
        return cell, contrast, r

    def _moving_bar_parse(key: str):
        if key.startswith("moving_bar_bright_"):
            contrast = "bright"
            task = "moving_bar_bright"
        elif key.startswith("moving_bar_dark_"):
            contrast = "dark"
            task = "moving_bar_dark"
        else:
            return None
        if key == f"{task}_DSI":
            return None, contrast, "DSI"
        for role in ("PD", "ND"):
            suf = f"_{role}"
            if key.endswith(suf) and key != f"{task}_{role}":
                cell = key[len(task) + 1: -len(suf)]
                return cell, contrast, role
        return None

    # role universe (for color indexing)
    spot_radii: set = set()
    moving_roles: set = set()
    other_role_ids_order: list = []
    other_role_ids_seen: set = set()

    curve_specs_by_cell: dict[str, list] = {}
    curve_specs_global: list = []

    for key in part_keys:
        curve = np.asarray(costs_by_part[key], dtype=np.float64)
        if curve.size == 0:
            continue

        parsed_spot = _spot_parse(key)
        if parsed_spot is not None:
            cell, contrast, r = parsed_spot
            spot_radii.add(r)
            role_id = ("spot_r", r)
            label = f"R{r} ({contrast})" if contrast else f"R{r}"
            curve_specs_by_cell.setdefault(cell, []).append((role_id, label, curve))
            continue

        parsed_moving = _moving_bar_parse(key)
        if parsed_moving is not None:
            cell, contrast, role = parsed_moving
            moving_roles.add(role)
            role_id = ("moving_role", role)
            label = f"{role} ({contrast})" if contrast else role
            if cell is None:
                curve_specs_global.append((role_id, label, curve))
            else:
                curve_specs_by_cell.setdefault(cell, []).append((role_id, label, curve))
            continue

        # unknown / future task parts:
        # - try to place into cell grid if the key contains a known cell name substring
        # - otherwise treat as global (global panels come after cell panels).
        role_id = ("other", key)
        if role_id not in other_role_ids_seen:
            other_role_ids_seen.add(role_id)
            other_role_ids_order.append(role_id)
        known_cells = {n for row in CELL_ORDER_ROWS for n in row}
        matches = [c for c in known_cells if c and c in key]
        if matches:
            cell = max(matches, key=len)
            curve_specs_by_cell.setdefault(cell, []).append((role_id, key, curve))
        else:
            curve_specs_global.append((role_id, key, curve))

    # Build deterministic color mapping:
    # - spot radii first: R0, R1, R2, ...
    # - then moving bar roles: PD, ND, DSI
    # - then other roles in the order they appear in `part_order`.
    palette = list(plt.get_cmap("tab20").colors)
    role_id_order: list = []
    for r in sorted(spot_radii, key=lambda x: (isinstance(x, float), x)):
        role_id_order.append(("spot_r", r))

    moving_role_order = [r for r in ("PD", "ND", "DSI") if r in moving_roles]
    extra_moving = sorted([r for r in moving_roles if r not in moving_role_order])
    moving_role_order.extend(extra_moving)
    for r in moving_role_order:
        role_id_order.append(("moving_role", r))

    role_id_order.extend(other_role_ids_order)

    role_id_to_color = {rid: palette[i % len(palette)] for i, rid in enumerate(role_id_order)}

    # Layout: [total log + parts log] then [total linear + parts linear].
    ncols = NCOLS_GT
    present_cells = set(curve_specs_by_cell.keys())
    order_rows = cell_order_rows(sorted(present_cells))
    n_cell_rows = len(order_rows)

    n_global_axes = len(curve_specs_global)
    n_global_rows = (n_global_axes + ncols - 1) // ncols if n_global_axes else 0
    n_part_rows = n_cell_rows + n_global_rows
    n_block_rows = 1 + n_part_rows
    nrows = 2 * n_block_rows

    fig = plt.figure(figsize=(PANEL_W * ncols, PANEL_H * nrows))
    gs = fig.add_gridspec(
        nrows, ncols,
        hspace=0.55, wspace=0.45,
        top=0.95, bottom=0.06, left=0.07, right=0.98,
    )

    cell_curves = [
        curve
        for specs in curve_specs_by_cell.values()
        for _, _, curve in specs
    ]

    def _sorted_specs(cell):
        specs = curve_specs_by_cell.get(cell) or []
        return sorted(
            specs,
            key=lambda x: role_id_order.index(x[0]) if x[0] in role_id_order else 10**9,
        )

    def _draw_total(row_idx, *, log):
        ax = fig.add_subplot(gs[row_idx, :])
        ax.plot(costs, color='steelblue', linewidth=2, linestyle='-')
        if log:
            _apply_cost_yscale(ax, costs)
        else:
            ax.set_ylim(*cost_ylim(costs))
        ax.set_title("total (weighted)")
        ax.set_ylabel("cost [% gt power]")
        ax.grid(True, alpha=0.3, which='both' if log else 'major')

    def _draw_part_block(row0, *, log, shared_cell_ylim, with_legend):
        legend_done = False
        for gi, row_cells in enumerate(order_rows):
            row_idx = row0 + gi
            start = (ncols - len(row_cells)) // 2
            for j, cell in enumerate(row_cells):
                ax = fig.add_subplot(gs[row_idx, start + j])
                specs = _sorted_specs(cell)
                curves = []
                for role_id, label, curve in specs:
                    curves.append(curve)
                    ax.plot(
                        curve, color=role_id_to_color.get(role_id),
                        linewidth=2, linestyle='-', label=label,
                    )
                if log and shared_cell_ylim and cell_curves:
                    _apply_cost_yscale(ax, *cell_curves)
                elif curves:
                    if log:
                        _apply_cost_yscale(ax, *curves)
                    else:
                        ax.set_ylim(*cost_ylim(*curves))
                if j == 0:
                    ax.set_ylabel("cost [% gt power]", fontsize=8)
                ax.set_title(str(cell), fontsize=8)
                ax.grid(True, alpha=0.3, which='both' if log else 'major')
                if with_legend and (not legend_done) and len(specs) > 1:
                    ax.legend(fontsize=7)
                    legend_done = True
                if gi == n_cell_rows - 1:
                    ax.set_xlabel("step")

        for gi, (role_id, label, curve) in enumerate(curve_specs_global):
            row_idx = row0 + n_cell_rows + gi // ncols
            col = gi % ncols
            ax = fig.add_subplot(gs[row_idx, col])
            ax.plot(curve, color=role_id_to_color.get(role_id), linewidth=2, linestyle='-')
            if log:
                _apply_cost_yscale(ax, curve)
            else:
                ax.set_ylim(*cost_ylim(curve))
            ax.set_title(label, fontsize=8)
            ax.grid(True, alpha=0.3, which='both' if log else 'major')
            if col == 0:
                ax.set_ylabel("cost [% gt power]", fontsize=8)
            if gi // ncols == n_global_rows - 1:
                ax.set_xlabel("step")

    _draw_total(0, log=True)
    _draw_part_block(1, log=True, shared_cell_ylim=True, with_legend=True)
    _draw_total(n_block_rows, log=False)
    _draw_part_block(n_block_rows + 1, log=False, shared_cell_ylim=False, with_legend=False)

    fig.suptitle(f'Training cost ({len(costs)} steps)', fontsize=12, y=1.01)
    fig.tight_layout()
    timer.end_draw()
    save_figure(fig, path, dpi=150, timer=timer)
