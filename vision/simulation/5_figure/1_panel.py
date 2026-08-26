"""Shared figure panel (no task-specific logic)."""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import network.path  # noqa: F401 — FAFB path on sys.path
import train

GT_COLOR = 'gray'
V_READOUT_COLOR = 'red'
STD_COLOR = 'pink'
TRACE_LINE_W = 1.5
N_COL_GT = 5
N_COL_ALL = 8
PANEL_W = 3.0
PANEL_H = 2.2


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def gt_trace_affine(readout, cell, gt_trace):
    """Plot gt as ``a_gt * gt + bias_gt`` from ``readout.gt_affine_by_cell``."""
    if gt_trace is None:
        return None
    a_gt, bias = readout.gt_affine_by_cell.get(str(cell), (1.0, 0.0))
    return float(a_gt) * np.asarray(gt_trace, dtype=float) + float(bias)


def cost_ylim(*costs, percentile=99.0, padding=1.1, floor=1.0, log=False):
    """Ylim from high percentile so cost spikes do not dominate.

    ``log=True`` returns a positive ``limit_low`` (for ``yscale('log')``).
    """
    part_vals = []
    for part_costs in costs:
        if part_costs is None:
            continue
        finite = np.asarray(part_costs, dtype=np.float64).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size:
            part_vals.append(finite)
    if not part_vals:
        return (floor, floor * 10.0) if log else (0.0, floor)
    ylim_high = max(
        float(np.percentile(
            concatenated := np.concatenate(part_vals), percentile,
        )) * padding,
        floor,
    )
    if not log:
        return 0.0, ylim_high
    positive = concatenated[concatenated > 0]
    if not positive.size:
        return floor, max(ylim_high, floor * 10.0)
    ylim_low = max(float(positive.min()) / padding, floor)
    if ylim_low >= ylim_high:
        ylim_high = ylim_low * 10.0
    return ylim_low, ylim_high


def _cost_yscale(ax, *costs):
    ax.set_yscale('log')
    ax.set_ylim(*cost_ylim(*costs, log=True))


def annotate_v_th(ax, v_th, *, e_leak=None):
    """Text annotation ``e_leak=…`` / ``v_th=…`` on time panels (no horizontal line)."""
    labels = []
    if v_th is not None and np.isfinite(v_th):
        labels.append(f"v_th={float(v_th):.1f}")
    if e_leak is not None and np.isfinite(e_leak):
        labels.append(f"e_leak={float(e_leak):.1f}")
    if not labels:
        return
    ax.text(
        0.98, 0.02,
        '\n'.join(labels),
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=6, color="k",
        clip_on=False,
    )


def mark_sti_on(ax, t_onset, t_sti_end):
    """White band for sti-on samples ``[t_onset, t_sti_end]`` (axes face is gray)."""
    if t_onset is None or t_sti_end is None:
        return
    t_onset = int(t_onset)
    t_sti_end = int(t_sti_end)
    if t_sti_end < t_onset:
        return
    # axvspan end is exclusive in continuous x; +1 covers inclusive last sample.
    ax.axvspan(t_onset, t_sti_end + 1, facecolor='white', edgecolor='none', zorder=0)


def is_single_hex_cost(session, task=None, contrast=None):
    """True when cost uses a single hex (no hexes STD band)."""
    return getattr(
        session.primary_pack if task is None and contrast is None
        else session.packs[task][contrast],
        "cost_radius", None,
    ) == 0


def pack_center_mask(pack, connectome):
    """Boolean mask over pack cost entries included in the cost radius."""
    entry_nodes = pack.entry_nodes.cpu().numpy()
    entry_radii = getattr(pack, "entry_radii", None)
    if entry_radii is not None:
        return entry_radii.cpu().numpy().astype(np.int64, copy=False) == 0
    cost_radius = getattr(pack, "cost_radius", None)
    if cost_radius is not None:
        import build_hex
        return build_hex.radius_mask(
            np.asarray(connectome.us)[entry_nodes],
            np.asarray(connectome.vs)[entry_nodes],
            int(cost_radius),
        )
    return np.ones(entry_nodes.shape[0], dtype=bool)


def std_from_traces(traces, single_hex=False):
    """Per-time STD across cost entries; zero when single-hex cost or one entry."""
    if single_hex or traces.shape[0] == 1:
        return np.zeros(traces.shape[1], dtype=np.float64)
    return traces.std(axis=0)


def v_th_from_z(z, session):
    return _param_from_z(z, session, 'v_th')


def e_leak_from_z(z, session):
    return _param_from_z(z, session, 'e_leak')


def _param_from_z(z, session, param):
    schema = train.schema_copy(session.schema)
    if param not in schema:
        return {}
    cell_vals = np.asarray(
        train.node_vals_from_z(z, schema)[param], dtype=np.float64,
    ).reshape(-1)
    cells = train.cells_from_connectome(session.connectome)
    if cell_vals.shape[0] != len(cells):
        raise ValueError(f"{param} length {cell_vals.shape[0]} != n_cell {len(cells)}")
    return {str(cell): float(val) for cell, val in zip(cells, cell_vals)}


def cell_ylabel(label, ca_n=None, n=None):
    """Row / cell axis label with ``n`` from *ca_n* or explicit *n*."""
    if n is None and ca_n is not None:
        for (cell_key, _), n_val in ca_n.items():
            if cell_key == label:
                n = n_val
                break
    if n is None:
        return label
    return f'{label} (n={int(n)})'


def network_hex_count(connectome):
    """Unique axial hexes on the connectome."""
    return len({(int(hex_u), int(hex_v)) for hex_u, hex_v in zip(connectome.us, connectome.vs)})


def readout_prep_s(*readouts):
    """Sum and clear ``prep_s`` on trace readouts (``None`` skipped).

    First figure that calls this receives the forward cost; later shared
    figures see ``0``.
    """
    prep_sum_s = 0.0
    for readout in readouts:
        if readout is None:
            continue
        prep_sum_s += float(getattr(readout, 'prep_s', 0.0) or 0.0)
        readout.prep_s = 0.0
    return prep_sum_s


class ElapsedTimer:
    """Shared prep / plot / save timing for trace figures.

    ``prior_prep`` is forward time already stored on ``TraceReadout.prep_s``.
    """

    __slots__ = ('start_time', '_prep_time', '_plot_time')

    def __init__(self, prior_prep=0.0):
        self.start_time = time.perf_counter() - float(prior_prep)
        self._prep_time = None
        self._plot_time = None

    def end_prep(self):
        self._prep_time = time.perf_counter()

    def end_plot(self):
        self._plot_time = time.perf_counter()

    def log(self, path):
        prep_time = self._prep_time if self._prep_time is not None else time.perf_counter()
        plot_time = self._plot_time if self._plot_time is not None else time.perf_counter()
        now = time.perf_counter()
        print(
            f'plot {path}: '
            + '  '.join(
                f'{name}={float(val):.1f}s'
                for name, val in {
                    'prep': prep_time - self.start_time,
                    'plot_s': plot_time - prep_time,
                    'save': now - plot_time,
                    'elapsed_s': now - self.start_time,
                }.items()
            )
        )


def ms_shown_axis_xlim(ms_shown, *, delta_ms, origin_t=0):
    """Inclusive t-index xlim from ``--ms-shown``; ``origin_t`` is onset t on the axis."""
    if ms_shown is None:
        return None
    limit_low = int(origin_t) + int(train.t_from_ms(float(ms_shown[0]), delta_ms=float(delta_ms)))
    limit_high = int(origin_t) + int(train.t_from_ms(float(ms_shown[1]), delta_ms=float(delta_ms)))
    if limit_low > limit_high:
        raise ValueError(f'ms-shown xlim START t={limit_low} > STOP t={limit_high}')
    return limit_low, limit_high


def at_xy_label(at_x, at_y):
    """Subtitle fragment for ``at_x`` / ``at_y`` hex-step filter."""
    parts = []
    if at_x is not None:
        parts.append('x=' + ','.join(
            label for label, _, _ in expand_at_xy(
                at_x if isinstance(at_x, (list, tuple)) else [at_x], None,
            )[0]
        ))
    if at_y is not None:
        parts.append('y=' + ','.join(
            label for label, _, _ in expand_at_xy(
                None, at_y if isinstance(at_y, (list, tuple)) else [at_y],
            )[0]
        ))
    return ', '.join(parts)


def expand_at_xy(at_xs, at_ys):
    """Expand optional x/y lists to ``[(label, at_x, at_y), ...]`` and ``at_xy_mode``."""
    if at_xs is not None and at_ys is not None:
        return [
            (
                f'({str(int(xn)) if (xn := float(at_x)).is_integer() else f"{xn:g}"},'
                f'{str(int(yn)) if (yn := float(at_y)).is_integer() else f"{yn:g}"})',
                at_x,
                at_y,
            )
            for at_x in at_xs
            for at_y in at_ys
        ], 'xy'
    if at_xs is not None:
        return [
            (
                str(int(number)) if (number := float(at_x)).is_integer() else f'{number:g}',
                at_x,
                None,
            )
            for at_x in at_xs
        ], 'x'
    if at_ys is not None:
        return [
            (
                str(int(number)) if (number := float(at_y)).is_integer() else f'{number:g}',
                None,
                at_y,
            )
            for at_y in at_ys
        ], 'y'
    return [], None


def at_xy_reds(n_label):
    """Red shades for per at_xy traces plus a darker primary mean v_readout trace."""
    return [plt.cm.Reds(v) for v in np.linspace(0.35, 0.95, n_label + 1)]


def plot_std_band(ax, t, v_readout, std, *, color=None, alpha=None, label=r'$\pm$STD'):
    """Shaded ±STD for continuous line traces."""
    if std is None or not np.any(std):
        return
    t = np.asarray(t)
    v_readout = np.asarray(v_readout, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    mask = np.isfinite(v_readout) & np.isfinite(std)
    if not np.any(mask):
        return
    ax.fill_between(
        t[mask],
        v_readout[mask] - std[mask],
        v_readout[mask] + std[mask],
        color=STD_COLOR if color is None else color,
        alpha=0.3 if alpha is None else alpha,
        linewidth=0,
        label=label,
        zorder=1,
    )


def _trace_points(t, y, ts=None):
    """Return finite ``(x, y)`` points, optionally subsampled by integer ``ts``."""
    if y is None:
        return None, None
    t = np.asarray(t)
    y = np.asarray(y, dtype=np.float64)
    if ts is not None:
        ts = np.asarray(ts, dtype=np.int64)
        ts = ts[(ts >= 0) & (ts < y.shape[0])]
        t = t[ts]
        y = y[ts]
    mask = np.isfinite(y)
    if not np.any(mask):
        return None, None
    return t[mask], y[mask]


def traces_with_cost_ts(traces, readouts, *, entry_radius=None):
    """Copy ``traces`` with sparse cost ``ts`` per contrast readout."""
    return [
        {
            **trace,
            "ts": (
                None if readouts[trace["contrast"]].session is None else train.pack_cost_abs_ts(
                    readouts[trace["contrast"]].session.primary_pack,
                    readouts[trace["contrast"]].t_onset,
                    entry_radius=entry_radius,
                )
            ),
        }
        for trace in traces
    ]


def plot_trace(
    ax,
    t,
    trace,
    *,
    t_onset=0,
    color=V_READOUT_COLOR,
    linestyle='-',
    linewidth=TRACE_LINE_W,
    label=None,
):
    """Plot a 1-D trace: dashed before ``t_onset``, solid after."""
    if trace is None:
        return
    t = np.asarray(t)
    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim != 1:
        raise ValueError(
            f'plot_trace expects 1-D trace, got trace={getattr(trace, "shape", None)}'
        )
    if t.shape[0] > trace.shape[0]:
        t = t[: trace.shape[0]]
    elif t.shape[0] < trace.shape[0]:
        raise ValueError(
            f'plot_trace expects t length >= trace length, '
            f'got t={getattr(t, "shape", None)} '
            f'trace={getattr(trace, "shape", None)}'
        )
    n = int(trace.shape[0])
    t_onset = max(0, min(int(t_onset or 0), n))
    if t_onset > 0:
        # Include the onset sample so dashed and solid traces meet.
        end_pre = min(t_onset + 1, n)
        ax.plot(
            t[:end_pre], trace[:end_pre],
            color=color, linewidth=linewidth, linestyle='--',
        )
    if t_onset >= n:
        return
    ax.plot(
        t[t_onset:], trace[t_onset:],
        color=color, linewidth=linewidth, linestyle=linestyle, label=label,
    )


def plot_timecourse(
    ax,
    t,
    traces,
    *,
    show_std=True,
    title=None,
    title_fontsize=7,
    v_th=None,
    e_leak=None,
    show_ylabel=False,
    ylabel='mV',
    ticksize=6,
    style_xaxis=None,
    t_onset=0,
    gt_from_t=None,
):
    """v_readout (red) vs gt (gray) time courses for one or more contrast traces.

    ``traces``: sequence of dicts with keys ``v_readout_mean_cell`` /
    ``v_readout_mean_cell_mean_radius`` / ``ca_mean_cell``, ``gt``, optional
    ``std``, ``linestyle`` (default ``'-'``), ``ts``.
    When ``ts`` is set, gray gt is drawn as open dots at those samples
    (still never draws ``[0, gt_from_t)`` via line); otherwise gt is a solid
    line from ``gt_from_t``. Red v_readout is dashed before ``t_onset`` and
    solid after. ``gt_from_t`` defaults to ``t_onset``; pass ``0`` to draw gt
    from the first sample while keeping the v_readout split at ``t_onset``.
    Y-limits / ticks: matplotlib autoscale.
    """
    traces = list(traces or ())
    t_onset = max(0, int(t_onset or 0))
    gt_start = t_onset if gt_from_t is None else max(0, int(gt_from_t))
    for trace in traces:
        v_readout = trace.get("v_readout_mean_cell_mean_radius")
        if v_readout is None:
            v_readout = trace.get("v_readout_mean_cell")
        if v_readout is None:
            v_readout = trace.get("ca_mean_cell")
        gt = trace.get("gt")
        std = trace.get("std")
        linestyle = trace.get("linestyle", "-")
        cost_ts = trace.get("ts")
        if cost_ts is not None:
            gt_time, gt_value = _trace_points(t, gt, ts=cost_ts)
            if gt_time is not None:
                ax.plot(
                    gt_time, gt_value, linestyle='none', marker='o', markersize=4,
                    fillstyle='none', markeredgewidth=1.0, color=GT_COLOR,
                )
        elif gt is not None:
            time_gt = np.asarray(t)
            gt = np.asarray(gt, dtype=np.float64)
            if time_gt.shape[0] > gt.shape[0]:
                time_gt = time_gt[: gt.shape[0]]
            n_gt = int(gt.shape[0])
            gt_start_clamped = max(0, min(gt_start, n_gt))
            if gt_start_clamped < n_gt:
                ax.plot(
                    time_gt[gt_start_clamped:], gt[gt_start_clamped:],
                    color=GT_COLOR, linestyle=linestyle, linewidth=TRACE_LINE_W,
                )
        if v_readout is not None:
            if show_std and std is not None:
                plot_std_band(ax, t, v_readout, std)
            plot_trace(
                ax, t, v_readout, t_onset=t_onset,
                color=V_READOUT_COLOR, linestyle=linestyle, linewidth=TRACE_LINE_W,
            )
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, pad=2)
    if style_xaxis is not None:
        style_xaxis(ax)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=ticksize)
    annotate_v_th(ax, v_th, e_leak=e_leak)


_MATPLOT_LINE_DASH = {
    '-': 'solid',
    '--': 'dash',
    '-.': 'dashdot',
    ':': 'dot',
    'None': 'solid',
    'none': 'solid',
    '': 'solid',
}


def _plotly_color(color, *, alpha=None, fallback=None):
    """Matplotlib artist color → plotly ``#rrggbb`` or ``rgba(...)`` (keeps alpha)."""
    from matplotlib.colors import to_rgba

    if color in (None, 'auto', 'none', 'None'):
        if fallback is None:
            raise ValueError(f'invalid plotly color: {color!r}')
        color = fallback
    red, green, blue, color_alpha = to_rgba(color, alpha=alpha)
    if color_alpha >= 1.0:
        return (
            f'#{int(round(red * 255)):02x}'
            f'{int(round(green * 255)):02x}'
            f'{int(round(blue * 255)):02x}'
        )
    return (
        f'rgba({int(round(red * 255))},{int(round(green * 255))},'
        f'{int(round(blue * 255))},{color_alpha:g})'
    )


def _save_interactive_html(fig, path):
    """Write standalone HTML: hover traces for x/y (plotly), matching PNG style."""
    import plotly.graph_objects as go
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    fig.canvas.draw()
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    traces = []
    shapes = []
    layout = {
        'autosize': False,
        'width': max(400, int(fig.get_figwidth() * 100)),
        'height': max(300, int(fig.get_figheight() * 100)),
        'margin': dict(l=40, r=20, t=60, b=40),
        'hovermode': 'closest',
        'showlegend': False,
        'plot_bgcolor': _plotly_color(
            visible_axes[0].get_facecolor()
            if visible_axes else plt.rcParams['axes.facecolor']
        ),
        'paper_bgcolor': _plotly_color(fig.get_facecolor()),
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
        gridlines = ax.xaxis.get_gridlines()
        axis_style = dict(
            showgrid=True,
            zeroline=False,
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor=_plotly_color(ax.spines['bottom'].get_edgecolor()),
            gridcolor=(
                _plotly_color(gridlines[0].get_color())
                if gridlines else _plotly_color(plt.rcParams['grid.color'])
            ),
        )
        layout[xkey] = dict(
            domain=[float(pos.x0), float(pos.x1)],
            anchor=yaxis,
            title=ax.get_xlabel() or None,
            **axis_style,
        )
        layout[ykey] = dict(
            domain=[float(pos.y0), float(pos.y1)],
            anchor=xaxis,
            title=ax.get_ylabel() or None,
            **axis_style,
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

        # Sti-on / axvspan bands (data-x, axes-y) → shapes under traces.
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
                fillcolor=_plotly_color(patch.get_facecolor()),
                line=dict(width=0),
                layer='below',
            ))

        # STD fill_between etc. before line traces (same stacking as PNG).
        for collection in ax.collections:
            if not isinstance(collection, PolyCollection):
                continue
            face_colors = collection.get_facecolors()
            if face_colors is None or len(face_colors) == 0:
                continue
            for path_index, collection_path in enumerate(collection.get_paths()):
                vertices = np.asarray(collection_path.vertices, dtype=float)
                if vertices.size == 0:
                    continue
                traces.append(go.Scatter(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    mode='lines',
                    fill='toself',
                    fillcolor=_plotly_color(
                        face_colors[min(path_index, len(face_colors) - 1)],
                    ),
                    line=dict(
                        width=0,
                        color=_plotly_color(
                            face_colors[min(path_index, len(face_colors) - 1)],
                        ),
                    ),
                    hoverinfo='skip',
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                ))

        for line_index, line in enumerate(ax.get_lines()):
            line_x = np.asarray(line.get_xdata(), dtype=float)
            line_y = np.asarray(line.get_ydata(), dtype=float)
            if line_x.size == 0:
                continue
            label = line.get_label()
            if not label or str(label).startswith('_'):
                label = ax.get_title() or f'trace{line_index}'
            line_style = line.get_linestyle()
            if isinstance(line_style, (tuple, list)):
                dash = (
                    'dash'
                    if line_style not in (None, 'None', 'none', '-', 'solid')
                    else 'solid'
                )
            else:
                dash = _MATPLOT_LINE_DASH.get(str(line_style), 'solid')
            mode = 'lines'
            if line.get_marker() not in (None, 'None', 'none', ''):
                mode = 'lines+markers' if dash != 'solid' or line.get_linewidth() else 'markers'
                if line.get_linestyle() in ('None', 'none', ''):
                    mode = 'markers'
            line_alpha = line.get_alpha()
            traces.append(go.Scatter(
                x=line_x,
                y=line_y,
                mode=mode,
                name=str(label),
                line=dict(
                    color=_plotly_color(line.get_color(), alpha=line_alpha),
                    width=float(line.get_linewidth() or 1.0),
                    dash=dash,
                ),
                marker=dict(
                    size=max(4.0, float(line.get_markersize() or 4.0)),
                    color=(
                        'rgba(0,0,0,0)'
                        if line.get_fillstyle() == 'none'
                        else _plotly_color(
                            line.get_markerfacecolor(),
                            alpha=line_alpha,
                            fallback=line.get_color(),
                        )
                    ),
                    line=dict(
                        color=_plotly_color(
                            line.get_markeredgecolor(),
                            alpha=line_alpha,
                            fallback=line.get_color(),
                        ),
                        width=float(line.get_markeredgewidth() or 0.0),
                    ),
                ),
                xaxis=xaxis,
                yaxis=yaxis,
                hovertemplate='x=%{x}<br>y=%{y}<extra>%{fullData.name}</extra>',
            ))

    if shapes:
        layout['shapes'] = shapes
    go.Figure(data=traces, layout=layout).write_html(
        path, include_plotlyjs=True, full_html=True, config={
            'displayModeBar': True,
            'scrollZoom': True,
        },
    )


def save_figure(fig, path, dpi=150, rasterize=False, *, timer=None):
    """Save figure: ``.png`` (default) or interactive ``.html`` (plotly hover x/y).

    Always prints prep/plot/save via :class:`ElapsedTimer`. Pass *timer* when the
    caller already marked prep/plot; otherwise only *save* is timed.
    """
    path = os.fspath(path)
    if timer is None:
        timer = ElapsedTimer()
        timer.end_prep()
        timer.end_plot()
    if path.lower().endswith('.html'):
        _save_interactive_html(fig, path)
    else:
        if rasterize:
            for ax in fig.axes:
                ax.set_rasterized(True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    timer.log(path)


def plot_cost_figure(
    costs,
    path,
    *,
    costs_by_cell,
    costs_global,
    series_order,
    colors_by_series,
    rows,
    timer=None,
):
    """Shared train-cost grid: total + per-cell parts (log then linear blocks)."""
    timer = timer or ElapsedTimer()
    if timer._prep_time is None:
        timer.end_prep()
    n_col = N_COL_GT
    n_global_axis = len(costs_global)
    n_global_row = (n_global_axis + n_col - 1) // n_col if n_global_axis else 0
    n_cell_row = len(rows)
    n_block_row = 1 + n_cell_row + n_global_row

    fig = plt.figure(figsize=(PANEL_W * n_col, PANEL_H * 2 * n_block_row))
    grid_spec = fig.add_gridspec(
        2 * n_block_row, n_col,
        hspace=0.55, wspace=0.45,
        top=0.95, bottom=0.06, left=0.07, right=0.98,
    )

    cell_costs = [
        part_costs
        for cell_costs_list in costs_by_cell.values()
        for _, _, part_costs in cell_costs_list
    ]

    def _sorted_costs(cell):
        cell_costs_list = costs_by_cell.get(cell) or []
        return sorted(
            cell_costs_list,
            key=lambda series_label_part_costs: (
                series_order.index(series_label_part_costs[0])
                if series_label_part_costs[0] in series_order else 10**9
            ),
        )

    def _plot_cost_total_row(row, *, log):
        ax = fig.add_subplot(grid_spec[row, :])
        ax.plot(costs, color='steelblue', linewidth=2, linestyle='-')
        if log:
            _cost_yscale(ax, costs)
        else:
            ax.set_ylim(*cost_ylim(costs))
        ax.set_title("weighted mean (sum of parts)")
        ax.set_ylabel("cost [% gt power]")
        ax.grid(True, alpha=0.3, which='both' if log else 'major')

    def _plot_cost_part_block(row0, *, log, shared_cell_ylim, with_legend):
        legend_done = False
        for row_group, row_cells in enumerate(rows):
            row = row0 + row_group
            start = (n_col - len(row_cells)) // 2
            for col, cell in enumerate(row_cells):
                ax = fig.add_subplot(grid_spec[row, start + col])
                cell_costs_list = _sorted_costs(cell)
                panel_costs = []
                for series, label, part_costs in cell_costs_list:
                    panel_costs.append(part_costs)
                    ax.plot(
                        part_costs, color=colors_by_series.get(series),
                        linewidth=2, linestyle='-', label=label,
                    )
                if log and shared_cell_ylim and cell_costs:
                    _cost_yscale(ax, *cell_costs)
                elif panel_costs:
                    if log:
                        _cost_yscale(ax, *panel_costs)
                    else:
                        ax.set_ylim(*cost_ylim(*panel_costs))
                if col == 0:
                    ax.set_ylabel("weighted cost contribution", fontsize=8)
                ax.set_title(str(cell), fontsize=8)
                ax.grid(True, alpha=0.3, which='both' if log else 'major')
                if with_legend and (not legend_done) and len(cell_costs_list) > 1:
                    ax.legend(fontsize=7)
                    legend_done = True
                if row_group == n_cell_row - 1:
                    ax.set_xlabel("step")

        for row_group, (series, label, part_costs) in enumerate(costs_global):
            row = row0 + n_cell_row + row_group // n_col
            col = row_group % n_col
            ax = fig.add_subplot(grid_spec[row, col])
            ax.plot(part_costs, color=colors_by_series.get(series), linewidth=2, linestyle='-')
            if log:
                _cost_yscale(ax, part_costs)
            else:
                ax.set_ylim(*cost_ylim(part_costs))
            ax.set_title(label, fontsize=8)
            ax.grid(True, alpha=0.3, which='both' if log else 'major')
            if col == 0:
                ax.set_ylabel("weighted cost contribution", fontsize=8)
            if row_group // n_col == n_global_row - 1:
                ax.set_xlabel("step")

    _plot_cost_total_row(0, log=True)
    _plot_cost_part_block(1, log=True, shared_cell_ylim=True, with_legend=True)
    _plot_cost_total_row(n_block_row, log=False)
    _plot_cost_part_block(n_block_row + 1, log=False, shared_cell_ylim=False, with_legend=False)

    fig.suptitle(f'Train cost ({len(costs)} steps)', fontsize=12, y=1.01)
    fig.tight_layout()
    timer.end_plot()
    save_figure(fig, path, dpi=150, timer=timer)


def plot_cost_total(costs, path, *, timer=None):
    """Plot train cost total only."""
    timer = timer or ElapsedTimer()
    timer.end_prep()
    if costs is None or not hasattr(costs, "__len__") or len(costs) == 0:
        raise ValueError("plot_cost_total requires non-empty `costs`")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(costs, color='steelblue', linewidth=2, linestyle='-')
    _cost_yscale(ax, costs)
    ax.set_xlabel('step')
    ax.set_ylabel('cost [% gt power]')
    ax.set_title(f'Train cost ({len(costs)} steps)')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    timer.end_plot()
    save_figure(fig, path, dpi=150, timer=timer)
