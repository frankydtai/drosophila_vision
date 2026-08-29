"""Static-bar plotting (network static-bar task).

Each panel: one cell at one bar-position on the motion axis.
Columns are bar ``mid`` positions from the cost pack; rows are cells
(``plot_gt``: sbar ``gt_cells``; ``plot_all``: config ``sbar_figure_cells``).

GT traces are Gruntman Fig.2 width-1 flash responses (T4/T5) plus Mi1/Mi4
position contributions from ``task.sbar.gt``, using the same timing as the pack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

import train
from config import FIGURE_PLOT
from task.sbar.gt import GT_CELLS, gt_trace_key, load_gt as load_sbar_gt
from task.sbar.pack import SbarPack
from task.sbar.sti_geo import node_us_vs
from figure.spread import (
    _session_task_timing,
    contrast_linestyle,
    contrast_order,
    plot_cell_time,
)
from network.construction import (
    active_gt_cells,
    cells_in_order,
    gt_cells_from_opts,
)
from figure.plot import session_from_task
from figure.panel import (
    ElapsedTimer,
    as_numpy,
    cell_ylabel,
    e_leak_from_z,
    gt_trace_affine,
    readout_prep_s,
    save_figure,
    sd_from_traces,
    traces_with_cost_ts,
    v_th_from_z,
)
from task.spread.sti_spec import t_sti_end


SBAR_DPI = 100


def sbar_figure_cells() -> tuple[str, ...]:
    """``sbar_figure_cells`` from config (rows for ``plot_all``)."""
    cells = FIGURE_PLOT.get("sbar_figure_cells")
    if not cells:
        raise ValueError("config sbar_figure_cells must be a non-empty list")
    return tuple(str(cell) for cell in cells)


def active_sbar_gt_cells(session, task=None, contrast=None):
    """Configured sbar gt cells (sti opts), not cost-pack-only."""
    if task is None and contrast is None:
        pack = session.primary_pack
    elif task is None or contrast is None:
        raise ValueError("task and contrast must be passed together")
    else:
        pack = session.packs[task][contrast]
    opts = dict((session.train_opts or {}).get(f"{pack.task}_sti_opts") or {})
    connectome = session.connectome
    return tuple(
        active_gt_cells(
            gt_cells_from_opts(opts),
            GT_CELLS,
            connectome.cells,
            context="sbar plot",
        )
    )


@dataclass
class SbarTraceReadout:
    """One sbar forward pass: cell × mid position traces."""

    task: str
    contrast: str
    cells: list
    mids: list
    n_t: int
    t_onset: int | None = None
    t_sti_end: int | None = None
    ms_shown: tuple[float, float] | None = None
    v_readout: dict = field(default_factory=dict)
    sd: dict = field(default_factory=dict)
    n_by_cell_mid: dict = field(default_factory=dict)
    v_th_by_cell: dict = field(default_factory=dict)
    e_leak_by_cell: dict = field(default_factory=dict)
    gt_affine_by_cell: dict = field(default_factory=dict)
    a_sti_mid: dict[str, float] = field(default_factory=dict)
    a_sti_mid_sigma: float | None = None
    session: object = None
    prep_s: float = 0.0


def _sbar_mids_from_pack(pack: SbarPack) -> list:
    """Sorted unique ``mid`` values from ``pack.entry_part_keys``."""
    mids = set()
    for key in pack.entry_part_keys:
        mid_str = key.rsplit("_mid", 1)[1]
        mids.add(float(mid_str))
    return sorted(mids)


def resolve_sbar_gts(sessions, gts=None):
    """``{contrast: {(cell, mid): gt}}`` — abs-time gt (length ``n_t``) for cost ``ts``."""
    if gts is not None:
        return gts
    if not sessions:
        return {}
    gts = {}
    for contrast, session in sessions.items():
        pack = session.primary_pack
        t_onset, n_t, n_t_gt, ms_sti, delta_ms = _session_task_timing(session)
        opts = dict((session.train_opts or {}).get(f"{pack.task}_sti_opts") or {})
        raw_gts = load_sbar_gt(
            t_onset=t_onset,
            ms_response=float(opts["ms_response"]),
            ms_sti=ms_sti,
            delta_ms=delta_ms,
            ms_post=float(opts.get("ms_post", 0.0)),
        )
        by_cell_mid = {}
        for cell in cells_in_order(GT_CELLS):
            if cell not in session.connectome.cells:
                continue
            for mid in _sbar_mids_from_pack(pack):
                key = gt_trace_key(cell, contrast, mid)
                if key not in raw_gts:
                    continue
                gt = np.full(n_t, np.nan, dtype=np.float64)
                gt[t_onset:t_onset + n_t_gt] = np.asarray(
                    raw_gts[key][t_onset:t_onset + n_t_gt], dtype=np.float64,
                )
                by_cell_mid[(cell, mid)] = gt
        gts[str(contrast)] = by_cell_mid
    return gts


@torch.no_grad()
def network_sbar_trace_readout(session, z, task, contrast, *, ms_shown=None):
    """Run one forward; return :class:`SbarTraceReadout`."""
    t_prep0 = time.perf_counter()
    pack: SbarPack = session.packs[task][contrast]
    params = train.params_from_z(z, session)
    a_sti_mid = {}
    a_sti_mid_sigma = None
    if "a_sti_mid" in params:
        spec = session.schema.get("a_sti_mid")
        if spec is not None:
            a_sti_mid_sigma = float(as_numpy(params["a_sti_mid"]).reshape(-1)[0])
            a_sti_mid = {
                str(mid): float(np.exp(-0.5 * (float(mid) / a_sti_mid_sigma) ** 2))
                for mid in (spec.get("mids") or ())
            }
    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    trace = train.forward_pack(session, params, i_sti, pack)
    trace = as_numpy(trace)

    connectome = session.connectome
    n_t = int(i_sti.shape[1])
    entry_bs = as_numpy(pack.entry_bs)
    entry_nodes = as_numpy(pack.entry_nodes)
    entry_part_keys = pack.entry_part_keys
    node_cells = as_numpy(connectome.node_cells[entry_nodes])
    cells = list(connectome.cells)
    all_cells = cells_in_order(connectome.cells)

    mids = _sbar_mids_from_pack(pack)

    v_readout: dict = {}
    sd_out: dict = {}
    n_by_cell_mid: dict = {}

    for cell in all_cells:
        cell_entry_mask = node_cells == cells.index(cell)
        if not np.any(cell_entry_mask):
            continue
        v_readout[cell] = {}
        sd_out[cell] = {}
        n_by_cell_mid[cell] = {}
        for mid in mids:
            mid_entry_mask = cell_entry_mask.copy()
            for entry_idx, key in enumerate(entry_part_keys):
                mid_str = key.rsplit("_mid", 1)[1]
                if float(mid_str) != mid:
                    mid_entry_mask[entry_idx] = False
            if not np.any(mid_entry_mask):
                continue
            entry_traces = trace[
                entry_bs[mid_entry_mask], :, entry_nodes[mid_entry_mask]
            ]
            v_readout[cell][mid] = entry_traces.mean(axis=0)
            sd_out[cell][mid] = sd_from_traces(
                entry_traces, single_hex=(entry_traces.shape[0] == 1),
            )
            n_by_cell_mid[cell][mid] = int(entry_traces.shape[0])

    # Plot-only: sample ``sbar_figure_cells`` at the same (b, hex) as pack entries.
    us, vs = node_us_vs(connectome)
    for cell in sbar_figure_cells():
        if cell in v_readout or cell not in connectome.cells:
            continue
        cell_v: dict = {}
        cell_sd: dict = {}
        cell_n: dict = {}
        for mid in mids:
            mid_traces = []
            seen_b_uv: set[tuple[int, int, int]] = set()
            for entry_idx, key in enumerate(entry_part_keys):
                mid_str = key.rsplit("_mid", 1)[1]
                if float(mid_str) != mid:
                    continue
                entry_node = int(entry_nodes[entry_idx])
                b = int(entry_bs[entry_idx])
                u, v = int(us[entry_node]), int(vs[entry_node])
                b_uv = (b, u, v)
                if b_uv in seen_b_uv:
                    continue
                seen_b_uv.add(b_uv)
                for cell_node in connectome.nodes_at_uv(u, v, cell):
                    mid_traces.append(trace[b, :, int(cell_node)])
            if not mid_traces:
                continue
            entry_traces = np.stack(mid_traces, axis=0)
            cell_v[mid] = entry_traces.mean(axis=0)
            cell_sd[mid] = sd_from_traces(
                entry_traces, single_hex=(entry_traces.shape[0] == 1),
            )
            cell_n[mid] = int(entry_traces.shape[0])
        if not cell_v:
            continue
        v_readout[cell] = cell_v
        sd_out[cell] = cell_sd
        n_by_cell_mid[cell] = cell_n

    readout_cells = cells_in_order(list(v_readout))
    v_th = v_th_from_z(z, session)
    e_leak = e_leak_from_z(z, session)
    v_th_by_cell = {cell: v_th.get(cell, np.nan) for cell in readout_cells}
    e_leak_by_cell = {cell: e_leak.get(cell, np.nan) for cell in readout_cells}
    gt_affine_by_cell = {
        str(cell): train.gt_affine_from_cell(
            params, cell, session.connectome, session=session,
        )
        for cell in readout_cells
    }

    opts = dict((session.train_opts or {}).get(f"{task}_sti_opts") or {})
    t_onset = int(train.pack_t_onset(pack))
    ms_sti = opts.get("ms_sti")
    delta_ms = float(opts["delta_ms"])

    return SbarTraceReadout(
        task=task,
        contrast=contrast,
        cells=readout_cells,
        mids=mids,
        n_t=n_t,
        t_onset=t_onset,
        t_sti_end=t_sti_end(t_onset, n_t, ms_sti, delta_ms=delta_ms),
        ms_shown=ms_shown,
        v_readout=v_readout,
        sd=sd_out,
        n_by_cell_mid=n_by_cell_mid,
        v_th_by_cell=v_th_by_cell,
        e_leak_by_cell=e_leak_by_cell,
        gt_affine_by_cell=gt_affine_by_cell,
        a_sti_mid=a_sti_mid,
        a_sti_mid_sigma=a_sti_mid_sigma,
        session=session,
        prep_s=time.perf_counter() - t_prep0,
    )


def _sbar_filter_readout(readout, cells):
    """Keep only ``cells`` that have traces in ``readout``."""
    keep = [cell for cell in cells_in_order(cells) if cell in readout.v_readout]
    return SbarTraceReadout(
        task=readout.task,
        contrast=readout.contrast,
        cells=keep,
        mids=readout.mids,
        n_t=readout.n_t,
        t_onset=readout.t_onset,
        t_sti_end=readout.t_sti_end,
        ms_shown=readout.ms_shown,
        v_readout={cell: readout.v_readout[cell] for cell in keep},
        sd={cell: readout.sd[cell] for cell in keep if cell in readout.sd},
        n_by_cell_mid={
            cell: readout.n_by_cell_mid[cell]
            for cell in keep if cell in readout.n_by_cell_mid
        },
        v_th_by_cell={
            cell: readout.v_th_by_cell[cell]
            for cell in keep if cell in readout.v_th_by_cell
        },
        e_leak_by_cell={
            cell: readout.e_leak_by_cell[cell]
            for cell in keep if cell in readout.e_leak_by_cell
        },
        gt_affine_by_cell={
            cell: readout.gt_affine_by_cell[cell]
            for cell in keep if cell in readout.gt_affine_by_cell
        },
        a_sti_mid=dict(readout.a_sti_mid),
        a_sti_mid_sigma=readout.a_sti_mid_sigma,
        session=readout.session,
        prep_s=readout.prep_s,
    )


def _sbar_gt_readout(readout):
    """Gt figure rows: configured active gt cells (not cost-pack-only)."""
    session = readout.session
    return _sbar_filter_readout(
        readout,
        active_sbar_gt_cells(
            session,
            session.primary_pack.task,
            session.primary_pack.contrast,
        ),
    )


def _sbar_all_readout(readout):
    """All figure rows: config ``sbar_figure_cells``."""
    return _sbar_filter_readout(readout, sbar_figure_cells())


def _sbar_mid_label(mid: float) -> str:
    """Format mid as column label."""
    return f"{int(mid):+d}" if float(mid).is_integer() else f"{float(mid):+.1f}"


def _panel_a_sti_mid(readout: SbarTraceReadout, mid: float) -> str:
    """Effective symmetric stimulus amplitude shown in one sbar panel."""
    if np.isclose(float(mid), 0.0):
        return "1"
    key = str(float(abs(mid))).removesuffix(".0")
    value = readout.a_sti_mid.get(key)
    return "n/a" if value is None else f"{float(value):.4g}"


def _sbar_figure(n_row, n_col):
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(2.0 * n_col, 1.8 * n_row),
        sharex=True,
        sharey="row",
    )
    if n_row == 1:
        axes = np.asarray([axes])
    if n_col == 1:
        axes = axes[:, None]
    return fig, axes


def _plot_figure(path, *, timer, readouts, title, gts=None, cost_parts=None):
    """Plot sbar figure from ``readouts`` (contrast → SbarTraceReadout)."""
    order = contrast_order(readouts)
    if not order:
        raise ValueError("_plot_figure requires at least one readout")
    primary = readouts[order[0]]
    cells = primary.cells
    n_row = len(cells)
    n_t = primary.n_t
    t_onset = primary.t_onset
    t_sti_end = primary.t_sti_end
    delta_ms = float(primary.session.delta_ms)
    delta_ms_pre = float(primary.session.delta_ms_pre)
    ms_shown = primary.ms_shown

    timer.end_prep()
    sessions = {contrast: readouts[contrast].session for contrast in order}
    gt_by_contrast = resolve_sbar_gts(sessions, gts)
    mids_by_cell = {
        cell: sorted({
            float(mid)
            for contrast in order
            for mid in readouts[contrast].v_readout.get(cell, {})
        })
        for cell in cells
    }
    n_col = max((len(cell_mids) for cell_mids in mids_by_cell.values()), default=0)
    if n_col == 0:
        raise ValueError("_plot_figure requires at least one cell/mid trace")
    fig, axes = _sbar_figure(n_row, n_col)

    for row, cell in enumerate(cells):
        cell_mids = mids_by_cell[cell]
        start = (n_col - len(cell_mids)) // 2
        for col in range(n_col):
            if not (start <= col < start + len(cell_mids)):
                axes[row, col].axis("off")
        for col, mid in enumerate(cell_mids, start=start):
            ax = axes[row, col]
            n_entry = primary.n_by_cell_mid.get(cell, {}).get(mid)
            mid_label = _sbar_mid_label(mid)
            n_label = str(int(n_entry)) if n_entry is not None else "n/a"
            panel_title = (
                f"{mid_label}\n"
                f"n={n_label}\n"
                f"a_sti_mid={_panel_a_sti_mid(primary, mid)}"
            )
            traces = []
            for contrast in order:
                contrast_readout = readouts[contrast]
                v_readout = contrast_readout.v_readout.get(cell, {}).get(mid)
                if v_readout is None:
                    continue
                gt_by_cell_mid = gt_by_contrast.get(contrast) or {}
                traces.append({
                    "contrast": contrast,
                    "v_readout_mean_cell": v_readout,
                    "gt": gt_trace_affine(
                        contrast_readout, cell, gt_by_cell_mid.get((cell, mid)),
                    ),
                    "sd": contrast_readout.sd.get(cell, {}).get(mid),
                    "linestyle": contrast_linestyle(contrast),
                })
            traces = traces_with_cost_ts(traces, readouts)
            if not traces:
                ax.axis("off")
                continue
            plot_cell_time(
                ax, traces,
                title=panel_title,
                show_xlabels=(row == n_row - 1),
                show_ylabel=(col == start),
                v_th=primary.v_th_by_cell.get(cell),
                e_leak=primary.e_leak_by_cell.get(cell),
                n_t=n_t,
                t_onset=t_onset,
                t_sti_end=t_sti_end,
                delta_ms=delta_ms,
                delta_ms_pre=delta_ms_pre,
                ms_shown=ms_shown,
            )
            ax.tick_params(labelsize=6)
        axes[row, start].set_ylabel(cell_ylabel(cell, None), fontsize=8, labelpad=12)
        for col in range(n_col):
            axes[row, col].tick_params(labelleft=(col == start))

    opts = dict((primary.session.train_opts or {}).get(f"{primary.task}_sti_opts") or {})
    bar_dist = opts.get("bar_dist")
    bar_directions = opts.get("bar_directions")
    shift_mid = opts.get("shift_mid", opts.get("shift_radius"))
    subtitle = ""
    if bar_dist is not None and bar_directions is not None:
        subtitle = f"  [bar_dist={bar_dist}, directions={bar_directions}]"
    if shift_mid is not None:
        subtitle += f"  [shift_mid={shift_mid}]"
    if primary.a_sti_mid_sigma is not None:
        values = ", ".join(
            f"{mid}={value:.4g}" for mid, value in primary.a_sti_mid.items()
        )
        subtitle += (
            f"  [a_sti_mid σ={primary.a_sti_mid_sigma:.4g}: {values}]"
        )
    elif primary.a_sti_mid:
        values = ", ".join(
            f"{mid}={value:.4g}" for mid, value in primary.a_sti_mid.items()
        )
        subtitle += f"  [a_sti_mid: {values}]"
    fig.suptitle(title + subtitle, fontsize=12)
    fig.subplots_adjust(top=0.92, bottom=0.10, hspace=0.85, wspace=0.35)
    timer.end_plot()
    save_figure(fig, path, dpi=SBAR_DPI, rasterize=True, timer=timer)


@torch.no_grad()
def plot_gt(path, *, readouts, title, gts=None, cost_parts=None):
    """Plot sbar gt figure from contrast → :class:`SbarTraceReadout`."""
    gt_readouts = {contrast: _sbar_gt_readout(readout) for contrast, readout in readouts.items()}
    _plot_figure(
        path,
        timer=ElapsedTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts=gt_readouts,
        title=title,
        gts=gts,
        cost_parts=cost_parts,
    )


@torch.no_grad()
def plot_all(path, *, readouts, title, gts=None, cost_parts=None):
    """Plot sbar all figure: config ``sbar_figure_cells`` rows."""
    all_readouts = {
        contrast: _sbar_all_readout(readout) for contrast, readout in readouts.items()
    }
    _plot_figure(
        path,
        timer=ElapsedTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts=all_readouts,
        title=title,
        gts=gts,
        cost_parts=cost_parts,
    )


_TASK = "sbar"


def build_readout(session, z, contrast, **readout_kwargs):
    return network_sbar_trace_readout(
        session_from_task(session, _TASK, contrast), z, _TASK, contrast, **readout_kwargs,
    )


def figure_titles(session, suffix, token, *, contrast=None):
    if contrast is None:
        return (
            f"Static-bar {token}-gt ({suffix})",
            f"Static-bar {token}-all ({suffix})",
        )
    return (
        f"sbar {contrast} {token}-gt ({suffix})",
        f"sbar {contrast} {token}-all ({suffix})",
    )
