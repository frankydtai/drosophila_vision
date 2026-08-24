"""Static-bar plotting (network static-bar task).

Each panel: one cell at one bar-position on the motion axis.
Layout: 8 rows (T4a,b,c,d + T5a,b,c,d) × 5 columns (bar positions from cost pack).

GT traces are Gruntman Fig.2 digitized width-1 flash responses, loaded from
``task.sbar.gt`` using the same timing as the pack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

import train
from task.sbar.gt import load_gt as load_sbar_gt
from task.sbar.pack import SbarPack
from figure.spread import (
    _session_task_timing,
    contrast_linestyle,
    contrast_order,
    plot_cell_time,
)
from network.construction import cells_in_order
from figure.plot import session_from_task
from figure.panel import (
    ElapsedTimer,
    as_numpy,
    cell_ylabel,
    e_leak_from_z,
    gt_trace_affine,
    readout_prep_s,
    save_figure,
    std_from_traces,
    traces_with_cost_ts,
    v_th_from_z,
)
from task.spread.sti_spec import t_sti_end


SBAR_DPI = 100


def sbar_gt_trace_key(cell: str, contrast: str, mid: float) -> str:
    """Build Gruntman GT trace_id: ``{cell_prefix}_{PC|NC}_{pos}_w1``."""
    prefix = cell[:2]
    if (cell.startswith("T4") and contrast == "bright") or (
        cell.startswith("T5") and contrast == "dark"
    ):
        pathway = "PC"
    else:
        pathway = "NC"
    pos_str = f"{int(mid):+d}" if float(mid).is_integer() else f"{float(mid):+.1f}"
    return f"{prefix}_{pathway}_pos{pos_str}_w1"


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
    std: dict = field(default_factory=dict)
    n_by_cell_mid: dict = field(default_factory=dict)
    v_th_by_cell: dict = field(default_factory=dict)
    e_leak_by_cell: dict = field(default_factory=dict)
    gt_affine_by_cell: dict = field(default_factory=dict)
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
        for cell in cells_in_order(session.connectome.cells):
            for mid in _sbar_mids_from_pack(pack):
                key = sbar_gt_trace_key(cell, contrast, mid)
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
    std_out: dict = {}
    n_by_cell_mid: dict = {}

    for cell in all_cells:
        cell_entry_mask = node_cells == cells.index(cell)
        if not np.any(cell_entry_mask):
            continue
        v_readout[cell] = {}
        std_out[cell] = {}
        n_by_cell_mid[cell] = {}
        for mid in mids:
            mid_entry_mask = cell_entry_mask.copy()
            for entry_idx, key in enumerate(entry_part_keys):
                mid_str = key.rsplit("_mid", 1)[1]
                if float(mid_str) != mid:
                    mid_entry_mask[entry_idx] = False
            if not np.any(mid_entry_mask):
                continue
            entry_traces = trace[entry_bs[mid_entry_mask], :, entry_nodes[mid_entry_mask]]
            v_readout[cell][mid] = entry_traces.mean(axis=0)
            std_out[cell][mid] = std_from_traces(entry_traces, single_hex=(entry_traces.shape[0] == 1))
            n_by_cell_mid[cell][mid] = int(np.sum(mid_entry_mask))

    v_th = v_th_from_z(z, session)
    e_leak = e_leak_from_z(z, session)
    v_th_by_cell = {cell: v_th.get(cell, np.nan) for cell in all_cells}
    e_leak_by_cell = {cell: e_leak.get(cell, np.nan) for cell in all_cells}
    gt_affine_by_cell = {
        str(cell): train.gt_affine_from_cell(
            params, cell, session.connectome, session=session,
        )
        for cell in all_cells
    }

    opts = dict((session.train_opts or {}).get(f"{task}_sti_opts") or {})
    t_onset = int(train.pack_t_onset(pack))
    ms_sti = opts.get("ms_sti")
    delta_ms = float(opts["delta_ms"])

    return SbarTraceReadout(
        task=task,
        contrast=contrast,
        cells=all_cells,
        mids=mids,
        n_t=n_t,
        t_onset=t_onset,
        t_sti_end=t_sti_end(t_onset, n_t, ms_sti, delta_ms=delta_ms),
        ms_shown=ms_shown,
        v_readout=v_readout,
        std=std_out,
        n_by_cell_mid=n_by_cell_mid,
        v_th_by_cell=v_th_by_cell,
        e_leak_by_cell=e_leak_by_cell,
        gt_affine_by_cell=gt_affine_by_cell,
        session=session,
        prep_s=time.perf_counter() - t_prep0,
    )


def _sbar_mid_label(mid: float) -> str:
    """Format mid as column label."""
    return f"{int(mid):+d}" if float(mid).is_integer() else f"{float(mid):+.1f}"


def _sbar_figure(n_row, n_col):
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(2.0 * n_col, 1.8 * n_row),
        sharex=True,
        sharey=True,
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
    mids = primary.mids
    n_col = len(mids)
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
    fig, axes = _sbar_figure(n_row, n_col)

    for row, cell in enumerate(cells):
        for col, mid in enumerate(mids):
            ax = axes[row, col]
            n_entry = primary.n_by_cell_mid.get(cell, {}).get(mid, 0)
            mid_label = _sbar_mid_label(mid)
            n_label = f" (n={int(n_entry)})" if n_entry else ""
            panel_title = mid_label + n_label if row == 0 else mid_label
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
                    "std": contrast_readout.std.get(cell, {}).get(mid),
                    "linestyle": contrast_linestyle(contrast),
                })
            traces = traces_with_cost_ts(traces, readouts)
            if not traces:
                ax.axis("off")
                if row == 0:
                    ax.set_title(panel_title, fontsize=7, pad=2)
                continue
            plot_cell_time(
                ax, traces,
                title=panel_title if row == 0 else None,
                show_xlabels=(row == n_row - 1),
                show_ylabel=(col == 0),
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
        axes[row, 0].set_ylabel(cell_ylabel(cell, None), fontsize=8, labelpad=12)

    opts = dict((primary.session.train_opts or {}).get(f"{primary.task}_sti_opts") or {})
    bar_dist = opts.get("bar_dist")
    bar_directions = opts.get("bar_directions")
    subtitle = ""
    if bar_dist is not None and bar_directions is not None:
        subtitle = f"  [bar_dist={bar_dist}, directions={bar_directions}]"
    fig.suptitle(title + subtitle, fontsize=12)
    fig.subplots_adjust(top=0.92, bottom=0.10, hspace=0.50, wspace=0.35)
    timer.end_plot()
    save_figure(fig, path, dpi=SBAR_DPI, rasterize=True, timer=timer)


@torch.no_grad()
def plot_gt(path, *, readouts, title, gts=None, cost_parts=None):
    """Plot sbar gt figure from contrast → :class:`SbarTraceReadout`."""
    _plot_figure(
        path,
        timer=ElapsedTimer(prior_prep=readout_prep_s(*readouts.values())),
        readouts=readouts,
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
