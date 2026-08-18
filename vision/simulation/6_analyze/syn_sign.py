"""Histogram + Δv vs %% n_syn+ (excitatory fraction) for a trained run.

Per cell column (``spot_gt_v`` order):
  1. hist: red=init %% n_syn+, blue=×trained ``syn_strength_cell``
  2. plot per cost radius=0,1,… (same radii as spot time panels):
     x = blue %% n_syn+, y = model ``v[t_sti_end-1] - v[t_onset]``

Writes ``<run>/pre_syn/syn_{gt,all}.png`` (incoming), or ``post_syn/`` with ``post=true``.

Usage (from ``simulation/``)::

  ../.venv/bin/python -m analyze.syn_sign
  ../.venv/bin/python -m analyze.syn_sign post=true
"""
from __future__ import annotations

from config import (
    ANALYZE_RUNS,
    ANALYZE_SYN_SIGN,
)

import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import import_bootstrap  # noqa: F401
import hydra
import matplotlib.pyplot as plt
import network.path  # noqa: F401
import numpy as np
import train
import train.implementation as train_mod
import figure.plot as plot
import figure.spot as spot
from figure.panel import (
    N_COL_ALL,
    N_COL_GT,
    PANEL_H,
    PANEL_W,
    ElapsedTimer,
    save_figure,
)
from network.connectivity import build_cell_pair_idxs
from network.construction import (
    cell_rows,
    gt_cells_from_opts,
    active_gt_cells,
    load_network_json,
)
from task.spread.gt import GT_CELLS
from network import path  # noqa: F401 -- FAFBv783 on sys.path
import build_hex


def _syn_strength_from_edges(edges, cells, syn_strength_cell, pairs):
    """Map (src_type_i, tar_type_i) -> trained syn_strength_cell."""
    cell_idx = dict(zip(cells, range(len(cells))))
    n_cell = len(cells)
    src_t = np.array([cell_idx[edge["source_cell"]] for edge in edges], dtype=np.int64)
    tar_t = np.array([cell_idx[edge["target_cell"]] for edge in edges], dtype=np.int64)
    _, n_pair, pairs = build_cell_pair_idxs(src_t, tar_t, n_cell)
    syn = np.asarray(syn_strength_cell, dtype=np.float64).reshape(-1)
    if syn.shape[0] != n_pair:
        raise SystemExit(
            f"syn_strength_cell length {syn.shape[0]} != n_pair {n_pair}"
        )
    if pairs is not None:
        expected = [
            f"{cells[source]}{train.PAIR_SEP}{cells[target]}" for source, target in pairs
        ]
        if list(pairs) != expected:
            raise SystemExit("pairs in best_param.npz do not match network.json")
    return dict(zip(pairs, map(float, syn))), cell_idx


def syn_plus_by_id(
    edges,
    cell: str,
    *,
    direction: str,
    cell_idx: dict[str, int],
    strength_by_pair: dict[tuple[int, int], float] | None,
) -> dict[int, float]:
    """id -> %% n_syn+ (optional per-pair strength weighting)."""
    if direction == "post":
        self_cell_field, self_id_field = "source_cell", "src"
    else:
        self_cell_field, self_id_field = "target_cell", "tar"
    syn_p = defaultdict(float)
    syn_t = defaultdict(float)
    for edge in edges:
        if edge.get(self_cell_field) != cell:
            continue
        try:
            sid = int(edge[self_id_field])
        except (KeyError, TypeError, ValueError):
            continue
        src = edge.get("source_cell")
        tar = edge.get("target_cell")
        if src not in cell_idx or tar not in cell_idx:
            continue
        ns = float(edge.get("n_syn", 0))
        if strength_by_pair is not None:
            ns *= strength_by_pair[(cell_idx[src], cell_idx[tar])]
        syn_t[sid] += ns
        try:
            sign = float(edge.get("syn_sign", 0))
        except (TypeError, ValueError):
            sign = 0.0
        if sign > 0:
            syn_p[sid] += ns
    return {
        sid: 100.0 * syn_p[sid] / tot
        for sid, tot in syn_t.items()
        if tot > 0
    }


def _side_by_side_hist(ax, pct_init, pct_trained, edges_bins, *, legend: bool):
    """Red = init, blue = syn_strength_cell-weighted; paired bars per bin."""
    c_init, _ = np.histogram(pct_init, bins=edges_bins)
    c_tr, _ = np.histogram(pct_trained, bins=edges_bins)
    bin_midpoints = 0.5 * (edges_bins[:-1] + edges_bins[1:])
    w = (edges_bins[1] - edges_bins[0]) * 0.4
    ax.bar(
        bin_midpoints - w / 2, c_init, width=w, color="red",
        edgecolor="white", linewidth=0.3, label="init",
    )
    ax.bar(
        bin_midpoints + w / 2, c_tr, width=w, color="C0",
        edgecolor="white", linewidth=0.3, label="× syn_strength_cell",
    )
    ax.set_xlim(0, 100)
    if legend:
        ax.legend(fontsize=7, loc="upper right")


def _active_spot_gt_cells(opts, available):
    """Same cell set as ``spot_gt_v``."""
    sti_opts = opts.get("spot_sti_opts") or {}
    requested = gt_cells_from_opts(sti_opts)
    if requested is None:
        requested = gt_cells_from_opts(opts)
    return active_gt_cells(
        requested, GT_CELLS, available, context="syn_sign spot-gt cells",
    )


def _spot_session_z(run_dir, contrast: str = "bright"):
    """Best ``z`` on ``spot`` × ``contrast`` (or raise if no spot task)."""
    session, z, _cost = plot.load_best(run_dir)
    tasks = list((session.train_opts or {}).get("tasks") or [])
    if "spot" not in tasks:
        raise SystemExit(f"run has no spot task (tasks={tasks})")
    return plot.session_from_task(session, "spot", contrast), z


def load_delta_v_tables(session, z):
    """cell -> radius_k -> [(id, Δv)]; radii from pack cost radii."""
    readout = spot._forward_spot_readout(session, z)
    t_onset = readout.get("t_onset")
    t_sti_end = readout.get("t_sti_end")
    if t_onset is None or t_sti_end is None or int(t_sti_end) <= int(t_onset):
        raise SystemExit("spot timing missing t_onset / t_sti_end")
    t0 = int(t_onset)
    t1 = int(t_sti_end) - 1
    traces = np.asarray(readout["figure_traces"], dtype=np.float64)
    type_idx = np.asarray(readout["type_idx"], dtype=np.int64)
    nodes = np.asarray(readout["nodes"], dtype=np.int64)
    du = np.asarray(readout["du"], dtype=np.int64)
    dv = np.asarray(readout["dv"], dtype=np.int64)
    cells = list(readout["cells"])
    ids = np.asarray(session.connectome.node_ids, dtype=np.int64)[nodes]
    radii = sorted({
        int(radius)
        for radius in spot.pack_spot_cost_radii(readout["pack"])
    })
    entry_radii = np.asarray(
        [int(build_hex.hex_radius(int(a), int(b))) for a, b in zip(du, dv)],
        dtype=np.int64,
    )
    delta = traces[:, t1] - traces[:, t0]
    delta_v_tables = {name: {radius: [] for radius in radii} for name in cells}
    for type_at, radius, delta_v, id_val in zip(type_idx, entry_radii, delta, ids):
        cell = cells[int(type_at)]
        if int(radius) not in delta_v_tables[cell] or not np.isfinite(delta_v):
            continue
        delta_v_tables[cell][int(radius)].append((int(id_val), float(delta_v)))
    return delta_v_tables, radii


def plot_syn_sign(
    path, *,
    active,
    n_col,
    panel_w,
    panel_h,
    edges,
    direction,
    cell_idx,
    strength_by_pair,
    edges_bins,
    run_label,
    delta_tables,
    radii,
):
    """Draw hist + per-radius Δv plots for ``active`` cells."""
    timer = ElapsedTimer()
    timer.end_prep()
    rows = cell_rows(active)
    flow = "out of" if direction == "post" else "onto"
    n_sub = 1 + len(radii)
    n_row = len(rows) * n_sub
    fig = plt.figure(figsize=(panel_w * n_col, panel_h * n_row))
    gs = fig.add_gridspec(
        n_row, n_col,
        hspace=0.55, wspace=0.45, top=0.96, bottom=0.04, left=0.06, right=0.98,
    )
    fig.suptitle(
        f"%% n_syn+ {flow} cell  |  hist: red=init blue=×α  |  "
        f"plot: x=×α %%  y=Δv  |  {run_label}",
        fontsize=11,
    )
    legend_done = False
    for gi, row_cells in enumerate(rows):
        base = gi * n_sub
        for ci in range(n_col):
            if ci >= len(row_cells):
                for sub in range(n_sub):
                    fig.add_subplot(gs[base + sub, ci]).set_axis_off()
                continue
            cell = row_cells[ci]
            pct_init_map = syn_plus_by_id(
                edges, cell, direction=direction,
                cell_idx=cell_idx, strength_by_pair=None,
            )
            pct_tr_map = syn_plus_by_id(
                edges, cell, direction=direction,
                cell_idx=cell_idx, strength_by_pair=strength_by_pair,
            )
            pct_init = np.asarray(list(pct_init_map.values()), dtype=np.float64)
            pct_tr = np.asarray(list(pct_tr_map.values()), dtype=np.float64)
            ax_h = fig.add_subplot(gs[base, ci])
            if pct_init.size == 0 and pct_tr.size == 0:
                ax_h.set_title(f"{cell} (empty)")
                ax_h.set_axis_off()
                for sub in range(1, n_sub):
                    fig.add_subplot(gs[base + sub, ci]).set_axis_off()
                continue
            _side_by_side_hist(
                ax_h, pct_init, pct_tr, edges_bins, legend=not legend_done,
            )
            legend_done = True
            ax_h.set_xlabel("% n_syn+")
            ax_h.set_ylabel("count")
            ax_h.set_title(
                f"{cell}  n={pct_init.size}  "
                f"init μ={pct_init.mean():.1f}  "
                f"×α μ={pct_tr.mean():.1f}",
                fontsize=9,
            )
            ax_h.tick_params(labelsize=7)
            by_r = delta_tables.get(cell) or {}
            for si, rk in enumerate(radii):
                ax = fig.add_subplot(gs[base + 1 + si, ci])
                xs, ys = [], []
                for id, d_v in by_r.get(rk) or []:
                    percent = pct_tr_map.get(id)
                    if percent is None:
                        continue
                    xs.append(percent)
                    ys.append(d_v)
                if xs:
                    ax.scatter(
                        xs, ys, s=8, c="C0", alpha=0.65, edgecolors="none",
                    )
                ax.set_xlim(0, 100)
                ax.set_xlabel("% n_syn+ (×α)", fontsize=7)
                ax.set_ylabel("Δv (mV)", fontsize=7)
                ax.set_title(f"radius={rk}  n={len(xs)}", fontsize=8)
                ax.tick_params(labelsize=6)
                ax.axhline(0.0, color="0.6", linewidth=0.5)
    timer.end_plot()
    save_figure(fig, path, timer=timer)


def save_syn_sign_figures(run_dir, *, post=False, bins=None) -> None:
    """Write ``pre_syn/syn_{gt,all}.png`` (or ``post_syn/`` when *post*)."""
    if bins is None:
        bins = int(ANALYZE_SYN_SIGN["bins"])
    opts = plot.load_train_opts(run_dir)
    if not opts:
        raise SystemExit(f"missing train_opts.json under {run_dir}")
    if opts.get("model", "borst") not in ("borst", "hp_lp"):
        raise SystemExit(
            f"syn_strength_cell requires borst/hp_lp, got {opts.get('model')!r}"
        )
    if opts.get("syn_mode", "per_cell") != "per_cell":
        raise SystemExit(
            f"syn_sign needs --syn-mode per_cell, got {opts.get('syn_mode')!r}"
        )
    network_json = opts.get("network_json")
    if not network_json:
        raise SystemExit("train_opts.json missing network_json")

    try:
        node_vals, cells_npz, pairs = train_mod.load_best_node_vals(run_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if "syn_strength_cell" not in node_vals:
        raise SystemExit("best_param.npz missing syn_strength_cell")

    _nodes, edges, cells, _meta = load_network_json(network_json)
    if list(cells) != list(cells_npz):
        raise SystemExit("cells mismatch: network.json vs best_param.npz")

    strength_by_pair, cell_idx = _syn_strength_from_edges(
        edges, cells, node_vals["syn_strength_cell"], pairs,
    )
    session, z = _spot_session_z(run_dir)
    delta_tables, radii = load_delta_v_tables(session, z)

    direction = "post" if post else "pre"
    syn_dir = os.path.join(run_dir, "post_syn" if post else "pre_syn")
    os.makedirs(syn_dir, exist_ok=True)
    figure_kwargs = dict(
        edges=edges,
        direction=direction,
        cell_idx=cell_idx,
        strength_by_pair=strength_by_pair,
        edges_bins=np.linspace(0.0, 100.0, bins + 1),
        run_label=os.path.basename(run_dir),
        delta_tables=delta_tables,
        radii=radii,
    )
    plot_syn_sign(
        os.path.join(syn_dir, "syn_gt.png"),
        active=_active_spot_gt_cells(opts, cells),
        n_col=N_COL_GT, panel_w=PANEL_W, panel_h=PANEL_H,
        **figure_kwargs,
    )
    plot_syn_sign(
        os.path.join(syn_dir, "syn_all.png"),
        active=list(cells),
        n_col=N_COL_ALL, panel_w=PANEL_W, panel_h=PANEL_H,
        **figure_kwargs,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(hydra_config) -> None:
    from config import apply_config

    apply_config(hydra_config)
    post = bool(ANALYZE_SYN_SIGN["post"])
    bins = int(ANALYZE_SYN_SIGN["bins"])
    for run in ANALYZE_RUNS:
        save_syn_sign_figures(
            plot.resolve_run_dir(run),
            post=post,
            bins=bins,
        )


if __name__ == "__main__":
    main()
