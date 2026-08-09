"""Histogram + Δv vs %% n_syn+ (excitatory fraction) for a trained run.

Per cell column (``spot_gt_v`` order):
  1. hist: red=init %% n_syn+, blue=×trained ``syn_strength_cell``
  2. scatter per cost radius r=0,1,… (same radii as spot time panels):
     x = blue %% n_syn+, y = model ``v[t_spot_end-1] - v[t_onset]``

Writes ``<run>/pre_syn/syn_{gt,all}.png`` (incoming), or ``post_syn/`` with ``--post``.

Usage (from ``simulation/``)::

  ../.venv/bin/python 6_analyze/syn_sign.py
  ../.venv/bin/python -m analyze.syn_sign --post
"""
from __future__ import annotations

import argparse
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
import matplotlib.pyplot as plt
import network.path  # noqa: F401
import numpy as np
import training
import training.implement as train_mod
import figure.plot_run as plot_trained
import figure.spot as spot_plot
from figure.util import (
    NCOLS_ALL,
    NCOLS_GT,
    PANEL_H,
    PANEL_W,
    PlotTimer,
    save_figure,
)
from network.connectivity import build_cell_pair_index
from network.construction import (
    cell_order_rows,
    gt_cells_from_opts,
    present_gt_cells,
    read_network_json,
)
from param_defaults import DEFAULT_RUN_PATH
from task.spot.gt import GT_CELLS
from task.spot.input import euclid_hex_dist

DEFAULT_BINS = 20


def _pair_strength_lookup(edges, cell_names, syn_strength_cell, pair_names):
    """Map (src_type_i, tar_type_i) -> trained syn_strength_cell."""
    name_to_i = {n: i for i, n in enumerate(cell_names)}
    n_cells = len(cell_names)
    src_t = np.array([name_to_i[e["source_cell"]] for e in edges], dtype=np.int64)
    tar_t = np.array([name_to_i[e["target_cell"]] for e in edges], dtype=np.int64)
    _, n_pairs, pair_keys = build_cell_pair_index(src_t, tar_t, n_cells)
    syn = np.asarray(syn_strength_cell, dtype=np.float64).reshape(-1)
    if syn.shape[0] != n_pairs:
        raise SystemExit(
            f"syn_strength_cell length {syn.shape[0]} != n_pairs {n_pairs}"
        )
    if pair_names is not None:
        expected = [
            f"{cell_names[s]}{training.PAIR_SEP}{cell_names[t]}" for s, t in pair_keys
        ]
        if list(pair_names) != expected:
            raise SystemExit("pair_names in best_param.npz do not match network.json")
    return {k: float(syn[i]) for i, k in enumerate(pair_keys)}, name_to_i


def instance_syn_plus_by_id(
    edges,
    cell_name: str,
    *,
    direction: str,
    name_to_i: dict[str, int],
    strength_by_pair: dict[tuple[int, int], float] | None,
) -> dict[int, float]:
    """root_id -> %% n_syn+ (optional per-pair strength weighting)."""
    if direction == "post":
        self_cell_field, self_id_field = "source_cell", "src"
    else:
        self_cell_field, self_id_field = "target_cell", "tar"
    syn_p = defaultdict(float)
    syn_t = defaultdict(float)
    for e in edges:
        if e.get(self_cell_field) != cell_name:
            continue
        try:
            sid = int(e[self_id_field])
        except (KeyError, TypeError, ValueError):
            continue
        src = e.get("source_cell")
        tar = e.get("target_cell")
        if src not in name_to_i or tar not in name_to_i:
            continue
        ns = float(e.get("n_syn", 0))
        if strength_by_pair is not None:
            ns *= strength_by_pair[(name_to_i[src], name_to_i[tar])]
        syn_t[sid] += ns
        try:
            sign = float(e.get("syn_sign", 0))
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
    centers = 0.5 * (edges_bins[:-1] + edges_bins[1:])
    width = (edges_bins[1] - edges_bins[0]) * 0.4
    ax.bar(
        centers - width / 2, c_init, width=width, color="red",
        edgecolor="white", linewidth=0.3, label="init",
    )
    ax.bar(
        centers + width / 2, c_tr, width=width, color="C0",
        edgecolor="white", linewidth=0.3, label="× syn_strength_cell",
    )
    ax.set_xlim(0, 100)
    if legend:
        ax.legend(fontsize=7, loc="upper right")


def _spot_gt_cells(opts, available):
    """Same cell set as ``spot_gt_v``."""
    stim = opts.get("spot_bright_stimulus_opts") or {}
    requested = gt_cells_from_opts(stim)
    if requested is None:
        requested = gt_cells_from_opts(opts)
    return present_gt_cells(
        requested, GT_CELLS, available, context="syn_sign spot-gt cells",
    )


def _spot_bright_session_z(outdir):
    """Best ``z`` on ``spot_bright`` (or primary spot task)."""
    session, z, _cost = plot_trained.load_best(outdir)
    tasks = list((session.train_opts or {}).get("tasks") or [])
    if "spot_bright" in tasks:
        return plot_trained.session_for_task(session, "spot_bright"), z
    if not any(t.startswith("spot_") for t in tasks):
        raise SystemExit(f"run has no spot task (tasks={tasks})")
    return session, z


def load_delta_v_tables(session, z):
    """cell -> radius_k -> [(root_id, Δv)]; radii from pack cost radii."""
    rows = spot_plot._spot_forward_rows(session, z)
    t_onset = rows.get("t_onset")
    t_spot_end = rows.get("t_spot_end")
    if t_onset is None or t_spot_end is None or int(t_spot_end) <= int(t_onset):
        raise SystemExit("spot timing missing t_onset / t_spot_end")
    t0 = int(t_onset)
    t1 = int(t_spot_end) - 1
    traces = np.asarray(rows["plot_traces"], dtype=np.float64)
    type_idx = np.asarray(rows["type_idx"], dtype=np.int64)
    node_idx = np.asarray(rows["node_idx"], dtype=np.int64)
    du = np.asarray(rows["du"], dtype=np.int64)
    dv = np.asarray(rows["dv"], dtype=np.int64)
    cell_names = list(rows["cell_names"])
    root_ids = np.asarray(session.backend.network.node_ids, dtype=np.int64)[node_idx]
    radii = sorted({
        int(round(float(r)))
        for r in spot_plot.pack_spot_cost_radii(rows["pack"])
    })
    r_k = np.asarray(
        [int(round(euclid_hex_dist(int(a), int(b)))) for a, b in zip(du, dv)],
        dtype=np.int64,
    )
    delta = traces[:, t1] - traces[:, t0]
    out = {name: {r: [] for r in radii} for name in cell_names}
    for i in range(len(type_idx)):
        rk = int(r_k[i])
        name = cell_names[int(type_idx[i])]
        if rk not in out[name] or not np.isfinite(delta[i]):
            continue
        out[name][rk].append((int(root_ids[i]), float(delta[i])))
    return out, radii


def plot_syn_sign(
    path, *,
    present,
    ncols,
    panel_w,
    panel_h,
    edges,
    direction,
    name_to_i,
    strength_by_pair,
    edges_bins,
    outdir_name,
    delta_tables,
    radii,
):
    """Draw hist + per-radius Δv scatters for ``present`` cells."""
    timer = PlotTimer()
    timer.end_prep()
    order_rows = cell_order_rows(present)
    flow = "out of" if direction == "post" else "onto"
    n_sub = 1 + len(radii)
    total_rows = len(order_rows) * n_sub
    fig = plt.figure(figsize=(panel_w * ncols, panel_h * total_rows))
    gs = fig.add_gridspec(
        total_rows, ncols,
        hspace=0.55, wspace=0.45, top=0.96, bottom=0.04, left=0.06, right=0.98,
    )
    fig.suptitle(
        f"%% n_syn+ {flow} cell  |  hist: red=init blue=×α  |  "
        f"scatter: x=×α %%  y=Δv  |  {outdir_name}",
        fontsize=11,
    )
    legend_done = False
    for gi, row in enumerate(order_rows):
        base = gi * n_sub
        for ci in range(ncols):
            if ci >= len(row):
                for s in range(n_sub):
                    fig.add_subplot(gs[base + s, ci]).set_axis_off()
                continue
            cell = row[ci]
            pct_init_map = instance_syn_plus_by_id(
                edges, cell, direction=direction,
                name_to_i=name_to_i, strength_by_pair=None,
            )
            pct_tr_map = instance_syn_plus_by_id(
                edges, cell, direction=direction,
                name_to_i=name_to_i, strength_by_pair=strength_by_pair,
            )
            pct_init = np.asarray(list(pct_init_map.values()), dtype=np.float64)
            pct_tr = np.asarray(list(pct_tr_map.values()), dtype=np.float64)
            ax_h = fig.add_subplot(gs[base, ci])
            if pct_init.size == 0 and pct_tr.size == 0:
                ax_h.set_title(f"{cell} (empty)")
                ax_h.set_axis_off()
                for s in range(1, n_sub):
                    fig.add_subplot(gs[base + s, ci]).set_axis_off()
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
                for root_id, d_v in by_r.get(rk) or []:
                    p = pct_tr_map.get(root_id)
                    if p is None:
                        continue
                    xs.append(p)
                    ys.append(d_v)
                if xs:
                    ax.scatter(
                        xs, ys, s=8, c="C0", alpha=0.65, edgecolors="none",
                    )
                ax.set_xlim(0, 100)
                ax.set_xlabel("% n_syn+ (×α)", fontsize=7)
                ax.set_ylabel("Δv (mV)", fontsize=7)
                ax.set_title(f"r={rk}  n={len(xs)}", fontsize=8)
                ax.tick_params(labelsize=6)
                ax.axhline(0.0, color="0.6", linewidth=0.5)
    timer.end_draw()
    save_figure(fig, path, timer=timer)


def write_syn_sign_plots(outdir, *, post=False, bins=DEFAULT_BINS) -> None:
    """Write ``pre_syn/syn_{gt,all}.png`` (or ``post_syn/`` when *post*)."""
    opts = plot_trained.load_train_opts(outdir)
    if not opts:
        raise SystemExit(f"missing train_opts.json under {outdir}")
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
        named, cell_names_npz, pair_names = train_mod.load_best_param_named(outdir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if "syn_strength_cell" not in named:
        raise SystemExit("best_param.npz missing syn_strength_cell")

    _nodes, edges, cell_names, _meta = read_network_json(network_json)
    if list(cell_names) != list(cell_names_npz):
        raise SystemExit("cell_names mismatch: network.json vs best_param.npz")

    strength_by_pair, name_to_i = _pair_strength_lookup(
        edges, cell_names, named["syn_strength_cell"], pair_names,
    )
    session, z = _spot_bright_session_z(outdir)
    delta_tables, radii = load_delta_v_tables(session, z)

    direction = "post" if post else "pre"
    syn_dir = os.path.join(outdir, "post_syn" if post else "pre_syn")
    os.makedirs(syn_dir, exist_ok=True)
    plot_kw = dict(
        edges=edges,
        direction=direction,
        name_to_i=name_to_i,
        strength_by_pair=strength_by_pair,
        edges_bins=np.linspace(0.0, 100.0, bins + 1),
        outdir_name=os.path.basename(outdir),
        delta_tables=delta_tables,
        radii=radii,
    )
    plot_syn_sign(
        os.path.join(syn_dir, "syn_gt.png"),
        present=_spot_gt_cells(opts, cell_names),
        ncols=NCOLS_GT, panel_w=PANEL_W, panel_h=PANEL_H,
        **plot_kw,
    )
    plot_syn_sign(
        os.path.join(syn_dir, "syn_all.png"),
        present=list(cell_names),
        ncols=NCOLS_ALL, panel_w=PANEL_W, panel_h=PANEL_H,
        **plot_kw,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Write syn_gt.png / syn_all.png: %% n_syn+ hist + "
            "Δv vs ×α %% scatters at spot cost radii."
        ),
    )
    ap.add_argument(
        "--run",
        default=DEFAULT_RUN_PATH,
        help="run under PARAMETER_DIR or absolute path (default: %(default)s)",
    )
    ap.add_argument(
        "--post",
        action="store_true",
        help="outgoing from CELL; default is incoming onto CELL",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help=f"histogram bins over [0, 100] (default: {DEFAULT_BINS})",
    )
    args = ap.parse_args(argv)
    write_syn_sign_plots(
        plot_trained.resolve_run_dir(args.run),
        post=bool(args.post),
        bins=args.bins,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
