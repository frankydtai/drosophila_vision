"""Plot per-source-cell ``v_syn`` contributions onto T4a/T4b for one sbar run.

Each panel (T4 cell × bar mid): one colored trace per presynaptic cell type

    y_S(t) = mean_{i in entries(C,mid)} [
        a_in[i] * sum_{e: j->i, cell(j)=S} v_out[j](t) * w_e * s_e
    ]

Red line = total ``v_syn`` on those entries (= sum of colored traces). No GT.

Run from ``vision/simulation``::

    ../.venv/bin/python scratch/plot_sbar_t4_input_contributions.py
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401,E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import figure.plot as figure_plot  # noqa: E402
from figure.panel import (  # noqa: E402
    TRACE_LINE_W,
    V_READOUT_COLOR,
    mark_sti_on,
    save_figure,
)
from figure.sbar import (  # noqa: E402
    _sbar_mid_label,
    _sbar_mids_from_pack,
    active_sbar_gt_cells,
)
from figure.spread import _style_time_axis  # noqa: E402
from network.construction import cells_in_order  # noqa: E402
from neuron.borst import syn_strength  # noqa: E402
from neuron.forward import forward_v, pack_t_onset  # noqa: E402
from task.sbar.pack import part_key  # noqa: E402
from task.spread.sti_spec import t_sti_end  # noqa: E402
from train.cost import entries_by_part  # noqa: E402
from train.param import params_from_z  # noqa: E402


DEFAULT_RUN = (
    "hp_lp/30225259-run_init_from-n_iter-500-tasks-sbar-a_lsd-1"
)
DEFAULT_SAVE = os.path.join(HERE, "sbar_t4_input_contributions.png")
POST_CELLS = ("T4a", "T4b")


def _source_cells_onto(connectome, post_cell: str) -> list[str]:
    """Presynaptic cell names with at least one edge onto ``post_cell``."""
    cells = list(connectome.cells)
    post_idx = cells.index(post_cell)
    names = set()
    for src_idx, tar_idx in connectome.conn.pairs:
        if int(tar_idx) == post_idx:
            names.add(cells[int(src_idx)])
    return cells_in_order(names)


def _entry_mask_for_part(pack, part: str) -> np.ndarray:
    part_idxs, part_keys = entries_by_part(pack)
    if part not in part_keys:
        return np.zeros(int(pack.entry_nodes.shape[0]), dtype=bool)
    part_idx = part_keys.index(part)
    return (part_idxs.detach().cpu().numpy() == part_idx)


def _a_sti_mid_by_mid(session, params) -> dict[str, float]:
    if "a_sti_mid" not in params:
        return {}
    spec = session.schema.get("a_sti_mid")
    if spec is None:
        return {}
    sigma = float(params["a_sti_mid"].detach().cpu().numpy().reshape(-1)[0])
    return {
        str(mid): float(np.exp(-0.5 * (float(mid) / sigma) ** 2))
        for mid in (spec.get("mids") or ())
    }


def _v_syn_from_source_cells(
    v: torch.Tensor,
    params: dict,
    connectome,
    source_cells: list[str],
    post_nodes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-source-cell ``a_in * signed_g`` at ``post_nodes``.

    ``v`` is ``(B, T, N)`` membrane voltage. Returns
    ``source_cell -> (B, T, n_post)`` float64 on CPU, plus ``\"__total__\"``.
    """
    conn = connectome.conn
    cells = list(connectome.cells)
    source_idxs = conn.source_idxs
    target_idxs = conn.target_idxs
    post_nodes = np.asarray(post_nodes, dtype=np.int64)
    post_local = {int(node): local for local, node in enumerate(post_nodes)}
    source_idx_set = {cells.index(name) for name in source_cells}

    tar_np = target_idxs.detach().cpu().numpy().astype(np.int64, copy=False)
    src_cell_np = conn.node_cells[source_idxs].detach().cpu().numpy()
    edge_keep_np = np.flatnonzero(
        np.isin(tar_np, post_nodes) & np.isin(src_cell_np, list(source_idx_set))
    )
    if edge_keep_np.size == 0:
        empty = np.zeros(
            (v.shape[0], v.shape[1], len(post_nodes)), dtype=np.float64,
        )
        out = {name: empty.copy() for name in source_cells}
        out["__total__"] = empty
        return out

    edge_keep_t = torch.as_tensor(edge_keep_np, dtype=torch.long, device=v.device)
    src_e = source_idxs.index_select(0, edge_keep_t)
    tar_e = target_idxs.index_select(0, edge_keep_t)
    w_e = conn.w_signed.index_select(0, edge_keep_t)
    s_full = syn_strength(params)
    s_e = conn._edge_syn_strength(s_full).index_select(0, edge_keep_t)
    tar_local = torch.as_tensor(
        [post_local[int(tar)] for tar in tar_np[edge_keep_np]],
        dtype=torch.long, device=v.device,
    )
    source_cell_of_edge = np.asarray(
        [cells[int(idx)] for idx in src_cell_np[edge_keep_np]],
    )

    v_out = torch.relu(v - params["v_th"]) * params["a_out"]
    xs = v_out.index_select(-1, src_e)
    edge_val = xs * w_e * s_e
    a_in_post = params["a_in"].index_select(0, torch.as_tensor(
        post_nodes, dtype=torch.long, device=v.device,
    ))

    n_b, n_t, n_post = int(v.shape[0]), int(v.shape[1]), len(post_nodes)
    out: dict[str, np.ndarray] = {}
    total = torch.zeros(
        n_b, n_t, n_post, dtype=v.dtype, device=v.device,
    )
    for name in source_cells:
        mask_np = source_cell_of_edge == name
        if not np.any(mask_np):
            out[name] = np.zeros((n_b, n_t, n_post), dtype=np.float64)
            continue
        mask = torch.as_tensor(mask_np, device=v.device)
        edge_s = edge_val[..., mask]
        tar_s = tar_local[mask]
        sums = torch.zeros(
            n_b, n_t, n_post, dtype=v.dtype, device=v.device,
        )
        sums.scatter_add_(
            -1, tar_s.expand(n_b, n_t, -1), edge_s,
        )
        contrib = a_in_post * sums
        total = total + contrib
        out[name] = contrib.detach().cpu().numpy().astype(np.float64, copy=False)
    out["__total__"] = total.detach().cpu().numpy().astype(np.float64, copy=False)
    return out


def _mean_over_entries(
    contrib_b_t_post: np.ndarray,
    entry_bs: np.ndarray,
    entry_local: np.ndarray,
) -> np.ndarray:
    """Mean contribution over cost entries → ``(T,)``."""
    traces = contrib_b_t_post[entry_bs, :, entry_local]
    return traces.mean(axis=0)


def _colors_for_sources(source_cells: list[str]) -> dict[str, tuple]:
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    # Keep red for total ``v_syn``.
    palette = [c for c in palette if not (
        c[0] > 0.7 and c[1] < 0.35 and c[2] < 0.35
    )]
    return {
        name: palette[idx % len(palette)]
        for idx, name in enumerate(source_cells)
    }


def plot_run(run_dir: str, save_path: str, *, contrast: str = "bright") -> None:
    run_dir = figure_plot.resolve_run_dir(run_dir)
    session, z, _ = figure_plot.load_best(run_dir, verbose=True)
    if not torch.is_tensor(z):
        z = torch.tensor(
            np.asarray(z), dtype=session.sim_dtype, device=session.device,
        )
    params = params_from_z(z, session)
    pack = session.packs["sbar"][contrast]
    post_cells = [
        cell for cell in POST_CELLS
        if cell in active_sbar_gt_cells(session, "sbar", contrast)
    ]
    if not post_cells:
        raise SystemExit("no T4a/T4b in active sbar gt cells")
    mids = _sbar_mids_from_pack(pack)
    a_sti_mid = _a_sti_mid_by_mid(session, params)

    sources_by_post = {
        cell: _source_cells_onto(session.connectome, cell) for cell in post_cells
    }
    all_sources = cells_in_order({
        name for names in sources_by_post.values() for name in names
    })
    colors = _colors_for_sources(all_sources)

    entry_bs = pack.entry_bs.detach().cpu().numpy().astype(np.int64)
    entry_nodes = pack.entry_nodes.detach().cpu().numpy().astype(np.int64)
    entry_masks = {
        (cell, mid): _entry_mask_for_part(
            pack, part_key(contrast, cell, mid),
        )
        for cell in post_cells
        for mid in mids
    }
    post_node_list = sorted({
        int(node)
        for mask in entry_masks.values()
        for node in entry_nodes[mask]
    })
    post_local = {node: local for local, node in enumerate(post_node_list)}

    print(f"forward_v sbar/{contrast} …")
    with torch.no_grad():
        i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
        v = forward_v(session, params, i_sti, pack=pack)
        contribs = _v_syn_from_source_cells(
            v, params, session.connectome, all_sources,
            np.asarray(post_node_list, dtype=np.int64),
        )

    n_t = int(v.shape[1])
    t_onset = int(pack_t_onset(pack))
    opts = dict((session.train_opts or {}).get("sbar_sti_opts") or {})
    delta_ms = float(opts["delta_ms"])
    delta_ms_pre = float(opts.get("delta_ms_pre", delta_ms))
    sti_end = t_sti_end(t_onset, n_t, opts.get("ms_sti"), delta_ms=delta_ms)
    t = np.arange(n_t)

    n_row = len(post_cells)
    n_col = len(mids)
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(2.0 * n_col, 1.8 * n_row),
        sharex=True, sharey="row", squeeze=False,
    )
    legend_handles = {}
    for row, cell in enumerate(post_cells):
        source_cells = sources_by_post[cell]
        for col, mid in enumerate(mids):
            ax = axes[row, col]
            mask = entry_masks[(cell, mid)]
            if not np.any(mask):
                ax.axis("off")
                continue
            bs = entry_bs[mask]
            locals_ = np.asarray(
                [post_local[int(node)] for node in entry_nodes[mask]],
                dtype=np.int64,
            )
            n_entry = int(mask.sum())
            if np.isclose(float(mid), 0.0):
                a_sti_label = "1"
            else:
                mid_key = str(float(abs(mid))).removesuffix(".0")
                a_sti = a_sti_mid.get(mid_key)
                a_sti_label = "n/a" if a_sti is None else f"{float(a_sti):.4g}"
            ax.set_title(
                f"{_sbar_mid_label(mid)}\nn={n_entry}\na_sti_mid={a_sti_label}",
                fontsize=8, pad=2,
            )
            mark_sti_on(ax, t_onset, sti_end)
            for source in source_cells:
                y = _mean_over_entries(contribs[source], bs, locals_)
                line, = ax.plot(
                    t, y, color=colors[source], linewidth=TRACE_LINE_W,
                    label=source,
                )
                legend_handles.setdefault(source, line)
            y_total = _mean_over_entries(contribs["__total__"], bs, locals_)
            line_total, = ax.plot(
                t, y_total, color=V_READOUT_COLOR, linewidth=TRACE_LINE_W + 0.5,
                label="v_syn", zorder=5,
            )
            legend_handles.setdefault("v_syn", line_total)
            _style_time_axis(
                ax, show_xlabel=(row == n_row - 1), n_t=n_t,
                delta_ms=delta_ms, delta_ms_pre=delta_ms_pre, t_onset=t_onset,
            )
            if col == 0:
                ax.set_ylabel(f"{cell}\nv_syn [mV]", fontsize=8)
            ax.tick_params(labelsize=6)

    handles = [legend_handles[name] for name in list(all_sources) + ["v_syn"]
               if name in legend_handles]
    labels = [h.get_label() for h in handles]
    fig.legend(
        handles, labels, loc="upper center", ncol=min(8, len(handles)),
        fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.02),
    )
    fig.subplots_adjust(top=0.88, bottom=0.10, hspace=0.85, wspace=0.35)
    save_figure(fig, save_path, dpi=100, rasterize=True)
    print(f"wrote {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--save", default=DEFAULT_SAVE)
    parser.add_argument("--contrast", default="bright", choices=("bright", "dark"))
    args = parser.parse_args()
    plot_run(args.run, os.path.abspath(args.save), contrast=args.contrast)


if __name__ == "__main__":
    main()
