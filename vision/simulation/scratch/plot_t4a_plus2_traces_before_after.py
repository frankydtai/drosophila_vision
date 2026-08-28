"""Plot T4a mid=+2 and mid=+0 pack traces from two trained runs.

Run from ``vision/simulation``::

    ../.venv/bin/python scratch/plot_t4a_plus2_traces_before_after.py
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
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import figure.plot as figure_plot  # noqa: E402
from neuron.readout import pack_traces  # noqa: E402
from task.sbar.sti_geo import node_us_vs  # noqa: E402
from train.cost import (  # noqa: E402
    _entry_cost_scales,
    _session_part_scale_sum,
    entries_by_part,
    forward_pack,
    gt_affine_from_pack,
)
from train.param import params_from_z  # noqa: E402


DEFAULT_BEFORE_RUN = (
    "hp_lp/29677084-run_init_from-n_iter-100-tasks-sbar-"
    "params.a_gt.mode-frozen-params.bias_gt.mode-frozen-"
    "params.a_in.mode-frozen-params.a_out.mode-frozen-"
    "params.e_leak.mode-frozen-params.v_th.mode-frozen-"
    "params.tau_lp.mode-frozen-params.tau_hp_rise.mode-frozen-par"
)
DEFAULT_AFTER_RUN = "hp_lp/29677691-run-n_iter-500-tasks-sbar"
DEFAULT_SAVE = os.path.join(HERE, "t4a_mid_plus2_plus0_traces_before_after.png")
MIDS = (2, 0)


def _readout(session, z, mid):
    """Return one mid pack's traces, affine GT, coordinates, and dispersion."""
    params = params_from_z(z, session)
    pack = session.packs["sbar"]["bright"]
    trace = forward_pack(session, params, pack.i_sti, pack)
    values = pack_traces(trace, pack)
    gts = pack.gts
    if pack.cost_ts is not None:
        cost_ts = pack.cost_ts.to(values.device)
        values = values.index_select(1, cost_ts)
        gts = gts.index_select(1, cost_ts)
        ms = cost_ts.detach().cpu().numpy() * float(session.delta_ms)
    else:
        ms = np.arange(values.shape[1], dtype=float) * float(session.delta_ms)

    a_gt, bias_gt = gt_affine_from_pack(params, pack, session)
    part_idxs, part_keys = entries_by_part(pack)
    part_key = f"sbar_bright_T4a_mid{int(mid):+d}"
    entries = torch.nonzero(
        part_idxs == part_keys.index(part_key), as_tuple=False,
    ).reshape(-1)
    values = values[entries]
    targets = a_gt[entries, None] * gts[entries] + bias_gt[entries, None]
    nodes = pack.entry_nodes[entries].detach().cpu().numpy()
    us, vs = node_us_vs(session.connectome)
    node_us = np.asarray(us)[nodes]
    node_vs = np.asarray(vs)[nodes]

    weights = _entry_cost_scales(pack)[entries]
    errors = values - targets
    mean_error = (weights[:, None] * errors).sum(0) / weights.sum()
    dispersion = (
        weights[:, None]
        * (errors - mean_error[None, :]) ** 2
        / (a_gt[entries, None] ** 2).clamp(min=torch.finfo(a_gt.dtype).tiny)
    ).sum() / _session_part_scale_sum(session)

    order = np.argsort(node_us)
    return {
        "ms": np.asarray(ms),
        "traces": values.detach().cpu().numpy()[order],
        "gt": targets.mean(0).detach().cpu().numpy(),
        "nodes": nodes[order],
        "us": node_us[order],
        "vs": node_vs[order],
        "dispersion": float(dispersion.item()),
    }


def _plot_panel(ax, data, title, cmap, norm, *, show_xlabel, show_ylabel):
    for trace, node, u, v in zip(
        data["traces"], data["nodes"], data["us"], data["vs"],
    ):
        ax.plot(
            data["ms"], trace,
            color=cmap(norm(float(u))), alpha=0.72, linewidth=0.9,
            label=f"node {node}, ({u},{v})",
        )
    mean_trace = data["traces"].mean(axis=0)
    sem_trace = data["traces"].std(axis=0, ddof=1) / np.sqrt(data["traces"].shape[0])
    ax.fill_between(
        data["ms"], mean_trace - sem_trace, mean_trace + sem_trace,
        color="red", alpha=0.20, linewidth=0,
        label="mean ± SEM", zorder=3,
    )
    ax.plot(
        data["ms"], mean_trace,
        color="red", linewidth=2.5,
        label=f"mean of {data['traces'].shape[0]} T4a traces", zorder=4,
    )
    ax.plot(
        data["ms"], data["gt"],
        color="black", linestyle="--", linewidth=2.0,
        label="affine GT", zorder=5,
    )
    ax.axvspan(0.0, 160.0, color="0.85", alpha=0.35, zorder=0)
    ax.axvline(160.0, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_title(
        f"{title} (n={data['traces'].shape[0]})\n"
        f"dispersion contribution = {data['dispersion']:.2f}"
    )
    if show_xlabel:
        ax.set_xlabel("time after onset [ms]")
    if show_ylabel:
        ax.set_ylabel("T4a membrane potential")
    ax.grid(alpha=0.18, linewidth=0.6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-run", default=DEFAULT_BEFORE_RUN)
    parser.add_argument("--after-run", default=DEFAULT_AFTER_RUN)
    parser.add_argument("--save", default=DEFAULT_SAVE)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    before_dir = figure_plot.resolve_run_dir(args.before_run)
    before_session, before_z, _ = figure_plot.load_best(before_dir, verbose=False)
    if not torch.is_tensor(before_z):
        before_z = torch.tensor(
            np.asarray(before_z), dtype=before_session.sim_dtype,
            device=before_session.device,
        )

    after_dir = figure_plot.resolve_run_dir(args.after_run)
    after_session, after_z, _ = figure_plot.load_best(after_dir, verbose=False)
    if not torch.is_tensor(after_z):
        after_z = torch.tensor(
            np.asarray(after_z), dtype=after_session.sim_dtype,
            device=after_session.device,
        )

    with torch.no_grad():
        before = {mid: _readout(before_session, before_z, mid) for mid in MIDS}
        after = {mid: _readout(after_session, after_z, mid) for mid in MIDS}

    all_us = np.concatenate([
        data["us"] for run_data in (before, after) for data in run_data.values()
    ])
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=float(all_us.min()), vmax=float(all_us.max()))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), sharex=True, sharey="row")
    for row, mid in enumerate(MIDS):
        _plot_panel(
            axes[row, 0], before[mid],
            f"mid={mid:+d} — Before: run 29677084", cmap, norm,
            show_xlabel=(row == len(MIDS) - 1), show_ylabel=True,
        )
        _plot_panel(
            axes[row, 1], after[mid],
            f"mid={mid:+d} — After: run 29677691 (500 iterations)", cmap, norm,
            show_xlabel=(row == len(MIDS) - 1), show_ylabel=False,
        )

    handles, labels = axes[0, 1].get_legend_handles_labels()
    axes[0, 1].legend(handles[-3:], labels[-3:], loc="best", fontsize=8)
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=axes,
        pad=0.02, fraction=0.035,
    )
    colorbar.set_label("T4a neuron u coordinate")
    fig.suptitle("sbar bright T4a pack traces: mid=+2 and mid=+0")
    fig.subplots_adjust(
        left=0.07, right=0.84, bottom=0.08, top=0.91,
        wspace=0.10, hspace=0.30,
    )

    save_path = os.path.abspath(args.save)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180)
    print(f"wrote {save_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
