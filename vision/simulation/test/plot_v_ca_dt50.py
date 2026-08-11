"""Plot spot center traces: v, ca (Δt=10), and v re-run at Δt=50 for 13 fit cells.

Usage (from ``SimulationCode/``):

    ../.venv/bin/python test/plot_v_ca_dt50.py
    ../.venv/bin/python test/plot_v_ca_dt50.py --show
    ../.venv/bin/python test/plot_v_ca_dt50.py --run-path borst/RUN_NAME
    ../.venv/bin/python test/plot_v_ca_dt50.py --ms-pre 500 --ms-response 1500
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import import_bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch

import training
from figure.plot_run import load_train_opts, session_for_target
from network.build import cell_family_rows, cell_names_in_family_order
from figure.util import TRACE_LW, TRACE_YLIM, save_figure
from neuron.params import DELTA_MS, set_delta_ms
from task.spot.gt import cell_list, resolve_spot_cost_radii, build_spot_center_readout
from task.spot.input import MS_PRE, MS_RESPONSE, spot_from_opts, spot_stimulus_batches
from training.config import PARAMETER_DIR

DEFAULT_RUN = (
    "hp_lp/"
    "28316594-train-nofsteps-1000-tau-hp-init-L1,L2,L4,L5-200"
)
DEFAULT_SAVE = os.path.join(HERE, "v_ca_dt50.png")
DT50_MS = 50.0


def _apply_spot_timing(opts: dict, *, ms_pre: float, ms_response: float) -> dict:
    """Set spot ``ms_pre`` / ``ms_response``; drop legacy ``t_on`` / ``n_t``."""
    out = copy.deepcopy(opts)
    for key in ("spot_bright_stimulus_opts", "spot_dark_stimulus_opts"):
        so = out.get(key)
        if so is None:
            continue
        so["ms_pre"] = float(ms_pre)
        so["ms_response"] = float(ms_response)
        so.pop("t_on", None)
        so.pop("n_t", None)
    return out


@torch.no_grad()
def _fit_center_traces(session, z, *, return_v_delta: bool) -> dict[str, np.ndarray]:
    """Mean center-hex (du=dv=0) trace per fit cell from one ``forward_full``.

    Apply ``out_scale`` like ``model_data_spot`` ca traces (same for v / vΔt
    so the three overlays share amplitude).
    """
    pack = session.primary_pack
    p = training.assign_params(z, list(session.schema), session.backend)
    sig = pack.signal if pack.signal.dim() == 3 else pack.signal.unsqueeze(0)
    if return_v_delta:
        trace_full = training.forward_full(
            session, p, sig, pack=pack, return_v_delta=True,
        )
    else:
        trace_full = training.forward_full(session, p, sig, pack=pack)

    opts = dict((session.train_opts or {}).get(f"{pack.name}_stimulus_opts") or {})
    connectome = session.backend.network
    spot = spot_from_opts(connectome, stimulus_opts=opts)
    batches = spot_stimulus_batches(spot)
    cost_radii = resolve_spot_cost_radii(stimulus_opts=opts)
    batch_idx, node_idx, _r, type_idx, _su, _sv, _du, _dv, center_row = (
        build_spot_center_readout(connectome, batches, cost_radii, pack.cost_extent)
    )
    raw = trace_full[batch_idx, :, node_idx]
    scale = training.out_scale_for_nodes(
        p, torch.as_tensor(node_idx, dtype=torch.long, device=z.device), session.backend,
    )
    scaled = scale[:, None] * raw

    cell_names = list(connectome.cell_names)
    out: dict[str, np.ndarray] = {}
    for name in cell_list:
        name = str(name)
        if name not in cell_names:
            continue
        ti = cell_names.index(name)
        mask = center_row & (type_idx == ti)
        if not np.any(mask):
            continue
        out[name] = scaled[mask].mean(dim=0).detach().cpu().numpy()
    return out


def _session_z_at_delta_ms(base_opts, model, named, cell_names, pair_names, dt_ms: float):
    """``set_delta_ms`` then rebuild spot session with same physical timing."""
    opts = copy.deepcopy(base_opts)

    set_delta_ms(dt_ms)
    for key in ("spot_bright_stimulus_opts", "spot_dark_stimulus_opts"):
        so = opts.get(key)
        if so is None:
            continue
        so["delta_ms"] = float(dt_ms)

    session = training.open_session_from_opts(opts, model=model)
    remapped = training.remap_named_node_values(
        named, cell_names, pair_names, list(session.schema), session.backend,
    )
    schema = training.attach_param_carry(list(session.schema), remapped)
    session = session.with_schema(schema)
    z = training.z_from_node_values(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )
    return session_for_target(session, "spot_bright"), z


def _plot(traces_v, traces_ca, traces_v50, dt10, dt50, save, show):
    present = [str(n) for n in cell_list if str(n) in traces_v]
    groups = [np.array(row) for row in cell_family_rows(present)]
    names = cell_names_in_family_order(present)
    nrows = len(groups)
    ncols = max(len(cell_group) for cell_group in groups)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.2 * ncols, 2.0 * nrows), squeeze=False,
    )
    t10 = np.arange(next(iter(traces_v.values())).shape[0]) * dt10 / 1000.0
    t50 = np.arange(next(iter(traces_v50.values())).shape[0]) * dt50 / 1000.0

    name_set = set(names)
    for r, group in enumerate(groups):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(group):
                ax.axis("off")
                continue
            name = str(group[c])
            if name not in name_set:
                ax.axis("off")
                continue
            ax.plot(t10, traces_v[name], color="C0", lw=TRACE_LW, label="v")
            ax.plot(t10, traces_ca[name], color="C1", lw=TRACE_LW, label="ca")
            if name in traces_v50:
                ax.plot(
                    t50, traces_v50[name], color="C2", lw=TRACE_LW,
                    label=f"v Δt={dt50:g}",
                )
            ax.set_title(name, fontsize=9)
            ax.set_ylim(*TRACE_YLIM)
            ax.axhline(0.0, color="0.7", lw=0.6)
            if r == nrows - 1:
                ax.set_xlabel("t (s)", fontsize=8)
            if c == 0:
                ax.set_ylabel("Δ mV", fontsize=8)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(
        f"spot center: v / ca (Δt={dt10:g} ms) vs v (Δt={dt50:g} ms)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, save)
    print(f"saved {save}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-path", default=DEFAULT_RUN)
    ap.add_argument("--save", default=DEFAULT_SAVE)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--dt50", type=float, default=DT50_MS)
    ap.add_argument(
        "--ms-pre", type=float, default=MS_PRE,
        help="pre-stimulus baseline in ms (default %(default)s)",
    )
    ap.add_argument(
        "--ms-response", type=float, default=MS_RESPONSE,
        help="post-onset ms_response in ms (default %(default)s)",
    )
    args = ap.parse_args()

    run_path = args.run_path
    if not os.path.isabs(run_path):
        run_path = os.path.join(str(PARAMETER_DIR), run_path)
    run_path = os.path.abspath(run_path)

    ms_pre = float(args.ms_pre)
    ms_response = float(args.ms_response)

    set_delta_ms(DELTA_MS)
    raw_opts = load_train_opts(run_path)
    if not raw_opts:
        raise SystemExit(f"missing train_opts.json under {run_path}")
    base_opts = _apply_spot_timing(raw_opts, ms_pre=ms_pre, ms_response=ms_response)
    model = base_opts.get("model")
    session0 = training.open_session_from_opts(base_opts, model=model)

    import training.implement as train_mod
    named, cell_names, pair_names = train_mod.load_best_param_named(run_path)
    remapped = training.remap_named_node_values(
        named, cell_names, pair_names, list(session0.schema), session0.backend,
    )
    schema = training.attach_param_carry(list(session0.schema), remapped)
    session0 = session0.with_schema(schema)
    base_opts = copy.deepcopy(session0.train_opts)

    one10, z10 = _session_z_at_delta_ms(
        base_opts, model, named, cell_names, pair_names, DELTA_MS,
    )
    traces_v = _fit_center_traces(one10, z10, return_v_delta=True)
    traces_ca = _fit_center_traces(one10, z10, return_v_delta=False)

    one50, z50 = _session_z_at_delta_ms(
        base_opts, model, named, cell_names, pair_names, float(args.dt50),
    )
    traces_v50 = _fit_center_traces(one50, z50, return_v_delta=True)

    set_delta_ms(DELTA_MS)
    _plot(
        traces_v, traces_ca, traces_v50,
        DELTA_MS, float(args.dt50),
        args.save, args.show,
    )


if __name__ == "__main__":
    main()
