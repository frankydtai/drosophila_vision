"""Plot individual T4a traces selected by the original moving-bar figure."""
from pathlib import Path
import os
import sys
import argparse

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM.parent))
os.chdir(SIM)
import import_bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import train
from figure.plot import load_best
from figure.mbar import _traces_from_forward, _windows_by_b
from task.mbar.pack import bar_specs_from_task
from task.sbar.sti_geo import node_us_vs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("right", "left"), default="right")
    parser.add_argument("--run", type=Path, default=SIM / "0_runs/hp_lp/34224104-run_init_from-n_iter-50-tasks-mbar-contrasts-bright-part_cost_scales-mbar-0,mbar_DSI-1")
    args = parser.parse_args()
    run = args.run.resolve()
    torch.set_num_threads(2)
    session, z, _ = load_best(str(run), verbose=True)
    pack = session.packs["mbar"]["bright"]
    with torch.no_grad():
        params = train.params_from_z(z, session)
        trace = train.forward_pack(session, params, pack.i_sti, pack).cpu().numpy()
    specs = bar_specs_from_task(session, "mbar", "bright")
    tokens = [s.token for s in specs]
    b = next(i for i, s in enumerate(specs) if s.direction == args.direction)
    token = tokens[b]
    readout, cells, _, _, _, _ = _traces_from_forward(
        session, "mbar", "bright", trace, specs, tokens,
    )
    nodes = np.flatnonzero((readout.cell_idxs == cells.index("T4a")) & (readout.t0_bn[b] >= 0))
    us, vs = node_us_vs(session.connectome)
    nodes = nodes[np.lexsort((vs[nodes], us[nodes]))]
    lengths = [readout.before_t[t] + readout.after_t[t] + 1 for t in tokens]
    windows = _windows_by_b(trace, readout.t0_bn, lengths)[b]
    values = windows[nodes]
    time_ms = (np.arange(values.shape[1]) - readout.before_t[token]) * session.delta_ms
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values.mean(axis=0), readout.ca_mean_cell["T4a", token], atol=1e-10)
    assert len(nodes) == readout.ca_n["T4a", token]
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, len(nodes)))
    rows = []
    for node, voltage, color in zip(nodes, values, colors):
        ax.plot(time_ms, voltage, color=color, lw=1.2, alpha=0.85,
                label=f"node {node} ({us[node]},{vs[node]})")
        rows.append(pd.DataFrame(dict(node=node, u=int(us[node]), v=int(vs[node]),
                                      time_ms=time_ms, voltage_mv=voltage)))
    ax.axvline(0, color="0.4", lw=0.9, ls="--")
    entries = np.flatnonzero((pack.entry_bs.cpu().numpy() == b)
                             & np.isin(pack.entry_nodes.cpu().numpy(), nodes))
    bounds = []
    for e in entries:
        node = int(pack.entry_nodes[e])
        first = int(readout.t0_bn[b, node]) + readout.before_t[token]
        lo = (int(pack.dsi_t0s[e]) - first) * session.delta_ms
        hi = (int(pack.dsi_t0s[e] + pack.dsi_n_ts[e]) - 1 - first) * session.delta_ms
        bounds.append((lo, hi))
    if len(set(bounds)) == 1:
        lo, hi = bounds[0]
        ax.axvspan(lo, hi, color="0.5", alpha=0.08, label="DSI window")
    preference = "PD" if args.direction == "right" else "ND"
    ax.set_title(f"Run {run.name.split('-')[0]} | T4a {args.direction} bright moving bar ({preference})\n"
                 f"All {len(nodes)} individual neuron traces; same selection and alignment as mbar_gt_v.png")
    ax.set_xlabel("Time from local first stimulus (ms)")
    ax.set_ylabel("Membrane potential (mV)")
    ax.set_xlim(time_ms[0], time_ms[-1])
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    fig.tight_layout()
    stem = run / f"mbar_t4a_{args.direction}_bright_individual_traces"
    fig.savefig(stem.with_suffix(".png"), dpi=170)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    pd.concat(rows, ignore_index=True).to_csv(stem.with_suffix(".csv"), index=False)
    print(f"Validated {len(nodes)} individual traces against the original mean")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
