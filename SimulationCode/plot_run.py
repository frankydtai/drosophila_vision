#!/usr/bin/env python
"""Re-plot a FiveCol parameter run (like vision ``figure.plot_run``).

Loads ``best_parameter.npy`` from a run folder under ``FiveCol_Parameter/``,
writes ``params.png``, ``model_vs_data.png``, and ``model_all_cells.png``.

Usage (from ``SimulationCode/``, project ``.venv``)::

    ../vision/.venv/bin/python plot_run.py
    ../vision/.venv/bin/python plot_run.py with_Ih
    ../vision/.venv/bin/python plot_run.py FiveCol_Parameter/with_Ih
    ../vision/.venv/bin/python plot_run.py /abs/path/to/with_Ih --all-cells-params
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import Medulla_Library as ml
import blindschleiche_py3 as bs
from FiveCol_MedSim_Pytorch import calc_cost, data, device, nofcells
import run_local_cpu_test as rlt

PARAMETER_ROOT = HERE / "FiveCol_Parameter"
DEFAULT_RUN = "with_Ih"
BEST_FNAME = "best_parameter.npy"


def resolve_run_dir(run_path: str) -> Path:
    """Resolve a run folder under ``FiveCol_Parameter/`` or an absolute path."""
    p = Path(run_path).expanduser()
    if p.is_file() and p.name == BEST_FNAME:
        return p.parent.resolve()
    if not p.is_absolute():
        cand = (HERE / p).resolve()
        if cand.is_dir():
            return cand
        cand = (PARAMETER_ROOT / p).resolve()
        if cand.is_dir():
            return cand
        raise SystemExit(f"run folder not found: {run_path!r}")
    p = p.resolve()
    if not p.is_dir():
        raise SystemExit(f"run folder not found: {p}")
    return p


def load_best_z(run_dir: Path) -> np.ndarray:
    path = run_dir / BEST_FNAME
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    z = np.load(path)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.size != 138:
        raise SystemExit(f"expected 138 params in {path}, got shape {z.shape}")
    return z


def plot_params(
    z: np.ndarray,
    path: Path,
    *,
    all_cells: bool = False,
    title: str = "",
) -> None:
    """Inp/out gain bars + Ih activation (Borst ``plot_params`` layout)."""
    ctype = np.asarray(np.load(HERE / "Circuits" / "ctype.npy", allow_pickle=True), dtype=str)
    cell_list = ml.cell_list
    cell_index = ml.cell_index

    if all_cells:
        plot_index = np.arange(nofcells)
        plot_list = ctype
    else:
        plot_index = cell_index
        plot_list = cell_list

    max_num = int(plot_index.shape[0])
    my_cmap = plt.get_cmap("viridis")
    fontsize_legend = 8

    fig = plt.figure(figsize=(7, 11))
    for i in range(2):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.bar(
            np.arange(max_num),
            z[plot_index + i * 65],
            color=my_cmap(np.arange(max_num) / float(max_num)),
        )
        if all_cells:
            ax.set_xticks(np.arange(max_num))
            ax.set_xticklabels(plot_list, rotation="vertical", fontsize=6)
        else:
            ax.set_xticks(np.arange(max_num))
            ax.set_xticklabels(list(plot_list))
        ax.set_yscale("log")
        ax.set_ylim(0.05, 500)
        ax.set_ylabel("input gain" if i == 0 else "output gain")
        if i == 0 and title:
            ax.set_title(title)

    ax = fig.add_subplot(3, 1, 3)
    ih_gmax = z[130:135]
    ih_midv = float(z[135])
    ih_slope = float(z[136])
    tau_midv = float(z[137])
    vm = np.arange(100) - 100
    ih_ss = 1.0 / (1.0 + np.exp((ih_midv - vm) * ih_slope))
    tau = 1.5 / (np.exp(-0.1 * (vm - tau_midv)) + np.exp(0.1 * (vm - tau_midv))) + 0.1
    ax.plot(vm, ih_ss, label="Ih Activation", linewidth=3)
    ax.plot(vm, tau, label="Ih time constant [s]", linewidth=3)
    ax.set_xlabel("membrane potential [mV]")
    ax.legend(loc=1, frameon=False, fontsize=fontsize_legend)
    ax.text(-20, 0.7, f"Ih_midv  = {int(ih_midv * 100) / 100.0}", fontsize=fontsize_legend)
    ax.text(-20, 0.6, f"Ih_slope = {int(ih_slope * 100) / 100.0}", fontsize=fontsize_legend)
    ax.text(-20, 0.5, f"tau_midv = {int(tau_midv * 100) / 100.0}", fontsize=fontsize_legend)

    bs.setmyaxes(0.2, 0.2, 0.2, 0.1)
    plt.bar(np.arange(5), ih_gmax)
    plt.xticks(np.arange(5), ["L1", "L2", "L3", "L4", "L5"], fontsize=fontsize_legend)
    plt.yticks(np.arange(5) * 20, np.arange(5) * 20, fontsize=fontsize_legend)
    plt.title("Ih_gmax", fontsize=fontsize_legend)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_run(
    run_dir: Path,
    *,
    all_cells_params: bool = False,
    skip_all_cells: bool = False,
) -> None:
    z_np = load_best_z(run_dir)
    z = torch.tensor(z_np, dtype=torch.float64, device=device)
    cost = float(calc_cost(z, data).item())
    title = f"{run_dir.name}, cost {cost:.2f}% of data power"
    print(f"outdir={run_dir}")
    print(f"params={run_dir / BEST_FNAME}")
    print(f"device={device}  cost={cost:.4f}")

    params_path = run_dir / "params.png"
    model_path = run_dir / "model_vs_data.png"
    all_path = run_dir / "model_all_cells.png"

    plot_params(z_np, params_path, all_cells=all_cells_params, title=title)
    print(f"saved: {params_path}")

    rlt.plot_model_vs_data(
        z, str(model_path), n_steps=0, title=f"Model vs data ({title})"
    )
    print(f"saved: {model_path}")

    if not skip_all_cells:
        rlt.plot_all_celltypes(
            z, str(all_path), n_steps=0, title=f"All cell types ({title})"
        )
        print(f"saved: {all_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_path",
        nargs="?",
        default=DEFAULT_RUN,
        help=(
            "Run folder under FiveCol_Parameter/ or absolute path "
            f"(default: {DEFAULT_RUN})"
        ),
    )
    ap.add_argument(
        "--all-cells-params",
        action="store_true",
        help="params.png: all 65 cell types instead of the 13 fit cells",
    )
    ap.add_argument(
        "--skip-all-cells",
        action="store_true",
        help="do not write model_all_cells.png (faster)",
    )
    args = ap.parse_args(argv)
    run_dir = resolve_run_dir(args.run_path)
    plot_run(
        run_dir,
        all_cells_params=args.all_cells_params,
        skip_all_cells=args.skip_all_cells,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
