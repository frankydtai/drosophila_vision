#!/usr/bin/env python
"""Unified training driver for the FiveCol medulla model.

`local` and `gpu` run the EXACT same code path (do_many_runs + make_plots);
they differ only in whether CUDA is disabled. Output folders are named by
model_type only (not by local/gpu). All results of a run land in one folder:

    FiveCol_Parameter/<model_type>/run_<id>/

where <id> is the SLURM job id (under SLURM) or a timestamp otherwise.

    # short LOCAL CPU smoke test (CUDA disabled)
    python run.py local --model_type adaptive --nofsteps 30 --lrs 0.1

    # full GPU training
    python run.py gpu --model_type conductance --nofruns 20 --nofsteps 10000 \
                      --lrs 0.1 0.01 0.001

Import-safe: importing this module does NOT parse argv or touch CUDA, so test
scripts can `import run` and reuse run_training / save_param_tables / etc.
"""
import argparse
import os
import sys
import time

# When executed as a script, run from this file's own directory so `fc` finds
# Circuits/ regardless of where it was launched (no need to cd first). Done
# before importing fc (which loads Circuits/ at import time). NOT done on
# `import run`, so importers keep control of cwd / CUDA_VISIBLE_DEVICES.
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # CLI `local` mode must disable CUDA *before* the model library is imported.
    if len(sys.argv) > 1 and sys.argv[1] == "local":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from FiveCol_MedSim_Pytorch import device, do_many_runs
import FiveCol_MedSim_Pytorch as fc
from plot_trained import plot_param_set, run_dir


def make_plots(fname, outdir):
    """Cost curve + model-vs-data + all-cell-types, all read from / saved to outdir."""
    costs_path = os.path.join(outdir, fname.replace(".npy", "") + "_costs.npy")
    costs = np.load(costs_path) if os.path.exists(costs_path) else None
    plot_param_set(np.load(os.path.join(outdir, fname)), outdir, costs=costs)


def ctype_labels():
    # When a connectome is active its own type vocabulary is the source of truth;
    # otherwise fall back to the 65-type Borst ctype.npy.
    if getattr(fc, "CONNECTOME", None) is not None:
        return np.asarray(fc.CONNECTOME.type_names)
    path = os.path.join(os.path.dirname(os.path.abspath(fc.__file__)), "Circuits", "ctype.npy")
    return np.load(path, allow_pickle=True)


def decompose_params(z_t, model_type):
    """Return (per_cell_cols, global_scalars) for one parameter vector, driven by the
    active schema so any parameter (incl. an added out_scale) shows up automatically.
    per_cell_cols: dict name -> (nofcells,) array;  global_scalars: dict name -> float."""
    n = fc.nofcells
    schema = fc.ADAPTIVE_SCHEMA if model_type == "adaptive" else fc.CONDUCTANCE_SCHEMA
    p = fc.assign_params(z_t, schema)
    cols, glob = {}, {}
    for seg in schema:
        name, v = seg["name"], p[seg["name"]]
        if seg["kind"] == "scalar":
            glob[name] = float(v.item() if torch.is_tensor(v) else v)
        elif seg["kind"] == "output":            # per-cell-type output param (65,)
            cols[name] = v.detach().cpu().numpy()
        else:                                    # full / lamina -> (325,), take one column
            cols[name] = v[:n].detach().cpu().numpy()
    if model_type == "adaptive":
        glob["gate_pivot"] = float(fc.GATE_PIVOT)
    return cols, glob


def write_param_table(z_t, model_type, table_path, extra_cols=None):
    """Write ONE clean rectangular csv (header + 65 rows, uniform columns) so the
    grid-style csv viewer renders it. Global scalars are repeated on every row.
    extra_cols: optional dict name -> (nofcells,) array of additional per-cell columns."""
    cols, glob = decompose_params(z_t, model_type)
    if extra_cols:
        cols.update(extra_cols)
    ctype = ctype_labels()
    cell_names = list(cols.keys())
    glob_names = list(glob.keys())
    with open(table_path, "w") as f:
        f.write("idx,ctype," + ",".join(cell_names + glob_names) + "\n")
        for i in range(fc.nofcells):
            row = ["%.6f" % cols[nm][i] for nm in cell_names] + ["%.6f" % glob[nm] for nm in glob_names]
            f.write("%d,%s," % (i, ctype[i]) + ",".join(row) + "\n")
    return table_path


def save_param_tables(fname, outdir, model_type):
    """Pick the best run in outdir/fname and write <fname>_table.csv into outdir."""
    all_params = np.load(os.path.join(outdir, fname))
    if all_params.ndim == 1:
        all_params = all_params[None, :]
    costs = np.array([fc.calc_cost(torch.tensor(all_params[i], dtype=torch.float64).to(device),
                                   fc.data).item() for i in range(all_params.shape[0])])
    best_i = int(np.argmin(costs))
    z_best = torch.tensor(all_params[best_i], dtype=torch.float64).to(device)
    table_path = os.path.join(outdir, fname.replace(".npy", "") + "_table.csv")
    write_param_table(z_best, model_type, table_path)
    print("wrote table: %s (best run #%d, cost=%.4f)" % (table_path, best_i, costs[best_i]))


def apply_param_modes(model_type, param_modes=None, param_fixes=None):
    """Override per-parameter mode (individual/shared/fixed) on the model's schema.
    Mutates the module-global schema so do_many_runs / calc_cost / tables all agree.
    Returns the active schema."""
    attr = "ADAPTIVE_SCHEMA" if model_type == "adaptive" else "CONDUCTANCE_SCHEMA"
    schema = fc.apply_modes(getattr(fc, attr), param_modes, param_fixes)
    setattr(fc, attr, schema)
    summary = ", ".join(f"{s['name']}:{fc.seg_mode(s)}({fc.seg_ntrain(s)})" for s in schema)
    print("param modes -> " + summary)
    return schema


def run_training(model_type, nofruns, nofsteps, lrs, fname=None, outdir=None,
                 no_plots=False, param_modes=None, param_fixes=None,
                 connectome=None, multi_column=False, share_edges=False,
                 sequential=None):
    """Full training pipeline (do_many_runs + tables + plots). Returns (fname, outdir)."""
    fc.MODEL_TYPE = model_type
    # Optional connectome backend (rebuilds CONN + schema + target). Must run
    # BEFORE apply_param_modes so the mode overrides act on the connectome schema.
    if connectome:
        fc.use_connectome(connectome, multi_column=multi_column,
                          share_edges=share_edges, sequential=sequential)
    apply_param_modes(model_type, param_modes, param_fixes)
    suffix = "" if model_type == "conductance" else f"_{model_type}"
    fname = fname or f"training{suffix or '_with_Ih'}.npy"
    outdir = outdir or run_dir(model_type)

    print(f"device={device}, model_type={model_type}, nofruns={nofruns}, nofsteps={nofsteps}, "
          f"lrs={lrs}, nparams={fc.schema_nparams(fc.active_schema())}, fname={fname}, outdir={outdir}")
    t0 = time.time()
    do_many_runs(nofruns, nofsteps, fname, lrs=lrs, outdir=outdir)
    print(f"done in {(time.time() - t0) / 3600:.2f} hours")

    save_param_tables(fname, outdir, model_type)
    if not no_plots:
        make_plots(fname, outdir)
    return fname, outdir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p, nofruns, nofsteps):
        p.add_argument("--model_type", default="conductance",
                       choices=["conductance", "adaptive"])
        p.add_argument("--nofruns", type=int, default=nofruns)
        p.add_argument("--nofsteps", type=int, default=nofsteps)
        p.add_argument("--lrs", type=float, nargs="+", default=[0.1, 0.01, 0.001],
                       help="learning-rate stages; each runs for --nofsteps steps")
        p.add_argument("--fname", default=None,
                       help="params filename (default derived from --model_type)")
        p.add_argument("--outdir", default=None,
                       help="output dir (default derived from --model_type)")
        p.add_argument("--no_plots", action="store_true",
                       help="skip automatic plotting after training")
        p.add_argument("--mode", nargs="+", default=[], metavar="NAME=MODE",
                       help="per-param mode override, e.g. --mode out_scale=shared "
                            "inp_gain=fixed (MODE in individual|shared|fixed)")
        p.add_argument("--fix", nargs="+", default=[], metavar="NAME=VALUE",
                       help="hold a param fixed at VALUE (implies fixed mode), "
                            "e.g. --fix Ih_midv=-50 out_scale=1.0")
        p.add_argument("--connectome", default=None, metavar="PATH",
                       help="path to a connectome network.json; enables the "
                            "ScatterConn backend (default: Borst 5-column dense)")
        p.add_argument("--multi_column", action="store_true",
                       help="connectome multi-column training (7-shift radial target)")
        p.add_argument("--share_edges", action="store_true",
                       help="full-graph tiling: 43 edge-sharing tiles (default 31 disjoint)")
        p.add_argument("--sequential", action="store_true", default=None,
                       help="force CPU-sequential multi-column cost (default: auto)")

    add_common(sub.add_parser("local", help="short local CPU run (CUDA disabled)"), 1, 100)
    add_common(sub.add_parser("gpu", help="full training run"), 20, 10000)
    return parser.parse_args()


def parse_kv(tokens, cast=str):
    """Parse ['a=1','b=2'] -> {'a': cast('1'), 'b': cast('2')}."""
    out = {}
    for tok in tokens:
        if "=" not in tok:
            raise SystemExit(f"expected NAME=VALUE, got {tok!r}")
        name, val = tok.split("=", 1)
        out[name] = cast(val)
    return out


def main():
    args = parse_args()
    param_modes = parse_kv(args.mode)
    param_fixes = parse_kv(args.fix, float)
    outdir = run_dir(args.model_type, parent=args.outdir)
    run_training(args.model_type, args.nofruns, args.nofsteps, args.lrs,
                 fname=args.fname, outdir=outdir, no_plots=args.no_plots,
                 param_modes=param_modes, param_fixes=param_fixes,
                 connectome=args.connectome, multi_column=args.multi_column,
                 share_edges=args.share_edges, sequential=args.sequential)


if __name__ == "__main__":
    main()
