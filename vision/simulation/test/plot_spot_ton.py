"""Re-plot spot model-gt with custom ``ms_pre`` / ``ms_response``.

Loads a trained run, overrides spot sti timing, re-runs the forward
pass, and saves ``model_data_spot_pre<ms>.png`` in the run dir.

Usage (from ``SimulationCode/``):

    ../.venv/bin/python test/plot_spot_ton.py
    ../.venv/bin/python test/plot_spot_ton.py --run-path borst/OTHER_RUN
    ../.venv/bin/python test/plot_spot_ton.py --ms-pre 1500 --ms-response 1500
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import import_bootstrap  # noqa: F401
import training
from figure.plot_run import session_for_target, spot_readout_fns
from training.config import PARAMETER_DIR, run_data_dir

DEFAULT_RUN = (
    "hp_lp/"
    "28307204-train-nofsteps-1000-tau-hp-indi-L1,L2,L4,L5"
)
DEFAULT_MS_PRE = 1000.0
DEFAULT_MS_RESPONSE = 1500.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-path", default=DEFAULT_RUN,
        help="run folder under PARAMETER_DIR or absolute path",
    )
    ap.add_argument(
        "--ms-pre", type=float, default=DEFAULT_MS_PRE,
        help="pre-sti baseline in ms (default %(default)s)",
    )
    ap.add_argument(
        "--ms-response", type=float, default=DEFAULT_MS_RESPONSE,
        help="post-onset ms_response in ms (default %(default)s)",
    )
    args = ap.parse_args()

    run_path = args.run_path
    if not os.path.isabs(run_path):
        run_path = os.path.join(str(PARAMETER_DIR), run_path)
    run_path = os.path.abspath(run_path)

    ms_pre = float(args.ms_pre)
    ms_response = float(args.ms_response)

    opts_path = os.path.join(run_data_dir(run_path), training.TRAIN_OPTS_FILE)
    with open(opts_path) as f:
        opts = json.load(f)

    orig_sti_opts = opts.get("spot_bright_sti_opts") or {}
    print(
        f"original ms_pre={orig_sti_opts.get('ms_pre')} "
        f"ms_response={orig_sti_opts.get('ms_response')}  "
        f"dt={orig_sti_opts.get('delta_ms', 10.0)} ms"
    )
    print(f"override ms_pre={ms_pre:g}  ms_response={ms_response:g}")

    for key in ("spot_bright_sti_opts", "spot_dark_sti_opts"):
        so = opts.get(key)
        if so is not None:
            so["ms_pre"] = ms_pre
            so["ms_response"] = ms_response
            so.pop("t_on", None)
            so.pop("n_t", None)

    session = training.open_session_from_opts(opts, model=opts.get("model"))

    import training.implement as train_mod
    named, cell_names, pair_names = train_mod.load_best_param_named(run_path)
    remapped = training.remap_named_node_values(
        named, cell_names, pair_names, list(session.schema), session.backend,
    )
    schema = training.attach_param_carry(list(session.schema), remapped)
    session = session.with_schema(schema)
    z = training.z_from_node_values(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )

    one = session_for_target(session, "spot_bright")

    cost = training.calc_cost(z, one).item()
    print(f"cost (ms_pre={ms_pre:.0f}) = {cost:.4f}% of gt power")

    make_readout, plot_gt, _plot_all = spot_readout_fns(one)
    readout = make_readout(one, z)

    out_png = os.path.join(run_path, f"model_spot_pre{int(ms_pre)}.png")
    plot_gt(
        out_png,
        readouts={'bright': readout},
        title=f"spot_bright model (ms_pre={ms_pre:.0f})",
    )
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
