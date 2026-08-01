"""Re-plot spot model-data with custom ``pre_ms`` / ``response_ms``.

Loads a trained run, overrides spot stimulus timing, re-runs the forward
pass, and saves ``model_data_spot_pre<ms>.png`` in the run dir.

Usage (from ``SimulationCode/``):

    ../.venv/bin/python test/plot_spot_ton.py
    ../.venv/bin/python test/plot_spot_ton.py --run-path borst/OTHER_RUN
    ../.venv/bin/python test/plot_spot_ton.py --pre-ms 1500 --response-ms 1500
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
from figure.plot_run import session_for_target, spot_bundle_fns
from training.config import PARAMETER_DIR, run_data_dir

DEFAULT_RUN = (
    "hp_lp/"
    "28307204-train-nofsteps-1000-tau-hp-indi-L1,L2,L4,L5"
)
DEFAULT_PRE_MS = 1000.0
DEFAULT_RESPONSE_MS = 1500.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-path", default=DEFAULT_RUN,
        help="run folder under PARAMETER_DIR or absolute path",
    )
    ap.add_argument(
        "--pre-ms", type=float, default=DEFAULT_PRE_MS,
        help="pre-stimulus baseline in ms (default %(default)s)",
    )
    ap.add_argument(
        "--response-ms", type=float, default=DEFAULT_RESPONSE_MS,
        help="post-onset response window in ms (default %(default)s)",
    )
    args = ap.parse_args()

    run_path = args.run_path
    if not os.path.isabs(run_path):
        run_path = os.path.join(str(PARAMETER_DIR), run_path)
    run_path = os.path.abspath(run_path)

    pre_ms = float(args.pre_ms)
    response_ms = float(args.response_ms)

    opts_path = os.path.join(run_data_dir(run_path), training.TRAIN_OPTS_FILE)
    with open(opts_path) as f:
        opts = json.load(f)

    orig_stim = opts.get("spot_bright_stimulus_opts") or {}
    print(
        f"original pre_ms={orig_stim.get('pre_ms')} "
        f"response_ms={orig_stim.get('response_ms')}  "
        f"dt={orig_stim.get('delta_ms', 10.0)} ms"
    )
    print(f"override pre_ms={pre_ms:g}  response_ms={response_ms:g}")

    for key in ("spot_bright_stimulus_opts", "spot_dark_stimulus_opts"):
        so = opts.get(key)
        if so is not None:
            so["pre_ms"] = pre_ms
            so["response_ms"] = response_ms
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
    z = training.node_values_to_z(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )

    one = session_for_target(session, "spot_bright")

    cost = training.calc_cost(z, one).item()
    print(f"cost (pre_ms={pre_ms:.0f}) = {cost:.4f}% of data power")

    make_bundle, plot_data, _plot_all = spot_bundle_fns(one)
    bundle = make_bundle(one, z)

    out_png = os.path.join(run_path, f"model_data_spot_pre{int(pre_ms)}.png")
    plot_data(
        out_png,
        bundle=bundle,
        title=f"spot_bright model (pre_ms={pre_ms:.0f})",
    )
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
