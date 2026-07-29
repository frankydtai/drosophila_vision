"""Re-plot spot model-data with a custom t_on.

Loads a trained run, overrides spot_bright_stimulus_opts.t_on and .maxtime,
re-runs the forward pass, and saves ``model_data_spot_ton<ms>.png`` in the
run dir.

Usage (from ``SimulationCode/``):

    ../.venv/bin/python test/plot_spot_ton.py
    ../.venv/bin/python test/plot_spot_ton.py --run-path conductance/OTHER_RUN
    ../.venv/bin/python test/plot_spot_ton.py --t-on-ms 1500 --maxtime-ms 3000
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

import FiveCol_MedSim_Pytorch as fc
import plot_trained
from training_config import PARAMETER_DIR, run_data_dir

DEFAULT_RUN = (
    "conductance/"
    "28256933-train-target-spot_bright-nofsteps-10000-lr-0.1,0.01,0.001"
)
DEFAULT_T_ON_MS = 1000.0
DEFAULT_POST_ONSET_MS = 1500.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-path", default=DEFAULT_RUN,
        help="run folder under PARAMETER_DIR or absolute path",
    )
    ap.add_argument(
        "--t-on-ms", type=float, default=DEFAULT_T_ON_MS,
        help="stimulus onset in ms (default %(default)s)",
    )
    ap.add_argument(
        "--maxtime-ms", type=float, default=None,
        help="total simulation time in ms (default: t_on + 1500)",
    )
    args = ap.parse_args()

    run_path = args.run_path
    if not os.path.isabs(run_path):
        run_path = os.path.join(str(PARAMETER_DIR), run_path)
    run_path = os.path.abspath(run_path)

    t_on_ms = args.t_on_ms
    maxtime_ms = args.maxtime_ms
    if maxtime_ms is None:
        maxtime_ms = t_on_ms + DEFAULT_POST_ONSET_MS

    opts_path = os.path.join(run_data_dir(run_path), fc.TRAIN_OPTS_FILE)
    with open(opts_path) as f:
        opts = json.load(f)

    orig_stim = opts.get("spot_bright_stimulus_opts") or {}
    dt = orig_stim.get("deltat_ms", 10.0)
    new_t_on = int(t_on_ms / dt)
    new_maxtime = int(maxtime_ms / dt)

    print(f"original t_on={orig_stim.get('t_on')} maxtime={orig_stim.get('maxtime')}  "
          f"dt={dt} ms")
    print(f"override t_on={new_t_on} ({t_on_ms} ms)  "
          f"maxtime={new_maxtime} ({maxtime_ms} ms)")

    for key in ("spot_bright_stimulus_opts", "spot_dark_stimulus_opts"):
        so = opts.get(key)
        if so is not None:
            so["t_on"] = new_t_on
            so["maxtime"] = new_maxtime

    session = fc.open_session_from_opts(opts, model=opts.get("model"))

    import train as train_mod
    named, type_names, pair_names = train_mod.load_best_param_named(run_path)
    remapped = fc.remap_named_unit_values(
        named, type_names, pair_names, list(session.schema), session.backend,
    )
    schema = fc.attach_param_carry(list(session.schema), remapped)
    session = session.with_schema(schema)
    z = fc.unit_values_to_z(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )

    one = plot_trained.session_for_target(session, "spot_bright")

    cost = fc.calc_cost(z, one).item()
    print(f"cost (t_on={t_on_ms:.0f} ms) = {cost:.4f}% of data power")

    make_bundle, plot_data, _plot_all = plot_trained.spot_bundle_fns(one)
    bundle = make_bundle(one, z)

    out_png = os.path.join(run_path, f"model_data_spot_ton{int(t_on_ms)}.png")
    plot_data(
        out_png,
        bundle=bundle,
        title=f"spot_bright model (t_on={t_on_ms:.0f} ms)",
    )
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
