#!/usr/bin/env python3
"""Smoke test: 16 moving bars -> photoreceptor current, then one forward pass.

Usage:
    ../.venv/bin/python test/run_moving_bar_probe.py
    ../.venv/bin/python test/run_moving_bar_probe.py --network path/to/network.json
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import FiveCol_MedSim_Pytorch as fc
from network.stimulus import build_moving_bar_signals
from connectome_io import NETWORK_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--network",
        default=str(NETWORK_DIR / "right_min_neuron1_extent2" / "network.json"),
    )
    args = ap.parse_args()

    fc.use_network(args.network, multi_column=False, sequential=True, dev="cpu")
    T = build_moving_bar_signals(fc.NETWORK, t_on=fc.t_on, device="cpu")
    fc.signal = T.signal
    maxtime = int(T.info["maxtime"])

    z = fc.guess_initial_params()
    out = fc._run_conductance_full(fc.assign_params(z, fc.CONDUCTANCE_SCHEMA), fc.signal)
    print("signal", tuple(fc.signal.shape))
    print("forward", tuple(out.shape))
    print("field_deg", T.info["field_deg"])
    print("maxtime", maxtime, f"sweep={T.info['sweep_steps']} steps ({T.info['sweep_time_s']:.2f} s)")
    print("n_photo_columns", T.info["n_photo_columns"])
    assert fc.signal.shape == (16, maxtime, fc.CONN.n_units)
    assert out.shape == (16, maxtime - fc.t_on, fc.CONN.n_units)
    assert maxtime < fc.maxtime, "moving-bar horizon should be shorter than Borst IMPULSE_MAXTIME"
    nz = int((fc.signal.abs().sum(dim=(1, 2)) > 0).sum())
    print(f"nonzero batches: {nz}/16")
    print("ok")


if __name__ == "__main__":
    main()
