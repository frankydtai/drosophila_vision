#!/usr/bin/env python3
"""Smoke test: moving-bar target via train.run_training (minimal steps).

Usage:
    ../.venv/bin/python test/run_moving_bar_training.py
    ../.venv/bin/python test/run_moving_bar_training.py --network right_min_neuron1_extent2
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

import network_bootstrap  # noqa: F401
import train
from training_config import MODEL_ALL_CELLS_PNG, MODEL_VS_DATA_PNG


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="right_min_neuron1_extent2",
                    help="built_network run folder name")
    args = ap.parse_args()

    fname, outdir, _session = train.run_training(
        "conductance",
        nofruns=1,
        nofsteps=5,
        lrs=[0.1],
        network=args.network,
        target_list=["moving_bar_bright"],
        sequential=True,
    )
    expected = {
        fname,
        fname.replace(".npy", "") + "_costs.npy",
        fname.replace(".npy", "") + "_final_costs.npy",
        fname.replace(".npy", "") + "_table.csv",
        "model.txt",
        "train_opts.json",
        MODEL_VS_DATA_PNG,
        MODEL_ALL_CELLS_PNG,
        "best_param.npy",
    }
    present = set(os.listdir(outdir))
    missing = expected - present
    assert not missing, f"missing in {outdir}: {missing}"
    print("outdir", outdir)
    print("ok")


if __name__ == "__main__":
    main()
