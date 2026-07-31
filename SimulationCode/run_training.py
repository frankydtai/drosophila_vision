#!/usr/bin/env python
"""Run parameter optimization as described in Borst (2025) section 2.4."""
import argparse
import time

from FiveCol_MedSim_Pytorch import do_many_runs, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nofruns", type=int, default=20)
    parser.add_argument("--nofsteps", type=int, default=10000)
    parser.add_argument("--fname", type=str, default="training_with_Ih.npy")
    args = parser.parse_args()

    print(f"device={device}, nofruns={args.nofruns}, nofsteps={args.nofsteps}, fname={args.fname}")
    t0 = time.time()
    do_many_runs(args.nofruns, args.nofsteps, args.fname)
    print(f"done in {(time.time() - t0) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
