#!/usr/bin/env python3
"""Smoke test: 16 moving bars -> photoreceptor current, then one forward pass."""
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
import FiveCol_MedSim_Pytorch as fc
import Medulla_Library as ml
from connectome_io import DEFAULT_NETWORK_RUN, resolve_network_json
from network.stimulus import build_moving_bar_signals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default=DEFAULT_NETWORK_RUN,
                    help="built_network run folder name")
    args = ap.parse_args()

    mb = fc.load_network_backend(str(resolve_network_json(args.network)), dev="cpu")
    session = fc.open_session(fc.make_train_opts(
        backend="network", target_list=["moving_bar_bright"], network=mb.network,
        multi_column=False, sequential=True, dev="cpu",
    ), "conductance")
    pack = session.primary_pack
    sig = pack.signal
    maxtime = int(pack.signal.shape[1])
    C = session.backend.network
    T = build_moving_bar_signals(C, t_on=fc.t_on, device="cpu")

    z = fc.guess_initial_params(session)
    schema = list(session.schema)
    out = fc._run_conductance_full(session, fc.assign_params(z, schema, session.backend), sig)
    print("signal", tuple(sig.shape))
    print("forward", tuple(out.shape))
    print("field_deg", T.info["field_deg"])
    print("maxtime", maxtime, f"sweep={T.info['sweep_steps']} steps ({T.info['sweep_time_s']:.2f} s)")
    print("n_sti_columns", T.info["n_sti_columns"])
    assert sig.shape == (16, maxtime, session.backend.n_units)
    assert out.shape == (16, maxtime - fc.t_on, session.backend.n_units)
    assert maxtime < ml.IMPULSE_MAXTIME
    nz = int((sig.abs().sum(dim=(1, 2)) > 0).sum())
    print(f"nonzero batches: {nz}/16")
    print("ok")


if __name__ == "__main__":
    main()
