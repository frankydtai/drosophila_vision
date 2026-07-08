#!/usr/bin/env python3
"""Tests for network.moving_bar_target."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import network_bootstrap  # noqa: F401
import torch
from connectome_io import NETWORK_DIR
from network.construction import load_network
from network.moving_bar_target import (
    ND_IDX,
    PD_IDX,
    build_moving_bar_target,
    load_fig1_traces,
)
from training_config import COST_WINDOW_STEPS


def test_fig1_traces_shape():
    fig1 = load_fig1_traces()
    assert len(fig1) == 16
    for arr in fig1.values():
        assert arr.shape == (COST_WINDOW_STEPS,)


def test_build_target_extent2():
    path = NETWORK_DIR / "right_min_neuron1_extent2" / "network.json"
    C = load_network(path, device="cpu")
    T = build_moving_bar_target(C, device="cpu", use_cache=True, contrasts=("bright",))
    assert T.signal.shape[0] == 8
    assert T.signal.shape[1] == T.maxtime
    assert T.data.shape[1] == COST_WINDOW_STEPS
    assert T.cost_t0.shape == T.readout_batch.shape == T.readout_unit.shape
    assert int(T.cost_t0.min()) >= 0
    assert int(T.cost_t0.max()) + COST_WINDOW_STEPS <= T.maxtime
    assert T.info["n_sti_columns"] == 19
    assert T.info["n_cost"] > 0
    assert T.info["skipped_orthogonal"] > 0
    assert T.cost_pd_nd.shape == (T.info["n_cost"],)
    assert int(T.info["n_cost_pd"]) + int(T.info["n_cost_nd"]) == T.info["n_cost"]
    assert int(T.info["n_cost_pd"]) > 0
    assert int(T.info["n_cost_nd"]) > 0


def test_expand_loss_weights_moving_bar_pd():
    import FiveCol_MedSim_Pytorch as fc
    out = fc.expand_loss_weights({"PD": 2.0})
    assert out["moving_bar_bright_PD"] == 2.0
    assert out["moving_bar_dark_PD"] == 2.0
    assert "moving_bar_bright_ND" not in out


def test_calc_cost_parts_pd_nd_split():
    import FiveCol_MedSim_Pytorch as fc
    path = str(NETWORK_DIR / "right_min_neuron1_extent2" / "network.json")
    mb = fc.load_network_backend(path, dev="cpu")
    session = fc.open_session(fc.make_train_opts(
        backend="network", target_list=["moving_bar_bright"], network=mb.network,
        sequential=True, dev="cpu",
    ), "conductance")
    pack = session.pack_for("moving_bar_bright")
    z = fc.guess_initial_params(session)
    parts = fc.calc_cost_parts(z, session)
    assert set(parts) == {"moving_bar_bright_PD", "moving_bar_bright_ND"}
    assert float(parts["moving_bar_bright_PD"]) >= 0.0
    assert float(parts["moving_bar_bright_ND"]) >= 0.0
    total = float(fc.calc_cost(z, session).item())
    manual = sum(
        float(parts[k].item()) * float(session.loss_weights.get(k, 1.0))
        for k in parts
    )
    assert abs(total - manual) < 1e-6
    assert pack.cost_pd_nd is not None
    assert int((pack.cost_pd_nd == PD_IDX).sum()) > 0
    assert int((pack.cost_pd_nd == ND_IDX).sum()) > 0


def test_build_target_cost_extent():
    path = NETWORK_DIR / "right_min_neuron1_extent2" / "network.json"
    C = load_network(path, device="cpu")
    T = build_moving_bar_target(C, device="cpu", use_cache=True, cost_extent=0)
    assert T.info["cost_extent"] == 0
    assert T.info["n_cost_columns"] == 1
    assert T.info["cost_column_uv"] == (0, 0)
    assert T.info["n_cost"] < 200
    assert T.data.shape[0] == T.info["n_cost"]


def test_resolve_cost_extent_by_target():
    import FiveCol_MedSim_Pytorch as fc
    out = fc.resolve_cost_extent_by_target(
        ["moving_bar_bright", "moving_bar_dark"],
        2,
        {"moving_bar_bright": 0},
    )
    assert out == {"moving_bar_bright": 0, "moving_bar_dark": 2}


def test_cost_extent_requires_network():
    import train as train_mod
    import argparse
    args = argparse.Namespace(
        model_type="conductance",
        nofruns=1,
        nofsteps=5,
        lrs="0.1",
        fname=None,
        outdir=None,
        mode="",
        fix="",
        ih_off="on",
        per_type=False,
        network=None,
        target="moving_bar_bright",
        loss_weight="",
        cost_extent="0",
        shift=False,
        i_baseline="",
        i_bright="",
        i_dark="",
    )
    try:
        train_mod.training_kwargs_from_args(args)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "requires --network" in str(exc)


def test_use_network_moving_bar_cost():
    import FiveCol_MedSim_Pytorch as fc
    path = str(NETWORK_DIR / "right_min_neuron1_extent2" / "network.json")
    mb = fc.load_network_backend(path, dev="cpu")
    session = fc.open_session(fc.make_train_opts(
        backend="network", target_list=["moving_bar_bright"], network=mb.network,
        sequential=True, dev="cpu",
    ), "conductance")
    pack = session.pack_for("moving_bar_bright")
    assert list(session.target_list) == ["moving_bar_bright"]
    assert pack.cost_t0 is not None
    assert pack.data.shape[1] == COST_WINDOW_STEPS
    z = fc.guess_initial_params(session)
    cost = float(fc.calc_cost(z, session))
    assert cost >= 0.0


def test_readout_window_pre_ton_zero():
    import FiveCol_MedSim_Pytorch as fc
    path = str(NETWORK_DIR / "right_min_neuron1_extent2" / "network.json")
    mb = fc.load_network_backend(path, dev="cpu")
    session = fc.open_session(fc.make_train_opts(
        backend="network", target_list=["moving_bar_bright"], network=mb.network,
        sequential=True, dev="cpu",
    ), "conductance")
    pack = session.pack_for("moving_bar_bright")
    schema = list(session.schema)
    p = fc.assign_params(fc.guess_initial_params(session), schema, session.backend)
    model_full = fc._run_conductance_full(session, p, pack.signal)
    sel = fc._readout_model_traces_pack(model_full, pack)
    t0 = pack.cost_t0
    t_rel = t0[:, None] - fc.t_on + torch.arange(sel.shape[1], dtype=torch.long)
    pre = t_rel < 0
    assert bool(pre.any())
    assert torch.all(sel[pre] == 0.0)


if __name__ == "__main__":
    test_fig1_traces_shape()
    test_build_target_extent2()
    test_build_target_cost_extent()
    test_resolve_cost_extent_by_target()
    test_cost_extent_requires_network()
    test_expand_loss_weights_moving_bar_pd()
    test_calc_cost_parts_pd_nd_split()
    test_use_network_moving_bar_cost()
    test_readout_window_pre_ton_zero()
    print("ok")
