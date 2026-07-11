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
import numpy as np
import torch
from connectome_io import NETWORK_DIR
from network.construction import load_network
from network.moving_bar_target import (
    ND_IDX,
    PD_IDX,
    build_moving_bar_signals,
    build_moving_bar_target,
    build_moving_bar_t0_grids,
    load_fig1_traces,
    moving_bar_window_t_rel,
    moving_bar_window_t_rel_torch,
    resolve_i_baseline,
)
from training_config import COST_WINDOW, T_ON


def test_fig1_traces_shape():
    fig1 = load_fig1_traces()
    assert len(fig1) == 16
    for arr in fig1.values():
        assert arr.shape == (COST_WINDOW,)


def test_build_target_extent2():
    path = NETWORK_DIR / "right_min_neuron1_extent2" / "network.json"
    C = load_network(path, device="cpu")
    T = build_moving_bar_target(C, device="cpu", use_cache=True, contrasts=("bright",))
    assert T.signal.shape[0] == 8
    assert T.signal.shape[1] == T.maxtime
    assert T.data.shape[1] == COST_WINDOW
    assert T.cost_t0.shape == T.readout_batch.shape == T.readout_unit.shape
    assert int(T.cost_t0.min()) >= 0
    assert int(T.cost_t0.max()) + COST_WINDOW <= T.maxtime
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
        model="conductance",
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
    assert pack.data.shape[1] == COST_WINDOW
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


def test_moving_bar_signal_pr_only():
    path = NETWORK_DIR / "right_min_neuron1_extent2" / "network.json"
    C = load_network(path, device="cpu")
    stim = build_moving_bar_signals(C, device="cpu", use_cache=True)
    sig = stim.signal.detach().cpu().numpy()
    is_input = np.asarray(C.is_input, dtype=bool)
    assert sig.shape[1] == stim.info["maxtime"]
    assert not is_input.all()
    assert np.all(sig[:, :, ~is_input] == 0.0)
    # PR units carry baseline before stimulus onset
    pr_sig = sig[:, :T_ON, is_input]
    assert pr_sig.size > 0
    assert np.allclose(pr_sig, float(stim.info["i_baseline"]))


def test_moving_bar_window_t_rel_matches_core():
    import FiveCol_MedSim_Pytorch as fc

    t0 = np.array([[42, 55, -1]], dtype=np.int64)
    win = 5
    t_rel_np, pre_np = moving_bar_window_t_rel(t0, fc.t_on, win)
    t0_t = torch.as_tensor(t0)
    t_rel_t, pre_t = moving_bar_window_t_rel_torch(t0_t, fc.t_on, win)
    assert np.array_equal(t_rel_np, t_rel_t.numpy())
    assert np.array_equal(pre_np, pre_t.numpy())


def test_resolve_i_baseline_and_session():
    import FiveCol_MedSim_Pytorch as fc
    from Medulla_Library import I_BASELINE

    assert resolve_i_baseline(None) == float(I_BASELINE)
    assert resolve_i_baseline(15.0) == 15.0
    assert fc.session_moving_bar_i_baseline({}) == float(I_BASELINE)
    assert fc.session_moving_bar_i_baseline({
        "moving_bar_bright_stimulus_opts": {"i_baseline": 22.0},
    }) == 22.0


def test_t0_grid_slice_uses_global_horizon():
    import FiveCol_MedSim_Pytorch as fc
    from plot.moving_bar import _bar_specs_for_session, _moving_bar_t0_grids

    path = str(NETWORK_DIR / "right_min_neuron1_extent2" / "network.json")
    mb = fc.load_network_backend(path, dev="cpu")
    session = fc.open_session(fc.make_train_opts(
        backend="network", target_list=["moving_bar_bright"], network=mb.network,
        sequential=True, dev="cpu",
    ), "conductance")
    specs = _bar_specs_for_session(session, "moving_bar_bright")
    pack = session.pack_for("moving_bar_bright")
    maxtime = int(session.maxtime)
    _, _, _, before_all, _, _, _ = _moving_bar_t0_grids(
        session, specs, pack.cost_extent, maxtime,
    )
    _, _, _, before_slice, _, _, _ = _moving_bar_t0_grids(
        session, specs, pack.cost_extent, maxtime, at_x=0.0, at_y=0.0,
    )
    assert before_slice == before_all


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
    test_moving_bar_signal_pr_only()
    test_moving_bar_window_t_rel_matches_core()
    test_resolve_i_baseline_and_session()
    test_t0_grid_slice_uses_global_horizon()
    print("ok")
