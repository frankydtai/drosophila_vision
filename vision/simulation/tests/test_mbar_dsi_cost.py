from __future__ import annotations

import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SIMULATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMULATION_ROOT not in sys.path:
    sys.path.insert(0, SIMULATION_ROOT)

import import_bootstrap  # noqa: E402,F401
import train.cost as cost  # noqa: E402
from task.mbar.pack import (  # noqa: E402
    MbarPack, build_mbar_gt, build_mbar_pack, dsi_cost_parts, dsi_groups, dsi_values,
    mbar_cost_sti_hexes,
)
from task.mbar.sti_geo import cost_motion_view_deg, view_bounds  # noqa: E402
from task.mbar.sti_spec import MbarSpec, hex_dsi_window  # noqa: E402
from task.sbar.sti_geo import sti_hexes  # noqa: E402
from train.session import materialize_pack  # noqa: E402


KEY_A = "mbar_DSI_bright_T4a"
KEY_B = "mbar_DSI_bright_T4b"


def make_pack():
    return MbarPack(
        task="mbar", contrast="bright",
        i_sti=torch.zeros((2, 12, 2), dtype=torch.float64),
        gts=torch.zeros((4, 3), dtype=torch.float64),
        cost_scales=torch.ones(4, dtype=torch.float64),
        entry_bs=torch.tensor([0, 0, 1, 1]),
        entry_nodes=torch.tensor([0, 1, 0, 1]),
        # Right = PD for T4a, left = PD for T4b.
        cost_pd_nds=torch.tensor([0, 1, 1, 0]),
        cost_t0s=torch.tensor([2, 3, 4, 5]),
        entry_part_keys=(
            "mbar_right_bright_w1_T4a_PD", "mbar_right_bright_w1_T4b_ND",
            "mbar_left_bright_w1_T4a_ND", "mbar_left_bright_w1_T4b_PD",
        ),
        entry_dsi_part_keys=(KEY_A, KEY_B, KEY_A, KEY_B),
        dsi_t0s=torch.tensor([2, 3, 4, 5]),
        dsi_n_ts=torch.tensor([3, 4, 3, 4]),
        t_onset=0,
    )


def response_trace():
    trace = torch.zeros((2, 12, 2), dtype=torch.float64)
    trace[0, 3, 0] = 4
    trace[1, 5, 0] = 1
    trace[1, 6, 1] = 6
    trace[0, 4, 1] = 2
    return trace


def make_session(packs, *, sequential=False, reduction="entry_sse", scales=None):
    return SimpleNamespace(
        tasks=("mbar",),
        iter_packs=lambda: iter(packs),
        part_cost_scales={"mbar": 0, "mbar_DSI": 1} if scales is None else scales,
        sequential=sequential, sim_dtype=torch.float64, device="cpu",
        train_opts={"cost_entry_reduce": reduction, "cost_norm": "a_gt2"},
    )


def test_hex_dsi_window_adds_one_actual_transit_on_each_side():
    stimulus = np.full(30, 20.0)
    stimulus[10:15] = [21, 30, 40, 30, 21]
    # Actual [10,14] -> [6,18], independent of the 900 ms GT window.
    assert hex_dsi_window(stimulus, 20) == (6, 13)
    assert hex_dsi_window(40 - stimulus, 20) == (6, 13)


def test_voltage_sums_use_local_windows_and_subtype_pd():
    pack = make_pack()
    baseline = torch.tensor([10.0, 20.0], dtype=torch.float64)
    trace = response_trace() + baseline
    trace[:, 0] = 1000  # Outside every local window.
    trace[:, 11] = 1000
    trace.requires_grad_()
    window_ts = {}
    values = dsi_values(pack, trace, window_ts=window_ts)
    assert window_ts == {
        KEY_A: {"PD": (0, 2), "ND": (0, 2)},
        KEY_B: {"PD": (-1, 2), "ND": (-1, 2)},
    }
    torch.testing.assert_close(values[KEY_A], trace.new_tensor(3.0))
    torch.testing.assert_close(values[KEY_B], trace.new_tensor(4.0))
    parts = dsi_cost_parts(pack, trace)

    torch.testing.assert_close(parts[KEY_A], trace.new_tensor(-3.0))
    torch.testing.assert_close(parts[KEY_B], trace.new_tensor(-4.0))
    sum(parts.values()).backward()
    assert torch.isfinite(trace.grad).all()
    assert torch.count_nonzero(trace.grad[:, [0, 11]]) == 0
    # Every in-window sample contributes, including samples away from the peak.
    torch.testing.assert_close(trace.grad[0, 2:5, 0], -torch.ones(3, dtype=trace.dtype))
    torch.testing.assert_close(trace.grad[1, 4:7, 0], torch.ones(3, dtype=trace.dtype))


def test_dsi_sums_aligned_mean_curves_over_window_intersection():
    pack = make_pack()
    # Replicate each entry onto a second hex of the same cell subtype.
    fields = {
        name: torch.cat((getattr(pack, name), getattr(pack, name)))
        for name in ("gts", "cost_scales", "entry_bs", "cost_pd_nds", "cost_t0s", "dsi_t0s", "dsi_n_ts")
    }
    fields.update(
        entry_nodes=torch.cat((pack.entry_nodes, pack.entry_nodes + 2)),
        entry_part_keys=pack.entry_part_keys * 2,
        entry_dsi_part_keys=pack.entry_dsi_part_keys * 2,
    )
    pack = replace(pack, **fields)
    second_hex = torch.zeros_like(response_trace())
    second_hex[0, 2, 0] = 2
    second_hex[1, 4, 0] = 4
    trace = torch.cat((response_trace(), second_hex), dim=2).requires_grad_()
    parts = dsi_cost_parts(pack, trace)
    # All PD samples sum to (4+2)/2=3; all ND samples to (1+4)/2=2.5.
    torch.testing.assert_close(parts[KEY_A], trace.new_tensor(-0.5))
    parts[KEY_A].backward()
    assert torch.isfinite(trace.grad).all()
    assert trace.grad[0, 3, 0] < 0
    assert trace.grad[1, 4, 2] > 0

    # The common window excludes later samples absent from the second hex.
    shortened = replace(pack, dsi_n_ts=pack.dsi_n_ts.clone())
    shortened.dsi_n_ts[4] = 1
    window_ts = {}
    values = dsi_values(shortened, trace, window_ts=window_ts)
    assert window_ts[KEY_A] == {"PD": (0, 0), "ND": (0, 2)}
    torch.testing.assert_close(values[KEY_A], trace.new_tensor(-1.5))


def test_zero_voltage_has_zero_loss_and_finite_gradients():
    pack = make_pack()
    trace = torch.zeros_like(pack.i_sti, requires_grad=True)
    parts = dsi_cost_parts(pack, trace)
    assert all(part == 0 for part in parts.values())
    sum(parts.values()).backward()
    assert torch.isfinite(trace.grad).all()


@pytest.mark.parametrize("v_pd,v_nd,expected_loss", [
    (-1.0, 1.0, 6.0),
    (1.0, -1.0, -6.0),
    (-1.0, -2.0, -3.0),
    (-50.0, -52.0, -6.0),
])
def test_raw_voltages_are_summed_without_rectification_or_normalization(v_pd, v_nd, expected_loss):
    pack = make_pack()
    trace = torch.zeros_like(pack.i_sti)
    trace[0, :, 0] = v_pd
    trace[1, :, 0] = v_nd
    trace.requires_grad_()
    parts = dsi_cost_parts(pack, trace)
    torch.testing.assert_close(parts[KEY_A], trace.new_tensor(expected_loss))
    sum(parts.values()).backward()
    assert torch.isfinite(trace.grad).all()
    assert trace.grad[0, 3, 0] == -1
    assert trace.grad[1, 5, 0] == 1


def test_dsi_preserves_negative_group_voltages_after_averaging_hexes():
    pack = make_pack()
    fields = {
        name: torch.cat((getattr(pack, name), getattr(pack, name)))
        for name in ("gts", "cost_scales", "entry_bs", "cost_pd_nds", "cost_t0s", "dsi_t0s", "dsi_n_ts")
    }
    pack = replace(
        pack, **fields,
        entry_nodes=torch.cat((pack.entry_nodes, pack.entry_nodes + 2)),
        entry_part_keys=pack.entry_part_keys * 2,
        entry_dsi_part_keys=pack.entry_dsi_part_keys * 2,
    )
    trace = torch.zeros((2, 12, 4), dtype=torch.float64)
    trace[0, :, 0] = -4.0
    trace[0, :, 2] = 2.0
    trace[1, :, [0, 2]] = 1.0
    # PD mean = -1; ND mean = 1; summing three time points gives DSI=-6.
    parts = dsi_cost_parts(pack, trace)
    torch.testing.assert_close(parts[KEY_A], trace.new_tensor(6.0))


def test_incomplete_direction_pair_does_not_add_dsi_part():
    single_direction = cost._subset_pack_bs(make_pack(), (0,))
    assert list(dsi_groups(single_direction)) == []


@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("reduction", ["entry_sse", "mean_trace"])
def test_dsi_only_forward_backward_and_fused_offsets(monkeypatch, sequential, reduction):
    first = replace(make_pack(), i_sti=response_trace())
    second = replace(first, i_sti=response_trace() * 1.5, contrast="dark",
                     entry_dsi_part_keys=tuple(key.replace("bright", "dark") for key in first.entry_dsi_part_keys),
                     entry_part_keys=tuple(key.replace("bright", "dark") for key in first.entry_part_keys))
    session = make_session([first, second], sequential=sequential, reduction=reduction)
    calls = []

    monkeypatch.setattr(cost, "params_from_z", lambda z, _session: {"gain": z})

    def forward(_session, params, drive, _pack):
        calls.append(drive.shape[0])
        return drive + params["gain"] * drive.square()

    monkeypatch.setattr(cost, "forward_pack", forward)
    monkeypatch.setattr(cost, "gt_affine_from_pack", lambda _params, pack, _session, **_kw: (
        torch.full_like(pack.cost_scales, 37.0), torch.full_like(pack.cost_scales, 123.0),
    ))
    z = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    reference_parts = {}
    for pack in (first, second):
        trace = pack.i_sti + z * pack.i_sti.square()
        reference_parts.update(dsi_cost_parts(pack, trace))
    reference = sum(reference_parts.values()) / len(reference_parts)
    expected_grad, = torch.autograd.grad(reference, z)

    parts = cost.calc_cost_parts(z, session)
    assert len(parts) == 4
    torch.testing.assert_close(sum(parts.values()), reference)
    total, backward_parts = cost.backward_part_sums(z, session)
    assert total == pytest.approx(float(reference.detach()))
    assert sum(backward_parts.values()) == pytest.approx(total)
    torch.testing.assert_close(z.grad, expected_grad)
    assert min(calls) >= 2  # Both directions must reach the same reduction.


def test_dsi_scale_zero_keeps_mse_and_exact_key_overrides():
    pack = make_pack()
    session = make_session([pack], scales={"mbar": 1, "mbar_DSI": 0, KEY_A: 2})
    assert cost._part_scale(session, KEY_A) == 2
    assert cost._part_scale(session, KEY_B) == 0
    assert cost._pack_has_active_dsi(pack, session)
    session.part_cost_scales = {"mbar": 1, "mbar_DSI": 0}
    assert not cost._pack_has_active_dsi(pack, session)
    assert cost._pack_has_active_cost(pack, session)


def test_materialize_converts_dsi_window_arrays():
    pack = replace(make_pack(), dsi_t0s=np.array([2, 3, 4, 5]), dsi_n_ts=np.array([3, 4, 3, 4]))
    converted = materialize_pack(pack, device="cpu", sim_dtype=torch.float64)
    assert converted.dsi_t0s.dtype == torch.long
    assert converted.dsi_n_ts.dtype == torch.long


def make_network():
    us = np.repeat([-1, 0, 1], 3)
    vs = np.zeros(9, dtype=np.int64)
    cell_ids = np.tile(np.arange(3), 3)
    return SimpleNamespace(
        cells=["R1-6", "T4a", "T4b"], us=us, vs=vs,
        node_cells=torch.tensor(cell_ids), n_node=9,
        sti_nodes=np.array([0, 3, 6]), meta={"side": "right"}, source_json=None,
        sti_nodes_at_uv=lambda u, v: np.flatnonzero((us == u) & (vs == v) & (cell_ids == 0)),
        nodes_at_uv=lambda u, v, cell, **_kw: np.flatnonzero(
            (us == u) & (vs == v) & (cell_ids == ["R1-6", "T4a", "T4b"].index(cell))
        ),
    )


def test_mbar_cost_mid_uses_only_motion_axis_strip():
    network = make_network()
    horizontal = [
        MbarSpec("right", "bright", 2.25),
        MbarSpec("left", "bright", 2.25),
    ]
    vertical = [
        MbarSpec("up", "bright", 2.25),
        MbarSpec("down", "bright", 2.25),
    ]

    horizontal_hexes = mbar_cost_sti_hexes(
        network, horizontal, cost_mid=0,
    )
    vertical_hexes = mbar_cost_sti_hexes(
        network, vertical, cost_mid=0,
    )

    assert len(horizontal_hexes) == 3
    assert all(hex.x == 0.0 for hex in horizontal_hexes)
    assert [(hex.x, hex.y) for hex in vertical_hexes] == [(0.0, 0.0)]


def test_mbar_sweep_follows_cost_mid_strip():
    cost_hexes = [
        SimpleNamespace(x=0.0, y=-1.0),
        SimpleNamespace(x=0.0, y=0.0),
        SimpleNamespace(x=0.0, y=1.0),
    ]
    view = cost_motion_view_deg(
        (-47.25, -47.25, 47.25, 47.25),
        cost_hexes,
        ["right", "left"],
    )
    # cost_mid=0 strip patch ±2.25°, plus 2*DEG margin -> ±11.25°.
    assert view[0] == -11.25
    assert view[2] == 11.25
    assert view[1] == -47.25
    assert view[3] == 47.25


def test_real_stimulus_pack_uses_individual_hex_transits_and_pads_both_sides(tmp_path):
    network = make_network()
    fig1 = tmp_path / "fig1.npz"
    np.savez(fig1, **{
        f"T4_PC_w4_{direction}__{field}": np.array(values)
        for direction in ("PD", "ND")
        for field, values in (("time_ms", [0, 900]), ("vm_mv", [0, 0]))
    })
    gt = build_mbar_gt(
        network, t_onset=162, delta_ms=2.0, use_cache=False,
        bar_dist=4, bar_ws_deg=[40.0], bar_directions=["right", "left"],
        multi_bar=False, i_baseline=20, i_sti=40, contrasts=["bright"],
        gt_cells=["T4a", "T4b"], fig1_path=fig1,
    )
    assert gt.t_onset > 162  # A wide bar needs extra pre-transit baseline.
    assert np.all(gt.dsi_t0s >= 0)
    assert np.all(gt.dsi_t0s + gt.dsi_n_ts <= gt.n_t)
    for b, node, start, length in zip(gt.entry_bs, gt.entry_nodes, gt.dsi_t0s, gt.dsi_n_ts):
        active = np.flatnonzero(~np.isclose(gt.i_sti[b, :, (node // 3) * 3], 20))
        first, last = int(active[0]), int(active[-1])
        assert start == first - (last - first)
        assert start + length - 1 == last + (last - first)


def test_dsi_only_pack_needs_no_digitized_gt_and_preserves_windows():
    pack, sti_opts, _ = build_mbar_pack(
        make_network(), contrast="bright", gt_amp=20,
        i_sti={"bright": 40, "dark": 0},
        sti_opts={
            "ms_pre": 324, "delta_ms_pre": 2, "delta_ms": 2,
            "bar_dist": 4, "bar_ws_deg": [40.0],
            "bar_directions": ["right", "left"], "multi_bar": False,
            "gt_cells": ["T4a", "T4b"],
        },
        opts={"part_cost_scales": {"mbar": 0, "mbar_DSI": 1}},
    )
    assert pack.gts.shape[1] == 0
    assert pack.t_onset * 2 == sti_opts["ms_pre"]
    assert pack.t_onset > 162
    pack = materialize_pack(pack, device="cpu", sim_dtype=torch.float64)
    assert len(list(dsi_groups(pack))) == 2
    session = make_session([pack], reduction="mean_trace")
    # Empty waveform readout must not suppress a DSI-only objective.
    parts = cost._pack_cost_parts_from_v_readout(
        pack, session, torch.ones_like(pack.cost_scales),
        torch.zeros_like(pack.cost_scales), torch.empty((len(pack.entry_bs), 0)),
        trace=pack.i_sti,
    )
    assert set(parts) == {KEY_A, KEY_B}
    assert set(parts) <= set(pack.cost_part_plot_specs)
