from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SIMULATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMULATION_ROOT not in sys.path:
    sys.path.insert(0, SIMULATION_ROOT)

import import_bootstrap  # noqa: E402,F401
from neuron.forward import _mbar_gaussian_overlay, pack_t_onset  # noqa: E402
from neuron.readout import pack_traces  # noqa: E402
from task.mbar.gt import fig1_trace_delta  # noqa: E402
from task.mbar.pack import MbarPack  # noqa: E402
from task.mbar.sti_spec import MbarSpec, build_bar_axis_distance_hex  # noqa: E402
from build_hex import DEG  # noqa: E402


def test_mbar_aligned_window_keeps_real_pre_stimulus_samples():
    trace = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    pack = MbarPack(
        task="mbar",
        contrast="bright",
        i_sti=torch.zeros((1, 7, 1)),
        gts=torch.zeros((1, 3)),
        cost_scales=torch.ones(1),
        entry_bs=torch.tensor([0]),
        entry_nodes=torch.tensor([0]),
        t_onset=4,
        cost_t0s=torch.tensor([1]),
    )

    assert pack_t_onset(pack) == 4
    torch.testing.assert_close(
        pack_traces(trace, pack), torch.tensor([[1.0, 2.0, 3.0]]),
    )


def test_other_aligned_packs_keep_legacy_pre_onset_zeroing():
    trace = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    pack = SimpleNamespace(
        i_sti=torch.zeros((1, 7, 1)),
        gts=torch.zeros((1, 3)),
        entry_bs=torch.tensor([0]),
        entry_nodes=torch.tensor([0]),
        cost_t0s=torch.tensor([1]),
        t_onset=4,
    )

    torch.testing.assert_close(pack_traces(trace, pack), torch.zeros((1, 3)))


def test_fig1_delta_uses_pre_stimulus_mean():
    trace = np.concatenate((np.full(150, 3.0), np.array([5.0, 7.0])))
    delta = fig1_trace_delta(trace, delta_ms=2.0)

    np.testing.assert_allclose(delta[:150], 0.0)
    np.testing.assert_allclose(delta[150:], [2.0, 4.0])


def test_mbar_gaussian_preserves_drive_outside_bar_support():
    """Gaussian mbar width must not add baseline current outside STI nodes."""
    pack = SimpleNamespace(
        bar_axis_distance=torch.tensor([[[0.0]]]),  # (B, T_post, n_sti)
        bar_axis_sti_nodes=torch.tensor([1]),
        t_onset=0,
        i_sti_baseline=20.0,
        i_sti_peak=40.0,
    )
    params = {"a_sti_mid": torch.tensor([1.0])}
    raw_drive = torch.tensor([[0.0, 20.0, 0.0]])

    drive = _mbar_gaussian_overlay(raw_drive, params, pack, t=0)

    torch.testing.assert_close(drive, torch.tensor([[0.0, 40.0, 0.0]]))


def test_mbar_gaussian_skips_pre_onset_and_keeps_finite_grad():
    pack = SimpleNamespace(
        bar_axis_distance=torch.tensor([[[0.0]]], dtype=torch.float32),
        bar_axis_sti_nodes=torch.tensor([0]),
        t_onset=1,
        i_sti_baseline=20.0,
        i_sti_peak=40.0,
    )
    sigma = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    raw_drive = torch.tensor([[5.0]], dtype=torch.float32)

    pre = _mbar_gaussian_overlay(raw_drive, {"a_sti_mid": sigma}, pack, t=0)
    torch.testing.assert_close(pre, raw_drive)

    post = _mbar_gaussian_overlay(raw_drive, {"a_sti_mid": sigma}, pack, t=1)
    post.sum().backward()
    assert torch.isfinite(sigma.grad).all()


@pytest.mark.parametrize("direction", ["right", "left", "up", "down"])
@pytest.mark.parametrize("peak", [40.0, 0.0])
def test_mbar_gaussian_one_mid_offset_matches_static_width(direction, peak):
    """A 4.5-degree offset has gain exp(-1/2) for sigma=1 mid."""
    hexes = [SimpleNamespace(x=x, y=y) for x, y in ((0, 0), (1, 0), (0, 1))]
    distances = build_bar_axis_distance_hex(
        hexes, [MbarSpec(direction, "bright" if peak else "dark", float(DEG))],
        n_t=5, bar_dist=4, multi_bar=False, t_onset=1, delta_ms=40,
        view_deg=(-float(DEG), -float(DEG), float(DEG), float(DEG)),
    )
    # The bar center reaches (0,0) at the final stored frame in all directions.
    offsets = [0.0, 1.0, 0.0] if direction in ("right", "left") else [0.0, 0.0, 1.0]
    np.testing.assert_allclose(distances[0, -1], offsets, atol=1e-12)
    pack = SimpleNamespace(
        bar_axis_distance=torch.tensor(distances, dtype=torch.float64),
        bar_axis_sti_nodes=torch.tensor([0, 1, 2]), t_onset=1,
        i_sti_baseline=20.0, i_sti_peak=peak,
    )
    sigma = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    raw = torch.tensor([[20.0, 20.0, 20.0, -7.0]], dtype=torch.float64)
    drive = _mbar_gaussian_overlay(raw, {"a_sti_mid": sigma}, pack, t=4)
    gain = np.exp(-0.5 * np.square(offsets))
    expected = np.concatenate((20.0 + (peak - 20.0) * gain, [-7.0]))
    torch.testing.assert_close(drive[0], torch.tensor(expected))
    drive.sum().backward()
    assert torch.isfinite(sigma.grad).all() and sigma.grad.abs().item() > 0
