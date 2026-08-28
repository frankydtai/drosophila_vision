from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch


SIMULATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMULATION_ROOT not in sys.path:
    sys.path.insert(0, SIMULATION_ROOT)

import import_bootstrap  # noqa: E402,F401
from neuron.forward import pack_t_onset  # noqa: E402
from neuron.readout import pack_traces  # noqa: E402
from task.mbar.gt import fig1_trace_delta  # noqa: E402
from task.mbar.pack import MbarPack  # noqa: E402


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
