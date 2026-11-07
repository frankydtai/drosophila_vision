from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import import_bootstrap  # noqa: E402,F401
from task.spread.pack import SpreadPack  # noqa: E402
from train.cost import _pack_cost_parts_from_v_readout  # noqa: E402
from train.param import resolve_val_from  # noqa: E402


KEY = "spread_bright_Mi4"


def loss(trace, *, scope="entry", b_offset=0, sparse=False):
    pack = SpreadPack(
        task="spread", contrast="bright", i_sti=trace,
        gts=torch.tensor([[0., 1., 1.]], dtype=torch.float64),
        entry_bs=torch.tensor([0]), entry_nodes=torch.tensor([0]),
        t_onset=0, cost_ts=torch.tensor([1, 2]) if sparse else None,
        entry_part_keys=(KEY,),
    )
    opts = {"source": "v_onset", "enabled": True}
    if scope is not None:
        opts["scope"] = scope
    session = SimpleNamespace(
        tasks=("spread",), part_cost_scales={},
        schema={"bias_gt": {"lo": -200., "hi": 200.}},
        train_opts={"cost_entry_reduce": "mean_trace", "cost_norm": "a_gt2",
                    "val_from": {"bias_gt": opts}},
    )
    global_bias = trace[b_offset, 0].mean().reshape(1)
    return _pack_cost_parts_from_v_readout(
        pack, session, torch.ones(1, dtype=torch.float64), global_bias,
        trace[b_offset:b_offset + 1, :, 0], trace=trace, b_offset=b_offset,
    )[KEY]


def test_entry_scope_removes_constant_spatial_offsets_and_their_gradient():
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    baseline = torch.stack((5 + 2 * theta, 20 - 3 * theta))
    trace = baseline[None, None, :].expand(1, 3, 2)
    legacy = loss(trace, scope="cell")
    legacy_grad = torch.autograd.grad(legacy, theta, retain_graph=True)[0]
    corrected = loss(trace)
    corrected_grad = torch.autograd.grad(corrected, theta)[0]
    assert float(corrected.detach()) == pytest.approx(2.)
    assert float(corrected_grad) == pytest.approx(0., abs=1e-12)
    assert abs(float(legacy_grad)) > 1.


@pytest.mark.parametrize("sparse", [False, True])
def test_fused_batch_offset_uses_the_entrys_own_onset(sparse):
    trace = torch.tensor([
        [[-30., 100.], [-40., 100.], [-50., 100.]],
        [[5., 20.], [6., 20.], [6., 20.]],
    ], dtype=torch.float64)
    assert float(loss(trace, b_offset=1, sparse=sparse)) == pytest.approx(0.)


def test_missing_scope_keeps_legacy_cell_baseline():
    trace = torch.tensor([[[5., 20.], [6., 20.], [6., 20.]]], dtype=torch.float64)
    assert float(loss(trace, scope=None)) == float(loss(trace, scope="cell"))


def test_entry_scope_preserves_gradient_of_the_actual_temporal_response():
    theta = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    response = theta * torch.tensor([0., 1., 1.], dtype=torch.float64)
    baseline = 5 + 3 * theta
    trace = torch.stack((baseline + response, response * 0 + 20), dim=-1)[None]
    result = loss(trace)
    grad = torch.autograd.grad(result, theta)[0]
    assert float(result.detach()) == pytest.approx(2.)
    assert float(grad) == pytest.approx(4.)


def test_invalid_scope_is_rejected():
    with pytest.raises(ValueError, match="scope must be cell or entry"):
        resolve_val_from({"bias_gt": {"scope": "typo"}})


def test_sd_supervision_does_not_penalize_constant_spatial_baseline_differences():
    trace = torch.tensor([[[5., 20.], [5., 20.], [5., 20.]]], dtype=torch.float64)
    pack = SimpleNamespace(
        task="spread", contrast="bright", i_sti=trace,
        gts=torch.zeros((2, 3), dtype=torch.float64),
        gt_stds=torch.zeros((2, 3), dtype=torch.float64),
        entry_bs=torch.tensor([0, 0]), entry_nodes=torch.tensor([0, 1]),
        t_onset=0, cost_ts=None, entry_part_keys=(KEY, KEY),
    )
    session = SimpleNamespace(
        tasks=("spread",), part_cost_scales={},
        schema={"bias_gt": {"lo": -200., "hi": 200.}},
        train_opts={"cost_entry_reduce": "mean_trace", "cost_norm": "a_gt2",
                    "a_lsd": 1., "val_from": {"bias_gt": {
                        "source": "v_onset", "enabled": True, "scope": "entry",
                    }}},
    )
    result = _pack_cost_parts_from_v_readout(
        pack, session, torch.ones(2, dtype=torch.float64),
        torch.full((2,), 12.5, dtype=torch.float64), trace[0].T, trace=trace,
    )[KEY]
    assert float(result) == pytest.approx(0., abs=1e-12)
