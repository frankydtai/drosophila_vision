"""Numerical checks for fine-part entry reduction."""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401,E402
import torch  # noqa: E402

from train.cost import _parts_from_entries  # noqa: E402


def _session(reduction: str, norm: str = "a_gt2", a_lsd: float = 1.0):
    return SimpleNamespace(
        train_opts={
            "cost_entry_reduce": reduction,
            "cost_norm": norm,
            "a_lsd": a_lsd,
        },
        part_cost_scales={},
        tasks=("toy",),
    )


def _part(values, *, reduction="mean_trace", norm="a_gt2", weights=None,
          gts=None, gt_stds=None, part_idxs=None, time_mask=None, a_lsd=1.0):
    values = torch.as_tensor(values, dtype=torch.float64)
    n_entry, n_t = values.shape
    a_gt = torch.ones(n_entry, dtype=values.dtype)
    bias_gt = torch.zeros_like(a_gt)
    gts = (
        torch.zeros_like(values) if gts is None else
        torch.as_tensor(gts, dtype=values.dtype)
    )
    weights = (
        torch.ones(n_entry, dtype=values.dtype) if weights is None else
        torch.as_tensor(weights, dtype=values.dtype)
    )
    part_idxs = (
        torch.zeros(n_entry, dtype=torch.long) if part_idxs is None else
        torch.as_tensor(part_idxs, dtype=torch.long)
    )
    keys = [f"toy_{i}" for i in range(int(part_idxs.max().item()) + 1)]
    return _parts_from_entries(
        a_gt, bias_gt, gts, weights, values, part_idxs, keys,
        _session(reduction, norm, a_lsd),
        time_mask=(
            None if time_mask is None else
            torch.as_tensor(time_mask, dtype=values.dtype)
        ),
        gt_stds=(
            None if gt_stds is None else
            torch.as_tensor(gt_stds, dtype=values.dtype)
        ),
    )


class CostEntryReduceTest(unittest.TestCase):
    def test_opposite_entries_cancel_only_after_mean(self):
        values = torch.tensor([[1.0], [-1.0]], requires_grad=True)
        new = _part(values)["toy_0"]
        old = _part(values, reduction="entry_sse")["toy_0"]
        self.assertEqual(float(new.item()), 0.0)
        self.assertEqual(float(old.item()), 2.0)

    def test_weighted_mean_retains_effective_weight(self):
        # Weighted mean=(1*0+3*4)/4=3; retained W=4, hence 4*3^2=36.
        cost = _part([[0.0], [4.0]], weights=[1.0, 3.0])["toy_0"]
        self.assertEqual(float(cost.item()), 36.0)

    def test_mean_trace_gradient(self):
        values = torch.tensor([[1.0], [3.0]], requires_grad=True)
        loss = _part(values)["toy_0"]
        loss.backward()
        self.assertTrue(torch.allclose(loss, loss.new_tensor(8.0)))
        self.assertTrue(torch.allclose(
            values.grad, values.grad.new_tensor([[4.0], [4.0]]),
        ))

    def test_gt_power_zero_when_mean_matches_target(self):
        cost = _part(
            [[0.0], [2.0]], gts=[[1.0], [1.0]], norm="gt_power",
        )["toy_0"]
        self.assertEqual(float(cost.item()), 0.0)

    def test_fine_parts_are_not_mixed(self):
        parts = _part([[1.0], [3.0]], part_idxs=[0, 1])
        self.assertEqual(float(parts["toy_0"].item()), 1.0)
        self.assertEqual(float(parts["toy_1"].item()), 9.0)

    def test_time_mask_uses_only_active_entries(self):
        parts = _part(
            [[0.0, 10.0], [2.0, 20.0]],
            time_mask=[[1.0, 0.0], [1.0, 1.0]],
        )
        # t0: W=2, mean=1 -> 2; t1: W=1, mean=20 -> 400.
        self.assertEqual(float(parts["toy_0"].item()), 402.0)

    def test_gt_std_matches_model_sample_std(self):
        # Values [0, 2] have mean 1 and sample SD sqrt(2).  Mean and SD both
        # match their targets, so the combined objective is zero (up to eps).
        target_sd = 2.0 ** 0.5
        cost = _part(
            [[0.0], [2.0]],
            gts=[[1.0], [1.0]],
            gt_stds=[[target_sd], [target_sd]],
        )["toy_0"]
        self.assertLess(float(cost.item()), 1e-12)

    def test_gt_std_penalizes_wrong_variation_not_individual_traces(self):
        # Mean is exactly zero; only the SD mismatch contributes.
        cost = _part(
            [[-1.0], [1.0]],
            gt_stds=[[0.0], [0.0]],
        )["toy_0"]
        self.assertGreater(float(cost.item()), 3.9)

    def test_a_lsd_zero_disables_std_loss(self):
        cost = _part(
            [[-1.0], [1.0]],
            gt_stds=[[0.0], [0.0]],
            a_lsd=0.0,
        )["toy_0"]
        self.assertEqual(float(cost.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
