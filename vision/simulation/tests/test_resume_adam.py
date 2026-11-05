from __future__ import annotations

import os
import sys
import unittest

SIMULATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMULATION_ROOT not in sys.path:
    sys.path.insert(0, SIMULATION_ROOT)

import import_bootstrap  # noqa: E402,F401
from train.implementation import (  # noqa: E402
    objective_diff,
    objective_fingerprint,
    resolve_resume_adam,
    same_train_objective,
)


def _opts(**overrides):
    base = {
        "tasks": ["mbar"],
        "contrasts": ["bright"],
        "i_sti": {"bright": 40.0, "dark": 0.0},
        "part_cost_scales": {"mbar": 0.0, "mbar_DSI": 1.0},
        "cost_norm": "a_gt2",
        "a_lsd": 1.0,
        "cost_entry_reduce": "mean_trace",
        "cost_ms": None,
    }
    base.update(overrides)
    return base


class ResumeAdamPolicyTest(unittest.TestCase):
    def test_same_objective_ignores_unrelated_keys(self):
        src = _opts(n_iter=50, network_json="/a")
        dst = _opts(n_iter=100, network_json="/b")
        self.assertTrue(same_train_objective(src, dst))
        self.assertEqual(objective_diff(src, dst), [])

    def test_tasks_change_is_different_objective(self):
        src = _opts(tasks=["sbar"])
        dst = _opts(tasks=["mbar"])
        self.assertFalse(same_train_objective(src, dst))
        self.assertEqual(objective_diff(src, dst), ["tasks"])

    def test_part_cost_scales_change_is_different_objective(self):
        src = _opts(part_cost_scales={"mbar": 1.0})
        dst = _opts(part_cost_scales={"mbar": 0.0, "mbar_DSI": 1.0})
        self.assertEqual(objective_diff(src, dst), ["part_cost_scales"])

    def test_fingerprint_normalizes_numeric_types(self):
        src = objective_fingerprint(_opts(a_lsd=1, i_sti={"bright": 40}))
        dst = objective_fingerprint(_opts(a_lsd=1.0, i_sti={"bright": 40.0}))
        self.assertEqual(src, dst)

    def test_entry_baseline_changes_objective_but_explicit_legacy_scope_does_not(self):
        legacy = _opts(val_from={"bias_gt": {"source": "v_onset", "enabled": True}})
        cell = _opts(val_from={"bias_gt": {
            "source": "v_onset", "enabled": True, "scope": "cell",
        }})
        entry = _opts(val_from={"bias_gt": {
            "source": "v_onset", "enabled": True, "scope": "entry",
        }})
        self.assertTrue(same_train_objective(legacy, cell))
        self.assertEqual(objective_diff(legacy, entry), ["bias_gt_scope"])

    def test_resolve_resume_adam_auto(self):
        self.assertEqual(
            resolve_resume_adam("auto", same_objective=True),
            (True, "auto"),
        )
        self.assertEqual(
            resolve_resume_adam(None, same_objective=False),
            (False, "auto"),
        )

    def test_resolve_resume_adam_forced(self):
        self.assertEqual(
            resolve_resume_adam(True, same_objective=False),
            (True, "forced"),
        )
        self.assertEqual(
            resolve_resume_adam("false", same_objective=True),
            (False, "forced"),
        )

    def test_resolve_resume_adam_rejects_unknown(self):
        with self.assertRaises(ValueError):
            resolve_resume_adam("maybe", same_objective=True)


if __name__ == "__main__":
    unittest.main()
