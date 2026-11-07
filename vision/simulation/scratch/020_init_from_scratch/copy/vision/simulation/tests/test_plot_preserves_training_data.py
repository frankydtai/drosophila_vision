from __future__ import annotations

import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


SIMULATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMULATION_ROOT not in sys.path:
    sys.path.insert(0, SIMULATION_ROOT)

import import_bootstrap  # noqa: E402,F401
import figure.plot as plot  # noqa: E402
import train  # noqa: E402


@dataclass(frozen=True)
class _Session:
    connectome: object
    schema: dict
    sim_dtype: torch.dtype
    device: str
    model: str
    tasks: tuple
    train_opts: dict

    def with_schema(self, schema):
        return replace(self, schema=schema)


def _saved_run(run_dir):
    data = run_dir / "data"
    data.mkdir()
    (data / "train_opts.json").write_text('{"model": "hp_lp"}')
    np.savez(
        data / "best_param.npz",
        syn_strength_cell=np.array([0.1]),
        cells=np.array(["Mi4", "T4a"], dtype=object),
        pairs=np.array(["Mi4:T4a"], dtype=object),
    )
    np.save(data / "costs.npy", [1.0])
    (run_dir / "syn_strength_cell.csv").write_text(
        ",Mi4,T4a\nMi4,,0.100000\nT4a,,\n",
    )
    session = _Session(
        connectome=SimpleNamespace(
            cells=["Mi4", "T4a"],
            conn=SimpleNamespace(pairs=[(0, 1)], node_cells=torch.tensor([0, 1])),
        ),
        schema={
            "syn_strength_cell": {
                "n_node": 1, "init": 1.0, "lo": 0.1, "hi": 10.0,
                "indi": [0], "shared": [], "fixed": [], "frozen": [],
            },
        },
        sim_dtype=torch.float32,
        device="cpu",
        model="hp_lp",
        tasks=("sbar",),
        train_opts={},
    )
    original = {
        p.relative_to(run_dir): p.read_bytes()
        for p in run_dir.rglob("*") if p.is_file()
    }
    return session, original


class PlotPreservesTrainingDataTest(unittest.TestCase):
    def setUp(self):
        temporary_run = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_run.cleanup)
        self.run_dir = Path(temporary_run.name)
        self.session, self.original = _saved_run(self.run_dir)

        def render(session, z, run_dir, suffix, model_all, *, task, **kwargs):
            value = train.params_from_z(z, session)["syn_strength_cell"].item()
            Path(run_dir, f"{task}{kwargs['file_suffix']}.txt").write_text(str(value))

        # Skip rendering and simulation; keep the real load/override/save path.
        for target, name, replacement in (
            (train, "session_from_run_dir", lambda *a, **kw: self.session),
            (train, "calc_cost_parts", lambda *a, **kw: {}),
            (train, "calc_cost", lambda *a, **kw: torch.tensor(1.0)),
            (plot, "plot_figures", render),
        ):
            patcher = patch.object(target, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_training_data_unchanged(self):
        for path, content in self.original.items():
            self.assertEqual((self.run_dir / path).read_bytes(), content, f"Plot modified {path}")

    def check_replot(self, strength=None):
        overrides = {} if strength is None else {"syn_strength_cell": {"Mi4:T4a": strength}}
        plot.plot_trained_run(
            str(self.run_dir), figure_kwargs={}, param_vals=overrides,
        )

        self.assert_training_data_unchanged()
        suffix = plot.param_filename_suffix(overrides)
        plotted_value = float((self.run_dir / f"sbar{suffix}.txt").read_text())
        self.assertAlmostEqual(plotted_value, 0.1 if strength is None else strength)

    def test_replot_preserves_training_data(self):
        self.check_replot()

    def test_parameter_override_changes_figure_only(self):
        self.check_replot(10.0)

    def test_direct_plot_preserves_saved_parameters(self):
        plot.plot_rf_t(
            np.array([[10.0]]), str(self.run_dir), session=self.session,
            final_costs=np.array([1.0]), file_suffix="_override",
        )

        self.assert_training_data_unchanged()
        self.assertEqual(float((self.run_dir / "sbar_override.txt").read_text()), 10.0)


if __name__ == "__main__":
    unittest.main()
