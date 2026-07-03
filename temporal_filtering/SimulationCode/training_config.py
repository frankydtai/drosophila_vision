"""Shared paths and constants for connectome training targets.

SimulationCode scripts import from here instead of hardcoding paths to
``MatlabFunctions/``, trained-parameter output folders, or cost-window sizes.
FAFB connectome build paths live in ``connectome_io``; this module covers
training data and training output paths.
"""

from __future__ import annotations

from pathlib import Path

# SimulationCode root (this file's directory).
SIMULATION_DIR = Path(__file__).resolve().parent
# Repo root: .../drosophila_vision
REPO_ROOT = SIMULATION_DIR.parent.parent

# Trained-parameter output root (``adaptive/`` and ``conductance/`` run_* subdirs).
PARAMETER_DIR = SIMULATION_DIR / "FiveCol_Parameter"

# Gruntman Fig. 1 Ci/Cii digitized population Vm traces (see MatlabFunctions/digitize_fig1_ci.py).
FIG1_CI_NPZ = REPO_ROOT / "MatlabFunctions" / "fig1_ci_digitized.npz"

# Per-neuron cost window: t_center ± 0.45 s at deltat = 10 ms.
COST_HALF_WINDOW_STEPS = 45
COST_WINDOW_STEPS = 2 * COST_HALF_WINDOW_STEPS  # 90 steps = 0.9 s

