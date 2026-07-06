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

# Per-run artifact subfolder (``.npy`` / ``.npz``, ``train_opts.json``, ``model_type.txt``).
RUN_DATA_SUBDIR = "data"


def run_data_dir(outdir: str | Path) -> str:
    return str(Path(outdir) / RUN_DATA_SUBDIR)

# Gruntman Fig. 1 Ci/Cii digitized population Vm traces (see MatlabFunctions/digitize_fig1_ci.py).
FIG1_CI_NPZ = REPO_ROOT / "MatlabFunctions" / "fig1_ci_digitized.npz"

# Simulation sampling interval (ms per discrete step).
DELTAT_MS = 10.0

# Moving-bar per-column cost window relative to bar centre.
COST_WINDOW_BEFORE_MS = 300.0
COST_WINDOW_AFTER_MS = 600.0

# Post-sweep tail: baseline after bar exit through ``t_center + after`` plus pad.
T_TAIL_PAD_MS = 50.0
MOVING_BAR_TAIL_MS = COST_WINDOW_AFTER_MS + T_TAIL_PAD_MS


def ms_to_steps(ms: float, *, deltat_ms: float = DELTAT_MS) -> int:
    """Convert milliseconds to simulation steps (rounded)."""
    return int(round(float(ms) / float(deltat_ms)))


def cost_window_before_steps(*, deltat_ms: float = DELTAT_MS) -> int:
    return ms_to_steps(COST_WINDOW_BEFORE_MS, deltat_ms=deltat_ms)


def cost_window_after_steps(*, deltat_ms: float = DELTAT_MS) -> int:
    return ms_to_steps(COST_WINDOW_AFTER_MS, deltat_ms=deltat_ms)


def cost_window_steps(*, deltat_ms: float = DELTAT_MS) -> int:
    span_ms = COST_WINDOW_BEFORE_MS + COST_WINDOW_AFTER_MS
    return int(span_ms / deltat_ms) + 1


def moving_bar_tail_steps(*, deltat_ms: float = DELTAT_MS) -> int:
    return ms_to_steps(MOVING_BAR_TAIL_MS, deltat_ms=deltat_ms)
