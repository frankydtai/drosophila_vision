"""Shared paths and constants for connectome training targets.

SimulationCode scripts import from here instead of hardcoding paths to
``MatlabFunctions/``, trained-parameter output folders, or cost-window sizes.
FAFB connectome build paths live in ``connectome_io``; this module covers
training data and training output paths.
"""

from __future__ import annotations

from pathlib import Path

import torch

# SimulationCode root (this file's directory).
SIMULATION_DIR = Path(__file__).resolve().parent
# Repo root: .../drosophila_vision
REPO_ROOT = SIMULATION_DIR.parent.parent

# Trained-parameter output root (``adaptive/`` and ``conductance/`` run_* subdirs).
PARAMETER_DIR = SIMULATION_DIR / "FiveCol_Parameter"

# Borst 5-column simulator matrices (``FiveCol_MedSim_*`` / ``Circuits/``).
BORST_CIRCUITS_DIR = SIMULATION_DIR / "Circuits"
BORST_CTYPE_NPY = BORST_CIRCUITS_DIR / "ctype.npy"
BORST_MULTI_COL_M_NPY = BORST_CIRCUITS_DIR / "multi_colM.npy"
BORST_INTRA_COL_M_NPY = BORST_CIRCUITS_DIR / "intra_colM.npy"
BORST_INTER_COL_M_NPY = BORST_CIRCUITS_DIR / "inter_colM.npy"
BORST_MC_CELL_INDEX_NPY = BORST_CIRCUITS_DIR / "mc_cell_index.npy"

# Per-run artifact subfolder (``.npy`` / ``.npz``, ``train_opts.json``).
RUN_DATA_SUBDIR = "data"


def run_data_dir(outdir: str | Path) -> str:
    return str(Path(outdir) / RUN_DATA_SUBDIR)

# Gruntman Fig. 1 Ci/Cii digitized population Vm traces (see MatlabFunctions/digitize_fig1_ci.py).
FIG1_CI_NPZ = REPO_ROOT / "MatlabFunctions" / "fig1_ci_digitized.npz"

# Simulation sampling interval (ms per discrete step).
DELTAT_MS = 10.0

SIM_DTYPE_DEFAULT = torch.float64


def sim_dtype_from_fp32(fp32: bool) -> torch.dtype:
    return torch.float32 if fp32 else SIM_DTYPE_DEFAULT

# Stimulus timing (canonical ms; step counts below use ms_to_steps at default DELTAT_MS).
T_ON_MS = 500.0
IMPULSE_MAXTIME_MS = 2000.0  # Borst spot / impulse horizon

# Moving-bar per-column cost window relative to first-stimulus alignment.
COST_WINDOW_MS = 900.0
COST_ALIGNED_FIRST_STI_MS = 300.0
COST_WINDOW_BEFORE_MS = COST_ALIGNED_FIRST_STI_MS
COST_WINDOW_AFTER_MS = COST_WINDOW_MS - COST_ALIGNED_FIRST_STI_MS

# Post-sweep tail: baseline after bar exit through ``t_first_sti + after`` plus pad.
T_TAIL_PAD_MS = 50.0
MOVING_BAR_TAIL_MS = COST_WINDOW_AFTER_MS + T_TAIL_PAD_MS


def ms_to_steps(ms: float, *, deltat_ms: float = DELTAT_MS) -> int:
    """Convert milliseconds to simulation steps (rounded)."""
    return int(round(float(ms) / float(deltat_ms)))


T_ON = ms_to_steps(T_ON_MS)
IMPULSE_MAXTIME = ms_to_steps(IMPULSE_MAXTIME_MS)
T_TAIL = ms_to_steps(MOVING_BAR_TAIL_MS)
COST_WINDOW_BEFORE = ms_to_steps(COST_WINDOW_BEFORE_MS)
COST_WINDOW_AFTER = ms_to_steps(COST_WINDOW_AFTER_MS)
COST_WINDOW = ms_to_steps(COST_WINDOW_MS) + 1
