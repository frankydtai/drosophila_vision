"""Shared paths and constants for connectome training targets.

SimulationCode scripts import from here instead of hardcoding paths to
``MatlabFunctions/``, trained-parameter output folders, or cost-window sizes.
FAFB connectome build paths live in ``connectome_io``; this module covers
training data and training output paths.

Spot / impulse step timing is set via ``train.py --t-on-ms`` and
``RESPONSE_DURATION_MS`` below.
"""

from __future__ import annotations

from pathlib import Path

import torch

# SimulationCode root (this file's directory).
SIMULATION_DIR = Path(__file__).resolve().parent
# Repo root: .../drosophila_vision
REPO_ROOT = SIMULATION_DIR.parent.parent

# Trained-parameter output root (``hp_lp/`` and ``conductance/`` run_* subdirs).
PARAMETER_DIR = SIMULATION_DIR / "0trained"

# Per-run artifact subfolder (``.npy`` / ``.npz``, ``train_opts.json``, ``param_schema.json``).
RUN_DATA_SUBDIR = "data"

# Per-run CSV summaries written next to PNGs under ``<run_name>/`` (not under data/).
PARAM_CSV = "param.csv"
SYN_STRENGTH_CSV = "syn_strength.csv"


def run_data_dir(outdir: str | Path) -> str:
    return str(Path(outdir) / RUN_DATA_SUBDIR)

# Gruntman Fig. 1 Ci/Cii digitized population Vm traces (see MatlabFunctions/digitize_fig1_ci.py).
FIG1_CI_NPZ = REPO_ROOT / "MatlabFunctions" / "fig1_ci_digitized.npz"

# Simulation sampling interval (ms per discrete step).
DELTAT_MS = 10.0

SIM_DTYPE_DEFAULT = torch.float64


def sim_dtype_from_fp32(fp32: bool) -> torch.dtype:
    return torch.float32 if fp32 else SIM_DTYPE_DEFAULT

# ---------------------------------------------------------------------------
# Spot / impulse timing (``DELTAT_MS`` = 10 ms per step at default settings).
#
# ``t_on`` (stimulus onset step) and ``maxtime`` (total simulation steps) are
# set per-session via ``train.py --t-on-ms`` (default 500 ms).  Response
# duration after onset is ``RESPONSE_DURATION_MS`` (1500 ms).
#
# Step axis (example at default --t-on-ms 500):
#
#   index:     0 … t_on-1       |  t_on … maxtime-1
#   wall time: 0–500 ms         |  500–2000 ms
#   count:     t_on (=50)       |  maxtime - t_on (=150)
#
#   maxtime: full simulation length; ``signal`` time dim.
#   maxtime - t_on: post-step response window;
#       ``TargetPack.data`` and MSE cost use this window.
#
# ``read_RecF_data()`` shape (13, 9, maxtime); indices ``[:t_on]`` are zero
# (PR step starts at ``t_on``; filtered ImpR is nonzero from ~``t_on+1``).
#
# Moving bar uses separate ``COST_WINDOW`` / ``maxtime``; not the above.
# ---------------------------------------------------------------------------
RESPONSE_DURATION_MS = 1500.0

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


T_TAIL = ms_to_steps(MOVING_BAR_TAIL_MS)
COST_WINDOW_BEFORE = ms_to_steps(COST_WINDOW_BEFORE_MS)
COST_WINDOW_AFTER = ms_to_steps(COST_WINDOW_AFTER_MS)
COST_WINDOW = ms_to_steps(COST_WINDOW_MS) + 1
