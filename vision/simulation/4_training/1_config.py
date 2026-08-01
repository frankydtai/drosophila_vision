# -*- coding: utf-8 -*-
"""Training vocabulary: task names, cost-part keys, CLI aliases, and run paths.

Pure data + parsing (no torch engine, no session objects), so both
:mod:`training.readout_pack` and :mod:`training.cost` can import it without a
cycle. Session assembly and stimulus-opts finalisation live in
:mod:`training.session`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import network.path  # noqa: F401 -- FAFB path on sys.path
from import_bootstrap import parse_comma_list

# simulation root (training/ → parent).
SIMULATION_DIR = Path(__file__).resolve().parent.parent

# Trained-parameter output root (``hp_lp/`` and ``borst/`` run_* subdirs).
PARAMETER_DIR = SIMULATION_DIR / "0_runs"

# Per-run artifact subfolder (``.npy`` / ``.npz``, ``train_opts.json``, ``param_schema.json``).
RUN_DATA_SUBDIR = "data"

# Per-run CSV summaries written next to PNGs under ``<run_name>/`` (not under data/).
PARAM_CSV = "param.csv"
SYN_STRENGTH_CELL_CSV = "syn_strength_cell.csv"
SYN_STRENGTH_EDGE_CSV = "syn_strength_edge.csv"


def run_data_dir(outdir: str | Path) -> str:
    return str(Path(outdir) / RUN_DATA_SUBDIR)


TRAIN_OPTS_FILE = "train_opts.json"

SPOT_TASKS = ("spot_bright", "spot_dark")
MOVING_BAR_TASKS = ("moving_bar_bright", "moving_bar_dark")
VALID_TASKS = SPOT_TASKS + MOVING_BAR_TASKS

_SPOT_BASELINE_KEY = "i_baseline_spot"
_MOVING_BAR_BASELINE_KEY = "i_baseline_moving_bar"
_SPOT_I_KEY = {"bright": "i_bright_spot", "dark": "i_dark_spot"}
_MOVING_BAR_I_KEY = {"bright": "i_bright_moving_bar", "dark": "i_dark_moving_bar"}

PD_ND_LABELS = ("PD", "ND")
PD_IDX, ND_IDX = 0, 1
MOVING_BAR_COST_PARTS = tuple(
    f"{t}_{lab}" for t in MOVING_BAR_TASKS for lab in (*PD_ND_LABELS, "DSI")
)
TASK_ALIASES = {
    "spot": SPOT_TASKS,
    "moving_bar": MOVING_BAR_TASKS,
}
CLI_TASK_NAMES = VALID_TASKS + tuple(TASK_ALIASES.keys())
TASK_I_FIELDS = {
    "spot_bright": frozenset({_SPOT_BASELINE_KEY, "i_bright_spot"}),
    "spot_dark": frozenset({_SPOT_BASELINE_KEY, "i_dark_spot"}),
    "moving_bar_bright": frozenset({_MOVING_BAR_BASELINE_KEY, "i_bright_moving_bar"}),
    "moving_bar_dark": frozenset({_MOVING_BAR_BASELINE_KEY, "i_dark_moving_bar"}),
}
I_CLI_BRIGHT_TASKS = {
    "spot": ("spot_bright",),
    "spot_bright": ("spot_bright",),
    "moving_bar": ("moving_bar_bright",),
    "moving_bar_bright": ("moving_bar_bright",),
}
I_CLI_DARK_TASKS = {
    "spot": ("spot_dark",),
    "spot_dark": ("spot_dark",),
    "moving_bar": ("moving_bar_dark",),
    "moving_bar_dark": ("moving_bar_dark",),
}
I_CLI_SIDECAR_FIELD = {
    ("i_baseline", "spot_bright"): _SPOT_BASELINE_KEY,
    ("i_baseline", "spot_dark"): _SPOT_BASELINE_KEY,
    ("i_baseline", "moving_bar_bright"): _MOVING_BAR_BASELINE_KEY,
    ("i_baseline", "moving_bar_dark"): _MOVING_BAR_BASELINE_KEY,
    ("i_bright", "spot_bright"): "i_bright_spot",
    ("i_bright", "moving_bar_bright"): "i_bright_moving_bar",
    ("i_dark", "spot_dark"): "i_dark_spot",
    ("i_dark", "moving_bar_dark"): "i_dark_moving_bar",
}
COST_WEIGHT_ALIASES = {
    "spot": SPOT_TASKS,
    "moving_bar": MOVING_BAR_COST_PARTS,
    "moving_bar_bright": (
        "moving_bar_bright_PD", "moving_bar_bright_ND", "moving_bar_bright_DSI",
    ),
    "moving_bar_dark": (
        "moving_bar_dark_PD", "moving_bar_dark_ND", "moving_bar_dark_DSI",
    ),
    "PD": ("moving_bar_bright_PD", "moving_bar_dark_PD"),
    "ND": ("moving_bar_bright_ND", "moving_bar_dark_ND"),
    "DSI": ("moving_bar_bright_DSI", "moving_bar_dark_DSI"),
}


def moving_bar_cost_part_key(task_name: str, part: str) -> str:
    return f"{task_name}_{part}"


def cost_part_keys_for_readout(task_name: str) -> Tuple[str, ...]:
    if task_name in MOVING_BAR_TASKS:
        return tuple(
            moving_bar_cost_part_key(task_name, lab)
            for lab in (*PD_ND_LABELS, "DSI")
        )
    return (task_name,)


def session_cost_part_keys(task_list) -> Tuple[str, ...]:
    keys = []
    for name in task_list:
        keys.extend(cost_part_keys_for_readout(name))
    return tuple(keys)


def expand_task_list(names) -> List[str]:
    """Expand ``--task`` ``TASK_ALIASES`` shorthands."""
    out = []
    for name in names:
        if name in TASK_ALIASES:
            out.extend(TASK_ALIASES[name])
        else:
            out.append(name)
    return out


def expand_cost_extent_dict(kv: Optional[dict]) -> Dict[str, int]:
    """Expand ``--cost-extent`` ``TASK_ALIASES`` keys."""
    if not kv:
        return {}
    out: Dict[str, int] = {}
    for name, val in kv.items():
        if name in TASK_ALIASES:
            for t in TASK_ALIASES[name]:
                out[t] = int(val)
        else:
            out[str(name)] = int(val)
    return out


def expand_gt_dict(kv: Optional[dict]) -> Dict[str, List[str]]:
    """Expand ``--gt`` ``TASK_ALIASES`` keys; values are cell-token lists."""
    if not kv:
        return {}
    out: Dict[str, List[str]] = {}
    for name, cells in kv.items():
        cell_list = [str(c) for c in cells]
        if name in TASK_ALIASES:
            for t in TASK_ALIASES[name]:
                out[t] = list(cell_list)
        else:
            out[str(name)] = cell_list
    return out


def resolve_cost_extent_by_task(task_list, default, by_task_kv) -> Dict[str, int]:
    """Map each concrete task to its explicitly requested cost extent."""
    expanded = expand_cost_extent_dict(by_task_kv or {})
    bad = [k for k in expanded if k not in VALID_TASKS]
    if bad:
        raise ValueError(
            f"unknown task(s) in --cost-extent: {bad} "
            f"(expected {'|'.join(CLI_TASK_NAMES)})",
        )
    out: Dict[str, int] = {}
    for tname in task_list:
        if tname in expanded:
            out[tname] = int(expanded[tname])
        elif default is not None:
            out[tname] = int(default)
    return out


def expand_cost_weight_dict(weights: Optional[dict]) -> Dict[str, float]:
    """Expand ``--cost-weight`` ``COST_WEIGHT_ALIASES`` keys."""
    if not weights:
        return {}
    out: Dict[str, float] = {}
    for name, val in weights.items():
        if name in COST_WEIGHT_ALIASES:
            for t in COST_WEIGHT_ALIASES[name]:
                out[t] = float(val)
        else:
            out[str(name)] = float(val)
    return out


def normalize_task_list(task_list) -> List[str]:
    if task_list is None:
        raise ValueError("task_list required")
    if isinstance(task_list, str):
        task_list = parse_comma_list(task_list)
    tl = expand_task_list(list(task_list))
    if not tl:
        raise ValueError("task_list must not be empty")
    bad = [t for t in tl if t not in VALID_TASKS]
    if bad:
        raise ValueError(
            f"unknown task(s) {bad!r} (expected {'|'.join(CLI_TASK_NAMES)})",
        )
    return tl
