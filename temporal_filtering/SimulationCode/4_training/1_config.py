# -*- coding: utf-8 -*-
"""Training vocabulary: target names, cost-part keys, CLI aliases, and run paths.

Pure data + parsing (no torch engine, no session objects), so both
:mod:`training.target_pack` and :mod:`training.cost` can import it without a
cycle. Session assembly and stimulus-opts finalisation live in
:mod:`training.session`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import network.path  # noqa: F401 -- connectome_io on sys.path
from connectome_io import parse_comma_list

# SimulationCode root (training/ → parent).
SIMULATION_DIR = Path(__file__).resolve().parent.parent

# Trained-parameter output root (``hp_lp/`` and ``borst/`` run_* subdirs).
PARAMETER_DIR = SIMULATION_DIR / "0_runs"

# Per-run artifact subfolder (``.npy`` / ``.npz``, ``train_opts.json``, ``param_schema.json``).
RUN_DATA_SUBDIR = "data"

# Per-run CSV summaries written next to PNGs under ``<run_name>/`` (not under data/).
PARAM_CSV = "param.csv"
SYN_STRENGTH_CSV = "syn_strength.csv"
EDGE_WEIGHT_CSV = "edge_weight.csv"


def run_data_dir(outdir: str | Path) -> str:
    return str(Path(outdir) / RUN_DATA_SUBDIR)


TRAIN_OPTS_FILE = "train_opts.json"

SPOT_TARGETS = ("spot_bright", "spot_dark")
MOVING_BAR_TARGETS = ("moving_bar_bright", "moving_bar_dark")
VALID_TARGETS = SPOT_TARGETS + MOVING_BAR_TARGETS

_SPOT_STEP_KEY = {"bright": "i_bright", "dark": "i_dark"}

PD_ND_LABELS = ("PD", "ND")
PD_IDX, ND_IDX = 0, 1
MOVING_BAR_COST_PARTS = tuple(
    f"{t}_{lab}" for t in MOVING_BAR_TARGETS for lab in (*PD_ND_LABELS, "DSI")
)
TARGET_ALIASES = {
    "spot": SPOT_TARGETS,
    "moving_bar": MOVING_BAR_TARGETS,
}
CLI_TARGET_NAMES = VALID_TARGETS + tuple(TARGET_ALIASES.keys())
TARGET_I_FIELDS = {
    "spot_bright": frozenset({"i_baseline", "i_bright"}),
    "spot_dark": frozenset({"i_baseline", "i_dark"}),
    "moving_bar_bright": frozenset({"i_baseline", "i_bright_bar"}),
    "moving_bar_dark": frozenset({"i_baseline", "i_dark_bar"}),
}
I_CLI_BRIGHT_TARGETS = {
    "spot": ("spot_bright",),
    "spot_bright": ("spot_bright",),
    "moving_bar": ("moving_bar_bright",),
    "moving_bar_bright": ("moving_bar_bright",),
}
I_CLI_DARK_TARGETS = {
    "spot": ("spot_dark",),
    "spot_dark": ("spot_dark",),
    "moving_bar": ("moving_bar_dark",),
    "moving_bar_dark": ("moving_bar_dark",),
}
I_CLI_SIDECAR_FIELD = {
    ("i_baseline", "spot_bright"): "i_baseline",
    ("i_baseline", "spot_dark"): "i_baseline",
    ("i_baseline", "moving_bar_bright"): "i_baseline",
    ("i_baseline", "moving_bar_dark"): "i_baseline",
    ("i_bright", "spot_bright"): "i_bright",
    ("i_bright", "moving_bar_bright"): "i_bright_bar",
    ("i_dark", "spot_dark"): "i_dark",
    ("i_dark", "moving_bar_dark"): "i_dark_bar",
}
COST_WEIGHT_ALIASES = {
    "spot": SPOT_TARGETS,
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


def moving_bar_cost_part_key(target_name: str, part: str) -> str:
    return f"{target_name}_{part}"


def cost_part_keys_for_target(target_name: str) -> Tuple[str, ...]:
    if target_name in MOVING_BAR_TARGETS:
        return tuple(
            moving_bar_cost_part_key(target_name, lab)
            for lab in (*PD_ND_LABELS, "DSI")
        )
    return (target_name,)


def session_cost_part_keys(target_list) -> Tuple[str, ...]:
    keys = []
    for name in target_list:
        keys.extend(cost_part_keys_for_target(name))
    return tuple(keys)


def expand_target_list(names) -> List[str]:
    """Expand ``--target`` ``TARGET_ALIASES`` shorthands."""
    out = []
    for name in names:
        if name in TARGET_ALIASES:
            out.extend(TARGET_ALIASES[name])
        else:
            out.append(name)
    return out


def expand_cost_extent_dict(kv: Optional[dict]) -> Dict[str, int]:
    """Expand ``--cost-extent`` ``TARGET_ALIASES`` keys."""
    if not kv:
        return {}
    out: Dict[str, int] = {}
    for name, val in kv.items():
        if name in TARGET_ALIASES:
            for t in TARGET_ALIASES[name]:
                out[t] = int(val)
        else:
            out[str(name)] = int(val)
    return out


def resolve_cost_extent_by_target(target_list, default, by_target_kv) -> Dict[str, int]:
    """Map each concrete target to its explicitly requested cost extent."""
    expanded = expand_cost_extent_dict(by_target_kv or {})
    bad = [k for k in expanded if k not in VALID_TARGETS]
    if bad:
        raise ValueError(
            f"unknown target(s) in --cost-extent: {bad} "
            f"(expected {'|'.join(CLI_TARGET_NAMES)})",
        )
    out: Dict[str, int] = {}
    for tname in target_list:
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


def normalize_target_list(target_list) -> List[str]:
    if target_list is None:
        raise ValueError("target_list required")
    if isinstance(target_list, str):
        target_list = parse_comma_list(target_list)
    tl = expand_target_list(list(target_list))
    if not tl:
        raise ValueError("target_list must not be empty")
    bad = [t for t in tl if t not in VALID_TARGETS]
    if bad:
        raise ValueError(
            f"unknown target(s) {bad!r} (expected {'|'.join(CLI_TARGET_NAMES)})",
        )
    return tl
