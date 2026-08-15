# -*- coding: utf-8 -*-
"""Train vocabulary: task names, cost-part keys, CLI aliases, and run paths.

Pure data + parsing (no torch engine, no session objects), so
:mod:`train.param`, :mod:`train.session`, and :mod:`train.cost` can import it
without a cycle. Session assembly and sti-opts finalisation live in
:mod:`train.session`.

**Enum allowed-token sets** (e.g. ``COST_NORMS``, ``SPOT_GT_MODES``) live here.
Matching **default scalars** live only in ``default_params`` (e.g. ``TRAIN_OPTIMIZATION['cost_norm']``,
``NEURON_FILTER['filter']``, ``SPOT_PACK['spot_gt_mode']``) — never put the ``(…)`` allowed tuple in ``default_params``.
"""
from __future__ import annotations

from default_params import (
    SPOT_PACK,
    TRAIN_OPTIMIZATION,
)

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import network.path  # noqa: F401 -- FAFB path on sys.path
from import_bootstrap import parse_comma_list

# Trained-parameter output root (``hp_lp/`` and ``borst/`` run_* subdirs).
PARAMETER_DIR = Path(__file__).resolve().parent.parent / "0_runs"

# Per-run data subfolder (``.npy`` / ``.npz``, ``train_opts.json``, ``param_schema.json``).
RUN_DATA_SUBDIR = "data"

# Per-run CSV summaries written next to PNGs under ``<run>/`` (not under data/).
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

# Waveform MSE normalization (``--cost-norm``); shared by spot + moving_bar MSE.
# gt_power: 100 * Σ w (v_readout−gt_aff)² / Σ w (a_gt·gt)²
# a_gt2:         Σ w (v_readout−gt_aff)² / a_gt²   (per-entry a_i²; bias not in denom)
COST_NORMS = ("gt_power", "a_gt2")

# Spot cost GT mode allowed tokens (``--spot-gt-mode``). Default scalar:
# ``default_params.SPOT_PACK['spot_gt_mode']`` (all | positive — comment only there).
# all: every active gt cell under both bright and dark (dark × contrast_sign −1).
# positive: only cells with rf_sign × contrast_sign > 0 (bright: ON; dark: OFF).
SPOT_GT_MODES = ("all", "pos")

# t=0 membrane pre steady (``--pre-steady``); not param init.


def _expand_choice(name, allowed: Tuple[str, ...], *, flag: str) -> str:
    key = str(name).strip()
    if key not in allowed:
        raise ValueError(f"{flag} must be one of {allowed}; got {key!r}")
    return key


def expand_cost_norm(name) -> str:
    """Validate ``--cost-norm`` token; canonical names only (no aliases)."""
    return _expand_choice(name, COST_NORMS, flag="cost_norm")


def expand_pre_steady(pre_steady) -> str:
    """Validate ``--pre-steady`` token (``probe`` | ``solve``)."""
    return _expand_choice(pre_steady, ("probe", "solve"), flag="pre_steady")


def expand_filter(filter) -> str:
    """Validate ``--filter`` token (``none`` | ``ca``)."""
    return _expand_choice(filter, ("none", "ca"), flag="filter")


def expand_spot_gt_mode(spot_gt_mode) -> str:
    """Validate ``--spot-gt-mode`` against :data:`SPOT_GT_MODES` (no aliases)."""
    return _expand_choice(spot_gt_mode, SPOT_GT_MODES, flag="spot_gt_mode")


TASK_ALIASES = {
    "spot": SPOT_TASKS,
    "moving_bar": MOVING_BAR_TASKS,
}
CLI_TASK_NAMES = VALID_TASKS + tuple(TASK_ALIASES.keys())
I_STI_KEYS = ("bright", "baseline", "dark")
I_STI_PARADIGMS = ("spot", "moving_bar")
TASK_I_FIELDS = {
    "spot_bright": frozenset({_SPOT_BASELINE_KEY, "i_bright_spot"}),
    "spot_dark": frozenset({_SPOT_BASELINE_KEY, "i_dark_spot"}),
    "moving_bar_bright": frozenset({_MOVING_BAR_BASELINE_KEY, "i_bright_moving_bar"}),
    "moving_bar_dark": frozenset({_MOVING_BAR_BASELINE_KEY, "i_dark_moving_bar"}),
}
PART_COST_SCALE_ALIASES = {
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


def moving_bar_cost_part_key(task: str, part: str) -> str:
    return f"{task}_{part}"


def spot_cost_part_key(task: str, cell: str, radius) -> str:
    """Fine spot part: ``{task}_{cell}_r{radius}`` (only radii with cost readout)."""
    r = float(radius)
    r_s = str(int(r)) if r == int(r) else str(r)
    return f"{task}_{cell}_r{r_s}"


def moving_bar_cell_cost_part_key(task: str, cell: str, part: str) -> str:
    """Fine moving-bar waveform part: ``{task}_{cell}_{PD|ND}``."""
    return f"{task}_{cell}_{part}"


def cost_part_keys_for_task(task: str) -> Tuple[str, ...]:
    """Coarse keys for CLI ``--part-cost-scale`` (before packs exist)."""
    if task in MOVING_BAR_TASKS:
        return tuple(
            moving_bar_cost_part_key(task, lab)
            for lab in (*PD_ND_LABELS, "DSI")
        )
    return (task,)


def cost_part_keys_for_pack(pack, backend) -> Tuple[str, ...]:
    """Fine keys from pack entries with ``cost_scales > 0`` (+ pack-level DSI)."""
    net = backend.network
    if net is None:
        raise ValueError("cost_part_keys_for_pack requires backend.network")
    w = pack.cost_scales
    entry_mask = w > 0
    cell_ids = net.node_cells[pack.entry_nodes]
    cells = net.cells
    keys: List[str] = []
    seen = set()

    def _add(key: str) -> None:
        if key not in seen:
            seen.add(key)
            keys.append(key)

    if pack.task in MOVING_BAR_TASKS:
        pd_nd = pack.cost_pd_nds
        if pd_nd is not None and bool(entry_mask.any()):
            for i in range(int(pack.entry_nodes.shape[0])):
                if not bool(entry_mask[i]):
                    continue
                cell = str(cells[int(cell_ids[i].item())])
                lab = PD_ND_LABELS[int(pd_nd[i].item())]
                _add(moving_bar_cell_cost_part_key(pack.task, cell, lab))
        if (
            pack.dsi_pos_ptr is not None
            and int(pack.dsi_pos_ptr.numel()) > 1
        ):
            _add(moving_bar_cost_part_key(pack.task, "DSI"))
        return tuple(keys)

    if pack.entry_radii is None or not bool(entry_mask.any()):
        return tuple(keys)
    radii = pack.entry_radii
    for i in range(int(pack.entry_nodes.shape[0])):
        if not bool(entry_mask[i]):
            continue
        cell = str(cells[int(cell_ids[i].item())])
        _add(spot_cost_part_key(pack.task, cell, float(radii[i].item())))
    return tuple(keys)


def session_cost_part_keys(tasks, session=None) -> Tuple[str, ...]:
    """Cost-part keys for ``tasks``.

    With ``session``, discover fine per-cell keys from packs; otherwise
    return coarse CLI keys (``spot_bright``, ``moving_bar_*_PD``, …).
    """
    if session is not None:
        keys: List[str] = []
        for task in session.tasks:
            keys.extend(cost_part_keys_for_pack(session.pack_for(task), session.backend))
        return tuple(keys)
    keys = []
    for task in tasks:
        keys.extend(cost_part_keys_for_task(task))
    return tuple(keys)


def coarse_scale_keys_for_part(part_key: str) -> Tuple[str, ...]:
    """Parent coarse keys a fine part inherits ``part_cost_scales`` from."""
    for lab in (*PD_ND_LABELS, "DSI"):
        suf = f"_{lab}"
        if not part_key.endswith(suf):
            continue
        body = part_key[: -len(suf)]
        for task in sorted(MOVING_BAR_TASKS, key=len, reverse=True):
            if body == task or body.startswith(f"{task}_"):
                return (moving_bar_cost_part_key(task, lab), task)
        return ()
    for task in sorted(SPOT_TASKS, key=len, reverse=True):
        if part_key == task:
            return ()
        prefix = f"{task}_"
        if part_key.startswith(prefix) and "_r" in part_key[len(prefix):]:
            return (task,)
    return ()


def expand_tasks(tasks) -> List[str]:
    """Expand ``--task`` ``TASK_ALIASES`` shorthands."""
    out = []
    for task in tasks:
        out.extend(TASK_ALIASES.get(task, (task,)))
    return out


def _expand_alias_dict(kv: Optional[dict], aliases: dict, map_value) -> dict:
    if not kv:
        return {}
    out = {}
    for key, val in kv.items():
        targets = aliases[key] if key in aliases else (str(key),)
        for t in targets:
            out[t] = map_value(val)
    return out


def expand_cost_radius_dict(kv: Optional[dict]) -> Dict[str, int]:
    """Expand ``--cost-radius`` ``TASK_ALIASES`` keys."""
    return _expand_alias_dict(kv, TASK_ALIASES, int)


def expand_gt_dict(kv: Optional[dict]) -> Dict[str, List[str]]:
    """Expand ``--gt`` ``TASK_ALIASES`` keys; values are cell-token lists."""
    return _expand_alias_dict(kv, TASK_ALIASES, lambda cells: [str(c) for c in cells])


def resolve_cost_radius_by_task(tasks, bare_cost_radius, by_task_kv) -> Dict[str, int]:
    """Map each concrete task to its explicitly requested cost radius."""
    expanded = expand_cost_radius_dict(by_task_kv or {})
    bad = [k for k in expanded if k not in VALID_TASKS]
    if bad:
        raise ValueError(
            f"unknown task(s) in --cost-radius: {bad} "
            f"(expected {'|'.join(CLI_TASK_NAMES)})",
        )
    out: Dict[str, int] = {}
    for task in tasks:
        if task in expanded:
            out[task] = int(expanded[task])
        elif bare_cost_radius is not None:
            out[task] = int(bare_cost_radius)
    return out


def expand_part_cost_scale_dict(scales: Optional[dict]) -> Dict[str, float]:
    """Expand ``--part-cost-scale`` ``PART_COST_SCALE_ALIASES`` keys."""
    return _expand_alias_dict(scales, PART_COST_SCALE_ALIASES, float)


def resolve_tasks(tasks) -> List[str]:
    """Expand task aliases and validate concrete task names."""
    if tasks is None:
        raise ValueError("tasks required")
    if isinstance(tasks, str):
        tasks = parse_comma_list(tasks)
    tl = expand_tasks(list(tasks))
    if not tl:
        raise ValueError("tasks must not be empty")
    bad = [t for t in tl if t not in VALID_TASKS]
    if bad:
        raise ValueError(
            f"unknown task(s) {bad!r} (expected {'|'.join(CLI_TASK_NAMES)})",
        )
    return tl
