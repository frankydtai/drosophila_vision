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

# Trained-parameter output root (``hp_lp/`` and ``borst/`` run_* subdirs).
PARAMETER_DIR = Path(__file__).resolve().parent.parent / "0_runs"

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

# Waveform MSE normalization (``--cost-norm``); shared by spot + moving_bar MSE.
# gt_power: 100 * Σ w (v_readout−gt_aff)² / Σ w (a_gt·gt)²
# a_gt2:         Σ w (v_readout−gt_aff)² / a_gt²   (per-entry a_i²; bias not in denom)
COST_NORMS = ("gt_power", "a_gt2")

# t=0 membrane pre steady (``--pre-steady MODE``); not param init.
# Shared by borst / hp_lp (default: param_defaults.PRE_STEADY).
PRE_STEADY_MODES = ("probe", "solve")


def _expand_choice(name, allowed: Tuple[str, ...], *, flag: str) -> str:
    key = str(name).strip()
    if key not in allowed:
        raise ValueError(f"{flag} must be one of {allowed}; got {key!r}")
    return key


def expand_cost_norm(name) -> str:
    """Validate ``--cost-norm`` token; canonical names only (no aliases)."""
    return _expand_choice(name, COST_NORMS, flag="cost_norm")


def expand_pre_steady_mode(mode) -> str:
    """Validate shared pre-steady mode token (``probe`` | ``solve``)."""
    return _expand_choice(mode, PRE_STEADY_MODES, flag="pre_steady")


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


def spot_cost_part_key(task_name: str, cell: str, radius) -> str:
    """Fine spot part: ``{task}_{cell}_r{radius}`` (only radii with cost readout)."""
    r = float(radius)
    r_s = str(int(r)) if r == int(r) else str(r)
    return f"{task_name}_{cell}_r{r_s}"


def moving_bar_cell_cost_part_key(task_name: str, cell: str, part: str) -> str:
    """Fine moving-bar waveform part: ``{task}_{cell}_{PD|ND}``."""
    return f"{task_name}_{cell}_{part}"


def cost_part_keys_for_readout(task_name: str) -> Tuple[str, ...]:
    """Coarse keys for CLI ``--cost-weight`` (before packs exist)."""
    if task_name in MOVING_BAR_TASKS:
        return tuple(
            moving_bar_cost_part_key(task_name, lab)
            for lab in (*PD_ND_LABELS, "DSI")
        )
    return (task_name,)


def cost_part_keys_for_pack(pack, backend) -> Tuple[str, ...]:
    """Fine keys from pack entries with ``cost_weight > 0`` (+ pack-level DSI)."""
    net = backend.network
    if net is None:
        raise ValueError("cost_part_keys_for_pack requires backend.network")
    w = pack.cost_weight
    active = w > 0
    cell_ids = net.node_cell[pack.readout_node]
    names = net.cell_names
    keys: List[str] = []
    seen = set()

    def _add(key: str) -> None:
        if key not in seen:
            seen.add(key)
            keys.append(key)

    if pack.name in MOVING_BAR_TASKS:
        pd_nd = pack.cost_pd_nd
        if pd_nd is not None and bool(active.any()):
            for i in range(int(pack.readout_node.shape[0])):
                if not bool(active[i]):
                    continue
                cell = str(names[int(cell_ids[i].item())])
                lab = PD_ND_LABELS[int(pd_nd[i].item())]
                _add(moving_bar_cell_cost_part_key(pack.name, cell, lab))
        if (
            pack.dsi_pos_ptr is not None
            and int(pack.dsi_pos_ptr.numel()) > 1
        ):
            _add(moving_bar_cost_part_key(pack.name, "DSI"))
        return tuple(keys)

    if pack.cost_radius is None or not bool(active.any()):
        return tuple(keys)
    radii = pack.cost_radius
    for i in range(int(pack.readout_node.shape[0])):
        if not bool(active[i]):
            continue
        cell = str(names[int(cell_ids[i].item())])
        _add(spot_cost_part_key(pack.name, cell, float(radii[i].item())))
    return tuple(keys)


def session_cost_part_keys(tasks, session=None) -> Tuple[str, ...]:
    """Cost-part keys for ``tasks``.

    With ``session``, discover fine per-cell keys from packs; otherwise
    return coarse CLI keys (``spot_bright``, ``moving_bar_*_PD``, …).
    """
    if session is not None:
        keys: List[str] = []
        for name in session.tasks:
            keys.extend(cost_part_keys_for_pack(session.pack_for(name), session.backend))
        return tuple(keys)
    keys = []
    for name in tasks:
        keys.extend(cost_part_keys_for_readout(name))
    return tuple(keys)


def coarse_weight_keys_for_part(part_key: str) -> Tuple[str, ...]:
    """Parent coarse keys a fine part inherits ``cost_weights`` from."""
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


def expand_tasks(names) -> List[str]:
    """Expand ``--task`` ``TASK_ALIASES`` shorthands."""
    out = []
    for name in names:
        out.extend(TASK_ALIASES.get(name, (name,)))
    return out


def _expand_alias_dict(kv: Optional[dict], aliases: dict, map_value) -> dict:
    if not kv:
        return {}
    out = {}
    for name, val in kv.items():
        targets = aliases[name] if name in aliases else (str(name),)
        for t in targets:
            out[t] = map_value(val)
    return out


def expand_cost_extent_dict(kv: Optional[dict]) -> Dict[str, int]:
    """Expand ``--cost-extent`` ``TASK_ALIASES`` keys."""
    return _expand_alias_dict(kv, TASK_ALIASES, int)


def expand_gt_dict(kv: Optional[dict]) -> Dict[str, List[str]]:
    """Expand ``--gt`` ``TASK_ALIASES`` keys; values are cell-token lists."""
    return _expand_alias_dict(kv, TASK_ALIASES, lambda cells: [str(c) for c in cells])


def resolve_cost_extent_by_task(tasks, default, by_task_kv) -> Dict[str, int]:
    """Map each concrete task to its explicitly requested cost extent."""
    expanded = expand_cost_extent_dict(by_task_kv or {})
    bad = [k for k in expanded if k not in VALID_TASKS]
    if bad:
        raise ValueError(
            f"unknown task(s) in --cost-extent: {bad} "
            f"(expected {'|'.join(CLI_TASK_NAMES)})",
        )
    out: Dict[str, int] = {}
    for tname in tasks:
        if tname in expanded:
            out[tname] = int(expanded[tname])
        elif default is not None:
            out[tname] = int(default)
    return out


def expand_cost_weight_dict(weights: Optional[dict]) -> Dict[str, float]:
    """Expand ``--cost-weight`` ``COST_WEIGHT_ALIASES`` keys."""
    return _expand_alias_dict(weights, COST_WEIGHT_ALIASES, float)


def normalize_tasks(tasks) -> List[str]:
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
