# -*- coding: utf-8 -*-
"""Train vocabulary: task names, cost part_keys, CLI aliases, and run paths.

Pure data + parsing (no torch engine, no session objects), so
:mod:`train.param`, :mod:`train.session`, and :mod:`train.cost` can import it
without a cycle. Session assembly and sti-opts finalisation live in
:mod:`train.session`.

**Enum allowed-token sets** (e.g. ``COST_NORMS``, ``SPOT_GT_MODES``) live here.
Matching **default scalars** live only in ``const_default`` (e.g. ``TRAIN_OPTIMIZATION['cost_norm']``,
``NEURON_SCHEMA['model']``, ``NEURON_SCHEMA['filter']``, ``SPOT_PACK['spot_gt_mode']``) — never put the ``(…)`` allowed tuple in ``const_default``.

``task`` ∈ {spot, moving_bar} and ``contrast`` ∈ {bright, dark} are independent axes.
``i_sti[task][contrast]`` holds only bright/dark currents; baseline is their midpoint.
"""
from __future__ import annotations

from const_default import (
    SPOT_PACK,
    TRAIN_CONFIG,
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


# Exactly two tasks; contrast is a separate axis (exactly two contrasts).
TASKS = ("spot", "moving_bar")
CONTRASTS = ("bright", "dark")

# Neuron dynamics model (``--model``). Default scalar: ``const_default.NEURON_SCHEMA['model']``.
MODELS = ("borst", "hp_lp")

# ``i_sti[task][contrast]`` keys only (baseline = midpoint; see
# ``task.moving_bar.sti_spec.i_baseline_from_i_sti``).

PD_ND_LABELS = ("PD", "ND")
PD_IDX, ND_IDX = 0, 1


MOVING_BAR_COST_PARTS = tuple(
    f"moving_bar_{contrast}_{part}"
    for contrast in CONTRASTS
    for part in (*PD_ND_LABELS, "DSI")
)

# Waveform MSE normalization (``--cost-norm``); shared by spot + moving_bar MSE.
# gt_power: 100 * Σ W (v_readout−gt_aff)² / Σ W (a_gt·gt)²
# a_gt2:         Σ W (v_readout−gt_aff)² / a_gt²   (per-entry a_i²; bias not in denom)
COST_NORMS = ("gt_power", "a_gt2")

# Spot cost GT mode allowed tokens (``--spot-gt-mode``). Default scalar:
# ``const_default.SPOT_PACK['spot_gt_mode']`` (all | positive — comment only there).
# all: every active gt cell under both bright and dark (dark × contrast_sign −1).
# positive: only cells with rf_sign × contrast_sign > 0 (bright: ON; dark: OFF).
SPOT_GT_MODES = ("all", "pos")

# t=0 pre steady (``--pre-steady``); not param init.


def _require_choice(token, allowed: Tuple[str, ...], *, flag: str) -> str:
    token = str(token).strip()
    if token not in allowed:
        raise ValueError(f"{flag} must be one of {allowed}; got {token!r}")
    return token


def expand_cost_norm(name) -> str:
    """Validate ``--cost-norm`` token; canonical names only (no aliases)."""
    return _require_choice(name, COST_NORMS, flag="cost_norm")


def expand_pre_steady(pre_steady) -> str:
    """Validate ``--pre-steady`` token (``probe`` | ``solve``)."""
    return _require_choice(pre_steady, ("probe", "solve"), flag="pre_steady")


def expand_filter(filter) -> str:
    """Validate ``--filter`` token (``none`` | ``ca``)."""
    return _require_choice(filter, ("none", "ca"), flag="filter")


def expand_spot_gt_mode(spot_gt_mode) -> str:
    """Validate ``--spot-gt-mode`` against :data:`SPOT_GT_MODES` (no aliases)."""
    return _require_choice(spot_gt_mode, SPOT_GT_MODES, flag="spot_gt_mode")


PART_COST_SCALE_ALIASES = {
    "spot": tuple(f"spot_{contrast}" for contrast in CONTRASTS),
    "moving_bar": MOVING_BAR_COST_PARTS,
    "PD": tuple(f"moving_bar_{contrast}_PD" for contrast in CONTRASTS),
    "ND": tuple(f"moving_bar_{contrast}_ND" for contrast in CONTRASTS),
    "DSI": tuple(f"moving_bar_{contrast}_DSI" for contrast in CONTRASTS),
}


def moving_bar_cost_part_key(task: str, contrast: str, part: str) -> str:
    return f"{task}_{contrast}_{part}"


def spot_cost_part_key(task: str, contrast: str, cell: str, radius) -> str:
    """Fine spot part: ``{task}_{contrast}_{cell}_r{radius}``."""
    return f"{task}_{contrast}_{cell}_r{int(radius)}"


def moving_bar_cell_cost_part_key(task: str, contrast: str, cell: str, part: str) -> str:
    """Fine moving-bar waveform part: ``{task}_{contrast}_{cell}_{PD|ND}``."""
    return f"{task}_{contrast}_{cell}_{part}"


def _spot_cost_part_keys_from_task(task: str, contrasts) -> Tuple[str, ...]:
    return tuple(f"{task}_{contrast}" for contrast in contrasts)


def _moving_bar_cost_part_keys_from_task(task: str, contrasts) -> Tuple[str, ...]:
    return tuple(
        moving_bar_cost_part_key(task, contrast, part)
        for contrast in contrasts
        for part in (*PD_ND_LABELS, "DSI")
    )


_COST_PART_KEYS_FROM_TASK = {
    "spot": _spot_cost_part_keys_from_task,
    "moving_bar": _moving_bar_cost_part_keys_from_task,
}


def cost_part_keys_from_task(task: str, contrasts=CONTRASTS) -> Tuple[str, ...]:
    """Coarse part_keys for CLI ``--part-cost-scale`` (before packs exist)."""
    return _COST_PART_KEYS_FROM_TASK[task](task, contrasts)


def _scale_at(cli: dict, *keys: str) -> float:
    for key in keys:
        if key in cli:
            return float(cli[key])
    return 1.0


def _spot_cost_part_keys_from_pack(pack, connectome, *, cli=None, scales=None):
    part_keys: List[str] = []
    seen = set()
    entry_mask = pack.cost_scales > 0
    if pack.entry_radii is None or not bool(entry_mask.any()):
        return tuple(part_keys)
    cell_idxs = connectome.node_cells[pack.entry_nodes]
    cells = connectome.cells
    radii = pack.entry_radii
    task, contrast = pack.task, pack.contrast
    coarse = f"{task}_{contrast}"
    for entry in range(int(pack.entry_nodes.shape[0])):
        if not bool(entry_mask[entry]):
            continue
        cell = str(cells[int(cell_idxs[entry].item())])
        key = spot_cost_part_key(task, contrast, cell, int(radii[entry].item()))
        if key in seen:
            continue
        seen.add(key)
        part_keys.append(key)
        if scales is not None:
            scales[key] = _scale_at(cli or {}, key, coarse)
    return tuple(part_keys)


def _moving_bar_cost_part_keys_from_pack(pack, connectome, *, cli=None, scales=None):
    part_keys: List[str] = []
    seen = set()
    entry_mask = pack.cost_scales > 0
    cell_idxs = connectome.node_cells[pack.entry_nodes]
    cells = connectome.cells
    task, contrast = pack.task, pack.contrast
    coarse = f"{task}_{contrast}"
    pd_nd = pack.cost_pd_nds
    if pd_nd is not None and bool(entry_mask.any()):
        for entry in range(int(pack.entry_nodes.shape[0])):
            if not bool(entry_mask[entry]):
                continue
            cell = str(cells[int(cell_idxs[entry].item())])
            part = PD_ND_LABELS[int(pd_nd[entry].item())]
            key = moving_bar_cell_cost_part_key(task, contrast, cell, part)
            part_key = moving_bar_cost_part_key(task, contrast, part)
            if key in seen:
                continue
            seen.add(key)
            part_keys.append(key)
            if scales is not None:
                scales[key] = _scale_at(cli or {}, key, part_key, coarse)
    if pack.dsi_pos_ptr is not None and int(pack.dsi_pos_ptr.numel()) > 1:
        key = moving_bar_cost_part_key(task, contrast, "DSI")
        if key not in seen:
            seen.add(key)
            part_keys.append(key)
            if scales is not None:
                scales[key] = _scale_at(cli or {}, key, coarse)
    return tuple(part_keys)


_COST_PART_KEYS_FROM_PACK = {
    "spot": _spot_cost_part_keys_from_pack,
    "moving_bar": _moving_bar_cost_part_keys_from_pack,
}


def cost_part_keys_from_pack(pack, connectome, *, cli=None, scales=None) -> Tuple[str, ...]:
    """Fine part_keys from pack entries; optional ``scales`` fill from ``cli``."""
    return _COST_PART_KEYS_FROM_PACK[pack.task](
        pack, connectome, cli=cli, scales=scales,
    )


def session_cost_part_keys(session=None, *, tasks=None, contrasts=None) -> Tuple[str, ...]:
    """Cost part_keys for session packs, or coarse keys from tasks×contrasts."""
    if session is not None:
        part_keys: List[str] = []
        for pack in session.iter_packs():
            part_keys.extend(cost_part_keys_from_pack(pack, session.connectome))
        return tuple(part_keys)
    if tasks is None:
        raise ValueError("tasks required when session is None")
    contrasts = contrasts if contrasts is not None else CONTRASTS
    part_keys = []
    for task in tasks:
        part_keys.extend(cost_part_keys_from_task(task, contrasts=contrasts))
    return tuple(part_keys)


def _expand_alias(by_token: Optional[dict], aliases: dict, map_value) -> dict:
    if not by_token:
        return {}
    out = {}
    for token, val in by_token.items():
        targets = aliases[token] if token in aliases else (str(token),)
        for target in targets:
            out[target] = map_value(val)
    return out


def expand_gt(by_task: Optional[dict]) -> Dict[str, List[str]]:
    """Copy ``--gt`` task→cells map; values become cell-token lists."""
    if not by_task:
        return {}
    return {
        str(task): [str(cell) for cell in cells]
        for task, cells in by_task.items()
    }


def cost_radius_by_task(tasks, bare_cost_radius, by_task: Optional[dict]) -> Dict[str, int]:
    """Map each task to cost radius from bare N and/or per-task tokens."""
    by_task = by_task or {}
    bad = [task for task in by_task if task not in TASKS]
    if bad:
        raise ValueError(
            f"unknown task(s) in --cost-radius: {bad} "
            f"(expected {'|'.join(TASKS)})",
        )
    out: Dict[str, int] = {}
    for task in tasks:
        if task in by_task:
            out[task] = int(by_task[task])
        elif bare_cost_radius is not None:
            out[task] = int(bare_cost_radius)
    return out


def expand_part_cost_scale(scales: Optional[dict]) -> Dict[str, float]:
    """Expand ``--part-cost-scale`` ``PART_COST_SCALE_ALIASES`` tokens."""
    return _expand_alias(scales, PART_COST_SCALE_ALIASES, float)


def parse_tasks(tasks) -> List[str]:
    """Parse comma-list or sequence into validated ``spot`` | ``moving_bar`` tasks."""
    if tasks is None:
        raise ValueError("tasks required")
    if isinstance(tasks, str):
        tasks = parse_comma_list(tasks)
    out = [str(task).strip() for task in tasks]
    if not out:
        raise ValueError("tasks must not be empty")
    bad = [task for task in out if task not in TASKS]
    if bad:
        raise ValueError(
            f"unknown task(s) {bad!r} (expected {'|'.join(TASKS)})",
        )
    return out


def parse_contrasts(contrasts) -> List[str]:
    """Parse comma-list or sequence into validated ``bright`` | ``dark`` contrasts."""
    if contrasts is None:
        raise ValueError("contrasts required")
    if isinstance(contrasts, str):
        contrasts = parse_comma_list(contrasts)
    out = [str(contrast).strip() for contrast in contrasts]
    if not out:
        raise ValueError("contrasts must not be empty")
    bad = [contrast for contrast in out if contrast not in CONTRASTS]
    if bad:
        raise ValueError(
            f"unknown contrast(s) {bad!r} (expected {'|'.join(CONTRASTS)})",
        )
    return out
