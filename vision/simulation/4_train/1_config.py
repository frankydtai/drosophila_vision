# -*- coding: utf-8 -*-
"""Train vocabulary: task names, cost part_keys, CLI aliases, and run paths.

Pure data + parsing (no torch engine, no session objects), so
:mod:`train.param`, :mod:`train.session`, and :mod:`train.cost` can import it
without a cycle. Session assembly and sti-opts finalisation live in
:mod:`train.session`.

**Enum allowed-token sets** (e.g. ``COST_NORMS``, ``SPREAD_GT_MODES``) live here.
Matching **default scalars** live only in ``config`` (e.g. ``TRAIN_OPTIMIZATION['cost_norm']``,
``NEURON_SCHEMA['model']``, ``NEURON_SCHEMA['filter']``, ``SPREAD_PACK['spread_gt_mode']``) — never put the ``(…)`` allowed tuple in ``config``.

``task`` ∈ {spread, spot, moving_bar} and ``contrast`` ∈ {bright, dark} are independent axes.
``i_sti[task][contrast]`` holds only bright/dark currents; baseline is their midpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


TASKS = ("spread", "spot", "moving_bar")
CONTRASTS = ("bright", "dark")

# Neuron dynamics model (``--model``). Default scalar: ``config.NEURON_SCHEMA['model']``.
MODELS = ("borst", "hp_lp")

PD_ND_LABELS = ("PD", "ND")
PD_IDX, ND_IDX = 0, 1

# Waveform MSE normalization (``--cost-norm``); shared by spot + moving_bar MSE.
# gt_power: 100 * Σ W (v_readout−gt_aff)² / Σ W (a_gt·gt)²
# a_gt2:         Σ W (v_readout−gt_aff)² / a_gt²   (per-entry a_i²; bias not in denom)
COST_NORMS = ("gt_power", "a_gt2")

# Spread cost GT mode allowed tokens (``--spread-gt-mode``). Default scalar:
# ``config.SPREAD_PACK['spread_gt_mode']`` (all | positive — comment only there).
SPREAD_GT_MODES = ("all", "pos")

# t=0 pre steady (``--pre-steady``); not param init.
PRE_STEADY_MODES = ("probe", "solve")

# Voltage vs Ca readout (``--filter``). Default scalar: ``config.NEURON_SCHEMA['filter']``.
FILTERS = ("none", "ca")


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
    return _require_choice(pre_steady, PRE_STEADY_MODES, flag="pre_steady")


def expand_filter(filter) -> str:
    """Validate ``--filter`` token (``none`` | ``ca``)."""
    return _require_choice(filter, FILTERS, flag="filter")


def expand_spread_gt_mode(spread_gt_mode) -> str:
    """Validate ``--spread-gt-mode`` against :data:`SPREAD_GT_MODES` (no aliases)."""
    return _require_choice(spread_gt_mode, SPREAD_GT_MODES, flag="spread_gt_mode")


def moving_bar_cost_part_key(task: str, contrast: str, part: str) -> str:
    return f"{task}_{contrast}_{part}"


def spread_cost_part_key(task: str, contrast: str, cell: str) -> str:
    """Fine spread part: ``{task}_{contrast}_{cell}``."""
    return f"{task}_{contrast}_{cell}"


def spot_cost_part_key(task: str, contrast: str, cell: str, radius) -> str:
    """Fine spot part: ``{task}_{contrast}_{cell}_r{radius}``."""
    return f"{task}_{contrast}_{cell}_r{int(radius)}"


def moving_bar_cell_cost_part_key(task: str, contrast: str, cell: str, part: str) -> str:
    """Fine moving-bar waveform part: ``{task}_{contrast}_{cell}_{PD|ND}``."""
    return f"{task}_{contrast}_{cell}_{part}"


def cost_part_keys_from_task(task: str, contrasts=CONTRASTS) -> Tuple[str, ...]:
    """Coarse part_keys for CLI ``--part-cost-scale`` (before packs exist)."""
    if task == "moving_bar":
        return tuple(
            moving_bar_cost_part_key(task, contrast, part)
            for contrast in contrasts
            for part in (*PD_ND_LABELS, "DSI")
        )
    if task not in TASKS:
        raise KeyError(task)
    return tuple(f"{task}_{contrast}" for contrast in contrasts)


PART_COST_SCALE_ALIASES = {
    **{task: cost_part_keys_from_task(task) for task in TASKS},
    "PD": tuple(f"moving_bar_{contrast}_PD" for contrast in CONTRASTS),
    "ND": tuple(f"moving_bar_{contrast}_ND" for contrast in CONTRASTS),
    "DSI": tuple(f"moving_bar_{contrast}_DSI" for contrast in CONTRASTS),
}


def _scale_at(cli: dict, *keys: str) -> float:
    for key in keys:
        if key in cli:
            return float(cli[key])
    return 1.0


def _cost_part_keys_from_pack_entries(
    pack, connectome, entry_part_key, *, cli=None, scales=None,
) -> Tuple[str, ...]:
    """Walk active cost entries; ``entry_part_key(entry, cell)`` → ``(key, *scale_keys)``."""
    entry_mask = pack.cost_scales > 0
    if not bool(entry_mask.any()):
        return ()
    part_keys: List[str] = []
    seen = set()
    cell_idxs = connectome.node_cells[pack.entry_nodes]
    cells = connectome.cells
    for entry in range(int(pack.entry_nodes.shape[0])):
        if not bool(entry_mask[entry]):
            continue
        cell = str(cells[int(cell_idxs[entry].item())])
        key, *scale_keys = entry_part_key(entry, cell)
        if key in seen:
            continue
        seen.add(key)
        part_keys.append(key)
        if scales is not None:
            scales[key] = _scale_at(cli or {}, key, *scale_keys)
    return tuple(part_keys)


def cost_part_keys_from_pack(pack, connectome, *, cli=None, scales=None) -> Tuple[str, ...]:
    """Fine part_keys from pack entries; optional ``scales`` fill from ``cli``."""
    task, contrast = pack.task, pack.contrast
    coarse = f"{task}_{contrast}"
    if task == "moving_bar":
        pd_nd = pack.cost_pd_nds
        if pd_nd is None:
            part_keys: Tuple[str, ...] = ()
        else:
            def entry_part_key(entry, cell):
                part = PD_ND_LABELS[int(pd_nd[entry].item())]
                return (
                    moving_bar_cell_cost_part_key(task, contrast, cell, part),
                    moving_bar_cost_part_key(task, contrast, part),
                    coarse,
                )
            part_keys = _cost_part_keys_from_pack_entries(
                pack, connectome, entry_part_key, cli=cli, scales=scales,
            )
        if pack.dsi_pos_ptr is not None and int(pack.dsi_pos_ptr.numel()) > 1:
            key = moving_bar_cost_part_key(task, contrast, "DSI")
            if key not in part_keys:
                part_keys = (*part_keys, key)
                if scales is not None:
                    scales[key] = _scale_at(cli or {}, key, coarse)
        return part_keys
    if task == "spot":
        if pack.entry_radii is None:
            return ()
        radii = pack.entry_radii

        def entry_part_key(entry, cell):
            return spot_cost_part_key(
                task, contrast, cell, int(radii[entry].item()),
            ), coarse
    elif task == "spread":
        def entry_part_key(entry, cell):
            return spread_cost_part_key(task, contrast, cell), coarse
    else:
        raise KeyError(task)
    return _cost_part_keys_from_pack_entries(
        pack, connectome, entry_part_key, cli=cli, scales=scales,
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
    if not scales:
        return {}
    out: Dict[str, float] = {}
    for token, val in scales.items():
        targets = (
            PART_COST_SCALE_ALIASES[token]
            if token in PART_COST_SCALE_ALIASES
            else (str(token),)
        )
        for target in targets:
            out[target] = float(val)
    return out


def _parse_allowed(values, allowed: Tuple[str, ...], *, noun: str) -> List[str]:
    if values is None:
        raise ValueError(f"{noun}s required")
    if isinstance(values, str):
        values = parse_comma_list(values)
    out = [str(token).strip() for token in values]
    if not out:
        raise ValueError(f"{noun}s must not be empty")
    bad = [token for token in out if token not in allowed]
    if bad:
        raise ValueError(
            f"unknown {noun}(s) {bad!r} (expected {'|'.join(allowed)})",
        )
    return out


def parse_tasks(tasks) -> List[str]:
    """Parse comma-list or sequence into validated spread | spot | moving_bar tasks."""
    return _parse_allowed(tasks, TASKS, noun="task")


def parse_contrasts(contrasts) -> List[str]:
    """Parse comma-list or sequence into validated ``bright`` | ``dark`` contrasts."""
    return _parse_allowed(contrasts, CONTRASTS, noun="contrast")
