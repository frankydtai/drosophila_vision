# -*- coding: utf-8 -*-
"""Train config parse helpers (Hydra config → runtime values).

No argparse registration. Callers: ``config``, ``figure``, ``analyze``.
"""
from __future__ import annotations

from config import MODEL, SPOT_INPUT_SPEC

import sys
from pathlib import Path

_SIMULATION_CODE = Path(__file__).resolve().parent.parent
if str(_SIMULATION_CODE) not in sys.path:
    sys.path.insert(0, str(_SIMULATION_CODE))

import import_bootstrap  # noqa: F401
from import_bootstrap import parse_comma_list
import train
from train.config import CONTRASTS, TASKS


def _format_filename_token(value):
    val = float(value)
    if val == int(val):
        return str(int(val))
    return "%g" % val


def euler_filename_suffix(euler=None):
    """PNG stem suffix for a non-``None`` euler override (``_im`` / ``_ex``)."""
    if euler is None:
        return ""
    token = str(euler)
    if token in ("implicit", "explicit"):
        token = "im" if token == "implicit" else "ex"
    elif token not in ("im", "ex"):
        token = train.expand_euler(token)
        token = "im" if token == "implicit" else "ex"
    return f"_{token}"


def filter_filename_suffix(filter=None):
    """PNG stem suffix for a non-``None`` filter override (``_v`` / ``_ca``)."""
    if filter is None:
        return ""
    expanded = train.expand_filter(filter)
    return f"_{'v' if expanded == 'none' else expanded}"


def param_filename_suffix(param_inits=None, param_vals=None):
    parts = []
    for key, bag in (("init", param_inits or ()), ("val", param_vals or ())):
        for param, node, number in bag:
            bits = [param, key]
            if node is not None:
                bits.append(str(node).replace(":", "_"))
            bits.append(_format_filename_token(number))
            parts.append("_".join(bits))
    if not parts:
        return ""
    return "_" + "_".join(parts)


def parse_i_sti(tokens, tasks=()):
    """Decode i_sti tokens into ``{task: {bright, dark}}`` (no defaults merge)."""
    if not tokens:
        return None
    out = {}
    for token in tokens:
        if "=" in token:
            name, val = token.split("=", 1)
            task = name.strip()
            if task not in TASKS:
                raise ValueError(
                    f"unknown task {task!r} in i_sti "
                    f"(expected {'|'.join(TASKS)})",
                )
            out[task] = _parse_i_sti_value(val.strip())
        else:
            val = _parse_i_sti_value(token.strip())
            for task in tasks:
                out[str(task)] = val
    return out or None


def _parse_i_sti_value(val):
    parts = [part.strip() for part in val.split(",")]
    if len(parts) != len(CONTRASTS):
        raise ValueError(
            f"i_sti expects {','.join(CONTRASTS)} "
            f"({len(CONTRASTS)} values), got {val!r}",
        )
    return {
        contrast: float(part)
        for contrast, part in zip(CONTRASTS, parts)
    }


STI_TIMING_KEYS = (
    "ms_pre",
    "ms_response",
    "ms_post",
    "ms_sti",
    "delta_ms",
    "delta_ms_pre",
)


def parse_sti_timing_keys(tokens) -> dict[str, float]:
    """Parse sti_timing KEY=MS tokens."""
    out: dict[str, float] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"sti_timing expected KEY=MS, got {token!r}")
        sti_timing_key, val = token.split("=", 1)
        sti_timing_key = sti_timing_key.strip()
        val = val.strip()
        if sti_timing_key not in STI_TIMING_KEYS:
            allowed = ", ".join(STI_TIMING_KEYS)
            raise ValueError(
                f"sti_timing unknown key {sti_timing_key!r}; allowed: {allowed}"
            )
        try:
            out[sti_timing_key] = float(val)
        except ValueError as exc:
            raise ValueError(
                f"sti_timing {sti_timing_key}={val!r} is not a number"
            ) from exc
    return out


def _resolve_sti_timing_literals() -> dict:
    sti_opts: dict = {}
    for sti_timing_key in STI_TIMING_KEYS:
        if sti_timing_key in ("delta_ms", "delta_ms_pre"):
            ms = MODEL[sti_timing_key]
        else:
            ms = SPOT_INPUT_SPEC.get(sti_timing_key)
        if ms is not None:
            sti_opts[sti_timing_key] = float(ms)
    return sti_opts


def resolve_train_sti_timing(tokens) -> dict:
    """Build full sti timing dict (defaults + optional sti_timing tokens)."""
    sti_opts = _resolve_sti_timing_literals()
    if tokens:
        sti_opts.update(parse_sti_timing_keys(tokens))
    return sti_opts


def resolve_sti_timing_kwargs(tokens=None):
    """Map sti_timing tokens to kwargs for :func:`figure.plot.override_session_sti_timing`."""
    empty = {sti_timing_key: None for sti_timing_key in STI_TIMING_KEYS}
    if not tokens:
        return empty
    if isinstance(tokens, dict):
        return {key: tokens.get(key) for key in STI_TIMING_KEYS}
    sti_timing = parse_sti_timing_keys(tokens)
    return {sti_timing_key: sti_timing.get(sti_timing_key) for sti_timing_key in STI_TIMING_KEYS}


def override_train_opts_timing(
    opts,
    *,
    ms_pre=None,
    ms_response=None,
    ms_post=None,
    ms_sti=None,
    delta_ms=None,
    delta_ms_pre=None,
):
    """Merge timing into train-opts spot/bar sti dicts."""
    from task.spot.sti_spec import override_sti_timing

    changed = {}
    so = opts.get("spot_sti_opts")
    if so is not None:
        changed = override_sti_timing(
            so,
            ms_pre=ms_pre,
            ms_response=ms_response,
            ms_post=ms_post,
            ms_sti=ms_sti,
            delta_ms=delta_ms,
            delta_ms_pre=delta_ms_pre,
        )
    if ms_pre is not None or delta_ms is not None or delta_ms_pre is not None:
        so = opts.get("moving_bar_sti_opts")
        if so is not None:
            changed_bar = override_sti_timing(
                so,
                ms_pre=ms_pre,
                ms_response=None,
                ms_post=None,
                ms_sti=None,
                delta_ms=delta_ms,
                delta_ms_pre=delta_ms_pre,
            )
            if not changed:
                changed = changed_bar
            else:
                changed.update(changed_bar)
    return changed


def parse_cost_radius(tokens):
    """Parse cost_radius: optional bare N plus task=N tokens."""
    if not tokens:
        return None, {}
    bare_cost_radius = None
    by_task = {}
    for token in tokens:
        if "=" in token:
            name, val = token.split("=", 1)
            by_task[name.strip()] = int(val.strip())
        else:
            if bare_cost_radius is not None:
                raise ValueError("only one bare radius allowed in cost_radius")
            bare_cost_radius = int(token)
    return bare_cost_radius, by_task


def resolve_gt(tokens):
    """Parse gt_by_task space-separated task=CELLS tokens."""
    if tokens is None:
        return None
    raw = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"expected task=CELLS, got {token!r}")
        name, val = token.split("=", 1)
        name = name.strip()
        types = parse_comma_list(val)
        if not types:
            raise ValueError(f"gt {name}=... must list at least one type")
        raw[name] = types
    return train.resolve_gt_cells_by_task(raw)


def resolve_part_cost_scales(tokens, tasks):
    """Parse part_cost_scales tokens."""
    if not tokens:
        return {}
    bare: list[str] = []
    explicit: dict[str, float] = {}
    for token in tokens:
        if "=" in token:
            name, val = token.split("=", 1)
            explicit[name.strip()] = float(val.strip())
        else:
            bare.append(token.strip())
    scales: dict[str, float] = {}
    if bare:
        scales = {part_key: 0.0 for part_key in train.session_cost_part_keys(tasks)}
        scales.update(train.expand_part_cost_scale({name: 1.0 for name in bare}))
    scales.update(train.expand_part_cost_scale(explicit))
    return scales


def param_in_modes(param_modes, param):
    return bool(param_modes and param in param_modes)
