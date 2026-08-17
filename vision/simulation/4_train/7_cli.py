# -*- coding: utf-8 -*-
"""Train config parse helpers (Hydra config → runtime values).

No argparse registration. Callers: ``config``, ``figure``, ``analyze``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SIMULATION_CODE = Path(__file__).resolve().parent.parent
if str(_SIMULATION_CODE) not in sys.path:
    sys.path.insert(0, str(_SIMULATION_CODE))

import import_bootstrap  # noqa: F401
from import_bootstrap import parse_comma_list
import train


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


def plot_ms_kwargs(figure_plot) -> dict:
    """Hydra ``plot_ms_*`` scalars for :func:`figure.plot.override_session`."""
    return {
        "ms_pre": figure_plot.get("ms_pre"),
        "ms_response": figure_plot.get("ms_response"),
        "ms_post": figure_plot.get("ms_post"),
        "ms_sti": figure_plot.get("ms_sti"),
        "delta_ms": figure_plot.get("delta_ms"),
        "delta_ms_pre": figure_plot.get("delta_ms_pre"),
    }


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
    """Merge timing into train-opts spread/spot/bar sti dicts."""
    from task.spread.sti_spec import override_sti_timing

    changed = {}
    for sti_opts_key in ("spread_sti_opts", "spot_sti_opts"):
        so = opts.get(sti_opts_key)
        if so is None:
            continue
        part = override_sti_timing(
            so,
            ms_pre=ms_pre,
            ms_response=ms_response,
            ms_post=ms_post,
            ms_sti=ms_sti,
            delta_ms=delta_ms,
            delta_ms_pre=delta_ms_pre,
        )
        if not changed:
            changed = part
        else:
            changed.update(part)
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


def param_in_modes(param_modes, param):
    return bool(param_modes and param in param_modes)
