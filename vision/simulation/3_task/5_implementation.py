# -*- coding: utf-8 -*-
"""Task implementation: wire spread / spot / mbar into session-ready packs and sti opts."""
from __future__ import annotations

from config import (
    MBAR_INPUT_SPEC,
    MODEL,
    SBAR_INPUT_GEO,
    SPREAD_INPUT_SPEC,
    SPOT_INPUT_GEO,
)
from task.mbar.gt import expand_gt_cells as expand_mbar_gt_cells
from task.mbar.pack import build_mbar_pack, resolve_mbar_sti_opts
from task.sbar.pack import build_sbar_pack, resolve_sbar_sti_opts
from task.spot.gt import expand_gt_cells as expand_spot_gt_cells
from task.spot.pack import build_spot_pack, resolve_spot_sti_opts
from task.spread.gt import expand_gt_cells as expand_spread_gt_cells
from task.spread.pack import build_spread_pack, resolve_spread_sti_opts

TASKS = ("spread", "spot", "mbar", "sbar")

_STI_TRAIN_OPT_KEYS = (
    ("spread", "spread_sti_opts"),
    ("spot", "spot_sti_opts"),
    ("mbar", "mbar_sti_opts"),
    ("sbar", "sbar_sti_opts"),
)

_STI_OPTS_BY_TASK = {
    "spread": {
        "ms_pre": SPREAD_INPUT_SPEC["ms_pre"],
        "ms_response": SPREAD_INPUT_SPEC["ms_response"],
        "ms_post": SPREAD_INPUT_SPEC["ms_post"],
        "ms_sti": SPREAD_INPUT_SPEC["ms_sti"],
        "delta_ms": MODEL["delta_ms"],
        "delta_ms_pre": MODEL["delta_ms_pre"],
    },
    "spot": {
        "ms_pre": SPREAD_INPUT_SPEC["ms_pre"],
        "ms_response": SPREAD_INPUT_SPEC["ms_response"],
        "ms_post": SPREAD_INPUT_SPEC["ms_post"],
        "ms_sti": SPREAD_INPUT_SPEC["ms_sti"],
        "delta_ms": MODEL["delta_ms"],
        "delta_ms_pre": MODEL["delta_ms_pre"],
        "shift_radius": SPOT_INPUT_GEO["shift_radius"],
        "spot_radius": SPOT_INPUT_GEO["spot_radius"],
        "multi_spot": SPOT_INPUT_GEO["multi_spot"],
        "fully_inside": SPOT_INPUT_GEO["fully_inside"],
    },
    "mbar": {
        "ms_pre": MBAR_INPUT_SPEC["ms_pre"],
        "delta_ms": MODEL["delta_ms"],
        "delta_ms_pre": MODEL["delta_ms_pre"],
        "bar_dist": SBAR_INPUT_GEO["bar_dist"],
        "bar_ws_deg": list(MBAR_INPUT_SPEC["bar_ws_deg"]),
        "bar_directions": list(MBAR_INPUT_SPEC["bar_directions"]),
        "multi_bar": SBAR_INPUT_GEO["multi_bar"],
    },
    "sbar": {
        "ms_pre": SPREAD_INPUT_SPEC["ms_pre"],
        "ms_response": SPREAD_INPUT_SPEC["ms_response"],
        "ms_post": SPREAD_INPUT_SPEC["ms_post"],
        "ms_sti": SPREAD_INPUT_SPEC["ms_sti"],
        "delta_ms": MODEL["delta_ms"],
        "delta_ms_pre": MODEL["delta_ms_pre"],
        "bar_dist": SBAR_INPUT_GEO["bar_dist"],
        "bar_directions": list(MBAR_INPUT_SPEC["bar_directions"]),
        "multi_bar": SBAR_INPUT_GEO["multi_bar"],
    },
}

_RESOLVE_STI_OPTS = {
    "spread": resolve_spread_sti_opts,
    "spot": resolve_spot_sti_opts,
    "mbar": resolve_mbar_sti_opts,
    "sbar": resolve_sbar_sti_opts,
}

_GT_CELLS_EXPAND = {
    "spread": expand_spread_gt_cells,
    "spot": expand_spot_gt_cells,
    "mbar": expand_mbar_gt_cells,
    "sbar": expand_mbar_gt_cells,
}


def resolve_gt_cells_by_task(by_task) -> dict[str, list[str]]:
    return {
        str(task): list(_GT_CELLS_EXPAND[task]([str(cell) for cell in cells]))
        for task, cells in (by_task or {}).items()
    }


def resolve_sti_opts(
    task: str,
    sti_opts,
    *,
    cost_radius=None,
) -> dict:
    if task not in _RESOLVE_STI_OPTS:
        raise ValueError(f"unknown task {task!r}")
    sti_opts = {**_STI_OPTS_BY_TASK[task], **(sti_opts or {})}
    sti_opts = _RESOLVE_STI_OPTS[task](sti_opts)
    if cost_radius is not None:
        sti_opts["cost_radius"] = int(cost_radius)
    else:
        sti_opts.pop("cost_radius", None)
    return sti_opts


def resolve_train_sti_opts(
    tasks,
    *,
    cost_radius=None,
    spread_sti_opts=None,
    spot_sti_opts=None,
    mbar_sti_opts=None,
    sbar_sti_opts=None,
) -> dict[str, dict | None]:
    """Merge defaults + overrides into per-task sti opts sidecar dicts."""
    sti_opts_by_task = {
        "spread": spread_sti_opts,
        "spot": spot_sti_opts,
        "mbar": mbar_sti_opts,
        "sbar": sbar_sti_opts,
    }
    out: dict[str, dict | None] = {}
    for task, sti_opts_key in _STI_TRAIN_OPT_KEYS:
        if task not in tasks and sti_opts_by_task[task] is None:
            out[sti_opts_key] = None
            continue
        out[sti_opts_key] = resolve_sti_opts(
            task,
            sti_opts_by_task[task],
            cost_radius=cost_radius,
        )
    return out


def build_task_pack(
    task: str,
    *,
    connectome,
    contrast: str,
    gt_amp: float,
    i_sti: dict,
    sti_opts: dict,
    opts: dict,
):
    if task == "spread":
        return build_spread_pack(
            connectome,
            contrast=contrast,
            gt_amp=gt_amp,
            i_sti=i_sti,
            sti_opts=sti_opts,
            opts=opts,
        )
    if task == "spot":
        return build_spot_pack(
            connectome,
            contrast=contrast,
            gt_amp=gt_amp,
            i_sti=i_sti,
            sti_opts=sti_opts,
            opts=opts,
        )
    if task == "mbar":
        return build_mbar_pack(
            connectome,
            contrast=contrast,
            gt_amp=gt_amp,
            i_sti=i_sti,
            sti_opts=sti_opts,
            opts=opts,
        )
    if task == "sbar":
        return build_sbar_pack(
            connectome,
            contrast=contrast,
            gt_amp=gt_amp,
            i_sti=i_sti,
            sti_opts=sti_opts,
            opts=opts,
        )
    raise ValueError(f"unknown task {task!r}")
