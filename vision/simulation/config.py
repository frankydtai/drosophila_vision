# -*- coding: utf-8 -*-
"""Load ``simulation/conf/config.yaml`` — sole config source (no train/figure imports at load)."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import OmegaConf

import import_bootstrap  # noqa: F401
from import_bootstrap import parse_comma_list

_ACTIVE: dict | None = None

RUN_NAME: str = ""
RUN_PATH: str = ""
MODEL: Dict[str, object] = {}
NEURON_SCHEMA: Dict[str, object] = {}
NEURON_FORWARD: Dict[str, object] = {}
NETWORK_PATH: Dict[str, object] = {}
SPREAD_INPUT_SPEC: Dict[str, object] = {}
SPREAD_PACK: Dict[str, object] = {}
SPOT_INPUT_GEO: Dict[str, object] = {}
SPOT_PACK: Dict[str, object] = {}
MOVING_BAR_INPUT_GEO: Dict[str, object] = {}
MOVING_BAR_INPUT_SPEC: Dict[str, object] = {}
TRAIN_CONFIG: Dict[str, object] = {}
VAL_FROM: Dict[str, object] = {}
TRAIN_OPTIMIZATION: Dict[str, object] = {}
TRAIN_SESSION: Dict[str, object] = {}
FIGURE_PLOT: Dict[str, object] = {}
FIGURE_PLOT_STI_SPOT: Dict[str, object] = {}
ANALYZE_RUNS: List[str] = []
ANALYZE_CELL_DYNAMICS: Dict[str, object] = {}
ANALYZE_SYN_SIGN: Dict[str, object] = {}
ANALYZE_TRACE: Dict[str, object] = {}
ANALYZE_COST_PART: Dict[str, object] = {}

_MODEL_KEYS = (
    "delta_ms", "delta_ms_pre", "cap", "g_leak", "e_exc", "e_inh", "e_h",
    "h_g_max", "gt_amp", "v_clamp", "a_syn_exc", "a_syn_inh", "euler",
)
_TRAIN_OPTIMIZATION_KEYS = (
    "part_cost_scales", "cost_norm", "cost_interval_ms", "cost_ms",
    "pre_steady", "pre_steady_n_iter", "pre_steady_damp", "n_run",
    "n_iter", "lrs", "checkpoint_interval",
)
_ANALYZE_TRACE_KEYS = (
    "trace_osc_min_f", "trace_osc_max_f", "trace_osc_peak_threshold",
    "trace_osc_z_threshold", "trace_osc_snr_min",
    "trace_drift_min_slope_mv_over_s", "trace_drift_min_r",
    "trace_baseline_ms", "trace_flat_max_abs",
    "trace_flat_v_peak_to_peak_max", "trace_flat_abs_mean",
)


def config_path() -> Path:
    return Path(__file__).resolve().with_name("conf") / "config.yaml"


def load_config_dict() -> dict:
    data = OmegaConf.to_container(OmegaConf.load(config_path()), resolve=True)
    if not isinstance(data, dict):
        raise ValueError("simulation/conf/config.yaml must decode to a dict")
    return data


def _bind_config(config_dict: dict) -> None:
    global RUN_NAME, RUN_PATH
    global MODEL, NEURON_SCHEMA, NEURON_FORWARD, NETWORK_PATH
    global SPREAD_INPUT_SPEC, SPREAD_PACK, SPOT_INPUT_GEO, SPOT_PACK
    global MOVING_BAR_INPUT_GEO, MOVING_BAR_INPUT_SPEC
    global TRAIN_CONFIG, VAL_FROM, TRAIN_OPTIMIZATION, TRAIN_SESSION
    global FIGURE_PLOT, FIGURE_PLOT_STI_SPOT, ANALYZE_RUNS
    global ANALYZE_CELL_DYNAMICS, ANALYZE_SYN_SIGN, ANALYZE_TRACE
    global ANALYZE_COST_PART

    model = str(config_dict["model"])
    RUN_NAME = str(config_dict["run_name"]).strip()
    RUN_PATH = f"{model}/{RUN_NAME}"

    MODEL = {key: config_dict[key] for key in _MODEL_KEYS}
    NEURON_SCHEMA = {
        "model": model,
        "filter": config_dict["filter"],
        "syn_mode": config_dict["syn_mode"],
        "a_lo": config_dict["a_lo"],
        "a_hi": config_dict["a_hi"],
        "h_cells": config_dict["h_cells"],
        "params": config_dict["params"],
    }
    NEURON_FORWARD = {"pre_grad": config_dict["pre_grad"]}
    NETWORK_PATH = {"network": config_dict["network"]}
    SPREAD_INPUT_SPEC = {
        "i_bright": config_dict["i_bright"],
        "i_dark": config_dict["i_dark"],
        "contrasts": config_dict["contrasts"],
        "ms_pre": config_dict["ms_pre"],
        "ms_sti": config_dict["ms_sti"],
        "ms_response": config_dict["ms_response"],
        "ms_post": config_dict["ms_post"],
    }
    SPREAD_PACK = {
        "spread_gt_mode": config_dict["spread_gt_mode"],
    }
    SPOT_INPUT_GEO = {
        "spot_radius": config_dict["spot_radius"],
        "fully_inside": config_dict["fully_inside"],
        "multi_spot": config_dict["multi_spot"],
        "shift_radius": config_dict["shift_radius"],
    }
    SPOT_PACK = {
        "spot_cost_radii": config_dict["spot_cost_radii"],
        "a_sti_radii": config_dict["a_sti_radii"],
        "spot_cost_radius_scale": config_dict["spot_cost_radius_scale"],
    }
    MOVING_BAR_INPUT_GEO = {
        "multi_bar": config_dict["multi_bar"],
        "bar_radius": config_dict["bar_radius"],
    }
    MOVING_BAR_INPUT_SPEC = {
        "i_bright": config_dict["i_bright"],
        "i_dark": config_dict["i_dark"],
        "ms_pre": config_dict["ms_pre"],
    }
    tasks = config_dict["tasks"]
    TRAIN_CONFIG = {
        "tasks": tasks,
        "contrasts": config_dict["contrasts"],
        "gt_by_task": config_dict.get("gt_by_task"),
        "cost_radius": config_dict.get("cost_radius"),
    }
    VAL_FROM = dict(config_dict.get("val_from") or {})
    TRAIN_OPTIMIZATION = {key: config_dict[key] for key in _TRAIN_OPTIMIZATION_KEYS}
    TRAIN_SESSION = {"fp": config_dict["fp"], "sequential": config_dict["sequential"]}
    FIGURE_PLOT = {
        "html": config_dict.get("html", False),
        "plot_right_only": config_dict.get("plot_right_only", True),
        "show_pre": config_dict.get("show_pre", True),
        "center_only": config_dict.get("center_only", False),
        "x": config_dict.get("x"),
        "y": config_dict.get("y"),
        "align_xy": config_dict.get("align_xy"),
        "ms_shown": config_dict.get("ms_shown"),
    }
    FIGURE_PLOT_STI_SPOT = {
        "spot_radii": config_dict.get("spot_radii"),
        "output": config_dict.get("spot_plot_output"),
    }
    analyze_runs = config_dict.get("analyze_runs")
    if not analyze_runs:
        ANALYZE_RUNS = [RUN_PATH]
    elif isinstance(analyze_runs, str):
        ANALYZE_RUNS = [str(analyze_runs)]
    else:
        ANALYZE_RUNS = [str(run) for run in analyze_runs]
    ANALYZE_CELL_DYNAMICS = {
        "cells": config_dict.get("cells"),
        "spec": config_dict.get("spec"),
        "node": config_dict.get("node"),
        "radius": int(config_dict.get("radius") or 0),
        "t_rel_start": config_dict.get("t_rel_start"),
        "t_rel_stop": config_dict.get("t_rel_stop"),
        "figure": bool(config_dict.get("analyze_figure", True)),
        "json": bool(config_dict.get("analyze_json", False)),
    }
    ANALYZE_SYN_SIGN = {
        "bins": config_dict["bins"],
        "after_train": config_dict["after_train"],
        "post": bool(config_dict.get("post", False)),
    }
    ANALYZE_TRACE = {key: config_dict[key] for key in _ANALYZE_TRACE_KEYS}
    ANALYZE_TRACE["check"] = config_dict.get("check")
    ANALYZE_TRACE["baseline_ms_shown"] = config_dict.get("baseline_ms_shown")
    ANALYZE_COST_PART = {
        "part": config_dict.get("part"),
        "cost_norm": config_dict.get("analyze_cost_norm"),
        "stride": int(config_dict.get("stride") or 1),
        "per_node": bool(config_dict.get("per_node", False)),
        "csv": config_dict.get("cost_part_csv"),
        "print_parts": bool(config_dict.get("print_parts", False)),
    }


def active_config() -> dict:
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load_config_dict()
        _bind_config(_ACTIVE)
    return _ACTIVE


def apply_config(hydra_config) -> dict:
    global _ACTIVE
    _ACTIVE = (
        OmegaConf.to_container(hydra_config, resolve=True)
        if OmegaConf.is_config(hydra_config)
        else dict(hydra_config)
    )
    _bind_config(_ACTIVE)
    return _ACTIVE


active_config()


def _as_float_list(values) -> List[float]:
    if values is None:
        return []
    if isinstance(values, str):
        values = parse_comma_list(values)
    return [float(x) for x in values]


def parse_cells(raw) -> List[str] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        cells = parse_comma_list(raw)
    else:
        cells = [str(cell).strip() for cell in raw if str(cell).strip()]
    return cells or None


def _resolve_init_from(init_from):
    if not init_from:
        return None
    import train
    init_from_path = Path(str(init_from)).expanduser()
    if not init_from_path.is_absolute():
        token = str(init_from).replace("\\", "/")
        parts = token.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "init_from must be MODEL/RUN under 0_runs "
                f"(models: {train.MODELS}) or an absolute path; got {init_from!r}"
            )
        src_model, run = parts
        if src_model not in train.MODELS:
            raise ValueError(
                f"init_from model {src_model!r} not in {train.MODELS}"
            )
        return f"{src_model}/{run}"
    return str(init_from)


def _resolve_run_name(config_dict: dict) -> str:
    if config_dict.get("run_name"):
        return str(config_dict["run_name"]).strip()
    prefix = os.environ.get("SLURM_JOB_ID") or time.strftime("%m%d_%H%M%S")
    return f"{prefix}-run"


def resolve_figure_kwargs(hydra_config) -> dict:
    from figure.plot import parse_align_xy, parse_axis_coords, parse_ms_shown_range

    apply_config(hydra_config)
    figure_plot = FIGURE_PLOT
    align_xy = parse_align_xy(figure_plot.get("align_xy"))
    align_at_x, align_at_y = align_xy if align_xy is not None else (None, None)
    ms_shown = None
    if figure_plot.get("ms_shown") is not None:
        ms_shown = parse_ms_shown_range(str(figure_plot["ms_shown"]))
    return dict(
        plot_right_only=bool(figure_plot.get("plot_right_only", True)),
        show_pre=bool(figure_plot.get("show_pre", True)),
        center_only=bool(figure_plot.get("center_only", False)),
        at_x=parse_axis_coords(figure_plot.get("x")),
        at_y=parse_axis_coords(figure_plot.get("y")),
        align_at_x=align_at_x,
        align_at_y=align_at_y,
        html=bool(figure_plot.get("html", False)),
        ms_shown=ms_shown,
    )


def resolve_run_kwargs(hydra_config) -> dict:
    """Map merged Hydra config to kwargs for :func:`run_train_and_plot`."""
    import train
    import train.cli as cli
    from train.config import expand_cost_norm, expand_pre_steady
    from train.implementation import run_dir

    apply_config(hydra_config)
    config_dict = active_config()

    model = str(NEURON_SCHEMA["model"])
    init_from = _resolve_init_from(config_dict.get("init_from"))
    cost_radius = TRAIN_CONFIG.get("cost_radius")
    if cost_radius is not None:
        cost_radius = int(cost_radius)

    param_tokens = list(config_dict.get("param_tokens") or [])
    param_init, param_vals, param_modes, param_clamps, param_jits = (
        train.parse_param_cli(param_tokens) if param_tokens else ([], [], {}, [], [])
    )
    if param_vals:
        raise ValueError(
            "param_tokens …val… is for plot/analyze only; use …init… for train"
        )

    param_init = param_init or None
    param_modes = param_modes or None
    param_clamps = param_clamps or None
    param_jits = param_jits or None

    syn_mode = str(NEURON_SCHEMA["syn_mode"])
    if param_modes:
        if syn_mode == "per_edge" and "syn_strength_cell" in param_modes:
            raise ValueError("param syn_strength_cell requires syn_mode per_cell")
        if syn_mode == "per_cell" and "syn_strength_edge" in param_modes:
            raise ValueError("param syn_strength_edge requires syn_mode per_edge")
        if "syn_strength_edge" in param_modes:
            train.validate_syn_strength_edge_param_mode(param_modes["syn_strength_edge"])

    filter = train.expand_filter(NEURON_SCHEMA["filter"])
    spread_gt_mode = train.expand_spread_gt_mode(SPREAD_PACK["spread_gt_mode"])
    val_from = train.resolve_val_from(VAL_FROM)
    val_from_opts = {"val_from": val_from}
    if filter != "ca":
        for param in ("v_th_ca", "a_ca", "tau_ca"):
            if cli.param_in_modes(param_modes, param):
                raise ValueError(f"param {param} requires filter ca")
        if train.val_from_enabled(val_from_opts, "v_th_ca") or train.val_from_enabled(val_from_opts, "a_ca"):
            raise ValueError("val_from v_th_ca / a_ca require filter ca")
        if param_modes:
            param_modes = {
                key: modes for key, modes in param_modes.items()
                if key not in ("v_th_ca", "a_ca", "tau_ca")
            } or None

    tasks = train.parse_tasks(TRAIN_CONFIG["tasks"])

    part_cost_scales = {
        str(part_key): float(scale)
        for part_key, scale in (TRAIN_OPTIMIZATION.get("part_cost_scales") or {}).items()
    }

    timing = {
        "ms_pre": float(SPREAD_INPUT_SPEC["ms_pre"]),
        "ms_response": float(SPREAD_INPUT_SPEC["ms_response"]),
        "ms_post": float(SPREAD_INPUT_SPEC["ms_post"]),
        "ms_sti": float(SPREAD_INPUT_SPEC["ms_sti"]),
        "delta_ms": float(MODEL["delta_ms"]),
        "delta_ms_pre": float(MODEL["delta_ms_pre"]),
    }
    moving_bar_sti_opts = {
        "multi_bar": MOVING_BAR_INPUT_GEO["multi_bar"],
        "ms_pre": timing["ms_pre"],
        "delta_ms": timing["delta_ms"],
        "delta_ms_pre": timing["delta_ms_pre"],
    }
    spot_sti_opts = dict(timing)
    spread_sti_opts = dict(timing)

    cost_interval_ms = float(TRAIN_OPTIMIZATION["cost_interval_ms"])
    if cost_interval_ms <= 0:
        raise ValueError("cost_interval_ms must be > 0")

    cost_ms_raw = dict(TRAIN_OPTIMIZATION["cost_ms"])
    cost_ms = {
        str(int(radius)): [float(x) for x in mss]
        for radius, mss in cost_ms_raw.items()
    }

    gt_tokens = TRAIN_CONFIG.get("gt_by_task")
    gt_by_task = cli.resolve_gt(gt_tokens) if gt_tokens else None
    if gt_by_task:
        gt_opts = {"moving_bar": moving_bar_sti_opts, "spot": spot_sti_opts, "spread": spread_sti_opts}
        for task, cells in gt_by_task.items():
            gt_opts[task]["gt_cells"] = list(cells)

    contrasts = train.parse_contrasts(TRAIN_CONFIG["contrasts"])

    lrs = _as_float_list(TRAIN_OPTIMIZATION.get("lrs"))
    if not lrs:
        raise ValueError("lrs must list at least one learning rate")

    cuda_available = torch.cuda.is_available()
    fp = int(TRAIN_SESSION["fp"])
    if not cuda_available and fp == 64:
        fp = 32

    n_iter = TRAIN_OPTIMIZATION["n_iter"]

    run_name = _resolve_run_name(config_dict)

    train_kwargs = dict(
        model=model,
        n_run=int(TRAIN_OPTIMIZATION["n_run"]),
        n_iter=int(n_iter),
        lrs=lrs,
        fname=config_dict.get("fname"),
        outdir=run_dir(model, parent=config_dict.get("outdir"), run=run_name),
        param_modes=param_modes,
        param_init=param_init,
        param_clamps=param_clamps,
        param_jits=param_jits,
        syn_mode=syn_mode,
        network=str(NETWORK_PATH["network"]),
        tasks=tasks,
        contrasts=contrasts,
        part_cost_scales=part_cost_scales,
        cost_norm=expand_cost_norm(TRAIN_OPTIMIZATION["cost_norm"]),
        cost_interval_ms=cost_interval_ms,
        cost_ms=cost_ms,
        cost_radius=cost_radius,
        shift_radius=SPOT_INPUT_GEO["shift_radius"],
        spot_radius=SPOT_INPUT_GEO["spot_radius"],
        multi_spot=SPOT_INPUT_GEO["multi_spot"],
        fully_inside=SPOT_INPUT_GEO["fully_inside"],
        moving_bar_sti_opts=moving_bar_sti_opts,
        spread_sti_opts=spread_sti_opts,
        spot_sti_opts=spot_sti_opts,
        euler=str(MODEL["euler"]),
        pre_steady=expand_pre_steady(TRAIN_OPTIMIZATION["pre_steady"]),
        pre_steady_n_iter=TRAIN_OPTIMIZATION["pre_steady_n_iter"],
        pre_steady_damp=TRAIN_OPTIMIZATION["pre_steady_damp"],
        fp=fp,
        pre_grad=bool(NEURON_FORWARD["pre_grad"]),
        val_from=val_from,
        filter=filter,
        spread_gt_mode=spread_gt_mode,
        sequential=bool(TRAIN_SESSION["sequential"]),
        init_from=init_from,
        checkpoint_interval=TRAIN_OPTIMIZATION.get("checkpoint_interval"),
        syn_sign=bool(ANALYZE_SYN_SIGN.get("after_train", False)),
    )
    train_kwargs["figure_kwargs"] = resolve_figure_kwargs(hydra_config)
    return train_kwargs
