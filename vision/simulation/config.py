# -*- coding: utf-8 -*-
"""Load ``simulation/conf/config.yaml`` — sole config source (no train/figure imports at load)."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import OmegaConf

import import_bootstrap  # noqa: F401

_ACTIVE: dict | None = None

RUN_NAME: str = ""
RUN_PATH: str = ""
MODEL: Dict[str, object] = {}
NEURON_SCHEMA: Dict[str, object] = {}
NEURON_FORWARD: Dict[str, object] = {}
NETWORK_PATH: Dict[str, object] = {}
SPREAD_INPUT_SPEC: Dict[str, object] = {}
SPREAD_GT: Dict[str, object] = {}
SPOT_INPUT_GEO: Dict[str, object] = {}
SPOT_PACK: Dict[str, object] = {}
SBAR_INPUT_GEO: Dict[str, object] = {}
SBAR_PACK: Dict[str, object] = {}
MBAR_INPUT_SPEC: Dict[str, object] = {}
TRAIN_CONFIG: Dict[str, object] = {}
VAL_FROM: Dict[str, object] = {}
TRAIN_OPTIMIZATION: Dict[str, object] = {}
TRAIN_SESSION: Dict[str, object] = {}
FIGURE_PLOT: Dict[str, object] = {}
FIGURE_PLOT_STI_SPOT: Dict[str, object] = {}
FIGURE_PLOT_STI_MBAR: Dict[str, object] = {}
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
    "part_cost_scales", "cost_norm", "cost_entry_reduce", "cost_ms",
    "pre_steady", "pre_steady_n_iter", "pre_steady_damp", "n_run",
    "n_iter", "lrs", "checkpoint_interval",
)
_RUN_MAX = 255

_ANALYZE_TRACE_KEYS = (
    "trace_osc_min_f", "trace_osc_max_f", "trace_osc_peak_threshold",
    "trace_osc_z_threshold", "trace_osc_snr_min",
    "trace_drift_min_slope_mv_over_s", "trace_drift_min_r",
    "trace_baseline_ms", "trace_flat_max_abs",
    "trace_flat_v_peak_to_peak_max", "trace_flat_abs_mean",
)


def config_path() -> Path:
    return Path(__file__).resolve().with_name("conf") / "config.yaml"


def load_config() -> dict:
    data = OmegaConf.to_container(OmegaConf.load(config_path()), resolve=True)
    if not isinstance(data, dict):
        raise ValueError("simulation/conf/config.yaml must decode to a dict")
    return data


def _list_tokens(value, *, key: str) -> List[str]:
    # Hydra `tasks=spread` overrides a YAML list with a str; treat as one token.
    # Multi-value: Hydra list syntax `tasks=[spread,sbar]` (not comma-split str).
    if isinstance(value, (list, tuple)) or OmegaConf.is_list(value):
        tokens = [str(token).strip() for token in value if str(token).strip()]
    elif isinstance(value, str):
        token = value.strip()
        tokens = [token] if token else []
    else:
        raise ValueError(f"{key} must be a list, got {value!r}")
    if not tokens:
        raise ValueError(f"{key} must list at least one token")
    return tokens


def _int_list(value, *, key: str) -> List[int]:
    return [int(token) for token in _list_tokens(value, key=key)]


def _float_list(value, *, key: str) -> List[float]:
    return [float(token) for token in _list_tokens(value, key=key)]


def _bind_config(config: dict) -> None:
    global RUN_NAME, RUN_PATH
    global MODEL, NEURON_SCHEMA, NEURON_FORWARD, NETWORK_PATH
    global SPREAD_INPUT_SPEC, SPREAD_GT, SPOT_INPUT_GEO, SPOT_PACK
    global SBAR_INPUT_GEO, SBAR_PACK, MBAR_INPUT_SPEC
    global TRAIN_CONFIG, VAL_FROM, TRAIN_OPTIMIZATION, TRAIN_SESSION
    global FIGURE_PLOT, FIGURE_PLOT_STI_SPOT, FIGURE_PLOT_STI_MBAR, ANALYZE_RUNS
    global ANALYZE_CELL_DYNAMICS, ANALYZE_SYN_SIGN, ANALYZE_TRACE
    global ANALYZE_COST_PART

    model = str(config["model"])
    RUN_NAME = str(config["run_name"]).strip()
    RUN_PATH = str(config["run_path"]).strip()

    MODEL = {key: config[key] for key in _MODEL_KEYS}
    NEURON_SCHEMA = {
        "model": model,
        "filter": config["filter"],
        "syn_mode": config["syn_mode"],
        "a_lo": config["a_lo"],
        "a_hi": config["a_hi"],
        "h_cells": _list_tokens(config["h_cells"], key="h_cells"),
        "params": config["params"],
    }
    NEURON_FORWARD = {"pre_grad": config["pre_grad"]}
    NETWORK_PATH = {"network": config["network"]}
    SPREAD_INPUT_SPEC = {
        "ms_pre": config["ms_pre"],
        "ms_sti": config["ms_sti"],
        "ms_response": config["ms_response"],
        "ms_post": config["ms_post"],
    }
    SPREAD_GT = {
        "spread_gt_mode": config["spread_gt_mode"],
    }
    SPOT_INPUT_GEO = {
        "spot_radius": config["spot_radius"],
        "fully_inside": config["fully_inside"],
        "multi_spot": config["multi_spot"],
        "shift_radius": config["shift_radius"],
    }
    SPOT_PACK = {
        "spot_cost_radii": _int_list(
            config["spot_cost_radii"], key="spot_cost_radii",
        ),
        "a_sti_radii": _int_list(config["a_sti_radii"], key="a_sti_radii"),
        "spot_cost_radius_scale": config["spot_cost_radius_scale"],
    }
    SBAR_INPUT_GEO = {
        "multi_bar": config["multi_bar"],
        "bar_dist": config["bar_dist"],
        "shift_mid": config["shift_mid"],
    }
    SBAR_PACK = {
        "a_sti_mids": _float_list(
            config["a_sti_mids"], key="a_sti_mids",
        ),
    }
    MBAR_INPUT_SPEC = {
        "ms_pre": config["ms_pre"],
        "bar_ws_deg": _float_list(config["bar_ws_deg"], key="bar_ws_deg"),
        "bar_directions": _list_tokens(config["bar_directions"], key="bar_directions"),
    }
    tasks = _list_tokens(config["tasks"], key="tasks")
    TRAIN_CONFIG = {
        "tasks": tasks,
        "contrasts": _list_tokens(config["contrasts"], key="contrasts"),
        "i_sti": {
            str(contrast): float(val)
            for contrast, val in config["i_sti"].items()
        },
        "gt_by_task": config.get("gt_by_task"),
        "cost_radius": config.get("cost_radius"),
    }
    VAL_FROM = dict(config.get("val_from") or {})
    TRAIN_OPTIMIZATION = {
        key: (
            _float_list(config[key], key=key)
            if key == "lrs"
            else config[key]
        )
        for key in _TRAIN_OPTIMIZATION_KEYS
    }
    TRAIN_SESSION = {"fp": config["fp"], "sequential": config["sequential"]}
    FIGURE_PLOT = {
        "tasks": (
            _list_tokens(config["figure_tasks"], key="figure_tasks")
            if config.get("figure_tasks") is not None
            else None
        ),
        "sbar_figure_cells": (
            _list_tokens(config["sbar_figure_cells"], key="sbar_figure_cells")
            if config.get("sbar_figure_cells") is not None
            else None
        ),
        "html": config.get("html", False),
        "plot_right_only": config.get("plot_right_only", True),
        "x": config.get("x"),
        "y": config.get("y"),
        "align_xy": config.get("align_xy"),
        "ms_shown": config.get("ms_shown"),
    }
    FIGURE_PLOT_STI_SPOT = {
        "spot_radii": (
            _float_list(config["spot_radii"], key="spot_radii")
            if config.get("spot_radii") is not None
            else None
        ),
        "output": config.get("spot_plot_output"),
    }
    plot_t = config.get("mbar_plot_t")
    FIGURE_PLOT_STI_MBAR = {
        "output": config.get("mbar_plot_output"),
        "gif": bool(config.get("mbar_plot_gif", False)),
        "gif_output": config.get("mbar_plot_gif_output"),
        "t_stride": int(config.get("mbar_plot_t_stride") or 2),
        "t": (
            _int_list(plot_t, key="mbar_plot_t")
            if plot_t not in (None, "")
            else None
        ),
        "direction": str(config.get("mbar_plot_direction", "left")),
    }
    analyze_runs = config.get("analyze_runs")
    if not analyze_runs:
        ANALYZE_RUNS = [RUN_PATH]
    elif isinstance(analyze_runs, (list, tuple)) or OmegaConf.is_list(analyze_runs):
        ANALYZE_RUNS = _list_tokens(analyze_runs, key="analyze_runs")
    else:
        raise ValueError(
            f"analyze_runs must be null or a list, got {analyze_runs!r}"
        )
    ANALYZE_CELL_DYNAMICS = {
        "cells": config.get("cells"),
        "spec": config.get("spec"),
        "mid": config.get("mid"),
        "node": config.get("node"),
        "radius": int(config.get("radius") or 0),
        "t_rel_start": config.get("t_rel_start"),
        "t_rel_stop": config.get("t_rel_stop"),
        "figure": bool(config.get("analyze_figure", True)),
        "json": bool(config.get("analyze_json", False)),
        "a_mid": config.get("a_mid"),
    }
    ANALYZE_SYN_SIGN = {
        "bins": config["bins"],
        "after_train": config["after_train"],
        "post": bool(config.get("post", False)),
    }
    ANALYZE_TRACE = {key: config[key] for key in _ANALYZE_TRACE_KEYS}
    ANALYZE_TRACE["check"] = config.get("check")
    ANALYZE_TRACE["baseline_ms_shown"] = config.get("baseline_ms_shown")
    ANALYZE_COST_PART = {
        "part": config.get("part"),
        "cost_norm": config.get("analyze_cost_norm"),
        "stride": int(config.get("stride") or 1),
        "per_node": bool(config.get("per_node", False)),
        "csv": config.get("cost_part_csv"),
        "print_parts": bool(config.get("print_parts", False)),
    }


def active_config() -> dict:
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load_config()
        _bind_config(_ACTIVE)
    return _ACTIVE


def resolve_config(hydra_config) -> dict:
    global _ACTIVE
    _ACTIVE = (
        OmegaConf.to_container(hydra_config, resolve=True)
        if OmegaConf.is_config(hydra_config)
        else dict(hydra_config)
    )
    _bind_config(_ACTIVE)
    return _ACTIVE


active_config()


def parse_cells(cells) -> List[str] | None:
    if cells is None:
        return None
    if isinstance(cells, str):
        value = cells.strip()
        if "," in value:
            raise ValueError(
                f"multiple values require Hydra list syntax: [{value}]"
            )
        values = [value] if value else []
    elif isinstance(cells, (list, tuple)) or OmegaConf.is_list(cells):
        values = [str(value).strip() for value in cells if str(value).strip()]
    else:
        raise ValueError(
            "value must be a scalar string, list, or null; "
            f"got {cells!r}"
        )
    return values or None


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


def _run_token(value) -> str:
    return re.sub(r"[^\w.,-]+", "-", str(value)).strip("-")


def _hydra_task_overrides() -> List[str]:
    from hydra.core.hydra_config import HydraConfig

    if not HydraConfig.initialized():
        return []
    return [str(token) for token in HydraConfig.get().overrides.task]


def _command_run(script_token: str, *, overrides: List[str] | None = None) -> str:
    """Build a run folder name token from Hydra task overrides (train only)."""
    prefix = os.environ.get("SLURM_JOB_ID") or time.strftime("%m%d_%H%M%S")
    parts = [prefix, script_token]
    for token in overrides or _hydra_task_overrides():
        token = str(token).lstrip("+~")
        key, sep, val = token.partition("=")
        if not sep or key == "run_name":
            continue
        parts.append(_run_token(key))
        parts.append(_run_token(val))
    run = "-".join(parts)
    if len(run) <= _RUN_MAX:
        return run
    return run[:_RUN_MAX].rstrip("-")


def _resolve_train_run_name(*, script_token: str) -> str:
    """Train ``run_dir`` name: explicit CLI ``run_name=`` else ``command_run``."""
    overrides = _hydra_task_overrides()
    for token in overrides:
        key, sep, val = str(token).lstrip("+~").partition("=")
        if sep and key == "run_name":
            return str(val).strip()
    return _command_run(script_token, overrides=overrides)


def resolve_figure_kwargs(hydra_config) -> dict:
    from figure.plot import parse_align_xy, parse_at_xs, parse_ms_shown

    resolve_config(hydra_config)
    figure_plot = FIGURE_PLOT
    align_xy = parse_align_xy(figure_plot.get("align_xy"))
    align_at_x, align_at_y = align_xy if align_xy is not None else (None, None)
    ms_shown = None
    if figure_plot.get("ms_shown") is not None:
        ms_shown = parse_ms_shown(figure_plot["ms_shown"])
    return dict(
        figure_tasks=figure_plot.get("tasks"),
        plot_right_only=bool(figure_plot.get("plot_right_only", True)),
        at_x=parse_at_xs(figure_plot.get("x")),
        at_y=parse_at_xs(figure_plot.get("y")),
        align_at_x=align_at_x,
        align_at_y=align_at_y,
        html=bool(figure_plot.get("html", False)),
        ms_shown=ms_shown,
    )


def session_kwargs_from_cli(hydra_config) -> dict:
    """CLI ``key=value`` bag for :func:`figure.plot.override_session`.

    Keys not on the Hydra CLI stay ``None`` so re-plot keeps ``train_opts.json``.
    """
    from hydra.core.hydra_config import HydraConfig

    keys = (
        "euler", "filter", "ms_pre", "ms_sti", "ms_response", "ms_post",
        "delta_ms", "delta_ms_pre",
    )
    hit = set()
    if HydraConfig.initialized():
        for token in HydraConfig.get().overrides.task:
            hit.add(str(token).lstrip("+~").split("=", 1)[0])
    if not isinstance(hydra_config, dict):
        hydra_config = OmegaConf.to_container(hydra_config, resolve=True)
    return {key: (hydra_config.get(key) if key in hit else None) for key in keys}


def resolve_run_kwargs(hydra_config, *, script_token: str = "run") -> dict:
    """Map merged Hydra config to kwargs for :func:`run_train_and_plot`."""
    import train
    from train.implementation import build_run_dir

    resolve_config(hydra_config)
    config = active_config()

    model = str(NEURON_SCHEMA["model"])
    init_from = _resolve_init_from(config.get("init_from"))
    cost_radius = TRAIN_CONFIG.get("cost_radius")

    syn_mode = str(NEURON_SCHEMA["syn_mode"])
    filter = str(NEURON_SCHEMA["filter"])
    val_from = train.resolve_val_from(VAL_FROM)
    val_from_opts = {"val_from": val_from}
    if filter != "ca":
        if train.val_from_enabled(val_from_opts, "v_th_ca") or train.val_from_enabled(val_from_opts, "a_ca"):
            raise ValueError("val_from v_th_ca / a_ca require filter ca")

    tasks = list(TRAIN_CONFIG["tasks"])
    contrasts = list(TRAIN_CONFIG["contrasts"])

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
    mbar_sti_opts = {
        "bar_dist": SBAR_INPUT_GEO["bar_dist"],
        "bar_ws_deg": list(MBAR_INPUT_SPEC["bar_ws_deg"]),
        "bar_directions": list(MBAR_INPUT_SPEC["bar_directions"]),
        "multi_bar": SBAR_INPUT_GEO["multi_bar"],
        "ms_pre": timing["ms_pre"],
        "delta_ms": timing["delta_ms"],
        "delta_ms_pre": timing["delta_ms_pre"],
    }
    sbar_sti_opts = {
        **timing,
        "bar_dist": SBAR_INPUT_GEO["bar_dist"],
        "bar_directions": list(MBAR_INPUT_SPEC["bar_directions"]),
        "multi_bar": SBAR_INPUT_GEO["multi_bar"],
        "shift_mid": SBAR_INPUT_GEO["shift_mid"],
        "a_sti_mids": list(SBAR_PACK["a_sti_mids"]),
    }
    spot_sti_opts = {
        **timing,
        "shift_radius": SPOT_INPUT_GEO["shift_radius"],
        "spot_radius": SPOT_INPUT_GEO["spot_radius"],
        "multi_spot": SPOT_INPUT_GEO["multi_spot"],
        "fully_inside": SPOT_INPUT_GEO["fully_inside"],
    }
    spread_sti_opts = dict(timing)

    cost_ms = TRAIN_OPTIMIZATION["cost_ms"]

    gt_cells_by_task = train.resolve_gt_cells_by_task(TRAIN_CONFIG.get("gt_by_task"))
    if gt_cells_by_task:
        gt_opts = {
            "mbar": mbar_sti_opts,
            "sbar": sbar_sti_opts,
            "spot": spot_sti_opts,
            "spread": spread_sti_opts,
        }
        for task, cells in gt_cells_by_task.items():
            gt_opts[task]["gt_cells"] = list(cells)

    lrs = list(TRAIN_OPTIMIZATION["lrs"])
    if not lrs:
        raise ValueError("lrs must list at least one learning rate")

    cuda_available = torch.cuda.is_available()
    fp = int(TRAIN_SESSION["fp"])
    if not cuda_available and fp == 64:
        fp = 32

    n_iter = TRAIN_OPTIMIZATION["n_iter"]

    run_name = _resolve_train_run_name(script_token=script_token)

    train_kwargs = dict(
        model=model,
        params=config["params"],
        n_run=int(TRAIN_OPTIMIZATION["n_run"]),
        n_iter=int(n_iter),
        lrs=lrs,
        run_dir=build_run_dir(model, run=run_name),
        syn_mode=syn_mode,
        network=str(NETWORK_PATH["network"]),
        tasks=tasks,
        contrasts=contrasts,
        i_sti=dict(TRAIN_CONFIG["i_sti"]),
        part_cost_scales=part_cost_scales,
        cost_norm=str(TRAIN_OPTIMIZATION["cost_norm"]),
        cost_entry_reduce=str(TRAIN_OPTIMIZATION["cost_entry_reduce"]),
        cost_ms=cost_ms,
        cost_radius=cost_radius,
        mbar_sti_opts=mbar_sti_opts,
        sbar_sti_opts=sbar_sti_opts,
        spread_sti_opts=spread_sti_opts,
        spot_sti_opts=spot_sti_opts,
        euler=str(MODEL["euler"]),
        pre_steady=str(TRAIN_OPTIMIZATION["pre_steady"]),
        pre_steady_n_iter=TRAIN_OPTIMIZATION["pre_steady_n_iter"],
        pre_steady_damp=TRAIN_OPTIMIZATION["pre_steady_damp"],
        fp=fp,
        pre_grad=bool(NEURON_FORWARD["pre_grad"]),
        val_from=val_from,
        filter=filter,
        sequential=bool(TRAIN_SESSION["sequential"]),
        init_from=init_from,
        checkpoint_interval=TRAIN_OPTIMIZATION.get("checkpoint_interval"),
        syn_sign=bool(ANALYZE_SYN_SIGN.get("after_train", False)),
    )
    train_kwargs["figure_kwargs"] = resolve_figure_kwargs(hydra_config)
    return train_kwargs
