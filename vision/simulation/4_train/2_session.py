# -*- coding: utf-8 -*-
"""Session types and assembly: ``TrainSession`` + open helpers.

Owns sti-opts finalisation (CLI tokens -> per-task sidecar dicts),
connectome loading, and delegates per-task×contrast ``Pack`` assembly to
:mod:`task.spread.pack`, :mod:`task.spot.pack`, and :mod:`task.moving_bar.pack`.
"""
from __future__ import annotations

from config import (
    MOVING_BAR_INPUT_GEO,
    MOVING_BAR_INPUT_SPEC,
    NEURON_FORWARD,
    MODEL,
    NEURON_SCHEMA,
    SPREAD_INPUT_SPEC,
    SPREAD_GT,
    SPOT_INPUT_GEO,
    SPOT_PACK,
    TRAIN_CONFIG,
    TRAIN_OPTIMIZATION,
    TRAIN_SESSION,
    VAL_FROM,
)

import copy
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from import_bootstrap import parse_comma_list
from neuron import (
    build_schema,
    expand_euler,
)
from train.param import (
    PARAM_MODES,
    SIM_DTYPE,
    active_device,
    schema_copy,
    schema_with_param_carry,
    resolve_val_from,
    schema_n_z,
    sim_dtype_from_fp,
    val_from_enabled,
)

from task.spread.gt import expand_gt_cells as expand_spread_gt_cells
from task.spread.pack import (
    Pack,
    build_spread_pack,
    resolve_spread_sti_opts,
)
from task.spot.gt import expand_gt_cells as expand_spot_gt_cells
from task.spot.pack import (
    build_spot_pack,
    resolve_spot_sti_opts,
)
from task.moving_bar.gt import expand_gt_cells as expand_moving_bar_gt_cells
from task.moving_bar.pack import (
    build_moving_bar_pack,
    resolve_moving_bar_sti_opts,
)
from network.construction import (
    load_network, Network,
)

TASKS = ("spread", "spot", "moving_bar")
RUN_DATA_SUBDIR = "data"


def run_data_dir(run_dir) -> str:
    return str(Path(run_dir) / RUN_DATA_SUBDIR)


def _tokens(values) -> List[str]:
    if isinstance(values, str):
        return parse_comma_list(values)
    return [str(token) for token in values]


@dataclass(frozen=True)
class TrainSession:
    """Immutable runtime context for one train / plotting run.

    Model / synapse scalars are flat fields (injected from
    ``config`` at session open). ``delta_ms`` / ``delta_ms_pre`` come
    only from sti opts — never nested under Physics.
    """

    connectome: Network
    model: str
    schema: dict
    packs: Dict[str, Dict[str, Pack]]
    tasks: Tuple[str, ...]
    contrasts: Tuple[str, ...]
    part_cost_scales: Dict[str, float]
    sequential: bool
    device: str
    delta_ms: float
    delta_ms_pre: float
    cap: float
    g_leak: float
    e_exc: float
    e_inh: float
    e_h: float
    h_g_max: float
    gt_amp: float
    v_clamp: float
    a_syn_exc: float
    a_syn_inh: float
    euler: str
    pre_steady: str = "solve"
    pre_steady_n_iter: int = 60
    pre_steady_damp: float = 1.0
    sim_dtype: torch.dtype = SIM_DTYPE
    train_opts: Optional[dict] = None

    def with_schema(self, schema) -> "TrainSession":
        return replace(self, schema=schema_copy(schema))

    def iter_packs(self) -> Iterator[Pack]:
        for task in self.tasks:
            for contrast in self.contrasts:
                yield self.packs[task][contrast]

    @property
    def primary_pack(self) -> Pack:
        return self.packs[self.tasks[0]][self.contrasts[0]]

    @property
    def n_t(self) -> int:
        i_sti = self.primary_pack.i_sti
        return int(i_sti.shape[1] if i_sti.dim() == 3 else i_sti.shape[0])

    def pack_i_sti(self, pack: Optional[Pack] = None) -> torch.Tensor:
        pack = pack or self.primary_pack
        i_sti = pack.i_sti
        if pack.task == "spot" and i_sti.dim() == 3 and int(i_sti.shape[0]) == 1:
            i_sti = i_sti.squeeze(0)
        return i_sti


def resolve_cell_idxs(cells, connectome: Network):
    """Map cells to idxs in the network vocabulary."""
    if connectome is None:
        raise ValueError("resolve_cell_idxs requires connectome")
    wanted = [str(n) for n in cells]
    vocab = list(connectome.cells)
    return [vocab.index(n) for n in wanted if n in vocab]


def load_train_connectome(
    network_json,
    device: Optional[str] = None,
    *,
    a_syn_exc: float,
    a_syn_inh: float,
    sim_dtype=SIM_DTYPE,
    syn_mode=NEURON_SCHEMA['syn_mode'],
    params=NEURON_SCHEMA['params'],
) -> Network:
    """Load connectome ``Network`` for train (print summary)."""
    device = device or active_device()
    mode = syn_mode
    connectome = load_network(
        network_json, device=device,
        a_syn_exc=a_syn_exc, a_syn_inh=a_syn_inh,
        dtype=sim_dtype, syn_mode=mode,
    )
    print(f"network: {network_json}")
    print(f"  n_node={connectome.n_node}, n_cell={connectome.n_cell}, "
          f"n_pair={connectome.conn.n_pair}, n_edge={connectome.conn.n_edge}, "
          f"syn_mode={mode}, "
          f"n_z={schema_n_z(build_schema('borst', connectome, syn_mode=mode, params=params, a_sti_radii=SPOT_PACK['a_sti_radii']))}")
    return connectome


_GT_CELLS_EXPAND = {
    "spread": expand_spread_gt_cells,
    "spot": expand_spot_gt_cells,
    "moving_bar": expand_moving_bar_gt_cells,
}


def resolve_gt_cells_by_task(by_task) -> Dict[str, List[str]]:
    """Map concrete tasks to final gt cell lists (task + cell aliases expanded)."""
    return {
        str(task): list(_GT_CELLS_EXPAND[task]([str(cell) for cell in cells]))
        for task, cells in (by_task or {}).items()
    }


_STI_TRAIN_OPT_KEYS = (
    ("spread", "spread_sti_opts"),
    ("spot", "spot_sti_opts"),
    ("moving_bar", "moving_bar_sti_opts"),
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
    "moving_bar": {
        "ms_pre": MOVING_BAR_INPUT_SPEC["ms_pre"],
        "delta_ms": MODEL["delta_ms"],
        "delta_ms_pre": MODEL["delta_ms_pre"],
        "multi_bar": MOVING_BAR_INPUT_GEO["multi_bar"],
    },
}


def resolve_i_sti(i_sti=None) -> Dict[str, float]:
    """Merge ``config`` ``i_sti`` defaults with optional contrast overrides."""
    out = dict(TRAIN_CONFIG["i_sti"])
    if i_sti:
        out.update({str(contrast): float(val) for contrast, val in i_sti.items()})
    return out


_RESOLVE_STI_OPTS = {
    "spread": resolve_spread_sti_opts,
    "spot": resolve_spot_sti_opts,
    "moving_bar": resolve_moving_bar_sti_opts,
}


def _resolve_sti_opts(
    opts,
    task,
    *,
    cost_radius,
    shift_radius,
    spot_radius,
    multi_spot,
    fully_inside,
):
    opts = {**_STI_OPTS_BY_TASK.get(task, {}), **(opts or {})}
    sti_opts = _RESOLVE_STI_OPTS[task](
        opts,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
    )
    if cost_radius is not None:
        sti_opts["cost_radius"] = int(cost_radius)
    else:
        sti_opts.pop("cost_radius", None)
    return sti_opts


def resolve_train_opts(
    tasks=None,
    contrasts=None,
    part_cost_scales=None,
    sequential=None,
    cost_radius=None,
    shift_radius=None,
    spot_radius=None,
    multi_spot=SPOT_INPUT_GEO['multi_spot'],
    fully_inside=SPOT_INPUT_GEO['fully_inside'],
    i_sti=None,
    cost_norm=TRAIN_OPTIMIZATION['cost_norm'],
    cost_ms=None,
    moving_bar_sti_opts=None,
    spread_sti_opts=None,
    spot_sti_opts=None,
    network_json=None,
    network=None,
    syn_mode=NEURON_SCHEMA['syn_mode'],
    device=None,
    packs=None,
    euler=MODEL['euler'],
    pre_steady=None,
    pre_steady_n_iter=TRAIN_OPTIMIZATION['pre_steady_n_iter'],
    pre_steady_damp=TRAIN_OPTIMIZATION['pre_steady_damp'],
    fp=TRAIN_SESSION['fp'],
    pre_grad=NEURON_FORWARD['pre_grad'],
    val_from=None,
    filter=NEURON_SCHEMA['filter'],
    spread_gt_mode=SPREAD_GT['spread_gt_mode'],
):
    """Canonical train opts for :func:`open_session`."""
    if network is None and network_json is None:
        raise ValueError("resolve_train_opts requires network or network_json")
    fp = int(fp)
    if fp not in (16, 32, 64):
        raise ValueError(f"fp must be 16, 32, or 64; got {fp!r}")
    filter = str(filter)
    if pre_steady is None:
        pre_steady = TRAIN_OPTIMIZATION['pre_steady']
    if sequential is None:
        sequential = TRAIN_SESSION['sequential']
    pre_steady_n_iter = int(pre_steady_n_iter)
    pre_steady_damp = float(pre_steady_damp)
    if pre_steady_n_iter < 1:
        raise ValueError(f"pre_steady_n_iter must be >= 1; got {pre_steady_n_iter}")
    if not (0.0 < pre_steady_damp <= 1.0):
        raise ValueError(
            f"pre_steady_damp must be in (0, 1]; got {pre_steady_damp}"
        )
    val_from = resolve_val_from(val_from=val_from)
    val_from_opts = {"val_from": val_from}
    if filter != "ca":
        if val_from_enabled(val_from_opts, "v_th_ca") or val_from_enabled(val_from_opts, "a_ca"):
            raise ValueError(
                "val_from v_th_ca / a_ca require filter ca "
                f"(got filter={filter!r})"
            )
    tasks = _tokens(tasks if tasks is not None else TRAIN_CONFIG["tasks"])
    contrasts = _tokens(
        contrasts if contrasts is not None else TRAIN_CONFIG["contrasts"]
    )
    if spot_radius is None:
        spot_radius = SPOT_INPUT_GEO['spot_radius']
    if shift_radius is None:
        shift_radius = SPOT_INPUT_GEO['shift_radius']
    finalize_kwargs = dict(
        cost_radius=cost_radius,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
    )
    sti_opts_by_task = {
        "spread": spread_sti_opts,
        "spot": spot_sti_opts,
        "moving_bar": moving_bar_sti_opts,
    }
    sti_opts = {}
    for task, sti_opts_key in _STI_TRAIN_OPT_KEYS:
        if task not in tasks and sti_opts_by_task[task] is None:
            sti_opts[sti_opts_key] = None
            continue
        sti_opts[sti_opts_key] = _resolve_sti_opts(
            sti_opts_by_task[task],
            task,
            **finalize_kwargs,
        )
    opts = {
        "tasks": tasks,
        "contrasts": contrasts,
        "i_sti": resolve_i_sti(i_sti),
        "part_cost_scales": {
            str(part_key): float(scale)
            for part_key, scale in (
                part_cost_scales
                if part_cost_scales is not None
                else TRAIN_OPTIMIZATION['part_cost_scales']
                or {}
            ).items()
        },
        "cost_norm": cost_norm,
        "cost_ms": copy.deepcopy(
            cost_ms if cost_ms is not None else TRAIN_OPTIMIZATION['cost_ms']
        ),
        "pre_steady": pre_steady,
        "pre_steady_n_iter": pre_steady_n_iter,
        "pre_steady_damp": pre_steady_damp,
        "sequential": sequential,
        **sti_opts,
        "euler": euler,
        "syn_mode": syn_mode,
        "pre_grad": pre_grad,
        "val_from": copy.deepcopy(val_from),
        "filter": filter,
        "spread_gt_mode": spread_gt_mode,
        "fp": fp,
        "packs": None,
        "params": copy.deepcopy(NEURON_SCHEMA["params"]),
        "network": network,
        "network_json": str(network_json) if network_json is not None else None,
        "device": device,
    }
    if packs is not None:
        opts["packs"] = packs
    return opts


def _cost_ms_sidecar(cost_ms):
    """JSON sidecar: interval float or explicit ``mss`` list."""
    if isinstance(cost_ms, bool) or cost_ms is None:
        raise ValueError("cost_ms must be an interval or a list of ms")
    if isinstance(cost_ms, dict):
        raise ValueError("cost_ms must be an interval or a list of ms, not a radius map")
    if isinstance(cost_ms, (int, float)):
        return float(cost_ms)
    if isinstance(cost_ms, str):
        tokens = parse_comma_list(cost_ms)
        if len(tokens) == 1:
            return float(tokens[0])
        return [float(x) for x in tokens]
    mss = [float(ms) for ms in cost_ms]
    if not mss:
        raise ValueError("cost_ms list must have at least one ms")
    return mss


def _sidecar_train_opts(opts, tasks, contrasts, resolved_sti, sequential_bool) -> dict:
    """Build JSON-serializable train_opts."""
    def _sti(sti_opts_key):
        got = resolved_sti.get(sti_opts_key)
        return got if got is not None else opts.get(sti_opts_key)

    train_opts = {
        "tasks": list(tasks),
        "contrasts": list(contrasts),
        "i_sti": {
            contrast: float(val)
            for contrast, val in (opts.get("i_sti") or {}).items()
        },
        "part_cost_scales": {
            str(part_key): float(scale)
            for part_key, scale in (opts.get("part_cost_scales") or {}).items()
        },
        "cost_norm": opts.get("cost_norm", TRAIN_OPTIMIZATION['cost_norm']),
        "cost_ms": _cost_ms_sidecar(
            opts.get("cost_ms", TRAIN_OPTIMIZATION['cost_ms'])
        ),
        "pre_steady": opts.get("pre_steady", TRAIN_OPTIMIZATION['pre_steady']),
        "pre_steady_n_iter": int(opts.get("pre_steady_n_iter", TRAIN_OPTIMIZATION['pre_steady_n_iter'])),
        "pre_steady_damp": float(opts.get("pre_steady_damp", TRAIN_OPTIMIZATION['pre_steady_damp'])),
        "sequential": sequential_bool,
        "network_json": str(opts["network_json"]),
        "spread_sti_opts": _sti("spread_sti_opts"),
        "spot_sti_opts": _sti("spot_sti_opts"),
        "moving_bar_sti_opts": _sti("moving_bar_sti_opts"),
    }
    if opts.get("params"):
        train_opts["params"] = copy.deepcopy(opts["params"])
    if "euler" not in opts:
        raise ValueError("train opts require euler (implicit|explicit)")
    train_opts["euler"] = opts["euler"]
    train_opts["syn_mode"] = opts.get("syn_mode", NEURON_SCHEMA['syn_mode'])
    train_opts["pre_grad"] = opts.get("pre_grad", NEURON_FORWARD['pre_grad'])
    train_opts["val_from"] = copy.deepcopy(opts.get("val_from", VAL_FROM))
    train_opts["filter"] = str(opts.get("filter", NEURON_SCHEMA['filter']))
    train_opts["spread_gt_mode"] = opts.get("spread_gt_mode", SPREAD_GT['spread_gt_mode'])
    train_opts["fp"] = int(opts.get("fp", TRAIN_SESSION['fp']))
    return train_opts


def resolve_schema(model, connectome, schema, train_opts):
    """Build the train schema from sidecar / YAML ``params``."""
    if schema is not None:
        return schema_copy(schema)
    filter = NEURON_SCHEMA['filter']
    if train_opts:
        filter = str(train_opts.get("filter", NEURON_SCHEMA['filter']))
    syn_mode = (train_opts or {}).get("syn_mode", NEURON_SCHEMA['syn_mode'])
    params = (train_opts or {}).get("params") or NEURON_SCHEMA["params"]
    schema = build_schema(
        model,
        connectome,
        syn_mode=syn_mode,
        params=params,
        filter=filter,
        a_sti_radii=SPOT_PACK['a_sti_radii'],
    )
    val_from = (train_opts or {}).get("val_from") or {}
    schema = schema_copy(schema)
    for target, entry in val_from.items():
        if not entry.get("enabled") or target not in schema:
            continue
        spec = schema[target]
        n_node = spec['n_node']
        for mode in PARAM_MODES:
            spec[mode] = []
        spec["frozen"] = list(range(n_node))
    return schema


def _build_session(
    connectome: Network,
    model: str,
    tasks: List[str],
    contrasts: List[str],
    packs: Dict[str, Dict[str, Pack]],
    *,
    delta_ms: float,
    delta_ms_pre: float,
    gt_amp: float,
    part_cost_scales=None,
    sequential=None,
    device=None,
    train_opts=None,
    schema: Optional[dict] = None,
    sim_dtype=SIM_DTYPE,
) -> TrainSession:
    device = device or active_device()
    sequential = False if sequential is None else bool(sequential)
    neuron_const = MODEL
    if train_opts is not None:
        train_opts["model"] = model
        train_opts["sequential"] = sequential
    if train_opts is None or "euler" not in train_opts:
        raise ValueError("train opts require euler (implicit|explicit)")
    euler = expand_euler(train_opts["euler"])
    pre_steady = str(
        train_opts.get("pre_steady", TRAIN_OPTIMIZATION['pre_steady']),
    )
    train_opts["pre_steady"] = pre_steady
    pre_steady_n_iter = int(
        train_opts.get("pre_steady_n_iter", TRAIN_OPTIMIZATION['pre_steady_n_iter'])
    )
    pre_steady_damp = float(
        train_opts.get("pre_steady_damp", TRAIN_OPTIMIZATION['pre_steady_damp'])
    )
    train_opts["pre_steady_n_iter"] = pre_steady_n_iter
    train_opts["pre_steady_damp"] = pre_steady_damp
    sch = resolve_schema(
        model, connectome, schema, train_opts,
    )
    sch = schema_with_param_carry(sch)
    cli_scales = {
        str(part_key): float(scale)
        for part_key, scale in (part_cost_scales or {}).items()
    }
    session = TrainSession(
        connectome=connectome,
        model=model,
        schema=sch,
        packs=dict(packs),
        tasks=tuple(tasks),
        contrasts=tuple(contrasts),
        part_cost_scales=cli_scales,
        sequential=sequential,
        device=device,
        delta_ms=float(delta_ms),
        delta_ms_pre=float(delta_ms_pre),
        cap=float(neuron_const['cap']),
        g_leak=float(neuron_const['g_leak']),
        e_exc=float(neuron_const['e_exc']),
        e_inh=float(neuron_const['e_inh']),
        e_h=float(neuron_const['e_h']),
        h_g_max=float(neuron_const['h_g_max']),
        gt_amp=float(gt_amp),
        v_clamp=float(neuron_const['v_clamp']),
        a_syn_exc=float(neuron_const['a_syn_exc']),
        a_syn_inh=float(neuron_const['a_syn_inh']),
        euler=euler,
        pre_steady=pre_steady,
        pre_steady_n_iter=pre_steady_n_iter,
        pre_steady_damp=pre_steady_damp,
        sim_dtype=sim_dtype,
        train_opts=train_opts,
    )
    return session


def open_session(
    opts: dict,
    model: str,
    *,
    schema: Optional[dict] = None,
    connectome: Optional[Network] = None,
) -> TrainSession:
    """Build a :class:`TrainSession` from canonical train opts."""
    opts = dict(opts)
    opts.pop("backend", None)
    filter = str(opts.get("filter", NEURON_SCHEMA['filter']))
    gt_amp = float(MODEL['gt_amp'])
    neuron_const = MODEL
    tasks = _tokens(opts.get("tasks"))
    contrasts = _tokens(opts.get("contrasts"))
    raw_i_sti = opts.get("i_sti")
    if raw_i_sti is None:
        raise ValueError("train opts require i_sti")
    i_sti = {str(contrast): float(val) for contrast, val in raw_i_sti.items()}
    opts["i_sti"] = i_sti
    device = opts.get("device") or active_device()
    sim_dtype = sim_dtype_from_fp(int(opts.get("fp", TRAIN_SESSION['fp'])))
    delta_ms, delta_ms_pre = _sti_delta_ms(opts)

    net = opts.get("network")
    syn_mode = opts.get("syn_mode", NEURON_SCHEMA['syn_mode'])
    if net is None:
        nj = opts.get("network_json")
        if not nj:
            raise ValueError("open_session requires opts['network'] or network_json")
        net = load_network(
            nj, device=device,
            a_syn_exc=float(neuron_const['a_syn_exc']),
            a_syn_inh=float(neuron_const['a_syn_inh']),
            dtype=sim_dtype, syn_mode=syn_mode,
        )
    if connectome is None:
        connectome = net
    elif connectome is not net:
        raise ValueError("connectome must be opts['network']")
    pack_kwargs = dict(
        connectome=connectome,
        device=device,
        sim_dtype=sim_dtype,
        i_sti=i_sti,
        gt_amp=gt_amp,
        filter=filter,
    )
    packs: Dict[str, Dict[str, Pack]] = {}
    resolved_sti = {}
    for task in tasks:
        packs[task] = {}
        for contrast in contrasts:
            if task == "spread":
                pack, sti_opts, _label = build_spread_pack(
                    spread_sti_opts=opts.get("spread_sti_opts"),
                    spread_gt_mode=str(
                        opts.get("spread_gt_mode", SPREAD_GT['spread_gt_mode']),
                    ),
                    cost_ms=opts.get("cost_ms", TRAIN_OPTIMIZATION['cost_ms']),
                    contrast=contrast, **pack_kwargs,
                )
            elif task == "moving_bar":
                pack, sti_opts, _label = build_moving_bar_pack(
                    moving_bar_sti_opts=opts.get("moving_bar_sti_opts"),
                    delta_ms=delta_ms,
                    delta_ms_pre=delta_ms_pre,
                    ms_pre=MOVING_BAR_INPUT_SPEC["ms_pre"],
                    multi_bar=MOVING_BAR_INPUT_GEO['multi_bar'],
                    contrast=contrast, **pack_kwargs,
                )
            elif task == "spot":
                pack, sti_opts, _label = build_spot_pack(
                    spot_sti_opts=opts.get("spot_sti_opts"),
                    spread_gt_mode=str(
                        opts.get("spread_gt_mode", SPREAD_GT['spread_gt_mode']),
                    ),
                    cost_ms=opts.get("cost_ms", TRAIN_OPTIMIZATION['cost_ms']),
                    cost_radius_scales=SPOT_PACK['spot_cost_radius_scale'],
                    spot_cost_radii=SPOT_PACK['spot_cost_radii'],
                    a_sti_radii=SPOT_PACK['a_sti_radii'],
                    contrast=contrast, **pack_kwargs,
                )
            else:
                raise ValueError(f"unknown task {task!r}")
            packs[task][contrast] = pack
            resolved_sti[f"{task}_sti_opts"] = sti_opts
    train_opts = _sidecar_train_opts(
        opts, tasks, contrasts, resolved_sti, bool(opts.get("sequential")),
    )
    return _build_session(
        connectome, model, tasks, contrasts, packs,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        gt_amp=gt_amp,
        part_cost_scales=opts.get("part_cost_scales"),
        sequential=opts.get("sequential"),
        device=device,
        train_opts=train_opts,
        schema=schema,
        sim_dtype=sim_dtype,
    )


def _sti_delta_ms(opts: dict) -> tuple[float, float]:
    """``delta_ms`` / ``delta_ms_pre`` from sti opts (required)."""
    delta_ms = None
    delta_ms_pre = None
    for _task, sti_opts_key in _STI_TRAIN_OPT_KEYS:
        so = opts.get(sti_opts_key)
        if not isinstance(so, dict):
            continue
        if delta_ms is None and so.get("delta_ms") is not None:
            delta_ms = float(so["delta_ms"])
        if delta_ms_pre is None and so.get("delta_ms_pre") is not None:
            delta_ms_pre = float(so["delta_ms_pre"])
    missing = [
        name for name, val in (("delta_ms", delta_ms), ("delta_ms_pre", delta_ms_pre))
        if val is None
    ]
    if missing:
        raise ValueError(
            f"train opts require {', '.join(missing)} in a sti opts dict "
            f"(one of {[key for _, key in _STI_TRAIN_OPT_KEYS]})"
        )
    if delta_ms <= 0 or delta_ms_pre <= 0:
        raise ValueError(
            f"sti opts delta_ms / delta_ms_pre must be > 0, "
            f"got {delta_ms}, {delta_ms_pre}"
        )
    return delta_ms, delta_ms_pre


def resolve_session(opts: dict, model: str | None = None, **kwargs) -> TrainSession:
    """Restore a session from a saved ``train_opts.json`` dict."""
    opts = dict(opts)
    opts.pop("backend", None)
    if model is None:
        model = opts.get("model")
        if not model:
            raise ValueError("train_opts requires model")
    opts["packs"] = None
    nj = opts.get("network_json")
    if not nj:
        raise ValueError("train_opts requires network_json")
    if not opts.get("tasks"):
        raise ValueError("train_opts requires tasks")
    if not opts.get("contrasts"):
        raise ValueError("train_opts requires contrasts")
    sim_dtype = sim_dtype_from_fp(int(opts.get("fp", TRAIN_SESSION['fp'])))
    neuron_const = MODEL
    syn_mode = opts.get("syn_mode", NEURON_SCHEMA['syn_mode'])
    mb = load_train_connectome(
        nj, device=opts.get("device") or active_device(), sim_dtype=sim_dtype,
        syn_mode=syn_mode,
        a_syn_exc=float(neuron_const['a_syn_exc']),
        a_syn_inh=float(neuron_const['a_syn_inh']),
    )
    opts["network"] = mb
    opts["syn_mode"] = syn_mode
    kwargs.setdefault("connectome", mb)
    return open_session(opts, model, **kwargs)


def session_from_run_dir(
    run_dir: str,
    model: str | None = None,
) -> TrainSession:
    """Load ``train_opts.json`` from a run folder and return a ready session."""
    opts_path = os.path.join(run_data_dir(os.path.abspath(run_dir)), "train_opts.json")
    if not os.path.isfile(opts_path):
        raise FileNotFoundError(f"missing {opts_path}")
    with open(opts_path) as f:
        opts = json.load(f)
    return resolve_session(opts, model)
