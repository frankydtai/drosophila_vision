# -*- coding: utf-8 -*-
"""Train CLI: argparse registration and argv → kwargs for implementation.

Does not run optimization or touch CUDA. Callers: ``simulation/run.py``,
``figure``, ``analyze``.

Branch-resolution contract (must keep):

* Do not add parameter-specific helper parsers/casters.
* CLI should pass branch-capable values through in a generic form.
* ``{v,ca}`` selection and branch-value casting belong to the unified
  post-parse path in ``open_session`` (via ``resolve_filter_branches``);
  non-branch CLI controls keep their normal scalar parsing.
"""
from __future__ import annotations

import re

from const_default import (
    MODEL,
    MOVING_BAR_INPUT,
    NETWORK_PATH,
    NEURON_FILTER,
    NEURON_FORWARD,
    NEURON_CONST,
    NEURON_SCHEMA,
    SPOT_INPUT,
    SPOT_PACK,
    STI_TIMING,
    TRAIN_CONFIG,
    TRAIN_OPTIMIZATION,
    TRAIN_OPTS,
    TRAIN_SESSION,
    VAL_FROM,
)

import argparse
import os
import re
import sys
import time
from pathlib import Path


def _cli_scalar_from_branch(val):
    """Return scalar CLI default from a value that may be a ``{v, ca}`` branch dict."""
    if isinstance(val, dict) and set(val) <= {"v", "ca"}:
        return val.get("v", next(iter(val.values())))
    return val

import torch

_SIMULATION_CODE = Path(__file__).resolve().parent.parent
if str(_SIMULATION_CODE) not in sys.path:
    sys.path.insert(0, str(_SIMULATION_CODE))

import import_bootstrap  # noqa: F401
from import_bootstrap import standardize_option_dashes, parse_bool, parse_comma_list
import network.path  # noqa: F401 — FAFB path on sys.path
from path import BUILT_NETWORKS_DIR
from task.spot.pack import (
    parse_cost_ms_tokens,
    resolve_spot_cost_radius_scale,
)
import train
from train.config import (
    COST_NORMS,
    I_STI_KEYS,
    SPOT_GT_MODES,
    TASKS,
    expand_cost_norm,
    expand_pre_steady,
)

RUN_MAX = 255


def _slug(token):
    """Filesystem-safe token for a CLI flag value."""
    return re.sub(r'[^\w.,-]+', '-', str(token)).strip('-')


def _argv_cli_tokens(argv):
    """Drop the script path; yield long-option tokens from *argv*."""
    if argv and argv[0].endswith('.py'):
        argv = argv[1:]
    argv = standardize_option_dashes(argv)
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ('-h', '--help'):
            i += 1
            continue
        if not token.startswith('--'):
            i += 1
            continue
        flag, sep, val = token[2:].partition('=')
        if sep:
            yield _slug(flag), _slug(val)
            i += 1
        elif i + 1 < len(argv) and not argv[i + 1].startswith('-'):
            yield _slug(flag), _slug(argv[i + 1])
            i += 2
        else:
            yield _slug(flag), None
            i += 1


def command_run(script_stem, argv=None):
    """Build a run folder stem from flags on the command line (``sys.argv``)."""
    if argv is None:
        argv = sys.argv[1:]
    prefix = os.environ.get('SLURM_JOB_ID') or time.strftime('%m%d_%H%M%S')
    parts = [prefix, script_stem]
    for flag, val in _argv_cli_tokens(argv):
        parts.append(flag)
        if val is not None:
            parts.append(val)
    run = '-'.join(parts)
    if len(run) <= RUN_MAX:
        return run
    return run[:RUN_MAX].rstrip('-')


def add_multi_spot_arguments(parser):
    """Spot center tiling flags (``--multi-spot``, ``--fully-inside``)."""
    parser.add_argument(
        "--multi-spot",
        type=_branch_cli_type(SPOT_INPUT['multi_spot']),
        default=SPOT_INPUT['multi_spot'],
        metavar="BOOL",
        help="tile simultaneous spot centers on network connectome "
             f"(default: {_format_branch_value(SPOT_INPUT['multi_spot'])}; false → center (0,0) only)",
    )
    parser.add_argument(
        "--fully-inside",
        type=_branch_cli_type(SPOT_INPUT['fully_inside']),
        default=SPOT_INPUT['fully_inside'],
        metavar="BOOL",
        help="with --multi-spot: keep only centers whose spot footprint lies inside "
             f"connectome radius (default: {_format_branch_value(SPOT_INPUT['fully_inside'])})",
    )


def _format_filename_token(value):
    if isinstance(value, dict) and set(value) <= {"v", "ca"}:
        val_v = float(value["v"])
        val_ca = float(value["ca"])
        if val_v == val_ca:
            return _format_filename_token(val_v)
        return f"v{val_v:g}-ca{val_ca:g}"
    val = float(value)
    if val == int(val):
        return str(int(val))
    return "%g" % val


def add_euler_argument(parser, *, default=None):
    if default is None:
        help = (
            "Euler (plot/analyze): im=implicit, ex=explicit "
            "(default: keep run train_opts.euler); i_h gates always explicit"
        )
    else:
        help = (
            "Euler: im=implicit (default), ex=explicit; "
            "i_h gates always explicit"
        )
    parser.add_argument(
        "--euler",
        default=default,
        choices=list(train.EULER_CLI),
        help=help,
    )


def add_filter_argument(parser, *, default=None):
    if default is None:
        help = (
            "readout filter (plot/analyze): none=v (schema skips v_th_ca/a_ca/tau_ca), "
            "ca=ca + Arenz digitized spot gt (default: keep run train_opts.filter)"
        )
    else:
        help = (
            "readout filter: none=v (schema skips v_th_ca/a_ca/tau_ca), "
            f"ca=ca + Arenz digitized spot gt (default: {default}; "
            "--sti-timing updates v or ca branch)"
        )
    parser.add_argument(
        "--filter",
        default=default,
        choices=("none", "ca"),
        help=help,
    )


def euler_filename_suffix(euler=None):
    """PNG stem suffix for a non-``None`` ``--euler`` (``_im`` / ``_ex``)."""
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
    """PNG stem suffix for a non-``None`` ``--filter`` (``_v`` / ``_ca``)."""
    if filter is None:
        return ""
    expanded = train.expand_filter(filter)
    return f"_{'v' if expanded == 'none' else expanded}"


_PARAM_HELP = (
    "example: a_h.init.L1=0.5 a_h.lo.L1=0.1 a_h.val.L1=0.8 a_h.mode=fixed "
    "(init/lo/hi/jit=schema numbers; val=current/trained for plot)"
)


def _val_from_literal_tokens():
    return " ".join(
        f"{target}={entry['source']}:{str(entry['enabled']).lower()}"
        for target, entry in VAL_FROM.items()
    )


def add_val_from_argument(parser):
    """Register ``--val-from TARGET=SOURCE:BOOL`` (space-separated, like ``--param``)."""
    parser.add_argument(
        "--val-from",
        dest="val_from",
        nargs="+",
        default=None,
        metavar="TARGET=SOURCE:BOOL",
        help=(
            "param copied from source at materialize (space-separated TARGET=SOURCE:BOOL). "
            f"Default: {_val_from_literal_tokens()}"
        ),
    )


def add_param_argument(parser, *, figure=False):
    help = _PARAM_HELP
    if figure:
        help += "; PNG stem suffix per init/val"
    parser.add_argument(
        "--param",
        nargs="+",
        default=None,
        metavar="PARAM.KEY[.NODES]=VALUE",
        help=help,
    )


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


def add_train_arguments(parser):
    """Register train CLI flags on *parser*."""
    parser.add_argument("--model", default=MODEL['model'], choices=list(train.MODELS))
    parser.add_argument(
        "--syn-mode",
        default=_cli_scalar_from_branch(NEURON_SCHEMA['syn_mode']),
        choices=list(train.SYN_MODES),
        help="synaptic edge weight: per_cell (syn_sign*n_syn + type→type syn_strength_cell; default) "
             "or per_edge (syn_sign only + per-edge syn_strength_edge magnitude)",
    )
    parser.add_argument("--n-run", type=int, default=TRAIN_OPTIMIZATION['n_run'])
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help=f"iters per learning-rate stage (default: {TRAIN_OPTIMIZATION['n_iter_gpu']} on GPU, "
             f"{TRAIN_OPTIMIZATION['n_iter_cpu']} on CPU)",
    )
    parser.add_argument("--lrs", default=TRAIN_OPTIMIZATION['lrs'],
                        help="comma-separated learning-rate stages; each runs for --n-iter iters")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=TRAIN_OPTIMIZATION['checkpoint_interval'],
        metavar="N",
        help="every N global train iters, snap to the interval-best params and write "
             "data/best_param_iter_XXXXX.npz, csv/param_XXXXX.csv, "
             "csv/syn_strength_cell_XXXXX.csv or csv/syn_strength_edge_XXXXX.csv, and png/*_XXXXX.png "
             f"(default: {TRAIN_OPTIMIZATION['checkpoint_interval']})",
    )
    parser.add_argument("--fname", default=None,
                        help="params filename (default derived from --model)")
    parser.add_argument("--outdir", default=None,
                        help="output dir (default derived from --model)")
    parser.add_argument("--init-from", dest="init_from", default=None, metavar="MODEL['model']/RUN",
                        help="prior run as MODEL['model']/RUN under 0_runs (e.g. borst/<run>); "
                             "or an absolute path; load named best_param.npz as z init "
                             "and best_adam.npz as adam m/v "
                             "(settings come from this CLI, not train_opts.json)")
    add_param_argument(parser)
    add_val_from_argument(parser)
    add_euler_argument(parser, default=_cli_scalar_from_branch(NEURON_CONST['euler']))
    parser.add_argument(
        "--pre-steady",
        default=_cli_scalar_from_branch(TRAIN_OPTIMIZATION['pre_steady']),
        choices=("probe", "solve"),
        help=(
            "t=0 pre steady shared by borst/hp_lp "
            f"(probe|solve; default: {TRAIN_OPTIMIZATION['pre_steady']}); "
            "solve uses fixed iters/damp from const_default"
        ),
    )
    parser.add_argument(
        "--fp",
        type=int,
        default=TRAIN_SESSION['fp'],
        choices=(16, 32, 64),
        metavar="N",
        help=f"simulation float w (default: {TRAIN_SESSION['fp']}); "
             "64 is forced to 32 when CUDA is unavailable",
    )
    parser.add_argument(
        "--pre-grad",
        type=parse_bool,
        default=_cli_scalar_from_branch(NEURON_FORWARD['pre_grad']),
        metavar="BOOL",
        help="include t < t_onset in BPTT "
             f"(default: {str(NEURON_FORWARD['pre_grad']).lower()}); "
             "false → no_grad pre + detach v / v_slow|u at onset",
    )
    add_filter_argument(parser, default=NEURON_FILTER['filter'])
    parser.add_argument(
        "--spot-gt-mode",
        default=_cli_scalar_from_branch(SPOT_PACK['spot_gt_mode']),
        choices=list(SPOT_GT_MODES),
        help="spot cost GT mode: all=every cell both contrasts, "
             "positive=only rf_sign×contrast_sign>0 "
             f"(bright ON / dark OFF; default: {SPOT_PACK['spot_gt_mode']})",
    )
    parser.add_argument(
        "--sequential",
        type=parse_bool,
        default=_cli_scalar_from_branch(TRAIN_SESSION['sequential']),
        metavar="BOOL",
        help=f"one sti b per forward (default: {str(TRAIN_SESSION['sequential']).lower()})",
    )
    parser.add_argument("--network", default=NETWORK_PATH['network'], metavar="RUN",
                        help=f"connectome backend: 4_built_networks run folder under {BUILT_NETWORKS_DIR} "
                             f"(default: {NETWORK_PATH['network']})")
    parser.add_argument(
        "--multi-bar",
        type=_branch_cli_type(MOVING_BAR_INPUT['multi_bar']),
        default=MOVING_BAR_INPUT['multi_bar'],
        metavar="BOOL",
        help="network moving-bar: tile simultaneous lane-clipped bars "
             f"(default: {_format_branch_value(MOVING_BAR_INPUT['multi_bar'])}); "
             "false → whole-view single bar over the full network view",
    )
    parser.add_argument(
        "--task",
        default=TRAIN_CONFIG['task'],
        help="task name(s): spot|moving_bar (comma-separated), e.g. spot,moving_bar",
    )
    parser.add_argument(
        "--contrast",
        default=",".join(TRAIN_OPTS["contrasts"]),
        help="contrast(s): bright|dark (comma-separated; "
             f"default: {','.join(TRAIN_OPTS['contrasts'])})",
    )
    parser.add_argument(
        "--part-cost-scale",
        default=None,
        nargs="+",
        metavar="NAME|NAME=VALUE",
        help="per-part cost scales (space-separated tokens). NAME=VALUE merges "
             "onto default 1; bare NAME (aliases: spot, moving_bar, "
             "PD/ND/DSI) zeros all parts for --task×--contrast then "
             "sets those to 1. e.g. DSI (=DSI-only), DSI=1 (PD/ND stay 1), "
             "DSI PD=0.2",
    )
    parser.add_argument(
        "--shift-radius",
        type=_branch_cli_type(SPOT_INPUT['shift_radius']),
        default=SPOT_INPUT['shift_radius'],
        help="spot sub-shift hex-disc radius for spot tasks in --task "
             f"(default: {_format_branch_value(SPOT_INPUT['shift_radius'])}; "
             "n_shift=1+3k(k+1); 0->1, 1->7, 2->19, 3->37, ...)",
    )
    parser.add_argument(
        "--spot-radius",
        type=_branch_cli_type(SPOT_INPUT['spot_radius']),
        default=SPOT_INPUT['spot_radius'],
        metavar="R",
        help=f"spot footprint / center-tiling radius (0.5 multiples; default {_format_branch_value(SPOT_INPUT['spot_radius'])}); "
             "radius=1 folds rf(2) into radius=1 gt a_radius and defaults cost scales "
             "to 0=1 1=1/6; radius 1.5/2 keep rf(radius) and 0=1 1=1/6 2=1/6",
    )
    add_multi_spot_arguments(parser)
    parser.add_argument(
        "--spot-cost-radius-scale",
        default=None,
        nargs="+",
        metavar="R|R=S",
        help="spot cost scales by hex-lattice radius from sti hex (space-separated). "
             "Same rules as --part-cost-scale: R=S merges onto radius defaults; bare R "
             "zeros all known radii then sets R=1. Omit → radius default "
             "(1→0=1 1=1/6; else 0=1 1=1/6 2=1/6). Keys: 0,1,2. "
             "Scales only (does not change rf gt)",
    )
    parser.add_argument(
        "--cost-radius",
        default=None,
        nargs="+",
        metavar="N|TRAIN_CONFIG['task']=N",
        help="network cost hex-disc radius (moving-bar default: network radius - 1; "
             "network radius 0/-1 and spot default to all hexes): bare N for all "
             "--task, or per-task space-separated e.g. moving_bar=0 "
             "(tasks: spot, moving_bar); -1 = all hexes; requires --network",
    )
    parser.add_argument(
        "--gt",
        default=None,
        nargs="+",
        metavar="TRAIN_CONFIG['task']=CELLS",
        help="final gt cell keep-set per task (space-separated TRAIN_CONFIG['task']=CELLS; "
             "CELLS comma-separated). Aliases: spot, moving_bar; moving-bar "
             "cell aliases T4, T5. e.g. --gt moving_bar=T4 spot=L1,L2,L3,L4,L5",
    )
    parser.add_argument(
        "--i-sti",
        default=None,
        nargs="+",
        metavar="[TASK=]bright,dark",
        help="sti currents in pA: bright,dark (baseline = midpoint); "
             "optional TASK=spot|moving_bar (default: apply to every --task)",
    )
    add_sti_timing_arguments(parser)
    parser.add_argument(
        "--cost-interval-ms",
        type=float,
        default=TRAIN_OPTIMIZATION['cost_interval_ms'],
        metavar="MS",
        help="spot: default post-onset times 0, interval, 2*interval, ... "
             f"through ms_response (default: {TRAIN_OPTIMIZATION['cost_interval_ms']}); "
             "overwritten per radius by --cost-ms",
    )
    _cost_ms_default = " ".join(
        f"{int(radius)}={','.join(str(x) for x in ms)}"
        for radius, ms in sorted(TRAIN_OPTIMIZATION['cost_ms'].items())
    ) or "none"
    parser.add_argument(
        "--cost-ms",
        default=None,
        nargs="+",
        metavar="R=MS,...",
        help="spot: explicit post-onset ms per hex-lattice radius (space-separated "
             "R=MS,...); overwrites --cost-interval-ms for those radii. "
             f"Omit → {_cost_ms_default}; none|off → all radii use interval",
    )
    parser.add_argument(
        "--cost-norm",
        default=_cli_scalar_from_branch(TRAIN_OPTIMIZATION['cost_norm']),
        choices=list(COST_NORMS),
        help="waveform MSE normalization: gt_power = 100*SSE/Σw(a_gt·gt)²; "
             f"a_gt2 = SSE/a_gt² (default: {TRAIN_OPTIMIZATION['cost_norm']})",
    )


def parse_i_sti(tokens, tasks=()):
    """Decode ``--i-sti`` tokens into ``{task: {bright, dark}}`` (no defaults merge)."""
    if not tokens:
        return None
    out = {}
    for token in tokens:
        if "=" in token:
            name, val = token.split("=", 1)
            task = name.strip()
            if task not in TASKS:
                raise ValueError(
                    f"unknown task {task!r} in --i-sti "
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
    if len(parts) != len(I_STI_KEYS):
        raise ValueError(
            f"--i-sti expects {','.join(I_STI_KEYS)} "
            f"({len(I_STI_KEYS)} values), got {val!r}",
        )
    return {
        i_sti_key: float(part)
        for i_sti_key, part in zip(I_STI_KEYS, parts)
    }


def _parse_filter_branch(branch: str) -> str:
    branch = str(branch).strip().lower()
    if branch in ("v", "none"):
        return "v"
    if branch == "ca":
        return "ca"
    raise ValueError(f"branch must be v or ca, got {branch!r}")


def _format_branch_value(val) -> str:
    if isinstance(val, dict):
        return f"v={val['v']}, ca={val['ca']}"
    return str(val)


def resolve_branch_value(token: str, default=None) -> dict:
    """Parse ``X`` or ``v=X,ca=Y`` into ``{v, ca}`` dict for any value type."""
    if isinstance(default, dict):
        out = dict(default)
    elif default is None:
        out = {"v": None, "ca": None}
    else:
        out = {"v": default, "ca": default}
    raw = str(token).strip()
    if not raw:
        return out
    if "=" not in raw:
        return {"v": raw, "ca": raw}
    for part in parse_comma_list(raw):
        if "=" not in part:
            raise ValueError(f"expected v=X or ca=X, got {part!r}")
        branch, val = part.split("=", 1)
        out[_parse_filter_branch(branch)] = val.strip()
    return out


def _branch_cli_type(default=None):
    def _parse(token: str) -> dict:
        return resolve_branch_value(token, default)
    return _parse


STI_TIMING_KEYS = (
    "ms_pre",
    "ms_response",
    "ms_post",
    "ms_sti",
    "delta_ms",
    "delta_ms_pre",
)

_BRANCH_SYNTAX = re.compile(r"(^|[,\s])(v|ca)=", re.IGNORECASE)


def parse_sti_timing_keys(tokens, *, filter: str) -> dict[str, dict[str, float]]:
    """Parse ``--sti-timing KEY=MS`` tokens; each value updates one filter branch only."""
    branch = "ca" if train.expand_filter(filter) == "ca" else "v"
    out: dict[str, dict[str, float]] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"--sti-timing expected KEY=MS, got {token!r}")
        sti_timing_key, val = token.split("=", 1)
        sti_timing_key = sti_timing_key.strip()
        val = val.strip()
        if sti_timing_key not in STI_TIMING_KEYS:
            allowed = ", ".join(STI_TIMING_KEYS)
            raise ValueError(f"--sti-timing unknown sti_timing_key {sti_timing_key!r}; allowed: {allowed}")
        if _BRANCH_SYNTAX.search(val):
            raise ValueError(
                f"--sti-timing {sti_timing_key}={val!r} must be a plain number, not v=/ca= syntax"
            )
        try:
            num = float(val)
        except ValueError as exc:
            raise ValueError(
                f"--sti-timing {sti_timing_key}={val!r} is not a number"
            ) from exc
        out[sti_timing_key] = {branch: num}
    return out


def _resolve_sti_timing_literals() -> dict:
    from task.spot.sti_spec import _merge_filter_ms

    sti_opts: dict = {}
    for sti_timing_key in STI_TIMING_KEYS:
        if sti_timing_key in ("delta_ms", "delta_ms_pre"):
            ms = NEURON_CONST[sti_timing_key]
        else:
            ms = STI_TIMING.get(sti_timing_key)
        if ms is not None:
            _merge_filter_ms(sti_opts, sti_timing_key, ms)
    return sti_opts


def resolve_train_sti_timing(filter: str, tokens) -> dict:
    """Build full sti timing dict for train (defaults + optional ``--sti-timing``)."""
    from task.spot.sti_spec import _merge_filter_ms

    sti_opts = _resolve_sti_timing_literals()
    if tokens:
        for sti_timing_key, ms in parse_sti_timing_keys(tokens, filter=filter).items():
            _merge_filter_ms(sti_opts, sti_timing_key, ms)
    return sti_opts


def add_sti_timing_arguments(parser):
    """Register ``--sti-timing KEY=MS`` (space-separated, like ``--param``)."""
    parser.add_argument(
        "--sti-timing",
        dest="sti_timing",
        nargs="+",
        default=None,
        metavar="KEY=MS",
        help=(
            "sti length KEY=MS tokens (space-separated). "
            f"Keys: {', '.join(STI_TIMING_KEYS)}. "
            "Plain numbers only; updates the current --filter (v or ca). "
            "Train: omit → const_default; plot/analyze: omit → keep run"
        ),
    )


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
    """Merge timing into train-opts spot/bar sti dicts.

    Spot opts go through :func:`task.spot.sti_spec.override_sti_timing`
    (standardize + drop derived ``t_onset``/``n_t``). Returns timing tokens that
    changed on spot opts (for filename suffixes); bar-only ``ms_pre`` /
    ``delta_ms`` / ``delta_ms_pre`` changes are included when no spot opts
    are present.
    """
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


def resolve_sti_timing_kwargs(args, *, filter=None):
    """Map ``--sti-timing`` to kwargs for :func:`figure.plot.override_session_sti_timing`."""
    tokens = getattr(args, "sti_timing", None)
    empty = {sti_timing_key: None for sti_timing_key in STI_TIMING_KEYS}
    if not tokens:
        return empty
    if filter is None:
        filter = getattr(args, "filter", None)
    if filter is None:
        filter = NEURON_FILTER['filter']
    sti_timing = parse_sti_timing_keys(tokens, filter=filter)
    return {sti_timing_key: sti_timing.get(sti_timing_key) for sti_timing_key in STI_TIMING_KEYS}


def parse_kv_tokens(tokens, cast=str):
    """Parse space-separated ``NAME=VALUE`` tokens (``nargs='+'``)."""
    if not tokens:
        return {}
    out = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"expected NAME=VALUE, got {token!r}")
        name, val = token.split("=", 1)
        out[name.strip()] = cast(val.strip())
    return out


def parse_cost_radius(tokens):
    """Parse ``--cost-radius``: optional bare ``N`` plus ``TRAIN_CONFIG['task']=N`` tokens."""
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
                raise ValueError("only one bare radius allowed in --cost-radius")
            bare_cost_radius = int(token)
    return bare_cost_radius, by_task


def resolve_gt(tokens):
    """Parse ``--gt`` space-separated ``TRAIN_CONFIG['task']=CELLS`` tokens (CELLS comma-separated).

    Values are the final keep-set (not a remove list). Returns ``None`` when
    omitted; otherwise a concrete-task → type-list map.
    """
    if tokens is None:
        return None
    raw = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"expected TRAIN_CONFIG['task']=CELLS, got {token!r}")
        name, val = token.split("=", 1)
        name = name.strip()
        types = parse_comma_list(val)
        if not types:
            raise ValueError(f"--gt {name}=... must list at least one type")
        raw[name] = types
    return train.resolve_gt_cells_by_task(raw)


def resolve_part_cost_scales(tokens, tasks):
    """Parse ``--part-cost-scale``: bare alias exclusive, ``NAME=VALUE`` merge.

    Bare tokens (aliases or concrete part_keys) zero every cost part for
    ``tasks``, then set those names to ``1``. Explicit ``NAME=VALUE``
    always applied last (merge onto defaults / exclusive map). Empty → ``{}``
    (runtime default scale 1).
    """
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


def resolve_train_kwargs(
    args,
    *,
    script_stem="train",
):
    """Parse a train CLI namespace into kwargs for :func:`train.implementation.run_train`."""
    model = args.model
    init_from = args.init_from
    if init_from:
        init_from_path = Path(str(init_from)).expanduser()
        if not init_from_path.is_absolute():
            token = str(init_from).replace("\\", "/")
            parts = token.split("/")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    "--init-from must be MODEL['model']/RUN under 0_runs "
                    f"(models: {train.MODELS}) or an absolute path; "
                    f"got {init_from!r}"
                )
            src_model, run = parts
            if src_model not in train.MODELS:
                raise ValueError(
                    f"--init-from model {src_model!r} not in {train.MODELS}"
                )
            init_from = f"{src_model}/{run}"
    param_init, param_vals, param_modes, param_clamps, param_jits = (
        train.parse_param_cli(args.param) if args.param else ([], [], {}, [], [])
    )
    if param_vals:
        raise ValueError(
            "--param …val… is for plot/analyze only; use …init… for train cold start"
        )
    param_init = param_init or None
    param_modes = param_modes or None
    param_clamps = param_clamps or None
    param_jits = param_jits or None
    syn_mode = getattr(args, "syn_mode", NEURON_SCHEMA['syn_mode'])
    if param_modes:
        if syn_mode == "per_edge" and "syn_strength_cell" in param_modes:
            raise ValueError("--param syn_strength_cell requires --syn-mode per_cell")
        if syn_mode == "per_cell" and "syn_strength_edge" in param_modes:
            raise ValueError("--param syn_strength_edge requires --syn-mode per_edge")
        if "syn_strength_edge" in param_modes:
            train.validate_syn_strength_edge_param_mode(param_modes["syn_strength_edge"])
    filter = train.expand_filter(args.filter)
    spot_gt_mode = train.expand_spot_gt_mode(args.spot_gt_mode)
    val_from = train.resolve_val_from(args.val_from)
    val_from_opts = {"val_from": val_from}
    if filter != "ca":
        for param in ("v_th_ca", "a_ca", "tau_ca"):
            if param_in_modes(param_modes, param):
                raise ValueError(f"--param {param} requires --filter ca")
        if train.val_from_enabled(val_from_opts, "v_th_ca") or train.val_from_enabled(val_from_opts, "a_ca"):
            raise ValueError("--val-from v_th_ca / a_ca require --filter ca")
        if param_modes:
            param_modes = {
                k: v for k, v in param_modes.items()
                if k not in ("v_th_ca", "a_ca", "tau_ca")
            } or None
    tasks = train.parse_tasks(args.task)
    part_cost_scales = resolve_part_cost_scales(args.part_cost_scale, tasks)
    bare_cost_radius, radius_by_task = parse_cost_radius(args.cost_radius)
    cost_radius_by_task = train.cost_radius_by_task(
        tasks, bare_cost_radius, radius_by_task,
    )
    if bare_cost_radius is not None and bare_cost_radius != -1 and bare_cost_radius < 0:
        raise ValueError("--cost-radius must be -1 or >= 0")
    if any(radius != -1 and radius < 0 for radius in radius_by_task.values()):
        raise ValueError("--cost-radius must be -1 or >= 0")
    shift_radius = args.shift_radius
    spot_radius = args.spot_radius
    if args.spot_cost_radius_scale:
        from train.session import resolve_filter_branches
        spot_radius_scalar = resolve_filter_branches(spot_radius, filter="none")
        # spot_radius == 1: fold radius=2 into radius=1 scales
        cost_radius_scales = dict(
            SPOT_PACK['spot_cost_radius_scale_radius1']
            if float(spot_radius_scalar) == 1
            else SPOT_PACK['spot_cost_radius_scale']
        )
        spot_cost_radius_scale = resolve_spot_cost_radius_scale(
            args.spot_cost_radius_scale,
            cost_radius_scales=cost_radius_scales,
            spot_cost_radii=SPOT_PACK['spot_cost_radii'],
        )
    else:
        spot_cost_radius_scale = None
    multi_spot = args.multi_spot
    fully_inside = args.fully_inside
    _timing = resolve_train_sti_timing(filter, args.sti_timing)
    multi_bar = args.multi_bar
    moving_bar_sti_opts = {
        "multi_bar": multi_bar,
        "ms_pre": _timing["ms_pre"],
        "delta_ms": _timing["delta_ms"],
        "delta_ms_pre": _timing["delta_ms_pre"],
    }
    spot_sti_opts = dict(_timing)
    if float(args.cost_interval_ms) <= 0:
        raise ValueError("--cost-interval-ms must be > 0")
    cost_ms_parsed = parse_cost_ms_tokens(args.cost_ms)
    if cost_ms_parsed is None:
        cost_ms_raw = dict(TRAIN_OPTIMIZATION['cost_ms'])
    else:
        cost_ms_raw = cost_ms_parsed
    cost_ms = {
        str(int(k)): [
            x
            if isinstance(x, dict) and set(x) and set(x) <= {"v", "ca"}
            else float(x)
            for x in v
        ]
        for k, v in cost_ms_raw.items()
    }
    gt_by_task = resolve_gt(args.gt)
    if gt_by_task:
        _gt_opts = {
            "moving_bar": moving_bar_sti_opts,
            "spot": spot_sti_opts,
        }
        for task, cells in gt_by_task.items():
            _gt_opts[task]["gt_cells"] = list(cells)
    contrasts = train.parse_contrasts(args.contrast)
    i_sti = parse_i_sti(args.i_sti, tasks=tasks)
    lrs = [float(x) for x in parse_comma_list(args.lrs)]
    if not lrs:
        raise ValueError("--lrs must list at least one learning rate")
    cuda_available = torch.cuda.is_available()
    fp = int(args.fp)
    if not cuda_available and fp == 64:
        fp = 32
    n_iter = args.n_iter
    if n_iter is None:
        n_iter = TRAIN_OPTIMIZATION['n_iter_gpu'] if cuda_available else TRAIN_OPTIMIZATION['n_iter_cpu']
    run = command_run(script_stem)
    from train.implementation import run_dir
    outdir = run_dir(model, parent=args.outdir, run=run)
    return dict(
        model=model,
        n_run=int(args.n_run),
        n_iter=n_iter,
        lrs=lrs,
        fname=args.fname,
        outdir=outdir,
        param_modes=param_modes,
        param_init=param_init,
        param_clamps=param_clamps,
        param_jits=param_jits,
        syn_mode=args.syn_mode,
        network=args.network,
        tasks=tasks,
        contrasts=contrasts,
        part_cost_scales=part_cost_scales,
        cost_norm=expand_cost_norm(args.cost_norm),
        cost_interval_ms=float(args.cost_interval_ms),
        cost_ms=cost_ms,
        cost_radius_by_task=cost_radius_by_task,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_scale=spot_cost_radius_scale,
        moving_bar_sti_opts=moving_bar_sti_opts,
        spot_sti_opts=spot_sti_opts,
        i_sti=i_sti,
        euler=args.euler,
        pre_steady=expand_pre_steady(args.pre_steady),
        pre_steady_n_iter=TRAIN_OPTIMIZATION['pre_steady_n_iter'],
        pre_steady_damp=TRAIN_OPTIMIZATION['pre_steady_damp'],
        fp=fp,
        pre_grad=bool(args.pre_grad),
        val_from=val_from,
        filter=filter,
        spot_gt_mode=spot_gt_mode,
        sequential=bool(args.sequential),
        init_from=init_from,
        checkpoint_interval=args.checkpoint_interval,
    )
