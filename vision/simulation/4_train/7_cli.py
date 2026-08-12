# -*- coding: utf-8 -*-
"""Train CLI: argparse registration and argv → kwargs for implementation.

Does not run optimization or touch CUDA. Callers: ``simulation/run.py``,
``figure``, ``analyze``.
"""
from __future__ import annotations

from default_params import (
    MODEL,
    MOVING_BAR_INPUT,
    NETWORK_PATH,
    NEURON_FORWARD,
    NEURON_PARAM,
    NEURON_SCHEMA,
    SPOT_INPUT,
    SPOT_PACK,
    TRAIN_CONFIG,
    TRAIN_OPTIMIZATION,
    TRAIN_SESSION,
)

import argparse
import os
import re
import sys
import time
from pathlib import Path


def _cli_default(val):
    """Return scalar CLI default from a value that may be a ``{v, ca}`` branch dict."""
    if isinstance(val, dict) and set(val) <= {"v", "ca"}:
        return val.get("v", next(iter(val.values())))
    return val

import torch

_SIMULATION_CODE = Path(__file__).resolve().parent.parent
if str(_SIMULATION_CODE) not in sys.path:
    sys.path.insert(0, str(_SIMULATION_CODE))

import import_bootstrap  # noqa: F401
from import_bootstrap import normalize_option_dashes, parse_bool, parse_comma_list
import network.path  # noqa: F401 — FAFB path on sys.path
from path import BUILT_NETWORKS_DIR
from task.spot.pack import (
    default_spot_cost_radius_scale,
    parse_cost_ms_tokens,
    parse_spot_cost_r_s_tokens,
)
from neuron.schema import spot_radius_key
import train
from train.config import (
    COST_NORMS,
    SPOT_GT_MODES,
    expand_cost_norm,
    expand_pre_steady,
)

RUN_NAME_MAX = 255


def _slug(text):
    """Filesystem-safe token for a CLI flag value."""
    return re.sub(r'[^\w.,-]+', '-', str(text)).strip('-')


def _argv_cli_tokens(argv):
    """Drop the script path; yield long-option tokens from *argv*."""
    if argv and argv[0].endswith('.py'):
        argv = argv[1:]
    argv = normalize_option_dashes(argv)
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ('-h', '--help'):
            i += 1
            continue
        if not tok.startswith('--'):
            i += 1
            continue
        key, sep, val = tok[2:].partition('=')
        if sep:
            yield _slug(key), _slug(val)
            i += 1
        elif i + 1 < len(argv) and not argv[i + 1].startswith('-'):
            yield _slug(key), _slug(argv[i + 1])
            i += 2
        else:
            yield _slug(key), None
            i += 1


def command_run_name(script_stem, argv=None):
    """Build a run folder name from flags on the command line (``sys.argv``)."""
    if argv is None:
        argv = sys.argv[1:]
    prefix = os.environ.get('SLURM_JOB_ID') or time.strftime('%m%d_%H%M%S')
    parts = [prefix, script_stem]
    for key, val in _argv_cli_tokens(argv):
        parts.append(key)
        if val is not None:
            parts.append(val)
    name = '-'.join(parts)
    if len(name) <= RUN_NAME_MAX:
        return name
    return name[:RUN_NAME_MAX].rstrip('-')


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


def add_train_arguments(parser):
    """Register train CLI flags on *parser*.

    Concrete omitted-flag values live in :mod:`default_params` and are
    wired here as ``default=CONST``. ``None`` only for omit-disabled flags.
    """
    parser.add_argument("--model", default=MODEL['model'], choices=list(train.KNOWN_MODELS))
    parser.add_argument(
        "--syn-mode",
        default=_cli_default(NEURON_SCHEMA['syn_mode']),
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
                             "and best_adam.npz as Adam m/v "
                             "(settings come from this CLI, not train_opts.json)")
    _indi_named_default = (
        "indi=" + ",".join(NEURON_SCHEMA['h_cells']) + " fixed=all"
    )

    def _box_train_mode_default(name):
        tm = NEURON_SCHEMA['param_boxes'][name]["train_mode"]
        if tm == "indi_named":
            return _indi_named_default
        return f"{tm}=all"

    _train_mode_help = (
        "indi=/shared=/fixed=/frozen= lists space-separated; 'all' in one train_mode = remainder; "
        "types or Src:Tar pairs (syn-strength-cell); init.NAMES=VAL / all=VAL overrides initial values. "
        "Example: indi=all init.L1,L2,L4,L5=200 all=10000"
    )
    _syn_strength_edge_help = (
        "only indi=all / fixed=all / frozen=all "
        "(--syn-mode per_edge; no shared= / named edges)"
    )
    _train_mode_kwargs = dict(default=None, nargs='+', metavar="MODE")
    parser.add_argument("--all-param", **_train_mode_kwargs,
                        help=f"apply train_modes to every parameter segment "
                             f"({_train_mode_help}; overridden by --i-h-shape and per-param flags)")
    parser.add_argument("--a-in", **_train_mode_kwargs,
                        help=f"a_in train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('a_in')})")
    parser.add_argument("--a-out", **_train_mode_kwargs,
                        help=f"a_out train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('a_out')})")
    parser.add_argument("--a-gt", **_train_mode_kwargs,
                        help=f"a_gt train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('a_gt')})")
    parser.add_argument("--bias-gt", **_train_mode_kwargs,
                        help=f"bias_gt train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('bias_gt')})")
    parser.add_argument("--syn-strength-cell", **_train_mode_kwargs,
                        help=f"syn_strength_cell train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('syn_strength_cell')}; "
                             f"--syn-mode per_cell only)")
    parser.add_argument("--syn-strength-edge", **_train_mode_kwargs,
                        help=f"syn_strength_edge train_modes ({_syn_strength_edge_help}; "
                             f"default {_box_train_mode_default('syn_strength_edge')})")
    parser.add_argument("--v-th", **_train_mode_kwargs,
                        help=f"v_th train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('v_th')})")
    parser.add_argument("--v-th-ca", **_train_mode_kwargs,
                        help=f"v_th_ca train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('v_th_ca')})")
    parser.add_argument("--a-ca", **_train_mode_kwargs,
                        help=f"a_ca train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('a_ca')})")
    parser.add_argument("--a-h", **_train_mode_kwargs,
                        help=f"a_h train_modes (borst i_h a_h / hp_lp HP mix; {_train_mode_help}; "
                             f"default {_box_train_mode_default('a_h')})")
    parser.add_argument("--a-h-rev", **_train_mode_kwargs,
                        help=f"a_h_rev train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('a_h_rev')})")
    parser.add_argument("--i-h-shape", **_train_mode_kwargs,
                        help="batch train_modes for v_mid_h_g/v_mid_h_tau/h_slope and rev "
                             f"({_train_mode_help}; default {_box_train_mode_default('v_mid_h_g')})")
    parser.add_argument("--v-mid-h-g", **_train_mode_kwargs,
                        help=f"v_mid_h_g train_modes (overrides --i-h-shape; {_train_mode_help})")
    parser.add_argument("--h-slope", **_train_mode_kwargs,
                        help=f"h_slope train_modes (overrides --i-h-shape; {_train_mode_help})")
    parser.add_argument("--v-mid-h-tau", **_train_mode_kwargs,
                        help=f"v_mid_h_tau train_modes (overrides --i-h-shape; {_train_mode_help})")
    parser.add_argument("--v-mid-h-g-rev", **_train_mode_kwargs,
                        help=f"v_mid_h_g_rev train_modes (overrides --i-h-shape; {_train_mode_help})")
    parser.add_argument("--h-slope-rev", **_train_mode_kwargs,
                        help=f"h_slope_rev train_modes (overrides --i-h-shape; {_train_mode_help})")
    parser.add_argument("--v-mid-h-tau-rev", **_train_mode_kwargs,
                        help=f"v_mid_h_tau_rev train_modes (overrides --i-h-shape; {_train_mode_help})")
    parser.add_argument("--tau-lp", **_train_mode_kwargs,
                        help=f"hp_lp tau_lp train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('tau_lp')})")
    parser.add_argument("--tau-ca", **_train_mode_kwargs,
                        help=f"tau_ca train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('tau_ca')})")
    parser.add_argument("--e-leak", **_train_mode_kwargs,
                        help=f"e_leak train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('e_leak')})")
    parser.add_argument("--tau-hp", **_train_mode_kwargs,
                        help=f"hp_lp tau_hp train_modes ({_train_mode_help}; "
                             f"default {_box_train_mode_default('tau_hp')})")
    parser.add_argument("--a-sti-radius", **_train_mode_kwargs,
                        help=(
                            "spot a_sti_radius train_modes "
                            f"(slots {','.join(spot_radius_key(r, aliases=SPOT_PACK['spot_cost_radius_key_aliases']) for r in SPOT_PACK['a_sti_radii'])}; "
                            "cost-radius scale==0 forces slot to 0 in forward; "
                            f"default {_box_train_mode_default('a_sti_radius')}; "
                            f"{_train_mode_help})"
                        ))
    parser.add_argument("--i-h-rev", default=_cli_default(NEURON_PARAM['i_h_rev']),
                        choices=list(train.I_H_REV_MODES),
                        help="rev-channel i_h: on (train a_h_rev+rev shape), "
                             "mirrored (rev copies forward), off (disable rev; default)")
    parser.add_argument(
        "--euler",
        default=_cli_default(NEURON_PARAM['euler']),
        choices=list(train.EULER_CLI),
        help="membrane Euler: im=implicit (default), ex=explicit; "
             "i_h gates always explicit",
    )
    parser.add_argument(
        "--pre-steady",
        default=_cli_default(TRAIN_OPTIMIZATION['pre_steady']),
        choices=("probe", "solve"),
        help=(
            "t=0 membrane pre steady shared by borst/hp_lp "
            f"(probe|solve; default: {TRAIN_OPTIMIZATION['pre_steady']}); "
            "solve uses fixed iters/damp from default_params"
        ),
    )
    parser.add_argument(
        "--fp",
        type=int,
        default=TRAIN_SESSION['fp'],
        choices=(16, 32, 64),
        metavar="N",
        help=f"simulation float width (default: {TRAIN_SESSION['fp']}); "
             "64 is forced to 32 when CUDA is unavailable",
    )
    parser.add_argument(
        "--pre-grad",
        type=parse_bool,
        default=_cli_default(NEURON_FORWARD['pre_grad']),
        metavar="BOOL",
        help="include t < t_onset in BPTT "
             f"(default: {str(NEURON_FORWARD['pre_grad']).lower()}); "
             "false → no_grad pre + detach state/v at onset",
    )
    parser.add_argument(
        "--bias-gt-from-v-onset",
        type=parse_bool,
        default=_cli_default(TRAIN_OPTIMIZATION['bias_gt_from_v_onset']),
        metavar="BOOL",
        help="write v at t_onset into bias_gt (cost/plot/CSV) "
             f"(default: {str(TRAIN_OPTIMIZATION['bias_gt_from_v_onset']).lower()}); "
             "forces bias_gt frozen=all",
    )
    parser.add_argument(
        "--bias-gt-from-v-onset-grad",
        type=parse_bool,
        default=_cli_default(TRAIN_OPTIMIZATION['bias_gt_from_v_onset_grad']),
        metavar="BOOL",
        help="with --bias-gt-from-v-onset: keep onset in the graph "
             f"(default: {str(TRAIN_OPTIMIZATION['bias_gt_from_v_onset_grad']).lower()}); "
             "ignored when --bias-gt-from-v-onset false",
    )
    parser.add_argument(
        "--v-th-ca-from-v-th",
        type=parse_bool,
        default=_cli_default(TRAIN_OPTIMIZATION['v_th_ca_from_v_th']),
        metavar="BOOL",
        help="write v_th into v_th_ca (forward/CSV) "
             f"(default: {str(TRAIN_OPTIMIZATION['v_th_ca_from_v_th']).lower()}); "
             "forces v_th_ca frozen=all",
    )
    parser.add_argument(
        "--a-ca-from-a-out",
        type=parse_bool,
        default=_cli_default(TRAIN_OPTIMIZATION['a_ca_from_a_out']),
        metavar="BOOL",
        help="write a_out into a_ca (forward/CSV) "
             f"(default: {str(TRAIN_OPTIMIZATION['a_ca_from_a_out']).lower()}); "
             "forces a_ca frozen=all",
    )
    parser.add_argument(
        "--filter",
        default=SPOT_PACK['filter'],
        choices=("none", "ca"),
        help="readout filter: none=v (schema skips v_th_ca/a_ca/tau_ca), "
             "ca=ca + Arenz digitized spot gt "
             f"(default: {SPOT_PACK['filter']}; ms_spot/ms_response select v or ca branch)",
    )
    parser.add_argument(
        "--spot-gt-mode",
        default=_cli_default(SPOT_PACK['spot_gt_mode']),
        choices=list(SPOT_GT_MODES),
        help="spot cost GT mode: all=every cell both contrasts, "
             "positive=only rf_sign×contrast_sign>0 "
             f"(bright ON / dark OFF; default: {SPOT_PACK['spot_gt_mode']})",
    )
    parser.add_argument(
        "--sequential",
        type=parse_bool,
        default=_cli_default(TRAIN_SESSION['sequential']),
        metavar="BOOL",
        help=f"one sti batch per forward (default: {str(TRAIN_SESSION['sequential']).lower()})",
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
             "false → whole-field single bar over the full network field",
    )
    parser.add_argument(
        "--task",
        default=TRAIN_CONFIG['task'],
        help="task name(s): spot (=spot_bright+spot_dark), moving_bar (=bright+dark), "
             "or explicit names / comma-separated list, e.g. spot,moving_bar",
    )
    parser.add_argument(
        "--part-cost-scale",
        default=None,
        nargs="+",
        metavar="NAME|NAME=VALUE",
        help="per-part cost scales (space-separated tokens). NAME=VALUE merges "
             "onto default 1; bare NAME (aliases: spot, moving_bar, "
             "moving_bar_bright/dark, PD/ND/DSI) zeros all parts for --task then "
             "sets those to 1. e.g. DSI (=DSI-only), DSI=1 (PD/ND stay 1), "
             "DSI PD=0.2",
    )
    parser.add_argument(
        "--shift-radius",
        type=_branch_cli_type(SPOT_INPUT['shift_radius']),
        default=SPOT_INPUT['shift_radius'],
        help="spot sub-shift hex-disc radius for spot tasks in --task "
             f"(default: {_format_branch_value(SPOT_INPUT['shift_radius'])}; "
             "n_shifts=1+3k(k+1); 0->1, 1->7, 2->19, 3->37, ...)",
    )
    parser.add_argument(
        "--spot-radius",
        type=_branch_cli_type(SPOT_INPUT['spot_radius']),
        default=SPOT_INPUT['spot_radius'],
        metavar="R",
        help=f"spot footprint / center-tiling radius (0.5 multiples; default {_format_branch_value(SPOT_INPUT['spot_radius'])}); "
             "radius=1 folds RecF(2) into r=1 gt a_radius and defaults cost scales "
             "to 0=1 1=1/6; radius 1.5/2 keep RecF(r) and 0=1 1=1/6 2=1/6",
    )
    add_multi_spot_arguments(parser)
    parser.add_argument(
        "--spot-cost-r-s",
        default=None,
        nargs="+",
        metavar="R|R=S",
        help="spot cost scales by Euclidean r from sti hex (space-separated). "
             "Same rules as --part-cost-scale: R=S merges onto radius defaults; bare R "
             "zeros all known radii then sets R=1. Omit → radius default "
             "(1→0=1 1=1/6; else 0=1 1=1/6 2=1/6). Keys: 0,1,2,sqrt3. "
             "Scales only (does not change RecF gt)",
    )
    parser.add_argument(
        "--cost-radius",
        default=None,
        nargs="+",
        metavar="N|TRAIN_CONFIG['task']=N",
        help="network cost hex-disc radius (moving-bar default: network radius - 1; "
             "network radius 0/-1 and spot default to all hexes): bare N for all "
             "--task, or per-task space-separated e.g. moving_bar_bright=0 "
             "(aliases: spot, moving_bar); -1 = all hexes; requires --network",
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
        "--i-baseline",
        default=None,
        nargs="+",
        metavar="TRAIN_CONFIG['task']=VALUE",
        help="per-task sti baseline (pA; space-separated TRAIN_CONFIG['task']=VALUE); "
             "aliases: spot, moving_bar",
    )
    parser.add_argument(
        "--i-bright",
        default=None,
        nargs="+",
        metavar="TRAIN_CONFIG['task']=VALUE",
        help="bright peak/step current (pA; space-separated TRAIN_CONFIG['task']=VALUE); "
             "tasks: spot_bright, moving_bar_bright (aliases spot, moving_bar)",
    )
    parser.add_argument(
        "--i-dark",
        default=None,
        nargs="+",
        metavar="TRAIN_CONFIG['task']=VALUE",
        help="dark peak/step current (pA; space-separated TRAIN_CONFIG['task']=VALUE); "
             "tasks: spot_dark, moving_bar_dark (aliases spot, moving_bar)",
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
        f"{spot_radius_key(r, aliases=SPOT_PACK['spot_cost_radius_key_aliases'])}="
        f"{','.join(str(x) for x in ms)}"
        for r, ms in sorted(TRAIN_OPTIMIZATION['cost_ms'].items())
    ) or "none"
    parser.add_argument(
        "--cost-ms",
        default=None,
        nargs="+",
        metavar="R=MS,...",
        help="spot: explicit post-onset ms per Euclidean r (space-separated "
             "R=MS,...); overwrites --cost-interval-ms for those radii. "
             f"Omit → {_cost_ms_default}; none|off → all radii use interval",
    )
    parser.add_argument(
        "--cost-norm",
        default=_cli_default(TRAIN_OPTIMIZATION['cost_norm']),
        choices=list(COST_NORMS),
        help="waveform MSE normalization: gt_power = 100*SSE/Σw(a_gt·gt)²; "
             f"a_gt2 = SSE/a_gt² (default: {TRAIN_OPTIMIZATION['cost_norm']})",
    )


def _parse_branch_key(key: str) -> str:
    branch = str(key).strip().lower()
    if branch in ("v", "none"):
        return "v"
    if branch == "ca":
        return "ca"
    raise ValueError(f"branch must be v or ca, got {key!r}")


def _format_branch_value(val) -> str:
    if isinstance(val, dict):
        return f"v={val['v']}, ca={val['ca']}"
    return str(val)


def parse_branch_value(text: str, default=None) -> dict:
    """Parse ``X`` or ``v=X,ca=Y`` into ``{v, ca}`` dict for any value type."""
    if isinstance(default, dict):
        out = dict(default)
    elif default is None:
        out = {"v": None, "ca": None}
    else:
        out = {"v": default, "ca": default}
    raw = str(text).strip()
    if not raw:
        return out
    if "=" not in raw:
        return {"v": raw, "ca": raw}
    for part in parse_comma_list(raw):
        if "=" not in part:
            raise ValueError(f"expected v=X or ca=X, got {part!r}")
        key, val = part.split("=", 1)
        out[_parse_branch_key(key)] = val.strip()
    return out


def _branch_cli_type(default=None):
    def _parse(text: str) -> dict:
        return parse_branch_value(text, default)
    return _parse


def add_sti_timing_arguments(
    parser,
    *,
    default_ms_pre=SPOT_INPUT['ms_pre'],
    default_ms_response=SPOT_INPUT['ms_response'],
    default_ms_post=SPOT_INPUT['ms_post'],
    default_ms_spot=SPOT_INPUT['ms_spot'],
    default_delta_ms=NEURON_PARAM['delta_ms'],
    default_delta_ms_pre=NEURON_PARAM['delta_ms_pre'],
):
    """Register ``--ms-pre`` / ``--ms-response`` / ``--ms-post`` / ``--ms-spot`` /
    ``--delta-ms`` / ``--delta-ms-pre``.

    Train uses ``default_params`` values. Plot / analyze pass ``None`` so
    omitted flags keep the run's ``train_opts.json``.
    """
    if default_ms_pre is None:
        pre_help = (
            "override pre-sti baseline in ms (spot + moving_bar; "
            "keep train if omitted)"
        )
    else:
        pre_help = (
            f"pre-sti baseline duration in ms (default: {default_ms_pre}; "
            "t_onset = t_from_ms(ms_pre, delta_ms=delta_ms_pre); "
            "n_t = t_onset+t_from_ms(ms_response)+t_from_ms(ms_post)+1)"
        )
    if default_ms_response is None:
        response_help = (
            "override spot post-onset ms_response cost/gt "
            "(v=MS,ca=MS or bare MS sets both; keep train if omitted)"
        )
    else:
        response_help = (
            "spot: post-onset ms_response cost/gt "
            f"(default: {_format_branch_value(default_ms_response)}; "
            "v=MS,ca=MS or bare MS sets both; excludes ms_post)"
        )
    if default_ms_post is None:
        post_help = (
            "override spot forward-only tail after response in ms "
            "(not in gt/cost; keep train if omitted)"
        )
    else:
        post_help = (
            f"spot: forward-only tail after response in ms (default: {_format_branch_value(default_ms_post)}; "
            "not in gt/cost)"
        )
    if default_ms_spot is None:
        spot_help = (
            "override spot-on width "
            "(v=MS,ca=MS or bare MS sets both; keep train if omitted; "
            "raises ms_response if shorter)"
        )
    else:
        spot_help = (
            "spot: bright/dark sti on-duration from onset "
            f"(default: {_format_branch_value(default_ms_spot)}; "
            "v=MS,ca=MS or bare MS sets both; raises ms_response if shorter)"
        )
    if default_delta_ms is None:
        delta_help = (
            "override post-onset simulation / sti time step "
            "(v=MS,ca=MS or bare MS sets both; writes delta_ms into all sti opts; "
            "keep train if omitted)"
        )
    else:
        delta_help = (
            f"post-onset simulation / sti time step "
            f"(default: {_format_branch_value(default_delta_ms)}; "
            "v=MS,ca=MS or bare MS sets both; writes delta_ms into all sti opts)"
        )
    if default_delta_ms_pre is None:
        delta_pre_help = (
            "override pre-onset simulation time step in ms "
            "(writes delta_ms_pre into all sti opts; keep train if omitted)"
        )
    else:
        delta_pre_help = (
            f"pre-onset simulation time step in ms "
            f"(default: {_format_branch_value(default_delta_ms_pre)}; "
            "v=MS,ca=MS or bare MS sets both; writes delta_ms_pre into all "
            "sti opts; t_onset = t_from_ms(ms_pre, delta_ms=delta_ms_pre))"
        )
    parser.add_argument(
        "--ms-pre",
        type=_branch_cli_type(default_ms_pre),
        default=default_ms_pre,
        metavar="MS",
        help=pre_help,
    )
    parser.add_argument(
        "--ms-response",
        type=_branch_cli_type(
            SPOT_INPUT['ms_response'] if default_ms_response is None else default_ms_response
        ),
        default=default_ms_response,
        metavar="MS",
        help=response_help,
    )
    parser.add_argument(
        "--ms-post",
        type=_branch_cli_type(default_ms_post),
        default=default_ms_post,
        metavar="MS",
        help=post_help,
    )
    parser.add_argument(
        "--ms-spot",
        type=_branch_cli_type(
            SPOT_INPUT['ms_spot'] if default_ms_spot is None else default_ms_spot
        ),
        default=default_ms_spot,
        metavar="MS",
        help=spot_help,
    )
    parser.add_argument(
        "--delta-ms",
        type=_branch_cli_type(
            NEURON_PARAM['delta_ms'] if default_delta_ms is None else default_delta_ms
        ),
        default=default_delta_ms,
        metavar="MS",
        help=delta_help,
    )
    parser.add_argument(
        "--delta-ms-pre",
        type=_branch_cli_type(
            NEURON_PARAM['delta_ms_pre']
            if default_delta_ms_pre is None else default_delta_ms_pre
        ),
        default=default_delta_ms_pre,
        metavar="MS",
        help=delta_pre_help,
    )


def apply_train_opts_timing(
    opts,
    *,
    ms_pre=None,
    ms_response=None,
    ms_post=None,
    ms_spot=None,
    delta_ms=None,
    delta_ms_pre=None,
):
    """Merge timing overrides into train-opts spot/bar sti dicts.

    Spot opts go through :func:`task.spot.input.apply_spot_timing_overrides`
    (normalize + drop derived ``t_onset``/``n_t``). Returns timing keys that
    changed on spot opts (for filename suffixes); bar-only ``ms_pre`` /
    ``delta_ms`` / ``delta_ms_pre`` changes are included when no spot opts
    are present.
    """
    from task.spot.input import apply_spot_timing_overrides

    def _timing_equal(a, b) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if isinstance(a, dict) and isinstance(b, dict):
            keys = set(a) | set(b)
            return all(_timing_equal(a.get(k), b.get(k)) for k in keys)
        if isinstance(a, dict) != isinstance(b, dict):
            return False
        return float(a) == float(b)

    changed = {}
    for key in ("spot_bright_sti_opts", "spot_dark_sti_opts"):
        so = opts.get(key)
        if so is None:
            continue
        changed = apply_spot_timing_overrides(
            so,
            ms_pre=ms_pre,
            ms_response=ms_response,
            ms_post=ms_post,
            ms_spot=ms_spot,
            delta_ms=delta_ms,
            delta_ms_pre=delta_ms_pre,
        )
    if ms_pre is not None or delta_ms is not None or delta_ms_pre is not None:
        for key in (
            "moving_bar_bright_sti_opts",
            "moving_bar_dark_sti_opts",
        ):
            so = opts.get(key)
            if so is None:
                continue
            before_pre = so.get("ms_pre")
            before_dt = so.get("delta_ms")
            before_dt_pre = so.get("delta_ms_pre")
            if ms_pre is not None:
                so["ms_pre"] = ms_pre
            if delta_ms is not None:
                so["delta_ms"] = delta_ms
            if delta_ms_pre is not None:
                so["delta_ms_pre"] = delta_ms_pre
            so.pop("t_onset", None)
            so.pop("n_t", None)
            if not changed:
                if ms_pre is not None and (
                    before_pre is None or not _timing_equal(before_pre, so.get("ms_pre"))
                ):
                    changed["ms_pre"] = so.get("ms_pre")
                if delta_ms is not None and before_dt != so.get("delta_ms"):
                    changed["delta_ms"] = so.get("delta_ms")
                if delta_ms_pre is not None and before_dt_pre != so.get("delta_ms_pre"):
                    changed["delta_ms_pre"] = so.get("delta_ms_pre")
    return changed


def sti_timing_kwargs_from_args(args):
    """Map parsed timing flags to kwargs for :func:`figure.plot.maybe_override_sti_timing`."""
    return dict(
        ms_pre=args.ms_pre,
        ms_response=args.ms_response,
        ms_post=args.ms_post,
        ms_spot=args.ms_spot,
        delta_ms=args.delta_ms,
        delta_ms_pre=args.delta_ms_pre,
    )


def parse_kv_tokens(tokens, cast=str):
    """Parse space-separated ``NAME=VALUE`` tokens (``nargs='+'``)."""
    if not tokens:
        return {}
    out = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"expected NAME=VALUE, got {tok!r}")
        name, val = tok.split("=", 1)
        out[name.strip()] = cast(val.strip())
    return out


def parse_tasks(text):
    """Parse comma-separated train tasks (with alias expansion)."""
    return train.normalize_tasks(parse_comma_list(text))


def parse_cost_radius(tokens):
    """Parse ``--cost-radius``: optional bare ``N`` plus ``TRAIN_CONFIG['task']=N`` tokens."""
    if not tokens:
        return None, {}
    default = None
    by_task = {}
    for tok in tokens:
        if "=" in tok:
            name, val = tok.split("=", 1)
            by_task[name.strip()] = int(val.strip())
        else:
            if default is not None:
                raise ValueError("only one bare radius allowed in --cost-radius")
            default = int(tok)
    return default, by_task


def parse_gt(tokens):
    """Parse ``--gt`` space-separated ``TRAIN_CONFIG['task']=CELLS`` tokens (CELLS comma-separated).

    Values are the final keep-set (not a remove list). Returns ``None`` when
    omitted; otherwise a concrete-task → type-list map.
    """
    if tokens is None:
        return None
    raw = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"expected TRAIN_CONFIG['task']=CELLS, got {tok!r}")
        name, val = tok.split("=", 1)
        name = name.strip()
        types = parse_comma_list(val)
        if not types:
            raise ValueError(f"--gt {name}=... must list at least one type")
        raw[name] = types
    return train.resolve_gt_cells_by_task(raw)


def parse_part_cost_scale(tokens, tasks):
    """Parse ``--part-cost-scale``: bare alias exclusive, ``NAME=VALUE`` merge.

    Bare tokens (aliases or concrete part keys) zero every cost part for
    ``tasks``, then set those names to ``1``. Explicit ``NAME=VALUE``
    always applied last (merge onto defaults / exclusive map). Empty → ``{}``
    (runtime default scale 1).
    """
    if not tokens:
        return {}
    bare: list[str] = []
    explicit: dict[str, float] = {}
    for tok in tokens:
        if "=" in tok:
            name, val = tok.split("=", 1)
            explicit[name.strip()] = float(val.strip())
        else:
            bare.append(tok.strip())
    scales: dict[str, float] = {}
    if bare:
        scales = {key: 0.0 for key in train.session_cost_part_keys(tasks)}
        scales.update(train.expand_part_cost_scale_dict({name: 1.0 for name in bare}))
    scales.update(train.expand_part_cost_scale_dict(explicit))
    return scales


def _train_mode_cli_text(parts):
    """Join space-separated train_mode CLI tokens into one parse string."""
    if parts is None:
        return None
    return ' '.join(parts)


def _train_mode_cli_map(args):
    """Build ``{seg_name: {indi/shared/fixed/frozen: tokens}}`` from train_mode CLI flags.

    Precedence: ``--all-param`` → ``--i-h-shape`` → per-param flags.
    Omitted segments keep schema defaults (not listed here).
    """
    syn_mode = train.normalize_syn_mode(getattr(args, "syn_mode", NEURON_SCHEMA['syn_mode']))
    syn_cell_text = _train_mode_cli_text(getattr(args, "syn_strength_cell", None))
    syn_edge_text = _train_mode_cli_text(getattr(args, "syn_strength_edge", None))
    if syn_mode == "per_edge" and syn_cell_text is not None:
        raise ValueError("--syn-strength-cell requires --syn-mode per_cell")
    if syn_mode == "per_cell" and syn_edge_text is not None:
        raise ValueError("--syn-strength-edge requires --syn-mode per_edge")
    texts = {}
    all_param = _train_mode_cli_text(getattr(args, "all_param", None))
    if all_param is not None:
        for name in train.PARAM_NAMES:
            if name == "syn_strength_cell" and syn_mode != "per_cell":
                continue
            if name == "syn_strength_edge" and syn_mode != "per_edge":
                continue
            texts[name] = all_param
    shape_text = _train_mode_cli_text(getattr(args, "i_h_shape", None))
    if shape_text is not None:
        for name in train.I_H_SHAPE_PARAM_NAMES:
            texts[name] = shape_text
    # Keys follow NEURON_SCHEMA['param_boxes'] / PARAM_NAMES order.
    per_param = {
        "a_gt": _train_mode_cli_text(getattr(args, "a_gt", None)),
        "bias_gt": _train_mode_cli_text(getattr(args, "bias_gt", None)),
        "syn_strength_cell": syn_cell_text,
        "syn_strength_edge": syn_edge_text,
        "a_in": _train_mode_cli_text(getattr(args, "a_in", None)),
        "a_out": _train_mode_cli_text(getattr(args, "a_out", None)),
        "e_leak": _train_mode_cli_text(getattr(args, "e_leak", None)),
        "v_th": _train_mode_cli_text(getattr(args, "v_th", None)),
        "v_th_ca": _train_mode_cli_text(getattr(args, "v_th_ca", None)),
        "a_ca": _train_mode_cli_text(getattr(args, "a_ca", None)),
        "tau_ca": _train_mode_cli_text(getattr(args, "tau_ca", None)),
        "tau_lp": _train_mode_cli_text(getattr(args, "tau_lp", None)),
        "tau_hp": _train_mode_cli_text(getattr(args, "tau_hp", None)),
        "a_h": _train_mode_cli_text(getattr(args, "a_h", None)),
        "v_mid_h_g": _train_mode_cli_text(getattr(args, "v_mid_h_g", None)),
        "v_mid_h_tau": _train_mode_cli_text(getattr(args, "v_mid_h_tau", None)),
        "h_slope": _train_mode_cli_text(getattr(args, "h_slope", None)),
        "a_h_rev": _train_mode_cli_text(getattr(args, "a_h_rev", None)),
        "v_mid_h_g_rev": _train_mode_cli_text(getattr(args, "v_mid_h_g_rev", None)),
        "v_mid_h_tau_rev": _train_mode_cli_text(getattr(args, "v_mid_h_tau_rev", None)),
        "h_slope_rev": _train_mode_cli_text(getattr(args, "h_slope_rev", None)),
        "a_sti_radius": _train_mode_cli_text(getattr(args, "a_sti_radius", None)),
    }
    for name, text in per_param.items():
        if text is not None:
            texts[name] = text
    out = {name: train.parse_train_mode_text(text) for name, text in texts.items()}
    if "syn_strength_edge" in out:
        train.validate_syn_strength_edge_train_mode(out["syn_strength_edge"])
    return out


def train_kwargs_from_args(
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
            text = str(init_from).replace("\\", "/")
            parts = text.split("/")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    "--init-from must be MODEL['model']/RUN under 0_runs "
                    f"(models: {train.KNOWN_MODELS}) or an absolute path; "
                    f"got {init_from!r}"
                )
            src_model, run_name = parts
            if src_model not in train.KNOWN_MODELS:
                raise ValueError(
                    f"--init-from model {src_model!r} not in {train.KNOWN_MODELS}"
                )
            init_from = f"{src_model}/{run_name}"
    train_modes = _train_mode_cli_map(args) or None
    filter = train.expand_filter(args.filter)
    spot_gt_mode = train.expand_spot_gt_mode(args.spot_gt_mode)
    if filter != "ca":
        for flag, attr in (
            ("--v-th-ca", "v_th_ca"),
            ("--a-ca", "a_ca"),
            ("--tau-ca", "tau_ca"),
        ):
            if _train_mode_cli_text(getattr(args, attr, None)) is not None:
                raise ValueError(f"{flag} requires --filter ca")
        if bool(args.v_th_ca_from_v_th) or bool(args.a_ca_from_a_out):
            raise ValueError(
                "--v-th-ca-from-v-th / --a-ca-from-a-out require --filter ca"
            )
        if train_modes:
            train_modes = {
                k: v for k, v in train_modes.items()
                if k not in ("v_th_ca", "a_ca", "tau_ca")
            } or None
    bias_gt_from_v_onset = bool(args.bias_gt_from_v_onset)
    bias_gt_from_v_onset_grad = bool(args.bias_gt_from_v_onset_grad)
    if not bias_gt_from_v_onset:
        bias_gt_from_v_onset_grad = False
    if bias_gt_from_v_onset:
        if _train_mode_cli_text(getattr(args, "bias_gt", None)) is not None:
            raise ValueError(
                "--bias-gt conflicts with --bias-gt-from-v-onset "
                "(bias_gt is forced frozen=all)"
            )
        train_modes = dict(train_modes or {})
        train_modes["bias_gt"] = train.parse_train_mode_text("frozen=all")
    v_th_ca_from_v_th = bool(args.v_th_ca_from_v_th)
    if v_th_ca_from_v_th:
        if _train_mode_cli_text(getattr(args, "v_th_ca", None)) is not None:
            raise ValueError(
                "--v-th-ca conflicts with --v-th-ca-from-v-th "
                "(v_th_ca is forced frozen=all)"
            )
        train_modes = dict(train_modes or {})
        train_modes["v_th_ca"] = train.parse_train_mode_text("frozen=all")
    a_ca_from_a_out = bool(args.a_ca_from_a_out)
    if a_ca_from_a_out:
        if _train_mode_cli_text(getattr(args, "a_ca", None)) is not None:
            raise ValueError(
                "--a-ca conflicts with --a-ca-from-a-out "
                "(a_ca is forced frozen=all)"
            )
        train_modes = dict(train_modes or {})
        train_modes["a_ca"] = train.parse_train_mode_text("frozen=all")
    tasks = parse_tasks(args.task)
    part_cost_scales = parse_part_cost_scale(args.part_cost_scale, tasks)
    default_radius, radius_kv = parse_cost_radius(args.cost_radius)
    cost_radius_by_task = train.resolve_cost_radius_by_task(
        tasks, default_radius, radius_kv,
    )
    if default_radius is not None and default_radius != -1 and default_radius < 0:
        raise ValueError("--cost-radius must be -1 or >= 0")
    if any(v != -1 and v < 0 for v in radius_kv.values()):
        raise ValueError("--cost-radius must be -1 or >= 0")
    shift_radius = args.shift_radius
    spot_radius = args.spot_radius
    if args.spot_cost_r_s:
        from task.spot.input import spot_radius_half_steps
        from train.session import resolve_filter_branches
        _sr_scalar = resolve_filter_branches(spot_radius, filter="none")
        spot_radius_half_steps(_sr_scalar)
        spot_cost_radius_scale = parse_spot_cost_r_s_tokens(
            args.spot_cost_r_s,
            default_scales=default_spot_cost_radius_scale(
                _sr_scalar,
                scales=SPOT_PACK['spot_cost_radius_scale'],
                scales_radius1=SPOT_PACK['spot_cost_radius_scale_radius1'],
            ),
            spot_cost_radii=SPOT_PACK['spot_cost_radii'],
            aliases=SPOT_PACK['spot_cost_radius_key_aliases'],
        )
    else:
        spot_cost_radius_scale = None
    multi_spot = args.multi_spot
    fully_inside = args.fully_inside
    ms_pre = args.ms_pre
    ms_response = args.ms_response
    ms_post = args.ms_post
    ms_spot = args.ms_spot
    delta_ms = args.delta_ms
    delta_ms_pre = args.delta_ms_pre
    multi_bar = args.multi_bar
    _timing = {
        "ms_pre": ms_pre,
        "ms_response": ms_response,
        "ms_post": ms_post,
        "delta_ms": delta_ms,
        "delta_ms_pre": delta_ms_pre,
        "ms_spot": ms_spot,
    }
    moving_bar_bright_sti_opts = {
        "multi_bar": multi_bar,
        "ms_pre": ms_pre,
        "delta_ms": delta_ms,
        "delta_ms_pre": delta_ms_pre,
    }
    moving_bar_dark_sti_opts = {
        "multi_bar": multi_bar,
        "ms_pre": ms_pre,
        "delta_ms": delta_ms,
        "delta_ms_pre": delta_ms_pre,
    }
    spot_bright_sti_opts = dict(_timing)
    spot_dark_sti_opts = dict(_timing)
    if float(args.cost_interval_ms) <= 0:
        raise ValueError("--cost-interval-ms must be > 0")
    cost_ms = parse_cost_ms_tokens(
        args.cost_ms, aliases=SPOT_PACK['spot_cost_radius_key_aliases'],
    )
    if cost_ms is None:
        cost_ms = dict(TRAIN_OPTIMIZATION['cost_ms'])
    for _o in (spot_bright_sti_opts, spot_dark_sti_opts):
        _o["cost_interval_ms"] = float(args.cost_interval_ms)
        _o["cost_ms"] = {
            str(float(k)): [float(x) for x in v] for k, v in cost_ms.items()
        }
    gt_by_task = parse_gt(args.gt)
    if gt_by_task:
        _gt_opts = {
            "moving_bar_bright": moving_bar_bright_sti_opts,
            "moving_bar_dark": moving_bar_dark_sti_opts,
            "spot_bright": spot_bright_sti_opts,
            "spot_dark": spot_dark_sti_opts,
        }
        for _tname, _types in gt_by_task.items():
            _gt_opts[_tname]["gt_cells"] = list(_types)
    i_cli = train.build_i_cli_by_task({
        "i_baseline": parse_kv_tokens(args.i_baseline, float),
        "i_bright": parse_kv_tokens(args.i_bright, float),
        "i_dark": parse_kv_tokens(args.i_dark, float),
    })
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
    run_name = command_run_name(script_stem)
    from train.implementation import run_dir
    outdir = run_dir(model, parent=args.outdir, name=run_name)
    return dict(
        model=model,
        n_run=int(args.n_run),
        n_iter=n_iter,
        lrs=lrs,
        fname=args.fname,
        outdir=outdir,
        train_modes=train_modes,
        syn_mode=train.normalize_syn_mode(args.syn_mode),
        network=args.network,
        tasks=tasks,
        part_cost_scales=part_cost_scales,
        cost_norm=expand_cost_norm(args.cost_norm),
        cost_radius_by_task=cost_radius_by_task,
        shift_radius=shift_radius,
        spot_radius=spot_radius,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_scale=spot_cost_radius_scale,
        moving_bar_bright_sti_opts=moving_bar_bright_sti_opts,
        moving_bar_dark_sti_opts=moving_bar_dark_sti_opts,
        spot_bright_sti_opts=spot_bright_sti_opts,
        spot_dark_sti_opts=spot_dark_sti_opts,
        i_cli=i_cli,
        i_h_rev=args.i_h_rev,
        euler=args.euler,
        pre_steady=expand_pre_steady(args.pre_steady),
        pre_steady_iters=TRAIN_OPTIMIZATION['pre_steady_iters'],
        pre_steady_damp=TRAIN_OPTIMIZATION['pre_steady_damp'],
        fp=fp,
        pre_grad=bool(args.pre_grad),
        bias_gt_from_v_onset=bias_gt_from_v_onset,
        bias_gt_from_v_onset_grad=bias_gt_from_v_onset_grad,
        v_th_ca_from_v_th=v_th_ca_from_v_th,
        a_ca_from_a_out=a_ca_from_a_out,
        filter=filter,
        spot_gt_mode=spot_gt_mode,
        sequential=bool(args.sequential),
        init_from=init_from,
        checkpoint_interval=args.checkpoint_interval,
    )
