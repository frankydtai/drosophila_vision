from default_params import (
    DEFAULT_RUN_PATH,
    SPOT_INPUT,
)
"""Simulation + plotting for the FiveCol medulla model."""
import argparse
import json
import os
import numpy as np
import torch

import import_bootstrap  # noqa: F401
import train
from task.spot.input import spot_from_opts
from figure import moving_bar as moving_bar_plot
from figure import spot as spot_plot
from figure.util import (
    network_hex_count,
    filter_plot_token,
    session_filter_plot_token,
)
from default_params import DEFAULT_RUN_PATH, SPOT_INPUT
from train.config import run_data_dir
from train.implementation import resolve_run_dir


def _plot_device_label():
    dev = train.active_device()
    if dev == 'cuda' and torch.cuda.is_available():
        return f'cuda ({torch.cuda.get_device_name(0)})'
    return dev


def spot_readout_fns(session):
    if session.backend.network is None:
        raise ValueError("spot_readout_fns requires session.backend.network")
    return (
        spot_plot.network_spot_trace_readout,
        spot_plot.plot_network_spot_gt,
        spot_plot.plot_network_spot_all,
    )


def _network_spot_tag(session, tname):
    """Subtitle suffix for network spot plots (exact spot/shift counts)."""
    if session.backend.network is None:
        return ''
    opts = (session.train_opts or {}).get(f'{tname}_sti_opts') or {}
    spot = spot_from_opts(session.backend.network, sti_opts=opts)
    n_spots = len(spot.centers)
    n_shifts = len(spot.shifts)
    n_hexes = network_hex_count(session.backend.network)
    return (
        f'  [avg over {n_spots} spots x {n_shifts} shifts = {n_spots * n_shifts}]\n'
        f'({n_hexes} hexes in network)'
    )


def load_train_opts(outdir):
    opts_path = os.path.join(
        run_data_dir(os.path.abspath(outdir)), train.TRAIN_OPTS_FILE,
    )
    if not os.path.isfile(opts_path):
        return None
    with open(opts_path) as f:
        return json.load(f)


def load_session(outdir, model=None):
    return train.open_session_from_outdir(outdir, model)


def session_for_task(base_session, tname):
    """Single-task session sharing backend/schema with a multi-task run."""
    if base_session.backend.network is None:
        raise ValueError("session_for_task requires base_session.backend.network")
    opts = dict(base_session.train_opts or {})
    opts['tasks'] = [tname]
    opts['packs'] = None
    opts['network'] = base_session.backend.network
    return train.open_session({**opts, 'backend': 'network'}, base_session.model,
                           schema=list(base_session.schema))


def resolve_model(outdir, override=None):
    if override is not None:
        model = override
    else:
        opts = load_train_opts(outdir)
        if not opts or 'model' not in opts:
            raise SystemExit(
                f'cannot determine model for {outdir!r}; '
                f'expected {train.TRAIN_OPTS_FILE} with "model" in '
                f'{train.KNOWN_MODELS}'
            )
        model = opts['model']
    if model not in train.KNOWN_MODELS:
        raise SystemExit(
            f'invalid model {model!r} in {outdir!r}; '
            f'expected one of {train.KNOWN_MODELS}'
        )
    return model


def select_best(params, session, *, final_costs=None, verbose=True):
    """Pick the best parameter row via ``argmin`` of costs; recompute only when not supplied."""
    params = np.atleast_2d(params)
    valid_mask = np.any(params != 0, axis=1)
    valid = params[valid_mask]
    if len(valid) == 0:
        raise SystemExit('no trained parameter sets found (file all zeros)')
    valid_idx = np.where(valid_mask)[0]

    note = ''
    if final_costs is not None:
        costs_arr = np.asarray(final_costs, dtype=np.float64)
        if costs_arr.shape[0] != params.shape[0]:
            raise ValueError(
                f'final_costs length {costs_arr.shape[0]} != params runs {params.shape[0]}',
            )
        valid_costs = costs_arr[valid_mask]
        best = int(np.argmin(valid_costs))
        run_i = int(valid_idx[best])
        selected = float(costs_arr[run_i])
        note = ' (from saved final costs)'
    else:
        costs_out = []
        for row in valid:
            z = torch.tensor(row, dtype=session.sim_dtype, device=session.device)
            costs_out.append(train.calc_cost(z, session).item())
        valid_costs = np.array(costs_out)
        best = int(np.argmin(valid_costs))
        run_i = int(valid_idx[best])
        selected = float(valid_costs[best])
    if verbose:
        print(f'{len(valid)} trained set(s); costs min={valid_costs.min():.4f} '
              f'max={valid_costs.max():.4f}; selected #{run_i}{note}')
    return valid[best], selected


def _session_z_from_best_named(session, run_dir):
    """Remap ``best_param.npz`` named values onto ``session``; return ``(session, z)``."""
    import train.implementation as train_mod

    named, cells, pair_names = train_mod.load_best_param_named(run_dir)
    remapped = train.remap_named_node_values(
        named, cells, pair_names, list(session.schema), session.backend,
    )
    schema = train.attach_param_carry(list(session.schema), remapped)
    session = session.with_schema(schema)
    z = train.z_from_node_values(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )
    return session, z


def load_best(outdir, *, model=None, verbose=False):
    """Load session and best ``z`` from a train run (``best_param.npz`` + costs)."""
    import train.implementation as train_mod

    outdir = os.path.abspath(outdir)
    if not os.path.isdir(outdir):
        raise SystemExit(f'run dir not found: {outdir}')
    model = resolve_model(outdir, override=model)
    session, z = _session_z_from_best_named(load_session(outdir, model=model), outdir)
    best_cost = None
    final_costs, _, _, _ = train_mod.load_stored_costs(outdir)
    if final_costs is not None and len(final_costs) > 0:
        best_cost = float(final_costs[int(np.argmin(final_costs))])
    if best_cost is None:
        best_cost = train.calc_cost(z, session).item()
    if verbose:
        print(f'loaded best_param.npz (cost={best_cost:.4f})')
    return session, z, float(best_cost)


def maybe_override_sti_timing(
    *,
    run_dir,
    session,
    z,
    ms_pre=None,
    ms_response=None,
    ms_post=None,
    ms_spot=None,
    delta_ms=None,
    delta_ms_pre=None,
    euler=None,
    filter=None,
):
    """Re-open session when any timing / euler / filter override is set; remap best ``z``.

    Unset flags keep values from the run's train opts. ``ms_pre`` /
    ``delta_ms`` / ``delta_ms_pre`` also update moving_bar sti opts;
    ``ms_response`` / ``ms_post`` / ``ms_spot`` are spot-only. ``euler`` is
    CLI ``im``/``ex`` (or already expanded ``implicit``/``explicit``).
    ``filter`` is ``none``/``ca``; active branch of ``{v, ca}`` ms_spot/ms_response
    is selected in :func:`train.open_session`.

    Returns ``(session, z, timing_changed)`` where ``timing_changed`` maps
    timing keys that differ from the run (for filename suffixes).
    """
    if (
        ms_pre is None
        and ms_response is None
        and ms_post is None
        and ms_spot is None
        and delta_ms is None
        and delta_ms_pre is None
        and euler is None
        and filter is None
    ):
        return session, z, {}

    if delta_ms is not None:
        if isinstance(delta_ms, dict):
            for branch, val in delta_ms.items():
                if float(val) <= 0:
                    raise SystemExit(f"--delta-ms {branch} must be > 0")
        elif float(delta_ms) <= 0:
            raise SystemExit("--delta-ms must be > 0")
    if delta_ms_pre is not None:
        if isinstance(delta_ms_pre, dict):
            for branch, val in delta_ms_pre.items():
                if float(val) <= 0:
                    raise SystemExit(f"--delta-ms-pre {branch} must be > 0")
        elif float(delta_ms_pre) <= 0:
            raise SystemExit("--delta-ms-pre must be > 0")
    if ms_post is not None and float(ms_post) < 0:
        raise SystemExit("--ms-post must be >= 0")

    import train.implementation as train_mod

    opts = load_train_opts(run_dir)
    if opts is None:
        raise SystemExit(f"missing train opts under {run_dir}")

    timing_changed = {}
    if (
        ms_pre is not None
        or ms_response is not None
        or ms_post is not None
        or ms_spot is not None
        or delta_ms is not None
        or delta_ms_pre is not None
    ):
        from train.cli import apply_train_opts_timing
        timing_changed = apply_train_opts_timing(
            opts,
            ms_pre=ms_pre,
            ms_response=ms_response,
            ms_post=ms_post,
            ms_spot=ms_spot,
            delta_ms=delta_ms,
            delta_ms_pre=delta_ms_pre,
        )

    if euler is not None:
        opts["euler"] = train.expand_euler(euler)

    if filter is not None:
        opts["filter"] = train.expand_filter(filter)

    session = train.open_session_from_opts(opts, model=opts.get("model"))
    session, z = _session_z_from_best_named(session, run_dir)
    return session, z, timing_changed


def _format_filename_token(value):
    if isinstance(value, dict) and set(value) <= {"v", "ca"}:
        v = float(value["v"])
        ca = float(value["ca"])
        if v == ca:
            return _format_filename_token(v)
        return f"v{v:g}-ca{ca:g}"
    v = float(value)
    if v == int(v):
        return str(int(v))
    return "%g" % v


def sti_timing_filename_suffix(
    *,
    ms_pre=None,
    ms_spot=None,
    ms_response=None,
    ms_post=None,
    delta_ms=None,
    delta_ms_pre=None,
):
    """PNG stem suffix for timing keys that differ from the run (plot / analyze).

    Order: pre, spot, response, post, delta, delta_pre. Example::

        _ms_post_2500

    Empty string when every value is unset.
    """
    parts = []
    for name, val in (
        ("ms_pre", ms_pre),
        ("ms_spot", ms_spot),
        ("ms_response", ms_response),
        ("ms_post", ms_post),
        ("delta", delta_ms),
        ("delta_pre", delta_ms_pre),
    ):
        if val is not None:
            parts.append(f"{name}_{_format_filename_token(val)}")
    if not parts:
        return ""
    return "_" + "_".join(parts)


def euler_filename_suffix(euler=None):
    """PNG stem suffix for a non-``None`` ``--euler`` override (``_im`` / ``_ex``)."""
    if euler is None:
        return ""
    token = str(euler)
    # Prefer CLI short form in filenames.
    if token in ("implicit", "explicit"):
        token = "im" if token == "implicit" else "ex"
    elif token not in ("im", "ex"):
        token = train.expand_euler(token)
        token = "im" if token == "implicit" else "ex"
    return f"_{token}"


def _cost_parts_for_plot(session, z):
    """Unscaled per-part costs at ``z`` for panel titles."""
    with torch.no_grad():
        parts = train.calc_cost_parts(z, session)
    return {k: float(v.item()) for k, v in parts.items()}


def _plot_path(outdir, stem, file_suffix="", *, html=False):
    from figure.util import plot_file_ext

    return os.path.join(outdir, f"{stem}{file_suffix}{plot_file_ext(html=html)}")


def _readout_plot_stem(prefix, session):
    """``spot_gt_v`` / ``spot_gt_ca`` (filter chooses ``v`` or ``ca``, not both)."""
    return f"{prefix}_{session_filter_plot_token(session)}"


def _plot_spot_tasks(session, z, outdir, spot_tasks, suffix, model_all,
                       gt_rts=None,
                       at_x=None, at_y=None, show_pre=True,
                       file_suffix="", html=False, ms_shown=None,
                       center_only=False):
    """Plot spot task(s); contrasts combined in one figure when both are trained."""
    spot_set = set(spot_tasks)
    build_readout, plot_gt, plot_all = spot_readout_fns(session)
    ref_t = 'spot_bright' if 'spot_bright' in spot_set else spot_tasks[0]
    net_tag = _network_spot_tag(session, ref_t)
    cost_parts = _cost_parts_for_plot(session, z)
    plot_kw = dict(gt_rts=gt_rts, cost_parts=cost_parts)
    token = session_filter_plot_token(session)
    readout_kw = dict(
        at_xs=at_x, at_ys=at_y,
        show_pre=show_pre,
        ms_shown=ms_shown,
        center_only=center_only,
    )
    if spot_set == set(train.SPOT_TASKS):
        readouts = {
            'bright': build_readout(
                session_for_task(session, 'spot_bright'), z, **readout_kw,
            ),
            'dark': build_readout(
                session_for_task(session, 'spot_dark'), z, **readout_kw,
            ),
        }
        mvd = _plot_path(outdir, _readout_plot_stem('spot_gt', session), file_suffix, html=html)
        plot_gt(
            mvd, readouts=readouts,
            title=f'Spot {token}-gt ({suffix}){net_tag}',
            **plot_kw,
        )
        allc = None
        if model_all:
            allc = _plot_path(outdir, _readout_plot_stem('spot_all', session), file_suffix, html=html)
            plot_all(
                allc, readouts=readouts,
                title=f'Spot {token}-all ({suffix}){net_tag}',
                **plot_kw,
            )
        return mvd, allc
    for tname in spot_tasks:
        _plot_one_task(
            session_for_task(session, tname), z, outdir, tname, suffix, model_all,
            gt_rts=gt_rts,
            at_x=at_x, at_y=at_y,
            show_pre=show_pre,
            cost_parts=cost_parts,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
            center_only=center_only,
        )


def _plot_bar_readouts(session, z, outdir, bar_readouts, suffix, model_all, *,
                      plot_right_only=True, at_x=None, at_y=None,
                      align_at_x=None, align_at_y=None,
                      show_pre=True, file_suffix="", html=False, ms_shown=None):
    """Plot moving-bar task(s); bright left | dark right when both are trained."""
    cost_parts = _cost_parts_for_plot(session, z)
    token = session_filter_plot_token(session)
    readout_kw = dict(
        at_xs=at_x, at_ys=at_y,
        align_at_x=align_at_x, align_at_y=align_at_y,
        show_pre=show_pre,
        ms_shown=ms_shown,
    )
    bar_set = set(bar_readouts)
    if bar_set == set(train.MOVING_BAR_TASKS):
        s_bright = session_for_task(session, 'moving_bar_bright')
        s_dark = session_for_task(session, 'moving_bar_dark')
        readout_bright = moving_bar_plot.moving_bar_trace_readout(
            s_bright, z, 'moving_bar_bright', **readout_kw,
        )
        readout_dark = moving_bar_plot.moving_bar_trace_readout(
            s_dark, z, 'moving_bar_dark', **readout_kw,
        )
        mvd = _plot_path(outdir, _readout_plot_stem('bar_gt', session), file_suffix, html=html)
        moving_bar_plot.plot_moving_bar_gt(
            mvd, readout=readout_bright, readout_2=readout_dark,
            title=f'Moving-bar {token}-gt ({suffix})',
            cost_parts=cost_parts,
        )
        allc = None
        if model_all:
            allc = _plot_path(outdir, _readout_plot_stem('bar_all', session), file_suffix, html=html)
            moving_bar_plot.plot_moving_bar_all(
                allc, readout=readout_bright, readout_2=readout_dark,
                title=f'Moving-bar {token}-all ({suffix})',
                right_only=plot_right_only,
                cost_parts=cost_parts,
            )
        return mvd, allc
    for tname in bar_readouts:
        one = session_for_task(session, tname)
        readout = moving_bar_plot.moving_bar_trace_readout(one, z, tname, **readout_kw)
        mvd = _plot_path(outdir, _readout_plot_stem('bar_gt', session), file_suffix, html=html)
        moving_bar_plot.plot_moving_bar_gt(
            mvd, readout=readout, title=f'{tname} {token}-gt ({suffix})',
            cost_parts=cost_parts,
        )
        allc = None
        if model_all:
            allc = _plot_path(outdir, _readout_plot_stem('bar_all', session), file_suffix, html=html)
            moving_bar_plot.plot_moving_bar_all(
                allc, readout=readout, title=f'{tname} {token}-all ({suffix})',
                right_only=plot_right_only,
                cost_parts=cost_parts,
            )
        return mvd, allc


def _plot_one_task(session, z, outdir, tname, suffix, model_all,
                     gt_rts=None,
                     at_x=None, at_y=None, show_pre=True,
                     cost_parts=None, file_suffix="", html=False, ms_shown=None,
                     center_only=False):
    if tname not in train.SPOT_TASKS:
        raise ValueError(f'unknown plot task {tname!r}')
    token = session_filter_plot_token(session)
    mvd = _plot_path(outdir, _readout_plot_stem('spot_gt', session), file_suffix, html=html)
    allc = _plot_path(outdir, _readout_plot_stem('spot_all', session), file_suffix, html=html)
    build_readout, plot_gt, plot_all = spot_readout_fns(session)
    net_tag = _network_spot_tag(session, tname)
    if cost_parts is None:
        cost_parts = _cost_parts_for_plot(session, z)
    plot_kw = dict(gt_rts=gt_rts, cost_parts=cost_parts)
    readout = build_readout(
        session, z,
        at_xs=at_x, at_ys=at_y,
        show_pre=show_pre,
        ms_shown=ms_shown,
        center_only=center_only,
    )
    from figure.gt import contrast_for_task
    readouts = {contrast_for_task(tname): readout}
    plot_gt(
        mvd, readouts=readouts, title=f'{tname} {token}-gt ({suffix}){net_tag}', **plot_kw,
    )
    if model_all:
        plot_all(allc, readouts=readouts, title=f'{tname} {token}-all ({suffix}){net_tag}', **plot_kw)
    return mvd, allc


def plot_param_set(params, outdir, model=None, model_all=True,
                   context_dir=None,
                   plot_tasks=None, session=None, *,
                   final_costs=None,
                   save_data=True,
                   gt_rts=None,
                   plot_right_only=True, at_x=None, at_y=None,
                   align_at_x=None, align_at_y=None,
                   show_pre=True, file_suffix="", html=False, ms_shown=None,
                   center_only=False):
    import train.implementation as train_mod

    os.makedirs(outdir, exist_ok=True)
    ctx = context_dir or outdir
    if model is None and session is not None:
        model = session.model
    if model is None:
        raise ValueError('model or session required')
    if session is None:
        session = load_session(ctx, model)

    params = np.atleast_2d(params)
    if final_costs is None:
        final_costs, _, _, _ = train_mod.load_stored_costs(outdir)

    print(f'plot device={_plot_device_label()}')
    best, best_cost = select_best(
        params, session, final_costs=final_costs,
    )
    z = torch.tensor(best, dtype=session.sim_dtype, device=session.device)

    suffix = f'trained, cost {best_cost:.2f}'
    cost_norm = (session.train_opts or {}).get("cost_norm", "a_gt2")
    if cost_norm == "gt_power":
        suffix = f'{suffix}% of gt power'
    else:
        suffix = f'{suffix} ({cost_norm})'
    tasks = list(session.tasks)
    if plot_tasks is not None:
        tasks = [t for t in tasks if t in plot_tasks]

    spot_tasks = [t for t in tasks if t in train.SPOT_TASKS]
    bar_readouts = [t for t in tasks if t in train.MOVING_BAR_TASKS]
    if (at_x is not None or at_y is not None) and not bar_readouts and not spot_tasks:
        raise SystemExit('--x/--y require a moving_bar or spot task in this run')
    if (align_at_x is not None or align_at_y is not None):
        if align_at_x is None or align_at_y is None:
            raise SystemExit('--align-xy requires X,Y')
        if at_x is None and at_y is None:
            raise SystemExit('--align-xy requires --x and/or --y')
        if not bar_readouts:
            raise SystemExit('--align-xy applies to moving_bar slice plots only')

    if spot_tasks:
        _plot_spot_tasks(
            session, z, outdir, spot_tasks, suffix, model_all,
            gt_rts=gt_rts,
            at_x=at_x, at_y=at_y,
            show_pre=show_pre,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
            center_only=center_only,
        )
    if bar_readouts:
        _plot_bar_readouts(
            session, z, outdir, bar_readouts, suffix, model_all,
            plot_right_only=plot_right_only,
            at_x=at_x, at_y=at_y,
            align_at_x=align_at_x, align_at_y=align_at_y,
            show_pre=show_pre,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
        )

    if save_data:
        os.makedirs(run_data_dir(os.path.abspath(outdir)), exist_ok=True)
        z_best = torch.tensor(best, dtype=session.sim_dtype, device=session.device)
        train_mod.save_best_param_named(outdir, z_best, session)
    return best, best_cost


def add_plot_arguments(parser):
    """Register plot-only CLI flags shared by ``run.py`` and ``figure.plot``.

    Timing overrides (``--ms-pre`` / …) are registered separately via
    :func:`add_plot_timing_arguments` so ``run.py`` does not double-register
    flags already on the train CLI.
    """
    from import_bootstrap import parse_bool

    parser.add_argument(
        '--html',
        action='store_true',
        help='save interactive plotly HTML (hover x/y) instead of PNG',
    )
    parser.add_argument(
        '--plot-right-only',
        nargs='?',
        const=True,
        default=True,
        type=parse_bool,
        metavar='BOOL',
        help='bar_all_{ca|v}: right-direction specs only (default true); '
             'pass false for all directions',
    )
    parser.add_argument(
        '--show-pre',
        nargs='?',
        const=True,
        default=True,
        type=parse_bool,
        metavar='BOOL',
        help='model pre-t_onset: dashed red when true (default); omit when false. '
             'Gray gt never draws pre.',
    )
    parser.add_argument(
        '--r0only',
        nargs='?',
        const=True,
        default=False,
        type=parse_bool,
        metavar='BOOL',
        help='spot_gt/spot_all: only plot center-radius (r=0) time row '
             '(default false: plot all trained r rows)',
    )
    parser.add_argument(
        '--x',
        default=None,
        metavar='X,...',
        help='{bar,spot}_all_{ca|v}: comma-separated x slices; with --y, one trace per (x,y) pair',
    )
    parser.add_argument(
        '--y',
        default=None,
        metavar='Y,...',
        help='{bar,spot}_all_{ca|v}: comma-separated y slices; with --x, one trace per (x,y) pair',
    )
    parser.add_argument(
        '--align-xy',
        default=None,
        metavar='X,Y',
        help='moving_bar slice plots: align --x/--y traces to ref hex hex (x,y); scope unchanged',
    )
    add_ms_shown_argument(parser)


def parse_axis_slices(text):
    """Parse comma-separated ``--x`` / ``--y`` values (empty -> ``None``)."""
    if not text:
        return None
    vals = [float(x) for x in import_bootstrap.parse_comma_list(text)]
    if not vals:
        raise ValueError("empty comma-separated axis slice")
    return vals


def parse_align_xy(text):
    """Parse ``--align-xy X,Y`` reference sti hex (empty -> ``None``)."""
    if not text:
        return None
    parts = import_bootstrap.parse_comma_list(text)
    if len(parts) != 2:
        raise ValueError("--align-xy requires exactly two comma-separated values X,Y")
    return float(parts[0]), float(parts[1])


def add_ms_shown_argument(parser):
    """Register ``--ms-shown START,STOP`` display / analyze time window."""
    parser.add_argument(
        "--ms-shown",
        default=None,
        metavar="START,STOP",
        help=(
            "absolute aligned ms START,STOP (not --ms-pre; not onset-relative). "
            "spot: 0=trial start, pre=0,ms_pre (e.g. 0,1000); "
            "bar: 0=t0 at node (neg START ok); omit = full trace"
        ),
    )


def parse_ms_shown_range(token, *, flag="--ms-shown"):
    """Parse ``START,STOP`` ms (comma; one token)."""
    parts = import_bootstrap.parse_comma_list(token)
    if len(parts) != 2:
        raise ValueError(f"{flag} must be START,STOP")
    start, stop = float(parts[0]), float(parts[1])
    if start > stop:
        raise ValueError(f"{flag} START={start} > STOP={stop}")
    return start, stop


def add_plot_timing_arguments(parser):
    """Hang train sti-timing CLI onto plot / analyze (defaults: keep run)."""
    from train.cli import add_sti_timing_arguments

    add_sti_timing_arguments(
        parser,
        default_ms_pre=None,
        default_ms_response=None,
        default_ms_post=None,
        default_ms_spot=None,
        default_delta_ms=None,
        default_delta_ms_pre=None,
    )


def add_plot_euler_argument(parser):
    """``--euler im|ex`` for plot / analyze (default: keep run train opts)."""
    parser.add_argument(
        "--euler",
        default=None,
        choices=list(train.EULER_CLI),
        help="membrane Euler override: im=implicit, ex=explicit "
             "(default: keep run train_opts.euler); i_h gates always explicit",
    )


def add_plot_filter_argument(parser):
    """``--filter none|ca`` for plot / analyze (default: keep run train opts)."""
    parser.add_argument(
        "--filter",
        default=None,
        choices=("none", "ca"),
        help="readout filter override: none=v (no Ca schema), ca=ca + Arenz digitized spot gt "
             f"(default: keep run train_opts.filter; "
             f"ms_spot v={SPOT_INPUT['ms_spot']['v']:g}, ca={SPOT_INPUT['ms_spot']['ca']:g}, "
             f"ms_response v={SPOT_INPUT['ms_response']['v']:g}, ca={SPOT_INPUT['ms_response']['ca']:g})",
    )


def filter_filename_suffix(filter=None):
    """PNG stem suffix for a non-``None`` ``--filter`` override (``_v`` / ``_ca``)."""
    if filter is None:
        return ""
    return f"_{filter_plot_token(train.expand_filter(filter))}"


def add_param_argument(parser):
    """``--param NAME=VALUE`` / ``NAME.NODE=VALUE`` for plot / analyze."""
    parser.add_argument(
        "--param",
        nargs="+",
        default=None,
        metavar="NAME=VALUE|NAME.NODE=VALUE",
        help=(
            "overwrite schema params before forward; "
            "NAME=VALUE or NAME.all=VALUE sets every node; "
            "NAME.NODE=VALUE for one cell / SRC:TAR pair / eN; "
            "PNG stem gets _NAME_NODE_VALUE per edit (after timing suffixes)"
        ),
    )


def parse_param_tokens(tokens):
    """Parse ``--param NAME=VALUE`` / ``NAME.NODE=VALUE`` / ``NAME.all=VALUE``."""
    if not tokens:
        return []
    out = []
    for tok in tokens:
        if "=" not in tok:
            raise SystemExit(
                f"--param expected NAME=VALUE or NAME.NODE=VALUE, got {tok!r}"
            )
        left, val_s = tok.split("=", 1)
        if not left:
            raise SystemExit(
                f"--param expected NAME=VALUE or NAME.NODE=VALUE, got {tok!r}"
            )
        if "." in left:
            name, node = left.split(".", 1)
            if not name or not node:
                raise SystemExit(
                    f"--param expected NAME=VALUE or NAME.NODE=VALUE, got {tok!r}"
                )
            if node == "all":
                node = None
        else:
            name, node = left, None
        try:
            val = float(val_s)
        except ValueError as exc:
            raise SystemExit(f"--param bad VALUE {val_s!r}") from exc
        out.append((name, node, val))
    return out


def param_filename_suffix(edits):
    """PNG stem suffix for ``--param`` edits; empty when none.

    Example: ``hp_lp.L1=1000`` → ``_hp_lp_L1_1000``.
    """
    if not edits:
        return ""
    parts = []
    for name, node, val in edits:
        bits = [name]
        if node is not None:
            bits.append(str(node).replace(":", "_"))
        bits.append(_format_filename_token(val))
        parts.append("_".join(bits))
    return "_" + "_".join(parts)


def apply_param_overrides(z, schema, session, edits):
    """Apply ``--param`` edits; return ``(z, schema)`` (schema may gain frozen carry)."""
    if not edits:
        return z, schema
    if session.backend.network is None:
        raise SystemExit("--param requires a network backend")

    named = train.node_values_from_z(z, schema)
    seg_by_name = {s["name"]: s for s in schema}
    edited_idxs = {}

    for name, node, val in edits:
        seg = seg_by_name.get(name)
        if seg is None:
            avail = sorted(seg_by_name)
            raise SystemExit(
                f"--param unknown param {name!r}; schema has: {avail}"
            )
        if name not in named:
            raise SystemExit(f"--param schema missing values for {name!r}")
        labels = train.node_names_for_segment(seg, session.backend)
        i_from_label = {lab: i for i, lab in enumerate(labels)}
        arr = np.array(named[name], dtype=np.float64, copy=True)
        if node is None:
            idxs = list(range(len(labels)))
        else:
            if node not in i_from_label:
                raise SystemExit(
                    f"--param unknown node {node!r} for {name}; "
                    f"available: {labels}"
                )
            idxs = [i_from_label[node]]
        for i in idxs:
            arr[i] = val
            print(f"param {name}.{labels[i]} = {val:g}", flush=True)
        named[name] = arr
        edited_idxs.setdefault(name, set()).update(idxs)

    new_schema = []
    for seg in schema:
        s = dict(seg)
        hit = edited_idxs.get(s["name"])
        if hit:
            for mode in ("indi", "shared", "fixed", "frozen"):
                s[mode] = [i for i in (s.get(mode) or []) if i not in hit]
            s["frozen"] = sorted(set(s.get("frozen") or []) | hit)
        new_schema.append(s)

    new_schema = train.attach_param_carry(new_schema, named)
    z_new = train.z_from_node_values(
        named, new_schema, dtype=z.dtype, device=z.device,
    )
    return z_new, new_schema


def plot_kwargs_from_args(args):
    """Map a parsed CLI namespace to :func:`plot_param_set` plot kwargs."""
    align_xy = parse_align_xy(args.align_xy)
    align_at_x, align_at_y = align_xy if align_xy is not None else (None, None)
    ms_shown = None
    if args.ms_shown is not None:
        ms_shown = parse_ms_shown_range(args.ms_shown)
    return dict(
        plot_right_only=args.plot_right_only,
        show_pre=args.show_pre,
        center_only=bool(args.r0only),
        at_x=parse_axis_slices(args.x),
        at_y=parse_axis_slices(args.y),
        align_at_x=align_at_x,
        align_at_y=align_at_y,
        html=bool(args.html),
        ms_shown=ms_shown,
    )


def main():
    import train.implementation as train_mod

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        'run_path',
        nargs='?',
        default=DEFAULT_RUN_PATH,
        help='run folder under PARAMETER_DIR or absolute path (default: %(default)s)',
    )
    add_plot_arguments(ap)
    add_plot_timing_arguments(ap)
    add_plot_euler_argument(ap)
    add_plot_filter_argument(ap)
    add_param_argument(ap)
    args = ap.parse_args()
    try:
        plot_kw = plot_kwargs_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    from train.cli import sti_timing_kwargs_from_args
    timing_kw = sti_timing_kwargs_from_args(args)
    param_edits = parse_param_tokens(args.param)

    outdir = resolve_run_dir(args.run_path)
    session, z, best_cost = load_best(outdir, verbose=True)
    session, z, timing_changed = maybe_override_sti_timing(
        run_dir=outdir, session=session, z=z, **timing_kw,
        euler=args.euler,
        filter=args.filter,
    )
    z_t = (
        z if torch.is_tensor(z)
        else torch.tensor(np.asarray(z, dtype=np.float64), dtype=torch.float64,
                          device=session.device)
    )
    z_t, schema = apply_param_overrides(
        z_t, list(session.schema), session, param_edits,
    )
    session = session.with_schema(schema)
    # Filter is already in readout stems (``_v`` / ``_ca``); do not append again.
    file_suffix = (
        sti_timing_filename_suffix(**timing_changed)
        + euler_filename_suffix(args.euler)
        + param_filename_suffix(param_edits)
    )
    model = resolve_model(outdir)
    z_np = z_t.detach().cpu().numpy()
    print(f'params={train_mod.best_param_path(outdir)}')
    print(f'model={model} ({z_np.shape[-1]} params)')
    plot_param_set(
        np.atleast_2d(z_np),
        outdir,
        model=model,
        session=session,
        final_costs=np.array([best_cost]),
        file_suffix=file_suffix,
        save_data=not param_edits,
        **plot_kw,
    )


if __name__ == '__main__':
    main()
