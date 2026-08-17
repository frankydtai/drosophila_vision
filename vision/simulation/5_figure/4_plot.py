from config import (
    RUN_PATH,
)
"""Simulation + plotting for the FiveCol medulla model."""
import argparse
import json
import os
import numpy as np
import torch
import hydra

import import_bootstrap  # noqa: F401
import train
from task.spot.sti_geo import resolve_spot
from figure import moving_bar
from figure import spot
from figure.panel import (
    network_hex_count,
    filter_figure_token,
    session_filter_figure_token,
)
from train.config import run_data_dir
from train.implementation import resolve_run_dir


def _plot_device_label():
    device = train.active_device()
    if device == 'cuda' and torch.cuda.is_available():
        return f'cuda ({torch.cuda.get_device_name(0)})'
    return device


def spot_readout_fns(session):
    return (
        spot.network_spot_trace_readout,
        spot.plot_network_spot_gt,
        spot.plot_network_spot_all,
    )


def _network_spot_label(session, task="spot"):
    """Subtitle suffix for network spot plots (exact spot/shift counts)."""
    opts = (session.train_opts or {}).get(f'{task}_sti_opts') or {}
    spot = resolve_spot(session.connectome, sti_opts=opts)
    n_spot = len(spot.centers)
    n_shift = len(spot.shifts)
    n_hex = network_hex_count(session.connectome)
    return (
        f'  [avg over {n_spot} spots x {n_shift} shifts = {n_spot * n_shift}]\n'
        f'({n_hex} hexes in network)'
    )


def load_train_opts(outdir):
    opts_path = os.path.join(
        run_data_dir(os.path.abspath(outdir)), "train_opts.json",
    )
    if not os.path.isfile(opts_path):
        return None
    with open(opts_path) as f:
        return json.load(f)


def session_from_task(base_session, task, contrast):
    """Session with one ``task`` × one ``contrast``; shared connectome/schema."""
    opts = dict(base_session.train_opts or {})
    opts['tasks'] = [task]
    opts['contrasts'] = [contrast]
    opts['packs'] = None
    opts['network'] = base_session.connectome
    return train.open_session(opts, base_session.model,
                           schema=train.schema_copy(base_session.schema))


def resolve_model(outdir, model=None):
    if model is None:
        opts = load_train_opts(outdir)
        if not opts or 'model' not in opts:
            raise SystemExit(
                f'cannot determine model for {outdir!r}; '
                f'expected train_opts.json with "model" in '
                f'{train.MODELS}'
            )
        model = opts['model']
    if model not in train.MODELS:
        raise SystemExit(
            f'invalid model {model!r} in {outdir!r}; '
            f'expected one of {train.MODELS}'
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


def _session_z_from_best_param(session, run_dir):
    """Remap ``best_param.npz`` per-param vals onto ``session``; return ``(session, z)``."""
    import train.implementation as train_mod

    node_vals, cells, pairs = train_mod.load_best_node_vals(run_dir)
    remapped = train.remap_node_vals(
        node_vals, cells, pairs, train.schema_copy(session.schema), session.connectome,
    )
    schema = train.schema_with_param_carry(train.schema_copy(session.schema), remapped)
    session = session.with_schema(schema)
    z = train.z_from_node_vals(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )
    return session, z


def load_best(outdir, *, model=None, verbose=False):
    """Load session and best ``z`` from a train run (``best_param.npz`` + costs)."""
    import train.implementation as train_mod

    outdir = os.path.abspath(outdir)
    if not os.path.isdir(outdir):
        raise SystemExit(f'run dir not found: {outdir}')
    model = resolve_model(outdir, model=model)
    session, z = _session_z_from_best_param(train.session_from_outdir(outdir, model=model), outdir)
    best_cost = None
    final_costs, _, _, _ = train_mod.load_stored_costs(outdir)
    if final_costs is not None and len(final_costs) > 0:
        best_cost = float(final_costs[int(np.argmin(final_costs))])
    if best_cost is None:
        best_cost = train.calc_cost(z, session).item()
    if verbose:
        print(f'loaded best_param.npz (cost={best_cost:.4f})')
    return session, z, float(best_cost)


def override_session_sti_timing(
    *,
    run_dir,
    session,
    z,
    ms_pre=None,
    ms_response=None,
    ms_post=None,
    ms_sti=None,
    delta_ms=None,
    delta_ms_pre=None,
    euler=None,
    filter=None,
):
    """Re-open session when any timing / euler / filter token is set; remap best ``z``.

    Unset flags keep values from the run's train opts. ``ms_pre`` /
    ``delta_ms`` / ``delta_ms_pre`` also update moving_bar sti opts;
    ``ms_response`` / ``ms_post`` / ``ms_sti`` are spot-only. ``euler`` is
    CLI ``im``/``ex`` (or already expanded ``implicit``/``explicit``).
    ``filter`` is ``none``/``ca`` (readout filter; timing values are scalars).

    Returns ``(session, z, timing_changed)`` where ``timing_changed`` maps
    timing keys that differ from the run (for filename suffixes).
    """
    if (
        ms_pre is None
        and ms_response is None
        and ms_post is None
        and ms_sti is None
        and delta_ms is None
        and delta_ms_pre is None
        and euler is None
        and filter is None
    ):
        return session, z, {}

    if delta_ms is not None and float(delta_ms) <= 0:
        raise SystemExit(f"--sti-timing delta_ms={delta_ms} must be > 0")
    if delta_ms_pre is not None and float(delta_ms_pre) <= 0:
        raise SystemExit(f"--sti-timing delta_ms_pre={delta_ms_pre} must be > 0")
    if ms_post is not None and float(ms_post) < 0:
        raise SystemExit("--sti-timing ms_post must be >= 0")

    import train.implementation as train_mod

    opts = load_train_opts(run_dir)
    if opts is None:
        raise SystemExit(f"missing train opts under {run_dir}")

    timing_changed = {}
    if (
        ms_pre is not None
        or ms_response is not None
        or ms_post is not None
        or ms_sti is not None
        or delta_ms is not None
        or delta_ms_pre is not None
    ):
        from train.cli import override_train_opts_timing
        timing_changed = override_train_opts_timing(
            opts,
            ms_pre=ms_pre,
            ms_response=ms_response,
            ms_post=ms_post,
            ms_sti=ms_sti,
            delta_ms=delta_ms,
            delta_ms_pre=delta_ms_pre,
        )

    if euler is not None:
        opts["euler"] = train.expand_euler(euler)

    if filter is not None:
        opts["filter"] = train.expand_filter(filter)

    session = train.resolve_session(opts, model=opts.get("model"))
    session, z = _session_z_from_best_param(session, run_dir)
    return session, z, timing_changed


def _format_filename_token(value):
    val = float(value)
    if val == int(val):
        return str(int(val))
    return "%g" % val


def sti_timing_filename_suffix(
    *,
    ms_pre=None,
    ms_sti=None,
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
        ("ms_sti", ms_sti),
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
    from train.cli import euler_filename_suffix as _suffix
    return _suffix(euler)


def _figure_cost_parts(session, z):
    """Unscaled per-part costs at ``z`` for panel titles."""
    with torch.no_grad():
        parts = train.calc_cost_parts(z, session)
    return {k: float(v.item()) for k, v in parts.items()}


def _plot_path(outdir, stem, file_suffix="", *, html=False):
    from figure.panel import figure_file_ext

    return os.path.join(outdir, f"{stem}{file_suffix}{figure_file_ext(html=html)}")


def _readout_figure_stem(prefix, session):
    """``spot_gt_v`` / ``spot_gt_ca`` (filter chooses ``v`` or ``ca``, not both)."""
    return f"{prefix}_{session_filter_figure_token(session)}"


def _plot_spot_tasks(session, z, outdir, suffix, model_all,
                       gts=None,
                       at_x=None, at_y=None, show_pre=True,
                       file_suffix="", html=False, ms_shown=None,
                       center_only=False):
    """Plot spot; both contrasts in one figure when session has bright and dark."""
    build_readout, plot_gt, plot_all = spot_readout_fns(session)
    net_label = _network_spot_label(session, "spot")
    cost_parts = _figure_cost_parts(session, z)
    figure_kwargs = dict(gts=gts, cost_parts=cost_parts)
    token = session_filter_figure_token(session)
    readout_kwargs = dict(
        at_xs=at_x, at_ys=at_y,
        show_pre=show_pre,
        ms_shown=ms_shown,
        center_only=center_only,
    )
    contrasts = tuple(session.contrasts)
    if set(contrasts) >= {"bright", "dark"}:
        readouts = {
            contrast: build_readout(
                session_from_task(session, "spot", contrast), z, **readout_kwargs,
            )
            for contrast in ("bright", "dark")
        }
        mvd = _plot_path(outdir, _readout_figure_stem('spot_gt', session), file_suffix, html=html)
        plot_gt(
            mvd, readouts=readouts,
            title=f'Spot {token}-gt ({suffix}){net_label}',
            **figure_kwargs,
        )
        allc = None
        if model_all:
            allc = _plot_path(outdir, _readout_figure_stem('spot_all', session), file_suffix, html=html)
            plot_all(
                allc, readouts=readouts,
                title=f'Spot {token}-all ({suffix}){net_label}',
                **figure_kwargs,
            )
        return mvd, allc
    for contrast in contrasts:
        _plot_one_spot(
            session_from_task(session, "spot", contrast), z, outdir, contrast, suffix, model_all,
            gts=gts,
            at_x=at_x, at_y=at_y,
            show_pre=show_pre,
            cost_parts=cost_parts,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
            center_only=center_only,
        )


def _plot_bar_readouts(session, z, outdir, suffix, model_all, *,
                      plot_right_only=True, at_x=None, at_y=None,
                      align_at_x=None, align_at_y=None,
                      show_pre=True, file_suffix="", html=False, ms_shown=None):
    """Plot moving-bar; bright left | dark right when session has both contrasts."""
    cost_parts = _figure_cost_parts(session, z)
    token = session_filter_figure_token(session)
    readout_kwargs = dict(
        at_xs=at_x, at_ys=at_y,
        align_at_x=align_at_x, align_at_y=align_at_y,
        show_pre=show_pre,
        ms_shown=ms_shown,
    )
    contrasts = tuple(session.contrasts)
    if set(contrasts) >= {"bright", "dark"}:
        s_bright = session_from_task(session, "moving_bar", "bright")
        s_dark = session_from_task(session, "moving_bar", "dark")
        readout_bright = moving_bar.moving_bar_trace_readout(
            s_bright, z, "moving_bar", "bright", **readout_kwargs,
        )
        readout_dark = moving_bar.moving_bar_trace_readout(
            s_dark, z, "moving_bar", "dark", **readout_kwargs,
        )
        mvd = _plot_path(outdir, _readout_figure_stem('bar_gt', session), file_suffix, html=html)
        moving_bar.plot_moving_bar_gt(
            mvd, readout=readout_bright, readout_2=readout_dark,
            title=f'Moving-bar {token}-gt ({suffix})',
            cost_parts=cost_parts,
        )
        allc = None
        if model_all:
            allc = _plot_path(outdir, _readout_figure_stem('bar_all', session), file_suffix, html=html)
            moving_bar.plot_moving_bar_all(
                allc, readout=readout_bright, readout_2=readout_dark,
                title=f'Moving-bar {token}-all ({suffix})',
                right_only=plot_right_only,
                cost_parts=cost_parts,
            )
        return mvd, allc
    for contrast in contrasts:
        one = session_from_task(session, "moving_bar", contrast)
        readout = moving_bar.moving_bar_trace_readout(
            one, z, "moving_bar", contrast, **readout_kwargs,
        )
        mvd = _plot_path(outdir, _readout_figure_stem('bar_gt', session), file_suffix, html=html)
        moving_bar.plot_moving_bar_gt(
            mvd, readout=readout, title=f'moving_bar {contrast} {token}-gt ({suffix})',
            cost_parts=cost_parts,
        )
        allc = None
        if model_all:
            allc = _plot_path(outdir, _readout_figure_stem('bar_all', session), file_suffix, html=html)
            moving_bar.plot_moving_bar_all(
                allc, readout=readout, title=f'moving_bar {contrast} {token}-all ({suffix})',
                right_only=plot_right_only,
                cost_parts=cost_parts,
            )
        return mvd, allc


def _plot_one_spot(session, z, outdir, contrast, suffix, model_all,
                     gts=None,
                     at_x=None, at_y=None, show_pre=True,
                     cost_parts=None, file_suffix="", html=False, ms_shown=None,
                     center_only=False):
    token = session_filter_figure_token(session)
    mvd = _plot_path(outdir, _readout_figure_stem('spot_gt', session), file_suffix, html=html)
    allc = _plot_path(outdir, _readout_figure_stem('spot_all', session), file_suffix, html=html)
    build_readout, plot_gt, plot_all = spot_readout_fns(session)
    net_label = _network_spot_label(session, "spot")
    if cost_parts is None:
        cost_parts = _figure_cost_parts(session, z)
    figure_kwargs = dict(gts=gts, cost_parts=cost_parts)
    readout = build_readout(
        session, z,
        at_xs=at_x, at_ys=at_y,
        show_pre=show_pre,
        ms_shown=ms_shown,
        center_only=center_only,
    )
    readouts = {contrast: readout}
    plot_gt(
        mvd, readouts=readouts,
        title=f'spot {contrast} {token}-gt ({suffix}){net_label}', **figure_kwargs,
    )
    if model_all:
        plot_all(
            allc, readouts=readouts,
            title=f'spot {contrast} {token}-all ({suffix}){net_label}', **figure_kwargs,
        )
    return mvd, allc


def plot_rf_t(params, outdir, model=None, model_all=True,
                   context_dir=None,
                   plot_tasks=None, session=None, *,
                   final_costs=None,
                   save_data=True,
                   gts=None,
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
        session = train.session_from_outdir(ctx, model)

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

    has_spot = "spot" in tasks
    has_moving_bar = "moving_bar" in tasks
    if (at_x is not None or at_y is not None) and not has_moving_bar and not has_spot:
        raise SystemExit('--x/--y require a moving_bar or spot task in this run')
    if (align_at_x is not None or align_at_y is not None):
        if align_at_x is None or align_at_y is None:
            raise SystemExit('--align-xy requires X,Y')
        if at_x is None and at_y is None:
            raise SystemExit('--align-xy requires --x and/or --y')
        if not has_moving_bar:
            raise SystemExit('--align-xy applies to moving_bar overlay plots only')

    if has_spot:
        _plot_spot_tasks(
            session, z, outdir, suffix, model_all,
            gts=gts,
            at_x=at_x, at_y=at_y,
            show_pre=show_pre,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
            center_only=center_only,
        )
    if has_moving_bar:
        _plot_bar_readouts(
            session, z, outdir, suffix, model_all,
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
        train_mod.save_best_param(outdir, z_best, session)
    return best, best_cost


def add_ms_shown_argument(parser):
    """Register ``--ms-shown START,STOP`` (analyze / legacy argparse entrypoints)."""
    parser.add_argument(
        "--ms-shown",
        default=None,
        metavar="START,STOP",
        help=(
            "absolute aligned ms START,STOP (not sti_timing; not onset-relative). "
            "spot: 0=trial start, pre=0,ms_pre (e.g. 0,1000); "
            "bar: 0=t0 at node (neg START ok); omit = entire trace"
        ),
    )


def add_plot_session_override_arguments(parser):
    """Plot/analyze session overrides (not used by Hydra ``run.py`` / ``figure.plot``)."""
    parser.add_argument(
        "--sti-timing",
        dest="sti_timing",
        nargs="+",
        default=None,
        metavar="KEY=MS",
        help=f"sti length KEY=MS tokens. Keys: {', '.join(train.cli.STI_TIMING_KEYS)}",
    )
    parser.add_argument(
        "--euler",
        default=None,
        choices=list(train.EULER_CLI),
        help="Euler: im|ex (default: keep run train_opts.euler)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        choices=("none", "ca"),
        help="readout filter override (default: keep run train_opts.filter)",
    )
    parser.add_argument(
        "--param",
        nargs="+",
        default=None,
        metavar="PARAM.KEY[.NODES]=VALUE",
        help="param init/lo/hi/jit/val/mode overrides for plot/analyze",
    )


def parse_axis_coords(token):
    """Parse comma-separated ``--x`` / ``--y`` values (empty -> ``None``)."""
    if token is None or token == "":
        return None
    if isinstance(token, (list, tuple)):
        vals = [float(x) for x in token]
    else:
        vals = [float(x) for x in import_bootstrap.parse_comma_list(str(token))]
    if not vals:
        raise ValueError("empty comma-separated axis coord")
    return vals


def parse_align_xy(token):
    """Parse ``--align-xy X,Y`` reference sti hex (empty -> ``None``)."""
    if token is None or token == "":
        return None
    if isinstance(token, (list, tuple)):
        if len(token) != 2:
            raise ValueError("align_xy requires exactly two values X,Y")
        return float(token[0]), float(token[1])
    parts = import_bootstrap.parse_comma_list(str(token))
    if len(parts) != 2:
        raise ValueError("--align-xy requires exactly two comma-separated values X,Y")
    return float(parts[0]), float(parts[1])


def parse_ms_shown_range(token, *, flag="--ms-shown"):
    """Parse ``START,STOP`` ms (comma; one token)."""
    parts = import_bootstrap.parse_comma_list(token)
    if len(parts) != 2:
        raise ValueError(f"{flag} must be START,STOP")
    start, stop = float(parts[0]), float(parts[1])
    if start > stop:
        raise ValueError(f"{flag} START={start} > STOP={stop}")
    return start, stop


def parse_param_init_val_tokens(tokens):
    try:
        return train.parse_param_init_val_tokens(tokens)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def filter_filename_suffix(filter=None):
    from train.cli import filter_filename_suffix as _suffix
    return _suffix(filter)


def param_filename_suffix(param_inits=None, param_vals=None):
    from train.cli import param_filename_suffix as _suffix
    return _suffix(param_inits=param_inits, param_vals=param_vals)


def override_params(z, schema, session, param_vals=None, param_inits=None,
                    param_clamps=None, param_jits=None):
    try:
        return train.override_params(
            z, schema, session,
            param_vals=param_vals, param_inits=param_inits,
            param_clamps=param_clamps, param_jits=param_jits,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def plot_trained_run(
    outdir,
    *,
    figure_kwargs,
    euler=None,
    filter=None,
    sti_timing=None,
    param_tokens=None,
):
    """Re-plot one trained run (shared by Hydra ``figure.plot`` main)."""
    import train.implementation as train_mod
    from train.cli import resolve_sti_timing_kwargs

    param_inits, param_vals, param_clamps, param_jits = parse_param_init_val_tokens(
        param_tokens or [],
    )
    timing_kwargs = resolve_sti_timing_kwargs(sti_timing)
    session, z, best_cost = load_best(outdir, verbose=True)
    session, z, timing_changed = override_session_sti_timing(
        run_dir=outdir, session=session, z=z, **timing_kwargs,
        euler=euler,
        filter=filter,
    )
    z = (
        z if torch.is_tensor(z)
        else torch.tensor(np.asarray(z, dtype=np.float64), dtype=torch.float64,
                          device=session.device)
    )
    z, schema = override_params(
        z, train.schema_copy(session.schema), session,
        param_vals=param_vals, param_inits=param_inits,
        param_clamps=param_clamps, param_jits=param_jits,
    )
    session = session.with_schema(schema)
    file_suffix = (
        sti_timing_filename_suffix(**timing_changed)
        + euler_filename_suffix(euler)
        + param_filename_suffix(param_inits=param_inits, param_vals=param_vals)
    )
    model = resolve_model(outdir)
    z = z.detach().cpu().numpy()
    print(f'params={train_mod.best_param_path(outdir)}')
    print(f'model={model} ({z.shape[-1]} params)')
    plot_rf_t(
        np.atleast_2d(z),
        outdir,
        model=model,
        session=session,
        final_costs=np.array([best_cost]),
        file_suffix=file_suffix,
        save_data=not (param_inits or param_vals),
        **figure_kwargs,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    from config import (
        FIGURE_PLOT,
        RUN_PATH,
        active_config,
        apply_config,
        resolve_figure_kwargs,
    )

    apply_config(cfg)
    figure_kwargs = resolve_figure_kwargs(cfg)
    outdir = resolve_run_dir(RUN_PATH)
    plot_trained_run(
        outdir,
        figure_kwargs=figure_kwargs,
        euler=FIGURE_PLOT.get("euler"),
        filter=FIGURE_PLOT.get("filter"),
        sti_timing=FIGURE_PLOT.get("sti_timing"),
        param_tokens=list(active_config().get("param_tokens") or []),
    )


if __name__ == '__main__':
    main()
