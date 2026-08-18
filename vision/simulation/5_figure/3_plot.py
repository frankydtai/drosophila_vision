"""Simulation + plotting for the FiveCol medulla model."""
from config import (
    RUN_PATH,
)
import json
import os
import numpy as np
import torch
import hydra
import matplotlib.pyplot as plt

import import_bootstrap  # noqa: F401
import train
from task.spot.sti_geo import resolve_spot
from figure import moving_bar
from figure import spot
from figure import spread
from figure.panel import (
    network_hex_count,
    session_filter_figure_token,
    ElapsedTimer,
    plot_cost_figure,
    plot_cost_total,
)
from train.session import run_data_dir
from train.implementation import resolve_run_dir
from network.construction import CELL_ROWS, cell_rows


def plot_cost(costs, path, *, costs_by_part=None, part_order=None):
    """Plot train cost; delegates part parsing to task figure modules."""
    from itertools import cycle, islice

    timer = ElapsedTimer()
    timer.end_prep()
    if costs is None or not hasattr(costs, "__len__") or len(costs) == 0:
        raise ValueError("plot_cost requires non-empty `costs`")

    if not costs_by_part:
        plot_cost_total(costs, path, timer=timer)
        return

    part_keys = list(part_order) if part_order else list(costs_by_part.keys())
    part_keys = [key for key in part_keys if key in costs_by_part and len(costs_by_part[key])]

    if not part_keys:
        plot_cost_total(costs, path, timer=timer)
        return

    spot_radii = set()
    pd_nd_labels = set()
    other_series_order = []
    other_series_seen = set()
    costs_by_cell = {}
    costs_global = []

    for part_key in part_keys:
        part_costs = np.asarray(costs_by_part[part_key], dtype=np.float64)
        if part_costs.size == 0:
            continue

        parsed = spot.spot_cost(part_key, part_costs)
        if parsed is not None:
            cell, series, label, part_costs = parsed
            spot_radii.add(series[1])
            costs_by_cell.setdefault(cell, []).append((series, label, part_costs))
            continue

        parsed = moving_bar.moving_bar_cost(part_key, part_costs)
        if parsed is not None:
            cell, series, label, part_costs = parsed
            pd_nd_labels.add(series[1])
            if cell is None:
                costs_global.append((series, label, part_costs))
            else:
                costs_by_cell.setdefault(cell, []).append((series, label, part_costs))
            continue

        series = ("other", part_key)
        if series not in other_series_seen:
            other_series_seen.add(series)
            other_series_order.append(series)
        known_cells = {cell for row in CELL_ROWS for cell in row}
        matches = [cell for cell in known_cells if cell and cell in part_key]
        if matches:
            cell = max(matches, key=len)
            costs_by_cell.setdefault(cell, []).append((series, part_key, part_costs))
        else:
            costs_global.append((series, part_key, part_costs))

    palette = list(plt.get_cmap("tab20").colors)
    series_order = []
    for radius in sorted(spot_radii, key=lambda value: (isinstance(value, float), value)):
        series_order.append(("spot_radius", radius))

    pd_nd_label_order = [
        pd_nd_label for pd_nd_label in ("PD", "ND", "DSI") if pd_nd_label in pd_nd_labels
    ]
    extra_pd_nd = sorted([
        pd_nd_label for pd_nd_label in pd_nd_labels if pd_nd_label not in pd_nd_label_order
    ])
    pd_nd_label_order.extend(extra_pd_nd)
    for pd_nd_label in pd_nd_label_order:
        series_order.append(("moving_bar_pd_nd", pd_nd_label))

    series_order.extend(other_series_order)

    color_from_series = dict(zip(
        series_order, islice(cycle(palette), len(series_order)),
    ))

    rows = cell_rows(sorted(costs_by_cell.keys()))
    plot_cost_figure(
        costs,
        path,
        costs_by_cell=costs_by_cell,
        costs_global=costs_global,
        series_order=series_order,
        color_from_series=color_from_series,
        rows=rows,
        timer=timer,
    )


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


def spread_readout_fns(session):
    return (
        spread.network_spread_trace_readout,
        spread.plot_network_spread_gt,
        spread.plot_network_spread_all,
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


def load_train_opts(run_dir):
    opts_path = os.path.join(
        run_data_dir(os.path.abspath(run_dir)), "train_opts.json",
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


def resolve_model(run_dir, model=None):
    if model is None:
        opts = load_train_opts(run_dir)
        if not opts or 'model' not in opts:
            raise SystemExit(
                f'cannot determine model for {run_dir!r}; '
                f'expected train_opts.json with "model" in '
                f'{train.MODELS}'
            )
        model = opts['model']
    if model not in train.MODELS:
        raise SystemExit(
            f'invalid model {model!r} in {run_dir!r}; '
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
    valid_run_idxs = np.where(valid_mask)[0]

    note = ''
    if final_costs is not None:
        final_costs = np.asarray(final_costs, dtype=np.float64)
        if final_costs.shape[0] != params.shape[0]:
            raise ValueError(
                f'final_costs length {final_costs.shape[0]} != params runs {params.shape[0]}',
            )
        valid_costs = final_costs[valid_mask]
        best = int(np.argmin(valid_costs))
        run = int(valid_run_idxs[best])
        selected = float(final_costs[run])
        note = ' (from saved final costs)'
    else:
        costs_out = []
        for row in valid:
            z = torch.tensor(row, dtype=session.sim_dtype, device=session.device)
            costs_out.append(train.calc_cost(z, session).item())
        valid_costs = np.array(costs_out)
        best = int(np.argmin(valid_costs))
        run = int(valid_run_idxs[best])
        selected = float(valid_costs[best])
    if verbose:
        print(f'{len(valid)} trained set(s); costs min={valid_costs.min():.4f} '
              f'max={valid_costs.max():.4f}; selected #{run}{note}')
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


def load_best(run_dir, *, model=None, verbose=False):
    """Load session and best ``z`` from a train run (``best_param.npz`` + costs)."""
    import train.implementation as train_mod

    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f'run dir not found: {run_dir}')
    model = resolve_model(run_dir, model=model)
    session, z = _session_z_from_best_param(train.session_from_run_dir(run_dir, model=model), run_dir)
    best_cost = None
    final_costs, _, _, _ = train_mod.load_stored_costs(run_dir)
    if final_costs is not None and len(final_costs) > 0:
        best_cost = float(final_costs[int(np.argmin(final_costs))])
    if best_cost is None:
        best_cost = train.calc_cost(z, session).item()
    if verbose:
        print(f'loaded best_param.npz (cost={best_cost:.4f})')
    return session, z, float(best_cost)


def override_session(
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
    """Re-open session when any ms / euler / filter override is set; remap best ``z``.

    Unset flags keep values from the run's train opts. ``ms_pre`` /
    ``delta_ms`` / ``delta_ms_pre`` also update moving_bar sti opts;
    ``ms_response`` / ``ms_post`` / ``ms_sti`` are spot-only. ``euler`` is
    CLI ``im``/``ex`` (or already expanded ``implicit``/``explicit``).
    ``filter`` is ``none``/``ca``.

    Returns ``(session, z, ms_changed)`` where ``ms_changed`` maps
    ms keys that differ from the run (for filename suffixes).
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
        raise SystemExit(f"delta_ms={delta_ms} must be > 0")
    if delta_ms_pre is not None and float(delta_ms_pre) <= 0:
        raise SystemExit(f"delta_ms_pre={delta_ms_pre} must be > 0")
    if ms_post is not None and float(ms_post) < 0:
        raise SystemExit("ms_post must be >= 0")

    opts = load_train_opts(run_dir)
    if opts is None:
        raise SystemExit(f"missing train opts under {run_dir}")

    ms_changed = {}
    if (
        ms_pre is not None
        or ms_response is not None
        or ms_post is not None
        or ms_sti is not None
        or delta_ms is not None
        or delta_ms_pre is not None
    ):
        pairs = (
            ("ms_pre", ms_pre),
            ("ms_post", ms_post),
            ("delta_ms", delta_ms),
            ("delta_ms_pre", delta_ms_pre),
            ("ms_response", ms_response),
            ("ms_sti", ms_sti),
        )
        val = lambda ms: None if ms is None else float(ms)

        def merge_ms(sti_opts, merge_pairs):
            before = {key: val(sti_opts.get(key)) for key, _ in merge_pairs}
            for key, ms in merge_pairs:
                if ms is not None:
                    sti_opts[key] = float(ms)
            ms_sti_val = sti_opts.get("ms_sti")
            ms_response_val = sti_opts.get("ms_response")
            if ms_sti_val is not None and ms_response_val is not None:
                sti_opts["ms_response"] = max(
                    float(ms_response_val), float(ms_sti_val),
                )
            sti_opts.pop("n_t", None)
            return {
                key: sti_opts[key]
                for key, _ in merge_pairs
                if val(sti_opts.get(key)) != before[key]
            }

        for sti_opts_key in ("spread_sti_opts", "spot_sti_opts"):
            sti_opts = opts.get(sti_opts_key)
            if sti_opts is not None:
                ms_changed.update(merge_ms(sti_opts, pairs))
        if ms_pre is not None or delta_ms is not None or delta_ms_pre is not None:
            sti_opts = opts.get("moving_bar_sti_opts")
            if sti_opts is not None:
                ms_changed.update(merge_ms(
                    sti_opts,
                    (
                        ("ms_pre", ms_pre),
                        ("ms_post", None),
                        ("delta_ms", delta_ms),
                        ("delta_ms_pre", delta_ms_pre),
                        ("ms_response", None),
                        ("ms_sti", None),
                    ),
                ))

    if euler is not None:
        opts["euler"] = train.expand_euler(euler)

    if filter is not None:
        opts["filter"] = str(filter)

    session = train.resolve_session(opts, model=opts.get("model"))
    session, z = _session_z_from_best_param(session, run_dir)
    return session, z, ms_changed


def _format_filename_token(value):
    val = float(value)
    if val == int(val):
        return str(int(val))
    return "%g" % val


def ms_filename_suffix(
    *,
    ms_pre=None,
    ms_sti=None,
    ms_response=None,
    ms_post=None,
    delta_ms=None,
    delta_ms_pre=None,
):
    """PNG stem suffix for ms keys that differ from the run (plot / analyze).

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


def param_filename_suffix(param_vals=None):
    parts = []
    for param, bag in (param_vals or {}).items():
        if isinstance(bag, dict):
            for node, number in bag.items():
                parts.append("_".join([
                    str(param), "val", str(node).replace(":", "_"),
                    _format_filename_token(number),
                ]))
        else:
            parts.append("_".join([
                str(param), "val", _format_filename_token(bag),
            ]))
    if not parts:
        return ""
    return "_" + "_".join(parts)


def _stored_cost_parts(run_dir):
    """Best-run per-part costs from ``costs_by_part.npz`` (no forward)."""
    import train.implementation as train_mod

    final_costs, _, _, final_costs_by_part = train_mod.load_stored_costs(run_dir)
    if not final_costs_by_part or final_costs is None or len(final_costs) == 0:
        return {}
    run = int(np.argmin(np.asarray(final_costs)))
    cost_parts = {}
    for part_key, costs in final_costs_by_part.items():
        costs = np.asarray(costs).reshape(-1)
        if run < costs.size:
            cost_parts[part_key] = float(costs[run])
    return cost_parts


def _plot_path(run_dir, stem, file_suffix="", *, html=False):
    return os.path.join(run_dir, f"{stem}{file_suffix}{'.html' if html else '.png'}")


def _readout_figure_stem(prefix, session):
    """``spot_gt_v`` / ``spot_gt_ca`` (filter chooses ``v`` or ``ca``, not both)."""
    return f"{prefix}_{session_filter_figure_token(session)}"


def _plot_spread_tasks(session, z, run_dir, suffix, model_all,
                       gts=None,
                       file_suffix="", html=False, ms_shown=None,
                       cost_parts=None):
    """Plot spread; both contrasts in one figure when session has bright and dark."""
    build_readout, plot_gt, plot_all = spread_readout_fns(session)
    figure_kwargs = dict(gts=gts, cost_parts=cost_parts)
    token = session_filter_figure_token(session)
    readout_kwargs = dict(ms_shown=ms_shown)
    contrasts = tuple(session.contrasts)
    if set(contrasts) >= {"bright", "dark"}:
        readouts = {
            contrast: build_readout(
                session_from_task(session, "spread", contrast), z, **readout_kwargs,
            )
            for contrast in ("bright", "dark")
        }
        gt_path = _plot_path(run_dir, _readout_figure_stem('spread_gt', session), file_suffix, html=html)
        plot_gt(
            gt_path, readouts=readouts,
            title=f'Spread {token}-gt ({suffix})',
            **figure_kwargs,
        )
        all_path = None
        if model_all:
            all_path = _plot_path(run_dir, _readout_figure_stem('spread_all', session), file_suffix, html=html)
            plot_all(
                all_path, readouts=readouts,
                title=f'Spread {token}-all ({suffix})',
                **figure_kwargs,
            )
        return gt_path, all_path
    for contrast in contrasts:
        one = session_from_task(session, "spread", contrast)
        readout = build_readout(one, z, **readout_kwargs)
        readouts = {contrast: readout}
        gt_path = _plot_path(run_dir, _readout_figure_stem('spread_gt', session), file_suffix, html=html)
        plot_gt(
            gt_path, readouts=readouts,
            title=f'spread {contrast} {token}-gt ({suffix})',
            **figure_kwargs,
        )
        if model_all:
            all_path = _plot_path(run_dir, _readout_figure_stem('spread_all', session), file_suffix, html=html)
            plot_all(
                all_path, readouts=readouts,
                title=f'spread {contrast} {token}-all ({suffix})',
                **figure_kwargs,
            )


def _plot_spot_tasks(session, z, run_dir, suffix, model_all,
                       gts=None,
                       at_x=None, at_y=None,
                       file_suffix="", html=False, ms_shown=None,
                       cost_parts=None):
    """Plot spot; both contrasts in one figure when session has bright and dark."""
    build_readout, plot_gt, plot_all = spot_readout_fns(session)
    net_label = _network_spot_label(session, "spot")
    figure_kwargs = dict(gts=gts, cost_parts=cost_parts)
    token = session_filter_figure_token(session)
    readout_kwargs = dict(
        at_xs=at_x, at_ys=at_y,
        ms_shown=ms_shown,
    )
    contrasts = tuple(session.contrasts)
    if set(contrasts) >= {"bright", "dark"}:
        readouts = {
            contrast: build_readout(
                session_from_task(session, "spot", contrast), z, **readout_kwargs,
            )
            for contrast in ("bright", "dark")
        }
        gt_path = _plot_path(run_dir, _readout_figure_stem('spot_gt', session), file_suffix, html=html)
        plot_gt(
            gt_path, readouts=readouts,
            title=f'Spot {token}-gt ({suffix}){net_label}',
            **figure_kwargs,
        )
        all_path = None
        if model_all:
            all_path = _plot_path(run_dir, _readout_figure_stem('spot_all', session), file_suffix, html=html)
            plot_all(
                all_path, readouts=readouts,
                title=f'Spot {token}-all ({suffix}){net_label}',
                **figure_kwargs,
            )
        return gt_path, all_path
    for contrast in contrasts:
        _plot_one_spot(
            session_from_task(session, "spot", contrast), z, run_dir, contrast, suffix, model_all,
            gts=gts,
            at_x=at_x, at_y=at_y,
            cost_parts=cost_parts,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
        )


def _plot_bar_readouts(session, z, run_dir, suffix, model_all, *,
                      plot_right_only=True, at_x=None, at_y=None,
                      align_at_x=None, align_at_y=None,
                      file_suffix="", html=False, ms_shown=None,
                      cost_parts=None):
    """Plot moving-bar; bright left | dark right when session has both contrasts."""
    token = session_filter_figure_token(session)
    readout_kwargs = dict(
        at_xs=at_x, at_ys=at_y,
        align_at_x=align_at_x, align_at_y=align_at_y,
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
        gt_path = _plot_path(run_dir, _readout_figure_stem('bar_gt', session), file_suffix, html=html)
        moving_bar.plot_moving_bar_gt(
            gt_path, readout=readout_bright, paired_readout=readout_dark,
            title=f'Moving-bar {token}-gt ({suffix})',
            cost_parts=cost_parts,
        )
        all_path = None
        if model_all:
            all_path = _plot_path(run_dir, _readout_figure_stem('bar_all', session), file_suffix, html=html)
            moving_bar.plot_moving_bar_all(
                all_path, readout=readout_bright, paired_readout=readout_dark,
                title=f'Moving-bar {token}-all ({suffix})',
                right_only=plot_right_only,
                cost_parts=cost_parts,
            )
        return gt_path, all_path
    for contrast in contrasts:
        one = session_from_task(session, "moving_bar", contrast)
        readout = moving_bar.moving_bar_trace_readout(
            one, z, "moving_bar", contrast, **readout_kwargs,
        )
        gt_path = _plot_path(run_dir, _readout_figure_stem('bar_gt', session), file_suffix, html=html)
        moving_bar.plot_moving_bar_gt(
            gt_path, readout=readout, title=f'moving_bar {contrast} {token}-gt ({suffix})',
            cost_parts=cost_parts,
        )
        all_path = None
        if model_all:
            all_path = _plot_path(run_dir, _readout_figure_stem('bar_all', session), file_suffix, html=html)
            moving_bar.plot_moving_bar_all(
                all_path, readout=readout, title=f'moving_bar {contrast} {token}-all ({suffix})',
                right_only=plot_right_only,
                cost_parts=cost_parts,
            )
        return gt_path, all_path


def _plot_one_spot(session, z, run_dir, contrast, suffix, model_all,
                     gts=None,
                     at_x=None, at_y=None,
                     cost_parts=None, file_suffix="", html=False, ms_shown=None):
    token = session_filter_figure_token(session)
    gt_path = _plot_path(run_dir, _readout_figure_stem('spot_gt', session), file_suffix, html=html)
    all_path = _plot_path(run_dir, _readout_figure_stem('spot_all', session), file_suffix, html=html)
    build_readout, plot_gt, plot_all = spot_readout_fns(session)
    net_label = _network_spot_label(session, "spot")
    figure_kwargs = dict(gts=gts, cost_parts=cost_parts)
    readout = build_readout(
        session, z,
        at_xs=at_x, at_ys=at_y,
        ms_shown=ms_shown,
    )
    readouts = {contrast: readout}
    plot_gt(
        gt_path, readouts=readouts,
        title=f'spot {contrast} {token}-gt ({suffix}){net_label}', **figure_kwargs,
    )
    if model_all:
        plot_all(
            all_path, readouts=readouts,
            title=f'spot {contrast} {token}-all ({suffix}){net_label}', **figure_kwargs,
        )
    return gt_path, all_path


def plot_rf_t(params, run_dir, model=None, model_all=True,
                   context_dir=None,
                   plot_tasks=None, session=None, *,
                   final_costs=None,
                   save_data=True,
                   gts=None,
                   plot_right_only=True, at_x=None, at_y=None,
                   align_at_x=None, align_at_y=None,
                   file_suffix="", html=False, ms_shown=None,
                   recompute_cost=False):
    import train.implementation as train_mod

    os.makedirs(run_dir, exist_ok=True)
    ctx = context_dir or run_dir
    if model is None and session is not None:
        model = session.model
    if model is None:
        raise ValueError('model or session required')
    if session is None:
        session = train.session_from_run_dir(ctx, model)

    params = np.atleast_2d(params)
    if final_costs is None:
        final_costs, _, _, _ = train_mod.load_stored_costs(run_dir)

    print(f'plot device={_plot_device_label()}')
    best, best_cost = select_best(
        params, session, final_costs=final_costs,
    )
    z = torch.tensor(best, dtype=session.sim_dtype, device=session.device)
    if recompute_cost:
        with torch.no_grad():
            parts = train.calc_cost_parts(z, session)
            cost_parts = {k: float(v.item()) for k, v in parts.items()}
            best_cost = float(train.calc_cost(z, session, parts=parts).item())
    else:
        cost_parts = _stored_cost_parts(run_dir)

    suffix = f'trained, cost {best_cost:.2f}'
    cost_norm = (session.train_opts or {}).get("cost_norm", "a_gt2")
    if cost_norm == "gt_power":
        suffix = f'{suffix}% of gt power'
    else:
        suffix = f'{suffix} ({cost_norm})'
    tasks = list(session.tasks)
    if plot_tasks is not None:
        tasks = [t for t in tasks if t in plot_tasks]

    has_spread = "spread" in tasks
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
            raise SystemExit('--align-xy applies to moving_bar --x/--y plots only')

    if has_spread:
        _plot_spread_tasks(
            session, z, run_dir, suffix, model_all,
            gts=gts,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
            cost_parts=cost_parts,
        )
    if has_spot:
        _plot_spot_tasks(
            session, z, run_dir, suffix, model_all,
            gts=gts,
            at_x=at_x, at_y=at_y,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
            cost_parts=cost_parts,
        )
    if has_moving_bar:
        _plot_bar_readouts(
            session, z, run_dir, suffix, model_all,
            plot_right_only=plot_right_only,
            at_x=at_x, at_y=at_y,
            align_at_x=align_at_x, align_at_y=align_at_y,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
            cost_parts=cost_parts,
        )

    if save_data:
        os.makedirs(run_data_dir(os.path.abspath(run_dir)), exist_ok=True)
        z_best = torch.tensor(best, dtype=session.sim_dtype, device=session.device)
        train_mod.save_best_param(run_dir, z_best, session)
    return best, best_cost


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


def parse_ms_shown_range(token, *, flag="ms_shown"):
    """Parse ``START,STOP`` ms (comma string or two-value sequence)."""
    if isinstance(token, (list, tuple)):
        parts = [str(x) for x in token]
    else:
        parts = import_bootstrap.parse_comma_list(token)
    if len(parts) != 2:
        raise ValueError(f"{flag} must be START,STOP")
    start, stop = float(parts[0]), float(parts[1])
    if start > stop:
        raise ValueError(f"{flag} START={start} > STOP={stop}")
    return start, stop


def override_params(z, schema, session, param_vals=None):
    try:
        return train.override_params(
            z, schema, session, param_vals=param_vals,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def plot_trained_run(
    run_dir,
    *,
    figure_kwargs,
    euler=None,
    filter=None,
    param_vals=None,
    ms_pre=None,
    ms_response=None,
    ms_post=None,
    ms_sti=None,
    delta_ms=None,
    delta_ms_pre=None,
):
    """Re-plot one trained run (shared by Hydra ``figure.plot`` main)."""
    import train.implementation as train_mod

    param_vals = param_vals or {}
    session, z, best_cost = load_best(run_dir, verbose=True)
    session, z, ms_changed = override_session(
        run_dir=run_dir, session=session, z=z,
        ms_pre=ms_pre,
        ms_response=ms_response,
        ms_post=ms_post,
        ms_sti=ms_sti,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        euler=euler,
        filter=filter,
    )
    z = (
        z if torch.is_tensor(z)
        else torch.tensor(np.asarray(z, dtype=np.float64), dtype=torch.float64,
                          device=session.device)
    )
    z, schema = override_params(
        z, train.schema_copy(session.schema), session, param_vals=param_vals,
    )
    session = session.with_schema(schema)
    recompute_cost = bool(
        ms_changed or euler is not None or filter is not None or param_vals
    )
    file_suffix = (
        ms_filename_suffix(**ms_changed)
        + euler_filename_suffix(euler)
        + param_filename_suffix(param_vals=param_vals)
    )
    model = resolve_model(run_dir)
    z = z.detach().cpu().numpy()
    print(f'params={train_mod.best_param_path(run_dir)}')
    print(f'model={model} ({z.shape[-1]} params)')
    plot_rf_t(
        np.atleast_2d(z),
        run_dir,
        model=model,
        session=session,
        final_costs=np.array([best_cost]),
        file_suffix=file_suffix,
        save_data=not param_vals,
        recompute_cost=recompute_cost,
        **figure_kwargs,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(hydra_config):
    from config import (
        active_config,
        apply_config,
        resolve_figure_kwargs,
        session_kwargs_from_cli,
    )

    apply_config(hydra_config)
    figure_kwargs = resolve_figure_kwargs(hydra_config)
    run_dir = resolve_run_dir(active_config()["run_path"])
    session_kwargs = session_kwargs_from_cli(hydra_config)
    plot_trained_run(
        run_dir,
        figure_kwargs=figure_kwargs,
        param_vals=dict(active_config().get("param_vals") or {}),
        **session_kwargs,
    )


if __name__ == '__main__':
    main()
