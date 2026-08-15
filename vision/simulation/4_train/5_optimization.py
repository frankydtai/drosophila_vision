# -*- coding: utf-8 -*-
"""Adam / staged-lr optimization loop (minimizes ``cost``).

Consumes :func:`train.cost.calc_cost_parts` / :func:`train.cost.backward_accumulate_scaled_cost`
and returns in-memory :class:`TrainResult`. Persistence lives in
``train.implementation``.
"""
from __future__ import annotations

import copy
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from train.config import session_cost_part_keys
from train.cost import (
    backward_accumulate_scaled_cost,
    calc_cost_parts,
    _scaled_cost_from_parts,
)
from train.param import (
    active_device,
    schema_bounds,
    schema_guess,
    schema_nparams,
)
from train.session import TrainSession


@dataclass(frozen=True)
class TrainResult:
    """Output of :func:`do_many_runs` (in memory; persistence is ``train``)."""

    run_params: np.ndarray   # (n_run, n_params)
    final_costs: np.ndarray  # (n_run,) scaled total
    cost_curve: np.ndarray   # per-step scaled total for ``argmin(final_costs)``
    cost_curves_by_part: Dict[str, np.ndarray] = field(default_factory=dict)
    final_costs_by_part: Dict[str, np.ndarray] = field(default_factory=dict)
    # Per-run moments at best_z: exp_avg, exp_avg_sq (n_params,), step (int).
    run_adams: tuple = ()


def _float_parts(parts: Optional[Dict[str, torch.Tensor]], task_order=None):
    """Tensor/number cost parts → ``{key: float}`` (optional key order)."""
    if not parts:
        return None
    out = {k: float(v.item() if torch.is_tensor(v) else v) for k, v in parts.items()}
    if task_order:
        return {k: out[k] for k in task_order if k in out}
    return out


def _fmt_cost_parts(parts):
    if not parts:
        return ""
    return "  [" + "  ".join(f"{k}={v:.4f}" for k, v in parts.items()) + "]"


_TQDM_REFRESH_INTERVAL = 10


def gradient_network(z, lr=0.0001, cost_fn=None, n_iters=100, device="cpu", z_bounds=None,
                     cost_log=None, iter_log=None, float_last_parts=None, task_order=None,
                     backward_iter=None, eval_cost=None,
                     checkpoint_interval=None, on_interval_best=None, global_iter_start=0,
                     opt_init=None):

    a = time.time()

    z = nn.Parameter(z.clone().to(device))

    optimizer = torch.optim.Adam([z], lr=lr)
    if opt_init is not None:
        _load_moments(optimizer, z, opt_init)

    def _measure_cost(param_z):
        if eval_cost is not None:
            return eval_cost(param_z)
        return cost_fn(param_z).item()

    try:
        cost = _measure_cost(z)
    except RuntimeError as e:
        raise RuntimeError(f'non-finite at init: {e}') from e
    if not np.isfinite(cost):
        raise RuntimeError(f'non-finite cost at init: {cost}')
    best_cost = cost
    best_z = z.clone().detach()
    interval_best_cost = cost
    interval_best_z = z.clone().detach()
    # Adam m/v at the z that achieved interval_best / best (before the step that left it).
    interval_best_opt = copy.deepcopy(optimizer.state_dict())
    best_opt = copy.deepcopy(optimizer.state_dict())

    initial_cost = 1.0 * cost
    initial_parts = float_last_parts(task_order) if float_last_parts else None
    best_parts = initial_parts

    def _snapshot_interval_best(cost_value):
        nonlocal interval_best_cost, interval_best_z, interval_best_opt
        interval_best_cost = cost_value
        interval_best_z = z.clone().detach()
        interval_best_opt = copy.deepcopy(optimizer.state_dict())

    def _interval_from_z():
        _snapshot_interval_best(_measure_cost(z))

    def _commit_interval_checkpoint(global_iter):
        if on_interval_best is not None:
            on_interval_best(
                global_iter, interval_best_z, interval_best_cost,
                opt_state=interval_best_opt,
            )
        with torch.no_grad():
            z.copy_(interval_best_z)
        optimizer.load_state_dict(interval_best_opt)
        _interval_from_z()

    progress_bar = tqdm(
        range(n_iters),
        desc=f'Cost: {cost:.4f}' + _fmt_cost_parts(initial_parts),
        bar_format=(
            '{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
            '[{elapsed}<{remaining}, {rate_fmt}] {desc}'
        ),
        miniters=_TQDM_REFRESH_INTERVAL,
        maxinterval=60,
        file=sys.stderr,
    )
    aborted = None

    for i in progress_bar:

        optimizer.zero_grad()

        try:
            if backward_iter is not None:
                cost = backward_iter(z)
            else:
                cost_t = cost_fn(z)
                cost = cost_t.item()
                cost_t.backward()
        except RuntimeError as e:
            aborted = f'iter {i}: {e}'
            break

        if not np.isfinite(cost):
            aborted = f'iter {i}: non-finite cost={cost}'
            break
        if not torch.isfinite(z).all():
            aborted = f'iter {i}: non-finite z'
            break
        if z.grad is not None and not torch.isfinite(z.grad).all():
            aborted = f'iter {i}: non-finite grad'
            break

        if cost < best_cost:

            best_cost = cost
            best_z = z.clone().detach()
            best_opt = copy.deepcopy(optimizer.state_dict())
            if float_last_parts is not None:
                best_parts = float_last_parts(task_order)

        if cost < interval_best_cost:
            _snapshot_interval_best(cost)

        if cost_log is not None:
            cost_log.append(cost)
        if iter_log is not None:
            iter_log(z)

        optimizer.step()

        with torch.no_grad():

            z.clamp_(z_bounds[:, 0].to(device), z_bounds[:, 1].to(device))

        global_iter = global_iter_start + i + 1
        if checkpoint_interval and global_iter % checkpoint_interval == 0:
            _commit_interval_checkpoint(global_iter)

        iter_parts = float_last_parts(task_order) if float_last_parts else None
        if (i + 1) % _TQDM_REFRESH_INTERVAL == 0 or i == n_iters - 1:
            progress_bar.set_description(
                f'Cost: {cost:.4f}' + _fmt_cost_parts(iter_parts),
                refresh=False,
            )

    if aborted is None:
        try:
            cost = _measure_cost(z)
            final_parts = float_last_parts(task_order) if float_last_parts else None
        except RuntimeError as e:
            aborted = f'final eval: {e}'
            cost = float('nan')
            final_parts = None
        else:
            if np.isfinite(cost) and cost < best_cost:
                best_cost = cost
                best_z = z.clone().detach()
                best_opt = copy.deepcopy(optimizer.state_dict())
                best_parts = final_parts
    else:
        cost = float('nan')
        final_parts = None

    print()
    if aborted is not None:
        print('ABORT:', aborted)
    print('Initl cost =', format(initial_cost, '.4f') + _fmt_cost_parts(initial_parts))
    print('Final cost =', format(cost, '.4f') + _fmt_cost_parts(final_parts))
    print('Best  cost =', format(best_cost, '.4f') + _fmt_cost_parts(best_parts))

    b = time.time()

    print('time needed  =', format(b - a, '.2f'), ' sec')
    print()

    return best_z, best_opt


def moments_from_state_dict(state_dict, n_params, *, dtype, device):
    """Pull ``(exp_avg, exp_avg_sq, iter)`` from a single-param Adam ``state_dict``."""
    state = state_dict.get('state') or {}
    if not state:
        zeros = torch.zeros(n_params, dtype=dtype, device=device)
        return zeros, zeros.clone(), 0
    pstate = next(iter(state.values()))
    exp_avg = pstate.get('exp_avg')
    exp_avg_sq = pstate.get('exp_avg_sq')
    if exp_avg is None or exp_avg_sq is None:
        zeros = torch.zeros(n_params, dtype=dtype, device=device)
        return zeros, zeros.clone(), 0
    # torch.optim.Adam state key is ``step`` (library); our name is ``iter``.
    torch_step = pstate.get('step', 0)
    adam_iter = int(torch_step.item()) if torch.is_tensor(torch_step) else int(torch_step)
    return (
        exp_avg.detach().to(device=device, dtype=dtype).clone(),
        exp_avg_sq.detach().to(device=device, dtype=dtype).clone(),
        adam_iter,
    )


def _load_moments(optimizer, z, opt_init):
    """Install named/z moments into *optimizer* for parameter *z* (keep group lr)."""
    exp_avg = opt_init['exp_avg'].detach().to(device=z.device, dtype=z.dtype)
    exp_avg_sq = opt_init['exp_avg_sq'].detach().to(device=z.device, dtype=z.dtype)
    if exp_avg.shape != z.shape or exp_avg_sq.shape != z.shape:
        raise ValueError(
            f"moment shape {tuple(exp_avg.shape)}/{tuple(exp_avg_sq.shape)} "
            f"!= z shape {tuple(z.shape)}"
        )
    adam_iter = float(opt_init.get('iter', 0))
    # Match torch.optim.Adam: library key ``step`` is a CPU float scalar; moments match *z*.
    optimizer.state[z] = {
        'step': torch.tensor(adam_iter, dtype=torch.float32),
        'exp_avg': exp_avg.clone(),
        'exp_avg_sq': exp_avg_sq.clone(),
    }


def optimize_staged(z, cost_fn, z_bounds, lrs, niters, cost_log=None, iter_log=None,
                 float_last_parts=None, task_order=None,
                 backward_iter=None, eval_cost=None,
                 checkpoint_interval=None, on_interval_best=None, global_iter_start=0,
                 opt_init=None):
    global_iter = global_iter_start
    opt_state = None
    for stage_i, lr in enumerate(lrs):
        z, opt_state = gradient_network(
            z, lr=lr, n_iters=niters, device=active_device(),
            cost_fn=cost_fn, z_bounds=z_bounds, cost_log=cost_log,
            iter_log=iter_log, float_last_parts=float_last_parts,
            task_order=task_order,
            backward_iter=backward_iter, eval_cost=eval_cost,
            checkpoint_interval=checkpoint_interval,
            on_interval_best=on_interval_best,
            global_iter_start=global_iter,
            opt_init=opt_init if stage_i == 0 else None,
        )
        global_iter += niters
    return z, opt_state


def _build_iter_logger(session: TrainSession):
    """Build train iter hooks for :func:`gradient_network`."""
    part_keys = session_cost_part_keys(session.tasks, session=session)
    target_history = {part_key: [] for part_key in part_keys}
    _last_parts: Optional[Dict[str, float]] = None
    _last_total: Optional[float] = None

    def _set_last(parts, total):
        nonlocal _last_parts, _last_total
        _last_parts = dict(parts)
        _last_total = float(total)

    def _eval_parts(z, *, no_grad: bool):
        ctx = torch.no_grad() if no_grad else nullcontext()
        with ctx:
            parts = calc_cost_parts(z, session)
            total = _scaled_cost_from_parts(parts, session)
        _set_last({k: float(v.item()) for k, v in parts.items()}, float(total.item()))
        return total

    def cost_fn(z):
        return _eval_parts(z, no_grad=False)

    def eval_cost(z):
        return float(_eval_parts(z, no_grad=True).item())

    def backward_iter(z):
        total, part_sums = backward_accumulate_scaled_cost(z, session)
        _set_last(part_sums, total)
        return total

    def log_iter(z=None):
        if _last_parts is None or _last_total is None:
            raise RuntimeError("log_iter called before cost_fn in the same train iter")
        for part_key in part_keys:
            if part_key in _last_parts:
                target_history[part_key].append(float(_last_parts[part_key]))
            else:
                target_history[part_key].append(0.0)
        return float(_last_total)

    def float_last_parts(task_order=None):
        if _last_parts is None:
            raise RuntimeError("float_last_parts called before cost_fn")
        return _float_parts(_last_parts, task_order)

    if session.sequential:
        return cost_fn, target_history, log_iter, float_last_parts, backward_iter, eval_cost
    return cost_fn, target_history, log_iter, float_last_parts, None, None


def do_many_runs(session: TrainSession, n_run, n_iter, lrs=(0.1, 0.01, 0.001),
                 z_init=None, opt_init=None, checkpoint_interval=None, checkpoint_outdir=None,
                 build_checkpoint_callback=None, checkpoint_on_png=None) -> TrainResult:
    """Run ``n_run`` independent runs; return arrays (no file I/O)."""
    schema = train.schema_copy(session.schema)
    n_params = schema_nparams(schema)
    bounds = schema_bounds(schema, session.sim_dtype)

    run_params = np.zeros((n_run, n_params))
    run_adams = []
    final_costs = np.zeros(n_run)
    part_keys = session_cost_part_keys(session.tasks, session=session)
    final_costs_by_part = {part_key: np.zeros(n_run) for part_key in part_keys}
    cost_histories = [None] * n_run
    part_histories = [None] * n_run

    for i in range(n_run):
        print()
        print('round', i)
        print()

        z = z_init.clone() if z_init is not None else schema_guess(schema, session.sim_dtype)
        cost_history = []
        (cost_fn, target_history, log_iter, float_last_parts,
         backward_iter, eval_cost) = _build_iter_logger(session)

        def iter_log(z):
            cost_history.append(log_iter(z))

        on_interval_best = None
        if checkpoint_interval is not None:
            if checkpoint_outdir is None or build_checkpoint_callback is None:
                raise ValueError(
                    "checkpoint_interval requires checkpoint_outdir and build_checkpoint_callback"
                )
            on_interval_best = build_checkpoint_callback(
                checkpoint_outdir, session, run_i=i, n_run=n_run,
                on_png=checkpoint_on_png,
            )

        z_best, opt_state = optimize_staged(
            z, cost_fn, bounds, lrs, n_iter,
            iter_log=iter_log,
            float_last_parts=float_last_parts,
            task_order=list(part_keys),
            backward_iter=backward_iter,
            eval_cost=eval_cost,
            checkpoint_interval=checkpoint_interval,
            on_interval_best=on_interval_best,
            opt_init=opt_init,
        )

        run_params[i] = z_best.detach().cpu().numpy()
        exp_avg, exp_avg_sq, adam_iter = moments_from_state_dict(
            opt_state, n_params, dtype=z_best.dtype, device='cpu',
        )
        run_adams.append({
            'exp_avg': exp_avg.numpy().astype(np.float64),
            'exp_avg_sq': exp_avg_sq.numpy().astype(np.float64),
            'iter': int(adam_iter),
        })
        final_parts = calc_cost_parts(z_best, session)
        final_costs[i] = float(_scaled_cost_from_parts(final_parts, session).item())
        for part_key, part in final_parts.items():
            final_costs_by_part[part_key][i] = float(part.item())
        cost_histories[i] = np.array(cost_history, dtype=np.float64)
        part_histories[i] = {
            part_key: np.array(curve, dtype=np.float64)
            for part_key, curve in target_history.items()
        }

    run_i = int(np.argmin(final_costs)) if n_run else 0
    cost_curve = (
        cost_histories[run_i]
        if cost_histories[run_i] is not None
        else np.array([], dtype=np.float64)
    )
    cost_curves_by_part = part_histories[run_i] or {}

    return TrainResult(
        run_params=run_params,
        final_costs=final_costs,
        cost_curve=cost_curve,
        cost_curves_by_part=cost_curves_by_part,
        final_costs_by_part=final_costs_by_part,
        run_adams=tuple(run_adams),
    )
