# -*- coding: utf-8 -*-
"""Staged-lr optimization loop (minimizes ``cost``).

Consumes :func:`train.cost.calc_cost_parts` / :func:`train.cost.backward_part_sums`
and returns in-memory :class:`TrainResult`. Persistence lives in
``train.implementation``.
"""
from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from train.cost import session_cost_part_keys
from train.cost import (
    backward_part_sums,
    calc_cost_parts,
    _scaled_cost_from_parts,
)
from train.param import (
    active_device,
    schema_clamps,
    schema_copy,
    schema_n_z,
    z_init_from_schema,
)
from train.session import TrainSession


@dataclass(frozen=True)
class TrainResult:
    """Output of :func:`do_many_runs` (in memory; persistence is ``train``)."""

    run_params: np.ndarray   # (n_run, n_z)
    final_costs: np.ndarray  # (n_run,) scaled total
    costs: np.ndarray   # per-step scaled total for ``argmin(final_costs)``
    costs_by_part: Dict[str, np.ndarray] = field(default_factory=dict)
    final_costs_by_part: Dict[str, np.ndarray] = field(default_factory=dict)
    # Per-run adams at best_z: exp_avg, exp_avg_sq (n_z,), iter (int).
    run_adams: tuple = ()


def _float_parts(parts: Optional[Dict[str, torch.Tensor]], task_order=None):
    """Tensor/number cost parts → ``{key: float}`` (optional key order)."""
    if not parts:
        return None
    parts = {k: float(v.item() if torch.is_tensor(v) else v) for k, v in parts.items()}
    if task_order:
        return {k: parts[k] for k in task_order if k in parts}
    return parts


def _fmt_cost_parts(parts, *, mid0_only=False):
    if not parts:
        return ""
    if mid0_only:
        parts = {k: v for k, v in parts.items() if k.endswith("mid+0")}
    return "  [" + "  ".join(f"{k}={v:.4f}" for k, v in parts.items()) + "]"


_TQDM_REFRESH_INTERVAL = 10


def gradient_network(z, lr=0.0001, cost_fn=None, n_iter=100, device="cpu", z_clamps=None,
                     cost_log=None, iter_log=None, float_last_parts=None, task_order=None,
                     backward_iter=None, eval_cost=None,
                     checkpoint_interval=None, on_interval_best=None, global_iter_start=0,
                     adam_init=None):

    a = time.time()

    z = nn.Parameter(z.clone().to(device))

    optimizer = torch.optim.Adam([z], lr=lr)
    if adam_init is not None:
        _load_adams(optimizer, z, adam_init)

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
    # Adams at the z that achieved interval_best / best (before the step that left it).
    interval_best_adam = _adam_bag(optimizer, z)
    best_adam = _adam_bag(optimizer, z)

    initial_cost = 1.0 * cost
    initial_parts = float_last_parts(task_order) if float_last_parts else None
    best_parts = initial_parts

    def _snapshot_interval_best(cost_value):
        nonlocal interval_best_cost, interval_best_z, interval_best_adam
        interval_best_cost = cost_value
        interval_best_z = z.clone().detach()
        interval_best_adam = _adam_bag(optimizer, z)

    def _interval_from_z():
        _snapshot_interval_best(_measure_cost(z))

    def _commit_interval_checkpoint(global_iter):
        if on_interval_best is not None:
            on_interval_best(
                global_iter, interval_best_z, interval_best_cost,
                adam=interval_best_adam,
            )
        with torch.no_grad():
            z.copy_(interval_best_z)
        _load_adams(optimizer, z, interval_best_adam)
        _interval_from_z()

    progress_bar = tqdm(
        range(n_iter),
        desc=f'Cost: {cost:.4f}' + _fmt_cost_parts(initial_parts, mid0_only=True),
        bar_format=(
            '{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
            '[{elapsed}<{remaining}, {rate_fmt}] {desc}'
        ),
        miniters=_TQDM_REFRESH_INTERVAL,
        maxinterval=60,
        file=sys.stderr,
    )
    aborted = None

    for iter in progress_bar:

        optimizer.zero_grad()

        try:
            if backward_iter is not None:
                cost = backward_iter(z)
            else:
                cost_t = cost_fn(z)
                cost = cost_t.item()
                cost_t.backward()
        except RuntimeError as e:
            aborted = f'iter {iter}: {e}'
            break

        if not np.isfinite(cost):
            aborted = f'iter {iter}: non-finite cost={cost}'
            break
        if not torch.isfinite(z).all():
            aborted = f'iter {iter}: non-finite z'
            break
        if z.grad is not None and not torch.isfinite(z.grad).all():
            aborted = f'iter {iter}: non-finite grad'
            break

        if cost < best_cost:

            best_cost = cost
            best_z = z.clone().detach()
            best_adam = _adam_bag(optimizer, z)
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

            z.clamp_(z_clamps[:, 0].to(device), z_clamps[:, 1].to(device))

        global_iter = global_iter_start + iter + 1
        if checkpoint_interval and global_iter % checkpoint_interval == 0:
            _commit_interval_checkpoint(global_iter)

        if (iter + 1) % _TQDM_REFRESH_INTERVAL == 0 or iter == n_iter - 1:
            progress_bar.set_description(
                f'Cost: {cost:.4f}' + _fmt_cost_parts(
                    float_last_parts(task_order) if float_last_parts else None,
                    mid0_only=True,
                ),
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
                best_adam = _adam_bag(optimizer, z)
                best_parts = final_parts
    else:
        cost = float('nan')
        final_parts = None

    print()
    if aborted is not None:
        print('ABORT:', aborted)
    print('Initl cost =', format(initial_cost, '.4f') + _fmt_cost_parts(initial_parts, mid0_only=True))
    print('Final cost =', format(cost, '.4f') + _fmt_cost_parts(final_parts, mid0_only=True))
    print('Best  cost =', format(best_cost, '.4f') + _fmt_cost_parts(best_parts, mid0_only=True))

    print('time needed  =', format(time.time() - a, '.2f'), ' sec')
    print()

    return best_z, best_adam


def adams_from_optimizer(optimizer, n_z, *, dtype, device):
    """Pull ``(exp_avg, exp_avg_sq, adam_iter)`` from a single-param Adam optimizer."""
    # Library ``state_dict()`` / key ``'state'`` — not a project naming token.
    adam_by_param = optimizer.state_dict().get('state') or {}
    if not adam_by_param:
        zeros = torch.zeros(n_z, dtype=dtype, device=device)
        return zeros, zeros.clone(), 0
    adam = next(iter(adam_by_param.values()))
    exp_avg = adam.get('exp_avg')
    exp_avg_sq = adam.get('exp_avg_sq')
    if exp_avg is None or exp_avg_sq is None:
        zeros = torch.zeros(n_z, dtype=dtype, device=device)
        return zeros, zeros.clone(), 0
    # torch.optim.Adam key is ``step`` (library); our name is ``adam_iter``.
    torch_step = adam.get('step', 0)
    adam_iter = int(torch_step.item()) if torch.is_tensor(torch_step) else int(torch_step)
    return (
        exp_avg.detach().to(device=device, dtype=dtype).clone(),
        exp_avg_sq.detach().to(device=device, dtype=dtype).clone(),
        adam_iter,
    )


def _adam_bag(optimizer, z):
    """``{exp_avg, exp_avg_sq, iter}`` snapshot for one ``z``."""
    exp_avg, exp_avg_sq, adam_iter = adams_from_optimizer(
        optimizer, int(z.numel()), dtype=z.dtype, device=z.device,
    )
    return {
        'exp_avg': exp_avg,
        'exp_avg_sq': exp_avg_sq,
        'iter': int(adam_iter),
    }


def _load_adams(optimizer, z, adam_init):
    """Install named/z adams into *optimizer* for parameter *z* (keep group lr)."""
    exp_avg = adam_init['exp_avg'].detach().to(device=z.device, dtype=z.dtype)
    exp_avg_sq = adam_init['exp_avg_sq'].detach().to(device=z.device, dtype=z.dtype)
    if exp_avg.shape != z.shape or exp_avg_sq.shape != z.shape:
        raise ValueError(
            f"adam shape {tuple(exp_avg.shape)}/{tuple(exp_avg_sq.shape)} "
            f"!= z shape {tuple(z.shape)}"
        )
    optimizer.state[z] = {
        'step': torch.tensor(float(adam_init.get('iter', 0)), dtype=torch.float32),
        'exp_avg': exp_avg.clone(),
        'exp_avg_sq': exp_avg_sq.clone(),
    }


def optimize_staged(z, cost_fn, z_clamps, lrs, niters, cost_log=None, iter_log=None,
                 float_last_parts=None, task_order=None,
                 backward_iter=None, eval_cost=None,
                 checkpoint_interval=None, on_interval_best=None, global_iter_start=0,
                 adam_init=None):
    global_iter = global_iter_start
    adam = None
    for stage_i, lr in enumerate(lrs):
        z, adam = gradient_network(
            z, lr=lr, n_iter=niters, device=active_device(),
            cost_fn=cost_fn, z_clamps=z_clamps, cost_log=cost_log,
            iter_log=iter_log, float_last_parts=float_last_parts,
            task_order=task_order,
            backward_iter=backward_iter, eval_cost=eval_cost,
            checkpoint_interval=checkpoint_interval,
            on_interval_best=on_interval_best,
            global_iter_start=global_iter,
            adam_init=adam_init if stage_i == 0 else None,
        )
        global_iter += niters
    return z, adam


def _build_iter_logger(session: TrainSession):
    """Build train iter hooks for :func:`gradient_network`."""
    part_keys = session_cost_part_keys(session)
    target_history = {part_key: [] for part_key in part_keys}
    _last_parts: Optional[Dict[str, float]] = None
    _last_total: Optional[float] = None

    def _set_last(parts, total):
        nonlocal _last_parts, _last_total
        _last_parts = dict(parts)
        _last_total = float(total)

    def _eval_parts(z, *, no_grad: bool):
        with (torch.no_grad() if no_grad else nullcontext()):
            parts = calc_cost_parts(z, session)
            total = _scaled_cost_from_parts(parts, session)
        _set_last({k: float(v.item()) for k, v in parts.items()}, float(total.item()))
        return total

    def cost_fn(z):
        return _eval_parts(z, no_grad=False)

    def eval_cost(z):
        return float(_eval_parts(z, no_grad=True).item())

    def backward_iter(z):
        total, part_sums = backward_part_sums(z, session)
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
                 z_init=None, adam_init=None, checkpoint_interval=None, checkpoint_run_dir=None,
                 build_checkpoint_callback=None, checkpoint_on_png=None) -> TrainResult:
    """Run ``n_run`` independent runs; return ``TrainResult`` (no file I/O)."""
    schema = schema_copy(session.schema)
    n_z = schema_n_z(schema)
    z_clamps = schema_clamps(schema, session.sim_dtype)

    run_params = np.zeros((n_run, n_z))
    run_adams = []
    final_costs = np.zeros(n_run)
    part_keys = session_cost_part_keys(session)
    final_costs_by_part = {part_key: np.zeros(n_run) for part_key in part_keys}
    cost_histories = [None] * n_run
    part_histories = [None] * n_run

    for run in range(n_run):
        print()
        print('round', run)
        print()

        z = z_init.clone() if z_init is not None else z_init_from_schema(schema, session.sim_dtype)
        cost_history = []
        (cost_fn, target_history, log_iter, float_last_parts,
         backward_iter, eval_cost) = _build_iter_logger(session)

        def iter_log(z):
            cost_history.append(log_iter(z))

        on_interval_best = None
        if checkpoint_interval is not None:
            if checkpoint_run_dir is None or build_checkpoint_callback is None:
                raise ValueError(
                    "checkpoint_interval requires checkpoint_run_dir and build_checkpoint_callback"
                )
            on_interval_best = build_checkpoint_callback(
                checkpoint_run_dir, session, run_i=run, n_run=n_run,
                on_png=checkpoint_on_png,
            )

        z_best, adam = optimize_staged(
            z, cost_fn, z_clamps, lrs, n_iter,
            iter_log=iter_log,
            float_last_parts=float_last_parts,
            task_order=list(part_keys),
            backward_iter=backward_iter,
            eval_cost=eval_cost,
            checkpoint_interval=checkpoint_interval,
            on_interval_best=on_interval_best,
            adam_init=adam_init,
        )

        run_params[run] = z_best.detach().cpu().numpy()
        run_adams.append({
            'exp_avg': adam['exp_avg'].detach().cpu().numpy().astype(np.float64),
            'exp_avg_sq': adam['exp_avg_sq'].detach().cpu().numpy().astype(np.float64),
            'iter': int(adam['iter']),
        })
        final_parts = calc_cost_parts(z_best, session)
        final_costs[run] = float(_scaled_cost_from_parts(final_parts, session).item())
        for part_key, part in final_parts.items():
            final_costs_by_part[part_key][run] = float(part.item())
        cost_histories[run] = np.array(cost_history, dtype=np.float64)
        part_histories[run] = {
            part_key: np.array(part_costs, dtype=np.float64)
            for part_key, part_costs in target_history.items()
        }

    run_i = int(np.argmin(final_costs)) if n_run else 0

    return TrainResult(
        run_params=run_params,
        final_costs=final_costs,
        costs=(
            cost_histories[run_i]
            if cost_histories[run_i] is not None
            else np.array([], dtype=np.float64)
        ),
        costs_by_part=part_histories[run_i] or {},
        final_costs_by_part=final_costs_by_part,
        run_adams=tuple(run_adams),
    )
