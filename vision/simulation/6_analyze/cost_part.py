"""One fine cost part at best ``z`` (SSE / denom / per-t breakdown).

Reuses train cost internals — does not reimplement MSE or entry grouping.

* ``gt_power``: ``cost = 100 * SSE / Σ w (a_gt·gt)²``
* ``a_gt2``: ``cost = Σ_i sse_entry_i / a_i²`` (no ×100)

Usage (from ``vision/simulation/``)::

    ../.venv/bin/python analyze/cost_part.py \\
      hp_lp/28703323-... --cell L4 --radius 0 --stride 5 --per-node

    ../.venv/bin/python analyze/cost_part.py \\
      hp_lp/28703323-... --part spot_bright_L4_r0 --list-parts

    ../.venv/bin/python analyze/cost_part.py \\
      hp_lp/28703323-... --cell L4 --radius 0 --cost-norm a_gt2 --stride 5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401
import numpy as np
import torch

import figure.plot as plot
import train
from train.config import COST_NORMS, expand_cost_norm, spot_cost_part_key
from train.cost import (
    _gather_cost_time,
    _parts_from_entries,
    _pack_cost_forward,
    _session_cost_norm,
    _spot_entries_by_part,
    _scaled_mse_terms,
    calc_cost_parts,
)
from train.param import params_from_z


def _resolve_part_key(args) -> str:
    if args.part:
        return str(args.part).strip()
    if args.cell is None or args.radius is None:
        raise SystemExit("need --part KEY, or both --cell and --radius")
    return spot_cost_part_key(args.task, str(args.cell), float(args.radius))


def _apply_cost_norm_override(session, cost_norm: str | None):
    """Return session whose train_opts use ``cost_norm`` (run default if None)."""
    if cost_norm is None:
        return session, _session_cost_norm(session)
    key = expand_cost_norm(cost_norm)
    opts = dict(session.train_opts or {})
    opts["cost_norm"] = key
    return replace(session, train_opts=opts), key


def cost_part(session, z, part_key: str) -> dict:
    """Build per-part tensors using the same forward + grouping as train cost."""
    params = params_from_z(z, session)
    cost_norm = _session_cost_norm(session)
    task = None
    for tname in train.SPOT_TASKS:
        if part_key.startswith(f"{tname}_"):
            task = tname
            break
    if task is None:
        raise SystemExit(
            f"part {part_key!r} is not a spot fine key "
            f"(expected prefix in {train.SPOT_TASKS})",
        )
    pack = session.pack_for(task)
    fwd = _pack_cost_forward(params, pack, session)
    if fwd is None:
        raise SystemExit(f"no cost forward for pack {task!r}")
    a_gt, bias_gt, gts, scale, v_readout, _v_readout_dsi, _pd_nd = fwd
    if v_readout is None:
        raise SystemExit(f"waveform v_readout required for {task!r}")
    v_readout, gts, time_mask = _gather_cost_time(pack, v_readout, gts)

    part_indices, keys = _spot_entries_by_part(pack, session.backend)
    if part_key not in keys:
        raise SystemExit(
            f"part {part_key!r} not in pack parts; available:\n  "
            + "\n  ".join(keys),
        )
    part_slot_idx = keys.index(part_key)
    entry_indices = (part_indices == part_slot_idx).nonzero(as_tuple=False).reshape(-1)
    if entry_indices.numel() == 0:
        raise SystemExit(f"part {part_key!r} has zero active entries")

    a_gt_part = a_gt[entry_indices]
    bias_gt_part = bias_gt[entry_indices]
    cost_scales_part = scale[entry_indices]
    mask_r = None if time_mask is None else time_mask[entry_indices]
    gt_scaled, cost_scales_2d, sse_wt, power_wt = _scaled_mse_terms(
        a_gt_part, bias_gt_part, gts[entry_indices], cost_scales_part,
        v_readout[entry_indices], time_mask=mask_r,
    )
    sse = sse_wt.sum(dim=0)
    power_t = power_wt.sum(dim=0)
    n = int(entry_indices.numel())
    n_t = int(sse.numel())
    w_sum = float(cost_scales_part.sum().item())
    if w_sum <= 0:
        raise SystemExit(f"part {part_key!r} has zero scale sum")
    a0 = float(a_gt_part[0].item())
    b0 = float(bias_gt_part[0].item())
    a2 = max(a0 * a0, float(torch.finfo(a_gt_part.dtype).tiny))
    sse_sum = float(sse.sum().item())
    power_sum = float(power_t.sum().item())

    official = _parts_from_entries(
        a_gt, bias_gt, gts, scale, v_readout, part_indices, keys, session,
        time_mask=time_mask,
    )
    if part_key not in official:
        raise SystemExit(f"part {part_key!r} has zero cost scale")
    cost = float(official[part_key].item())

    # Per-t display (/ w_sum); scalar ``cost`` is from ``_parts_from_entries``.
    if cost_norm == "gt_power":
        cost_mean = torch.where(
            power_t > 0,
            100.0 * sse / power_t / w_sum,
            torch.full_like(sse, float("nan")),
        )
        denom_t = power_t
        denom_name = "POWER"
    elif cost_norm == "a_gt2":
        cost_mean = sse / a2 / w_sum
        denom_t = torch.full_like(sse, a2)
        denom_name = "a_gt2"
    else:
        raise SystemExit(f"unsupported cost_norm {cost_norm!r}")

    gt_aff = gt_scaled + bias_gt_part[:, None]
    gt_aff_mean = (cost_scales_2d * gt_aff).sum(dim=0) / w_sum
    v_readout_mean = (cost_scales_2d * v_readout[entry_indices]).sum(dim=0) / w_sum
    sse_mean = sse / w_sum

    # ``t_cost`` / ``ms_cost``: post-onset cost samples. Bare ``t`` / ``ms``: absolute.
    delta_ms = float(session.delta_ms)
    delta_ms_pre = float(session.delta_ms_pre)
    t_onset = train.pack_t_onset(pack)
    if pack.cost_time_indices is None:
        t_cost = np.arange(n_t, dtype=np.int64)
        t = t_onset + t_cost
    else:
        t_cost = pack.cost_time_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if t_cost.shape[0] != n_t:
            raise SystemExit(
                f"cost_time_indices length {t_cost.shape[0]} != n_t {n_t}",
            )
        t = train.pack_cost_abs_time_idx(pack, t_onset)
    ms_cost = t_cost.astype(float) * delta_ms
    ms = np.array(
        [
            train.ms_from_t(
                int(ti),
                t_onset=t_onset,
                delta_ms_pre=delta_ms_pre,
                delta_ms=delta_ms,
            )
            for ti in t
        ],
        dtype=float,
    )

    return {
        "part_key": part_key,
        "task": task,
        "cost_norm": cost_norm,
        "n": n,
        "w_sum": w_sum,
        "n_t": n_t,
        "a_gt": a0,
        "bias_gt": b0,
        "a_gt2": a2,
        "sse_sum": sse_sum,
        "power_sum": power_sum,
        "cost": cost,
        "denom_name": denom_name,
        "sse_mean": sse_mean.detach().cpu().numpy(),
        "denom_t": denom_t.detach().cpu().numpy(),
        "cost_mean": cost_mean.detach().cpu().numpy(),
        "gt_aff_mean": gt_aff_mean.detach().cpu().numpy(),
        "v_readout_mean": v_readout_mean.detach().cpu().numpy(),
        "delta_ms": delta_ms,
        "delta_ms_pre": delta_ms_pre,
        "t_onset": t_onset,
        "t": t,
        "ms": ms,
        "t_cost": t_cost,
        "ms_cost": ms_cost,
    }


def _print_summary(info: dict) -> None:
    print(
        f"part={info['part_key']}  cost_norm={info['cost_norm']}  "
        f"N={info['n']}  n_t={info['n_t']}  "
        f"a_gt={info['a_gt']:.10f}  bias_gt={info['bias_gt']:.10f}",
        flush=True,
    )
    print(
        f"SSE={info['sse_sum']:.10f}  "
        f"ΣPOWER(=Σ w(a·gt)²)={info['power_sum']:.10f}  "
        f"a_gt²={info['a_gt2']:.10f}",
        flush=True,
    )
    print(f"cost={info['cost']:.10f}", flush=True)


def _print_table(info: dict, *, stride: int, per_node: bool) -> None:
    w_sum = info["w_sum"]
    denom_name = info["denom_name"]
    den_lab = f"{denom_name}_avg" if per_node and denom_name == "POWER" else denom_name
    print(
        f"{'t':>4} {'ms':>7} {'t_cost':>6} {'ms_cost':>7} "
        f"{'cost_mean':>14} {'gt_aff':>14} "
        f"{'SSE_mean':>14} {den_lab:>14}",
        flush=True,
    )
    t = info["t"]
    ms = info["ms"]
    t_cost = info["t_cost"]
    ms_cost = info["ms_cost"]
    for j in range(0, info["n_t"], max(1, int(stride))):
        c = info["cost_mean"][j]
        c_s = "nan" if not np.isfinite(c) else f"{c:.10f}"
        sse_mean = info["sse_mean"][j]
        den = info["denom_t"][j]
        if per_node and denom_name == "POWER":
            den = den / w_sum
        print(
            f"{int(t[j]):4d} {ms[j]:7g} {int(t_cost[j]):6d} {ms_cost[j]:7g} "
            f"{c_s:>14} {info['gt_aff_mean'][j]:14.10f} "
            f"{sse_mean:14.6f} {den:14.6f}",
            flush=True,
        )


def _save_csv(path: str, info: dict, *, per_node: bool) -> None:
    w_sum = info["w_sum"]
    t = info["t"]
    ms = info["ms"]
    t_cost = info["t_cost"]
    ms_cost = info["ms_cost"]
    denom_name = info["denom_name"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t", "ms", "t_cost", "ms_cost",
                "cost_mean", "gt_aff_mean", "v_readout_mean",
                "SSE_mean", denom_name,
                f"{denom_name}_avg" if denom_name == "POWER" else denom_name,
                "cost_norm", "part",
            ],
        )
        for j in range(info["n_t"]):
            c = info["cost_mean"][j]
            c_s = "" if not np.isfinite(c) else f"{c:.10f}"
            sse_mean = float(info["sse_mean"][j])
            den = float(info["denom_t"][j])
            den_avg = den / w_sum if denom_name == "POWER" else den
            w.writerow(
                [
                    int(t[j]), f"{ms[j]:g}",
                    int(t_cost[j]), f"{ms_cost[j]:g}",
                    c_s,
                    f"{info['gt_aff_mean'][j]:.10f}",
                    f"{info['v_readout_mean'][j]:.10e}",
                    f"{sse_mean:.10f}", f"{den:.10f}",
                    f"{den_avg:.10f}",
                    info["cost_norm"], info["part_key"],
                ],
            )
    print(f"wrote {path}", flush=True)


def _list_parts(session, z) -> None:
    parts = calc_cost_parts(z, session)
    print(f"cost_norm={_session_cost_norm(session)}  n_parts={len(parts)}", flush=True)
    for k in sorted(parts):
        print(f"  {k}={float(parts[k].item()):.6f}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run",
        help="run folder under PARAMETER_DIR or absolute path",
    )
    ap.add_argument(
        "--part",
        default=None,
        help="fine part key, e.g. spot_bright_L4_r0",
    )
    ap.add_argument(
        "--task",
        default="spot_bright",
        choices=list(train.SPOT_TASKS),
        help="spot task for --cell/--radius (default: spot_bright)",
    )
    ap.add_argument("--cell", default=None, help="cell type, e.g. L4")
    ap.add_argument(
        "--radius",
        type=float,
        default=None,
        help="spot cost radius, e.g. 0",
    )
    ap.add_argument(
        "--cost-norm",
        default=None,
        choices=list(COST_NORMS),
        help="override run train_opts.cost_norm (default: keep run)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="print every N-th time index (default: 1 = all)",
    )
    ap.add_argument(
        "--per-node",
        action="store_true",
        help="print POWER/W for gt_power denom (SSE_mean always uses W=Σ w_i)",
    )
    ap.add_argument(
        "--csv",
        default=None,
        metavar="PATH",
        help="write full per-t table to CSV",
    )
    ap.add_argument(
        "--list-parts",
        action="store_true",
        help="print all fine part costs and exit (or also report if part set)",
    )
    args = ap.parse_args(argv)

    run_dir = plot.resolve_run_dir(args.run)
    session, z, best_cost = plot.load_best(run_dir, verbose=True)
    session, cost_norm = _apply_cost_norm_override(session, args.cost_norm)
    print(f"cost_norm={cost_norm}  saved_total={best_cost:.6f}", flush=True)

    if args.list_parts and args.part is None and args.cell is None:
        _list_parts(session, z)
        return

    if args.list_parts:
        _list_parts(session, z)
        print(flush=True)

    part_key = _resolve_part_key(args)
    info = cost_part(session, z, part_key)
    _print_summary(info)
    print(flush=True)
    _print_table(info, stride=max(1, args.stride), per_node=bool(args.per_node))
    if args.csv:
        _save_csv(args.csv, info, per_node=bool(args.per_node))


if __name__ == "__main__":
    main()
