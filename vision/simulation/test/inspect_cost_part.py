"""Inspect one fine cost part at best ``z`` (SSE / denom / per-t breakdown).

Reuses training cost internals — does not reimplement MSE or entry grouping.

* ``gt_power``: ``cost = 100 * SSE / Σ w (a_gt·gt)²``
* ``a_gt2``: ``cost = Σ_i sse_entry_i / a_i²`` (no ×100)

Usage (from ``vision/simulation/``)::

    ../.venv/bin/python test/inspect_cost_part.py \\
      hp_lp/28703323-... --cell L4 --radius 0 --stride 5 --per-node

    ../.venv/bin/python test/inspect_cost_part.py \\
      hp_lp/28703323-... --part spot_bright_L4_r0 --list-parts

    ../.venv/bin/python test/inspect_cost_part.py \\
      hp_lp/28703323-... --cell L4 --radius 0 --cost-norm a_gt2 --stride 5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401
import numpy as np
import torch

import figure.plot_run as plot_run
import training
from training.config import COST_NORMS, expand_cost_norm, spot_cost_part_key
from training.cost import (
    _pack_cost_forward,
    _session_cost_norm,
    _spot_entry_groups,
    calc_cost_parts,
)
from training.params import params_from_z


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
    return session.with_train_opts(opts) if hasattr(session, "with_train_opts") else _session_with_opts(
        session, opts,
    ), key


def _session_with_opts(session, opts: dict):
    """Fallback when TrainSession has no ``with_train_opts``."""
    from dataclasses import replace

    return replace(session, train_opts=opts)


def _subsample_cost_time(pack, v_readout, gt):
    """Match ``_pack_cost_parts_from_v_readout`` sparse ``cost_time_ix`` gather."""
    if pack.cost_time_ix is None:
        return v_readout, gt
    ix = pack.cost_time_ix.to(device=v_readout.device)
    return v_readout.index_select(1, ix), gt.index_select(1, ix)


def _delta_ms_for_pack(session, pack) -> float:
    opts = (session.train_opts or {}).get(f"{pack.name}_stimulus_opts") or {}
    if "delta_ms" in opts:
        return float(opts["delta_ms"])
    return float(getattr(session, "delta_ms", 5.0))


def inspect_part(session, z, part_key: str) -> dict:
    """Build per-part tensors using the same forward + grouping as training cost."""
    p = params_from_z(z, session)
    cost_norm = _session_cost_norm(session)
    # Spot fine keys: ``{task}_{cell}_r{radius}``.
    task = None
    for tname in training.SPOT_TASKS:
        if part_key.startswith(f"{tname}_"):
            task = tname
            break
    if task is None:
        raise SystemExit(
            f"part {part_key!r} is not a spot fine key "
            f"(expected prefix in {training.SPOT_TASKS})",
        )
    pack = session.pack_for(task)
    fwd = _pack_cost_forward(p, pack, session)
    if fwd is None:
        raise SystemExit(f"no cost forward for pack {task!r}")
    a_gt, bias_gt, gt, weight, v_readout, _v_readout_dsi, _pd_nd = fwd
    if v_readout is None:
        raise SystemExit(f"waveform v_readout required for {task!r}")
    v_readout, gt = _subsample_cost_time(pack, v_readout, gt)

    group_id, keys = _spot_entry_groups(pack, session.backend)
    if part_key not in keys:
        raise SystemExit(
            f"part {part_key!r} not in pack groups; available:\n  "
            + "\n  ".join(keys),
        )
    g = keys.index(part_key)
    idx = (group_id == g).nonzero(as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        raise SystemExit(f"part {part_key!r} has zero active entries")

    a = a_gt[idx]
    b = bias_gt[idx]
    w = weight[idx]
    gtr = gt[idx]
    v_readout_r = v_readout[idx]
    gt_scaled = a[:, None] * gtr
    gt_aff = gt_scaled + b[:, None]
    diff = v_readout_r - gt_aff

    sse = (w[:, None] * diff ** 2).sum(dim=0)
    power_t = (w[:, None] * gt_scaled ** 2).sum(dim=0)
    n = int(idx.numel())
    n_t = int(sse.numel())
    w_sum = float(w.sum().item())
    if w_sum <= 0:
        raise SystemExit(f"part {part_key!r} has zero total weight")
    a0 = float(a[0].item())
    b0 = float(b[0].item())
    a2 = max(a0 * a0, float(torch.finfo(a.dtype).tiny))

    sse_sum = float(sse.sum().item())
    power_sum = float(power_t.sum().item())
    if cost_norm == "gt_power":
        denom = power_sum if power_sum > 0 else 1.0
        cost = 100.0 * sse_sum / denom
        cost_mean = torch.where(
            power_t > 0,
            100.0 * sse / power_t / w_sum,
            torch.full_like(sse, float("nan")),
        )
        denom_t = power_t
        denom_name = "POWER"
    elif cost_norm == "a_gt2":
        cost = sse_sum / a2
        cost_mean = sse / a2 / w_sum
        denom_t = torch.full_like(sse, a2)
        denom_name = "a_gt2"
    else:
        raise SystemExit(f"unsupported cost_norm {cost_norm!r}")

    gt_aff_mean = (w[:, None] * gt_aff).sum(dim=0) / w_sum
    v_readout_mean = (w[:, None] * v_readout_r).sum(dim=0) / w_sum

    official = calc_cost_parts(z, session)
    official_val = float(official[part_key].item()) if part_key in official else None

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
        "official": official_val,
        "denom_name": denom_name,
        "sse": sse.detach().cpu().numpy(),
        "denom_t": denom_t.detach().cpu().numpy(),
        "cost_mean": cost_mean.detach().cpu().numpy(),
        "gt_aff_mean": gt_aff_mean.detach().cpu().numpy(),
        "v_readout_mean": v_readout_mean.detach().cpu().numpy(),
        "delta_ms": _delta_ms_for_pack(session, pack),
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
    if info["official"] is not None:
        ok = np.isclose(info["cost"], info["official"], rtol=1e-6, atol=1e-8)
        print(
            f"calc_cost_parts[{info['part_key']}]={info['official']:.10f}  "
            f"match={ok}",
            flush=True,
        )


def _print_table(info: dict, *, stride: int, per_node: bool) -> None:
    w_sum = info["w_sum"]
    denom_name = info["denom_name"]
    den_lab = f"{denom_name}_avg" if per_node and denom_name == "POWER" else denom_name
    print(
        f"{'t':>4} {'ms':>7} {'cost_mean':>14} {'gt_aff':>14} "
        f"{'SSE_mean':>14} {den_lab:>14}",
        flush=True,
    )
    ms = np.arange(info["n_t"], dtype=float) * float(info["delta_ms"])
    for t in range(0, info["n_t"], max(1, int(stride))):
        c = info["cost_mean"][t]
        c_s = "nan" if not np.isfinite(c) else f"{c:.10f}"
        sse_mean = info["sse"][t] / w_sum
        den = info["denom_t"][t]
        if per_node and denom_name == "POWER":
            den = den / w_sum
        print(
            f"{t:4d} {ms[t]:7g} {c_s:>14} {info['gt_aff_mean'][t]:14.10f} "
            f"{sse_mean:14.6f} {den:14.6f}",
            flush=True,
        )


def _write_csv(path: str, info: dict, *, per_node: bool) -> None:
    w_sum = info["w_sum"]
    ms = np.arange(info["n_t"], dtype=float) * float(info["delta_ms"])
    denom_name = info["denom_name"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t", "ms", "cost_mean", "gt_aff_mean", "v_readout_mean",
                "SSE_mean", denom_name,
                f"{denom_name}_avg" if denom_name == "POWER" else denom_name,
                "cost_norm", "part",
            ],
        )
        for t in range(info["n_t"]):
            c = info["cost_mean"][t]
            c_s = "" if not np.isfinite(c) else f"{c:.10f}"
            sse_mean = float(info["sse"][t]) / w_sum
            den = float(info["denom_t"][t])
            den_avg = den / w_sum if denom_name == "POWER" else den
            w.writerow(
                [
                    t, f"{ms[t]:g}", c_s,
                    f"{info['gt_aff_mean'][t]:.10f}",
                    f"{info['v_readout_mean'][t]:.10e}",
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
        choices=list(training.SPOT_TASKS),
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
        help="print all fine part costs and exit (or also inspect if part set)",
    )
    args = ap.parse_args(argv)

    run_dir = plot_run.resolve_run_dir(args.run)
    print(f"run={run_dir}", flush=True)
    session, z, best_cost = plot_run.load_best(run_dir, verbose=True)
    session, cost_norm = _apply_cost_norm_override(session, args.cost_norm)
    print(f"cost_norm={cost_norm}  saved_total={best_cost:.6f}", flush=True)

    if args.list_parts and args.part is None and args.cell is None:
        _list_parts(session, z)
        return

    if args.list_parts:
        _list_parts(session, z)
        print(flush=True)

    part_key = _resolve_part_key(args)
    info = inspect_part(session, z, part_key)
    _print_summary(info)
    print(flush=True)
    _print_table(info, stride=max(1, args.stride), per_node=bool(args.per_node))
    if args.csv:
        _write_csv(args.csv, info, per_node=bool(args.per_node))


if __name__ == "__main__":
    main()
