"""Classify spot radius-0 average ``v_post`` / ``ca`` on an absolute-ms window.

Uses ``analyze.cell_dynamics.analyze_spot_average`` only — one forward, no
rewrite of core dynamics. Series: ``filter=ca`` → report ``ca``, else
``v_post``. Spot time axis: ``0`` = trial start; onset ≈ ``ms_pre``. Analyze /
baseline windows are absolute ms (same as ``--ms-shown``), not locked to pre
vs pulse.

Checks
------
* ``oscillation``: FFT / v_peak_to_peak on ``--ms-shown``
* ``flat``: ``--ms-shown`` near ``--baseline-ms-shown`` mean
* ``drift``: linear trend on ``--ms-shown`` (rising / falling / none)
* ``stability``: osc → drift → flat priority on ``--ms-shown``

``--param`` / ``--filter`` reuse ``figure.plot`` (same as cell_dynamics).
Usage (from ``vision/simulation/``)::

    ../.venv/bin/python 6_analyze/trace.py \\
      --run hp_lp/28693664-... --check oscillation --ms-shown 0,1000

    ../.venv/bin/python 6_analyze/trace.py \\
      --check flat --filter ca --ms-shown 1000,1100 \\
      --baseline-ms-shown 800,1000

    ../.venv/bin/python 6_analyze/trace.py \\
      --run hp_lp/28704173-... --check stability --ms-shown 0,1000 \\
      --baseline-ms-shown 0,200

    ../.venv/bin/python 6_analyze/trace.py \\
      --run hp_lp/28704173-... --check drift --ms-shown 0,1000 \\
      --cells TmY11,Mi1,Tm3 --param a_slow.TmY11=1 a_slow.Mi1=1 a_slow.Tm3=1
"""
from __future__ import annotations

from default_params import (
    ANALYZE_TRACE_DEFAULT,
    DEFAULT_RUN_PATH,
)

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401
import numpy as np
import torch

import figure.plot as plot_trained
import train
from analyze.cell_dynamics import TimeWindow, analyze_spot_average
from import_bootstrap import parse_comma_list
from train.cli import sti_timing_kwargs_from_args

CHECK_OSCILLATION = "oscillation"
CHECK_FLAT = "flat"
CHECK_DRIFT = "drift"
CHECK_STABILITY = "stability"


def detect_oscillation(
    v_trace: np.ndarray,
    *,
    delta_ms: float,
    min_osc_f: float = ANALYZE_TRACE_DEFAULT['trace_osc_min_f'],
    max_osc_f: float = ANALYZE_TRACE_DEFAULT['trace_osc_max_f'],
    peak_threshold: float = ANALYZE_TRACE_DEFAULT['trace_osc_peak_threshold'],
    z_threshold: float = ANALYZE_TRACE_DEFAULT['trace_osc_z_threshold'],
    snr_min: float = ANALYZE_TRACE_DEFAULT['trace_osc_snr_min'],
) -> dict:
    """FFT / v_peak_to_peak metrics on one already-sliced mean v_post segment."""
    n = len(v_trace)
    if n < 10:
        return {"flag": False, "reason": "too_short", "n_samples": n}

    mean = float(np.mean(v_trace))
    std = float(np.std(v_trace))
    if std < 1e-6:
        return {"flag": False, "reason": "flat", "std": std, "n_samples": n}

    v_detrend = v_trace - mean
    dt = delta_ms / 1000.0
    fs = np.fft.rfftfreq(n, dt)
    fft_mag = np.abs(np.fft.rfft(v_detrend))
    mask = (fs >= min_osc_f) & (fs <= max_osc_f)
    if not np.any(mask):
        return {
            "flag": False,
            "reason": "no_power_in_band",
            "peak_f": 0.0,
            "peak_power": 0.0,
            "n_samples": n,
        }

    peak_i = int(np.argmax(fft_mag[mask]))
    peak_f = float(fs[mask][peak_i])
    peak_power = float(fft_mag[mask][peak_i])
    power_sum = float(np.sum(fft_mag[mask] ** 2))
    v_peak_to_peak = float(np.ptp(v_trace))
    v_peak_to_peak_over_std = v_peak_to_peak / std if std > 0 else 0.0
    mean_band_power = power_sum / float(np.sum(mask))
    snr = peak_power / np.sqrt(mean_band_power) if mean_band_power > 0 else 0.0
    flag = (
        v_peak_to_peak_over_std > z_threshold
        and peak_power > peak_threshold
        and snr > snr_min
    )
    return {
        "flag": flag,
        "reason": "oscillation" if flag else "stable",
        "mean": mean,
        "std": std,
        "v_peak_to_peak": v_peak_to_peak,
        "v_peak_to_peak_over_std": v_peak_to_peak_over_std,
        "peak_f": peak_f,
        "peak_power": peak_power,
        "snr": snr,
        "n_samples": n,
    }


def detect_flat(
    v_trace: np.ndarray,
    *,
    baseline: float,
    max_abs: float,
    v_peak_to_peak_max: float,
    abs_mean: float,
) -> dict:
    """Flatness of an already-sliced segment vs a scalar baseline (mV)."""
    n = len(v_trace)
    if n < 2:
        return {"flag": False, "reason": "too_short", "n_samples": n}

    d = v_trace - float(baseline)
    abs_mean_d = float(np.mean(np.abs(d)))
    v_peak_to_peak = float(np.ptp(v_trace))
    v_peak_to_peak_from_baseline = float(np.ptp(d))
    max_abs_d = float(np.max(np.abs(d)))
    flag = (
        max_abs_d < max_abs
        and v_peak_to_peak_from_baseline < v_peak_to_peak_max
        and abs_mean_d < abs_mean
    )
    return {
        "flag": flag,
        "reason": "flat" if flag else "responsive",
        "baseline": float(baseline),
        "delta_mean": float(np.mean(v_trace) - baseline),
        "abs_mean": abs_mean_d,
        "v_peak_to_peak": v_peak_to_peak,
        "v_peak_to_peak_from_baseline": v_peak_to_peak_from_baseline,
        "max_abs": max_abs_d,
        "n_samples": n,
    }


def detect_drift(
    v_trace: np.ndarray,
    *,
    delta_ms: float,
    min_slope_mv_per_s: float = ANALYZE_TRACE_DEFAULT['trace_drift_min_slope_mv_per_s'],
    min_r: float = ANALYZE_TRACE_DEFAULT['trace_drift_min_r'],
) -> dict:
    """Linear trend on an already-sliced segment: rising / falling / none."""
    n = len(v_trace)
    if n < 3:
        return {
            "flag": False,
            "reason": "too_short",
            "direction": "none",
            "n_samples": n,
        }

    t_s = np.arange(n, dtype=float) * (delta_ms / 1000.0)
    t_c = t_s - t_s.mean()
    v = np.asarray(v_trace, dtype=float)
    v_c = v - v.mean()
    denom = float(np.dot(t_c, t_c))
    if denom < 1e-18:
        return {
            "flag": False,
            "reason": "no_span",
            "direction": "none",
            "slope_mv_per_s": 0.0,
            "r": 0.0,
            "n_samples": n,
        }
    slope = float(np.dot(t_c, v_c) / denom)
    v_hat = slope * t_c
    ss_tot = float(np.dot(v_c, v_c))
    ss_res = float(np.dot(v_c - v_hat, v_c - v_hat))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r = float(np.sqrt(max(r2, 0.0)) * np.sign(slope))
    flag = abs(slope) >= min_slope_mv_per_s and abs(r) >= min_r
    if flag and slope > 0:
        direction = "rising"
    elif flag and slope < 0:
        direction = "falling"
    else:
        direction = "none"
    return {
        "flag": flag,
        "reason": direction if flag else "no_drift",
        "direction": direction,
        "slope_mv_per_s": slope,
        "r": r,
        "delta_end_start": float(v[-1] - v[0]),
        "n_samples": n,
    }


def detect_stability(
    v_trace: np.ndarray,
    *,
    baseline: float,
    delta_ms: float,
    min_osc_f: float,
    max_osc_f: float,
    z_threshold: float,
    min_slope_mv_per_s: float,
    min_r: float,
    max_abs: float,
    v_peak_to_peak_max: float,
    abs_mean: float,
) -> dict:
    """Priority: oscillation → drift → flat-to-baseline → unstable."""
    osc = detect_oscillation(
        v_trace,
        delta_ms=delta_ms,
        min_osc_f=min_osc_f,
        max_osc_f=max_osc_f,
        z_threshold=z_threshold,
    )
    drift = detect_drift(
        v_trace,
        delta_ms=delta_ms,
        min_slope_mv_per_s=min_slope_mv_per_s,
        min_r=min_r,
    )
    flat = detect_flat(
        v_trace,
        baseline=baseline,
        max_abs=max_abs,
        v_peak_to_peak_max=v_peak_to_peak_max,
        abs_mean=abs_mean,
    )
    if osc["flag"]:
        label = "oscillation"
    elif drift["flag"]:
        label = drift["direction"]
    elif flat["flag"]:
        label = "stable_baseline"
    else:
        label = "unstable"
    return {
        "flag": label == "stable_baseline",
        "reason": label,
        "oscillation": osc,
        "drift": drift,
        "flat": flat,
        "n_samples": len(v_trace),
    }


def _sti_ms(args, opts: dict) -> tuple[float, float, float]:
    from task.spot.input import spot_timing_from_opts

    ms_pre = float(args.ms_pre) if args.ms_pre is not None else float(opts["ms_pre"])
    override_opts = dict(opts)
    if args.ms_spot is not None:
        override_opts["ms_spot"] = float(args.ms_spot)
    if args.ms_response is not None:
        override_opts["ms_response"] = float(args.ms_response)
    timing = spot_timing_from_opts(override_opts)
    ms_spot = timing.ms_spot
    ms_response = timing.ms_response
    return ms_pre, ms_spot, ms_response


def _default_ms_shown(
    check: str, ms_pre: float, ms_spot: float, ms_response: float,
) -> tuple[float, float]:
    if check == CHECK_FLAT:
        return ms_pre, ms_pre + ms_spot + ms_response
    return 0.0, ms_pre


def _default_baseline_ms_shown(
    analyze: tuple[float, float],
    baseline_ms: float,
) -> tuple[float, float]:
    start, _stop = analyze
    if start > 0:
        return max(0.0, start - baseline_ms), start
    return 0.0, min(baseline_ms, _stop)


def _resolve_windows(args, ms_pre, ms_spot, ms_response):
    if args.ms_shown is not None:
        analyze = plot_trained.parse_ms_shown_range(args.ms_shown, flag="--ms-shown")
    else:
        analyze = _default_ms_shown(args.check, ms_pre, ms_spot, ms_response)

    need_baseline = args.check in (CHECK_FLAT, CHECK_STABILITY)
    if args.baseline_ms_shown is not None:
        baseline = plot_trained.parse_ms_shown_range(
            args.baseline_ms_shown, flag="--baseline-ms-shown",
        )
    elif need_baseline:
        baseline = _default_baseline_ms_shown(analyze, float(args.baseline_ms))
    else:
        baseline = None

    if baseline is not None and baseline[0] >= baseline[1]:
        raise SystemExit(
            f"baseline window empty: {baseline[0]:g},{baseline[1]:g} "
            f"(set --baseline-ms-shown START,STOP)"
        )
    if analyze[0] >= analyze[1]:
        raise SystemExit(f"--ms-shown empty: {analyze[0]:g},{analyze[1]:g}")
    return analyze, baseline


def _slice_ms(
    v_full: np.ndarray, start_ms: float, stop_ms: float, *, delta_ms: float,
) -> np.ndarray:
    i0 = train.t_from_ms(start_ms, delta_ms=delta_ms)
    i1 = train.t_from_ms(stop_ms, delta_ms=delta_ms)
    n = len(v_full)
    if i0 < 0 or i1 >= n or i0 > i1:
        raise SystemExit(
            f"slice {start_ms:g},{stop_ms:g} (t={i0}:{i1}) out of range "
            f"for v_post length {n}"
        )
    return np.asarray(v_full[i0 : i1 + 1], dtype=float)


def _baseline_mean(
    v_full: np.ndarray,
    baseline: tuple[float, float],
    *,
    delta_ms: float,
) -> float:
    seg = _slice_ms(v_full, baseline[0], baseline[1], delta_ms=delta_ms)
    return float(np.mean(seg))


def _format_param_edits(edits: list[tuple[str, str | None, float]]) -> str:
    if not edits:
        return "none"
    parts: list[str] = []
    for name, node, val in edits:
        key = name if node is None else f"{name}.{node}"
        parts.append(f"{key}={val:g}")
    return " ".join(parts)


def _trace_series(rep: dict) -> np.ndarray:
    """Full-length series from ``analyze_spot_average`` report: ``ca`` or ``v_post``."""
    if rep.get("filter") == "ca":
        if "ca" not in rep:
            raise SystemExit(
                f"report for {rep.get('cell')!r} has filter=ca but no ca series"
            )
        return np.asarray(rep["ca"], dtype=float)
    return np.asarray(rep["v_post"], dtype=float)


def _load_reports(args):
    """Shared run load + one ``analyze_spot_average`` forward."""
    run_dir = plot_trained.resolve_run_dir(args.run)
    session, z, _best_cost = plot_trained.load_best(run_dir)
    train_opts = plot_trained.load_train_opts(run_dir) or {}
    train_filter = train.expand_filter(train_opts.get("filter", "none"))
    timing_kw = sti_timing_kwargs_from_args(args)
    session, z, _timing_changed = plot_trained.maybe_override_sti_timing(
        run_dir=run_dir,
        session=session,
        z=z,
        **timing_kw,
        filter=args.filter,
    )
    z_t = torch.tensor(
        np.asarray(z, dtype=np.float64), dtype=torch.float64, device=session.device,
    )
    schema = list(session.schema)
    param_edits = plot_trained.parse_param_tokens(args.param)
    z_t, schema = plot_trained.apply_param_overrides(z_t, schema, session, param_edits)
    session = session.with_schema(schema)
    params = train.materialize_from_opts(
        train.assign_params(z_t, schema, session.backend), session,
    )

    param_csv = os.path.join(run_dir, "param.csv")
    if args.cells == "all":
        with open(param_csv) as param_csv_file:
            cells = [row["cell"] for row in csv.DictReader(param_csv_file)]
    else:
        cells = parse_comma_list(args.cells)

    sess_one = plot_trained.session_for_task(session, args.task)
    delta_ms = float(sess_one.delta_ms)
    opts = dict(
        (sess_one.train_opts or {}).get(f"{sess_one.primary_pack.name}_sti_opts")
        or {},
    )
    ms_pre, ms_spot, ms_response = _sti_ms(args, opts)
    analyze, baseline = _resolve_windows(args, ms_pre, ms_spot, ms_response)
    forward_stop = analyze[1]
    if baseline is not None:
        forward_stop = max(forward_stop, baseline[1])
    # Buffer is always from trial start; start=0 keeps indices absolute.
    time_window = TimeWindow(kind="ms", start=0.0, stop=forward_stop)
    base_s = (
        f"{baseline[0]:g},{baseline[1]:g}" if baseline is not None else "none"
    )
    print(
        f"check={args.check}  {args.task} radius={args.radius}  "
        f"filter={args.filter or 'run'}  "
        f"ms-shown={analyze[0]:g},{analyze[1]:g}  "
        f"baseline-ms-shown={base_s}  "
        f"forward TimeWindow(ms, 0, {forward_stop:g})  "
        f"ms_pre={ms_pre:g} ms_spot={ms_spot:g} ms_response={ms_response:g}  "
        f"param={_format_param_edits(param_edits)}  "
        f"n_cells={len(cells)}",
        flush=True,
    )
    reports = analyze_spot_average(
        sess_one,
        params =params,
        cells=cells,
        task=args.task,
        time_window=time_window,
        radius=args.radius,
        train_filter=train_filter,
    )
    return cells, reports, delta_ms, analyze, baseline


def _print_oscillation(cells, reports, delta_ms, analyze, args) -> None:
    hdr = (
        f"{'Cell':<12} {'Osc?':<6} {'Reason':<12} "
        f"{'v_peak_to_peak_over_std':>8} {'peak_f':>8} {'SNR':>8} "
        f"{'v_peak_to_peak':>8} {'std':>8} {'n':>6}"
    )
    print(f"\n{hdr}\n{'-' * len(hdr)}", flush=True)
    hit: list[tuple[str, dict]] = []
    for cell in cells:
        rep = reports.get(cell)
        if rep is None:
            print(f"{cell:<12} {'?':<6} {'no_report':<12}", flush=True)
            continue
        v = _slice_ms(
            _trace_series(rep), analyze[0], analyze[1], delta_ms=delta_ms,
        )
        result = detect_oscillation(
            v,
            delta_ms=delta_ms,
            min_osc_f=args.min_f,
            max_osc_f=args.max_f,
            z_threshold=args.z_threshold,
        )
        yes = "YES" if result["flag"] else "NO"
        print(
            f"{cell:<12} {yes:<6} {result['reason']:<12} "
            f"{result.get('v_peak_to_peak_over_std', 0):>8.2f} "
            f"{result.get('peak_f', 0):>8.2f} "
            f"{result.get('snr', 0):>8.2f} "
            f"{result.get('v_peak_to_peak', 0):>8.2f} "
            f"{result.get('std', 0):>8.2f} "
            f"{result.get('n_samples', 0):>6}",
            flush=True,
        )
        if result["flag"]:
            hit.append((cell, result))

    print(f"\nOscillating ({len(hit)}/{len(cells)}):", flush=True)
    for cell, r in hit:
        print(
            f"  {cell}: f={r['peak_f']:.2f}Hz  "
            f"v_peak_to_peak_over_std={r['v_peak_to_peak_over_std']:.2f}  "
            f"SNR={r['snr']:.2f}  v_peak_to_peak={r['v_peak_to_peak']:.2f}mV",
            flush=True,
        )


def _print_flat(cells, reports, delta_ms, analyze, baseline, args) -> None:
    hdr = (
        f"{'Cell':<12} {'Flat?':<6} {'Reason':<12} "
        f"{'base':>8} {'Δmean':>8} {'|μ|':>8} {'v_peak_to_peak':>9} "
        f"{'max|Δ|':>8} {'n':>6}"
    )
    print(f"\n{hdr}\n{'-' * len(hdr)}", flush=True)
    hit: list[tuple[str, dict]] = []
    for cell in cells:
        rep = reports.get(cell)
        if rep is None:
            print(f"{cell:<12} {'?':<6} {'no_report':<12}", flush=True)
            continue
        v_full = _trace_series(rep)
        base = _baseline_mean(v_full, baseline, delta_ms=delta_ms)
        v = _slice_ms(v_full, analyze[0], analyze[1], delta_ms=delta_ms)
        result = detect_flat(
            v,
            baseline=base,
            max_abs=args.max_abs,
            v_peak_to_peak_max=args.v_peak_to_peak_max,
            abs_mean=args.abs_mean,
        )
        yes = "YES" if result["flag"] else "NO"
        print(
            f"{cell:<12} {yes:<6} {result['reason']:<12} "
            f"{result.get('baseline', 0):>8.2f} "
            f"{result.get('delta_mean', 0):>8.2f} "
            f"{result.get('abs_mean', 0):>8.3f} "
            f"{result.get('v_peak_to_peak', 0):>9.3f} "
            f"{result.get('max_abs', 0):>8.2f} "
            f"{result.get('n_samples', 0):>6}",
            flush=True,
        )
        if result["flag"]:
            hit.append((cell, result))

    print(f"\nFlat ({len(hit)}/{len(cells)}):", flush=True)
    for cell, r in hit:
        print(
            f"  {cell}: max|Δ|={r['max_abs']:.3f}  "
            f"v_peak_to_peak={r['v_peak_to_peak']:.3f}  "
            f"Δmean={r['delta_mean']:.3f}  base={r['baseline']:.2f}",
            flush=True,
        )


def _print_drift(cells, reports, delta_ms, analyze, args) -> None:
    hdr = (
        f"{'Cell':<12} {'Drift?':<6} {'Reason':<12} "
        f"{'slope':>10} {'r':>8} {'Δend':>8} {'n':>6}"
    )
    print(f"\n{hdr}\n{'-' * len(hdr)}", flush=True)
    hit: list[tuple[str, dict]] = []
    for cell in cells:
        rep = reports.get(cell)
        if rep is None:
            print(f"{cell:<12} {'?':<6} {'no_report':<12}", flush=True)
            continue
        v = _slice_ms(
            _trace_series(rep), analyze[0], analyze[1], delta_ms=delta_ms,
        )
        result = detect_drift(
            v,
            delta_ms=delta_ms,
            min_slope_mv_per_s=args.min_slope,
            min_r=args.min_r,
        )
        yes = "YES" if result["flag"] else "NO"
        print(
            f"{cell:<12} {yes:<6} {result['reason']:<12} "
            f"{result.get('slope_mv_per_s', 0):>10.3f} "
            f"{result.get('r', 0):>8.3f} "
            f"{result.get('delta_end_start', 0):>8.2f} "
            f"{result.get('n_samples', 0):>6}",
            flush=True,
        )
        if result["flag"]:
            hit.append((cell, result))

    print(f"\nDrifting ({len(hit)}/{len(cells)}):", flush=True)
    for cell, r in hit:
        print(
            f"  {cell}: {r['direction']}  slope={r['slope_mv_per_s']:.3f}mV/s  "
            f"r={r['r']:.3f}  Δend={r['delta_end_start']:.2f}mV",
            flush=True,
        )


def _print_stability(cells, reports, delta_ms, analyze, baseline, args) -> None:
    hdr = (
        f"{'Cell':<12} {'Label':<16} {'Osc?':<5} {'Drift':<8} {'Flat?':<5} "
        f"{'v_peak_to_peak_over_std':>8} {'slope':>9} {'max|Δ|':>8} {'n':>6}"
    )
    print(f"\n{hdr}\n{'-' * len(hdr)}", flush=True)
    n_by_label: dict[str, int] = {}
    for cell in cells:
        rep = reports.get(cell)
        if rep is None:
            print(f"{cell:<12} {'?':<16} {'?':<5} {'?':<8} {'?':<5}", flush=True)
            continue
        v_full = _trace_series(rep)
        base = _baseline_mean(v_full, baseline, delta_ms=delta_ms)
        v = _slice_ms(v_full, analyze[0], analyze[1], delta_ms=delta_ms)
        result = detect_stability(
            v,
            baseline=base,
            delta_ms=delta_ms,
            min_osc_f=args.min_f,
            max_osc_f=args.max_f,
            z_threshold=args.z_threshold,
            min_slope_mv_per_s=args.min_slope,
            min_r=args.min_r,
            max_abs=args.max_abs,
            v_peak_to_peak_max=args.v_peak_to_peak_max,
            abs_mean=args.abs_mean,
        )
        osc = result["oscillation"]
        drift = result["drift"]
        flat = result["flat"]
        label = result["reason"]
        n_by_label[label] = n_by_label.get(label, 0) + 1
        print(
            f"{cell:<12} {label:<16} "
            f"{'YES' if osc['flag'] else 'NO':<5} "
            f"{drift['direction']:<8} "
            f"{'YES' if flat['flag'] else 'NO':<5} "
            f"{osc.get('v_peak_to_peak_over_std', 0):>8.2f} "
            f"{drift.get('slope_mv_per_s', 0):>9.3f} "
            f"{flat.get('max_abs', 0):>8.2f} "
            f"{result.get('n_samples', 0):>6}",
            flush=True,
        )

    print(f"\nStability summary ({sum(n_by_label.values())}/{len(cells)}):", flush=True)
    for label in (
        "stable_baseline", "oscillation", "rising", "falling", "unstable",
    ):
        if label in n_by_label:
            print(f"  {label}: {n_by_label[label]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        default=DEFAULT_RUN_PATH,
        help="run dir relative to PARAMETER_DIR or absolute (default: %(default)s)",
    )
    ap.add_argument(
        "--check",
        required=True,
        choices=(CHECK_OSCILLATION, CHECK_FLAT, CHECK_DRIFT, CHECK_STABILITY),
        help=(
            "oscillation|flat|drift|stability on --ms-shown "
            "(not locked to pre/pulse)"
        ),
    )
    ap.add_argument(
        "--cells",
        default="all",
        help="comma-separated cells or all (default: param.csv)",
    )
    plot_trained.add_ms_shown_argument(ap)
    ap.add_argument(
        "--baseline-ms-shown",
        default=None,
        metavar="START,STOP",
        help=(
            "absolute ms baseline window for flat/stability. "
            "default: [start-baseline_ms, start] if start>0 else [0, baseline_ms]"
        ),
    )
    plot_trained.add_plot_timing_arguments(ap)
    plot_trained.add_plot_filter_argument(ap)
    plot_trained.add_param_argument(ap)
    ap.add_argument("--task", default="spot_bright")
    ap.add_argument("--radius", type=int, default=0, choices=(0, 1))
    ap.add_argument(
        "--z-threshold", type=float, default=ANALYZE_TRACE_DEFAULT['trace_osc_z_threshold'],
    )
    ap.add_argument("--min-f", type=float, default=ANALYZE_TRACE_DEFAULT['trace_osc_min_f'])
    ap.add_argument("--max-f", type=float, default=ANALYZE_TRACE_DEFAULT['trace_osc_max_f'])
    ap.add_argument(
        "--min-slope",
        type=float,
        default=ANALYZE_TRACE_DEFAULT['trace_drift_min_slope_mv_per_s'],
        help="drift: min |slope| in mV/s",
    )
    ap.add_argument(
        "--min-r",
        type=float,
        default=ANALYZE_TRACE_DEFAULT['trace_drift_min_r'],
        help="drift: min |Pearson r| of linear fit",
    )
    ap.add_argument(
        "--baseline-ms",
        type=float,
        default=ANALYZE_TRACE_DEFAULT['trace_baseline_ms'],
        help="default baseline length when --baseline-ms-shown omitted",
    )
    ap.add_argument(
        "--max-abs",
        type=float,
        default=ANALYZE_TRACE_DEFAULT['trace_flat_max_abs'],
        help="flat: max |Δ| vs baseline",
    )
    ap.add_argument(
        "--v-peak-to-peak-max",
        type=float,
        default=ANALYZE_TRACE_DEFAULT['trace_flat_v_peak_to_peak_max'],
        help="flat: max v_peak_to_peak from baseline",
    )
    ap.add_argument(
        "--abs-mean",
        type=float,
        default=ANALYZE_TRACE_DEFAULT['trace_flat_abs_mean'],
        help="flat: max mean |Δ| vs baseline",
    )
    args = ap.parse_args()

    cells, reports, delta_ms, analyze, baseline = _load_reports(args)
    if args.check == CHECK_OSCILLATION:
        _print_oscillation(cells, reports, delta_ms, analyze, args)
    elif args.check == CHECK_FLAT:
        _print_flat(cells, reports, delta_ms, analyze, baseline, args)
    elif args.check == CHECK_DRIFT:
        _print_drift(cells, reports, delta_ms, analyze, args)
    else:
        _print_stability(cells, reports, delta_ms, analyze, baseline, args)


if __name__ == "__main__":
    main()
