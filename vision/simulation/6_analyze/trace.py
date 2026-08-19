"""Classify spot radius-0 average ``v_post`` / ``ca`` on an absolute-ms window.

Uses ``analyze.cell_dynamics.analyze_spot_average`` only — one forward, no
rewrite of core dynamics. Series: ``filter=ca`` → report ``ca``, else
``v_post``. Spot time axis: ``0`` = trial start; onset ≈ ``ms_pre``. Analyze /
baseline windows are absolute ms (``ms_shown``), not locked to pre vs sti-on.

Checks (``check=``)
-------------------
* ``oscillation``: FFT / v_peak_to_peak on ``ms_shown``
* ``flat``: ``ms_shown`` near ``baseline_ms_shown`` mean
* ``drift``: linear trend on ``ms_shown`` (rising / falling / none)
* ``stability``: osc → drift → flat priority on ``ms_shown``

``param_vals.a_h.L1=0.8`` and Hydra keys in ``config.yaml`` (``euler`` /
``filter`` / ``ms_pre`` / …) rebuild the session in memory via
``figure.plot.override_session``; they do not write ``train_opts.json``.
Keys omitted on the Hydra CLI keep the run's train opts.

Usage (from ``vision/simulation/``)::

    ../.venv/bin/python -m analyze.trace \\
      check=oscillation ms_shown=0,1000

    ../.venv/bin/python -m analyze.trace \\
      check=flat filter=ca ms_shown=1000,1100 baseline_ms_shown=800,1000

    ../.venv/bin/python -m analyze.trace \\
      analyze_runs=hp_lp/28704173-... check=stability ms_shown=0,1000 \\
      baseline_ms_shown=0,200

    ../.venv/bin/python -m analyze.trace \\
      analyze_runs=hp_lp/28704173-... check=drift ms_shown=0,1000 \\
      cells=TmY11,Mi1,Tm3 param_vals.a_h.TmY11=1
"""
from __future__ import annotations

from config import (
    ANALYZE_CELL_DYNAMICS,
    ANALYZE_RUNS,
    ANALYZE_TRACE,
    FIGURE_PLOT,
    TRAIN_CONFIG,
    active_config,
    apply_config,
    parse_cells,
    session_kwargs_from_cli,
)

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401
import hydra
import numpy as np
import torch

import figure.plot as plot
import train
from analyze.cell_dynamics import TimeWindow, analyze_spot_average
from import_bootstrap import parse_comma_list

CHECK_OSCILLATION = "oscillation"
CHECK_FLAT = "flat"
CHECK_DRIFT = "drift"
CHECK_STABILITY = "stability"


def detect_oscillation(
    v_trace: np.ndarray,
    *,
    delta_ms: float,
    min_osc_f: float = ANALYZE_TRACE['trace_osc_min_f'],
    max_osc_f: float = ANALYZE_TRACE['trace_osc_max_f'],
    peak_threshold: float = ANALYZE_TRACE['trace_osc_peak_threshold'],
    z_threshold: float = ANALYZE_TRACE['trace_osc_z_threshold'],
    snr_min: float = ANALYZE_TRACE['trace_osc_snr_min'],
) -> dict:
    """FFT / v_peak_to_peak metrics on one already-sliced mean v_post slice."""
    n = len(v_trace)
    if n < 10:
        return {"flag": False, "reason": "too_short", "n_sample": n}

    mean = float(np.mean(v_trace))
    std = float(np.std(v_trace))
    if std < 1e-6:
        return {"flag": False, "reason": "flat", "std": std, "n_sample": n}

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
            "n_sample": n,
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
        "n_sample": n,
    }


def detect_flat(
    v_trace: np.ndarray,
    *,
    baseline: float,
    max_abs: float,
    v_peak_to_peak_max: float,
    abs_mean: float,
) -> dict:
    """Flatness of an already-sliced trace vs a scalar baseline (mV)."""
    n = len(v_trace)
    if n < 2:
        return {"flag": False, "reason": "too_short", "n_sample": n}

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
        "n_sample": n,
    }


def detect_drift(
    v_trace: np.ndarray,
    *,
    delta_ms: float,
    min_slope_mv_over_s: float = ANALYZE_TRACE['trace_drift_min_slope_mv_over_s'],
    min_r: float = ANALYZE_TRACE['trace_drift_min_r'],
) -> dict:
    """Linear trend on an already-sliced trace: rising / falling / none."""
    n = len(v_trace)
    if n < 3:
        return {
            "flag": False,
            "reason": "too_short",
            "direction": "none",
            "n_sample": n,
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
            "slope_mv_over_s": 0.0,
            "r": 0.0,
            "n_sample": n,
        }
    slope = float(np.dot(t_c, v_c) / denom)
    v_hat = slope * t_c
    ss_tot = float(np.dot(v_c, v_c))
    ss_res = float(np.dot(v_c - v_hat, v_c - v_hat))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r = float(np.sqrt(max(r2, 0.0)) * np.sign(slope))
    flag = abs(slope) >= min_slope_mv_over_s and abs(r) >= min_r
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
        "slope_mv_over_s": slope,
        "r": r,
        "delta_end_start": float(v[-1] - v[0]),
        "n_sample": n,
    }


def detect_stability(
    v_trace: np.ndarray,
    *,
    baseline: float,
    delta_ms: float,
    min_osc_f: float,
    max_osc_f: float,
    z_threshold: float,
    min_slope_mv_over_s: float,
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
        min_slope_mv_over_s=min_slope_mv_over_s,
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
        "n_sample": len(v_trace),
    }


def _sti_ms(opts: dict) -> tuple[float, float, float]:
    ms_pre = float(opts["ms_pre"])
    ms_response = float(opts["ms_response"])
    ms_sti = opts.get("ms_sti")
    if ms_sti is not None:
        ms_response = max(ms_response, float(ms_sti))
    return ms_pre, float(ms_sti) if ms_sti is not None else 0.0, ms_response


def _resolve_ms_shown(
    check: str, ms_pre: float, ms_sti: float, ms_response: float,
) -> tuple[float, float]:
    if check == CHECK_FLAT:
        return ms_pre, ms_pre + ms_sti + ms_response
    return 0.0, ms_pre


def _resolve_baseline_ms_shown(
    analyze: tuple[float, float],
    baseline_ms: float,
) -> tuple[float, float]:
    start, _stop = analyze
    if start > 0:
        return max(0.0, start - baseline_ms), start
    return 0.0, min(baseline_ms, _stop)


def _resolve_windows(
    *,
    check: str,
    ms_pre: float,
    ms_sti: float,
    ms_response: float,
    ms_shown,
    baseline_ms_shown,
    baseline_ms: float,
):
    if ms_shown is not None:
        analyze = plot.parse_ms_shown(str(ms_shown), flag="ms_shown")
    else:
        analyze = _resolve_ms_shown(check, ms_pre, ms_sti, ms_response)

    need_baseline = check in (CHECK_FLAT, CHECK_STABILITY)
    if baseline_ms_shown is not None:
        baseline = plot.parse_ms_shown(
            str(baseline_ms_shown), flag="baseline_ms_shown",
        )
    elif need_baseline:
        baseline = _resolve_baseline_ms_shown(analyze, baseline_ms)
    else:
        baseline = None

    if baseline is not None and baseline[0] >= baseline[1]:
        raise SystemExit(
            f"baseline window empty: {baseline[0]:g},{baseline[1]:g} "
            f"(set baseline_ms_shown=START,STOP)"
        )
    if analyze[0] >= analyze[1]:
        raise SystemExit(f"ms_shown empty: {analyze[0]:g},{analyze[1]:g}")
    return analyze, baseline


def _slice_ms(
    trace: np.ndarray, start_ms: float, stop_ms: float, *, delta_ms: float,
) -> np.ndarray:
    i0 = train.t_from_ms(start_ms, delta_ms=delta_ms)
    i1 = train.t_from_ms(stop_ms, delta_ms=delta_ms)
    n = len(trace)
    if i0 < 0 or i1 >= n or i0 > i1:
        raise SystemExit(
            f"slice {start_ms:g},{stop_ms:g} (t={i0}:{i1}) out of range "
            f"for v_post length {n}"
        )
    return np.asarray(trace[i0 : i1 + 1], dtype=float)


def _baseline_mean(
    trace: np.ndarray,
    baseline: tuple[float, float],
    *,
    delta_ms: float,
) -> float:
    trace_slice = _slice_ms(trace, baseline[0], baseline[1], delta_ms=delta_ms)
    return float(np.mean(trace_slice))


def _format_param(param_vals) -> str:
    parts: list[str] = []
    for param, bag in (param_vals or {}).items():
        if isinstance(bag, dict):
            for node, number in bag.items():
                parts.append(f"{param}.{node}={number:g}")
        else:
            parts.append(f"{param}={float(bag):g}")
    return " ".join(parts) if parts else "none"


def _trace_from_report(rep: dict) -> np.ndarray:
    """Full-length trace from ``analyze_spot_average`` report: ``ca`` or ``v_post``."""
    if rep.get("filter") == "ca":
        if "ca" not in rep:
            raise SystemExit(
                f"report for {rep.get('cell')!r} has filter=ca but no ca trace"
            )
        return np.asarray(rep["ca"], dtype=float)
    return np.asarray(rep["v_post"], dtype=float)


def _load_reports(
    run_arg,
    *,
    check: str,
    task: str,
    contrast: str,
    radius: int,
    cells: list[str] | None,
    session_kwargs: dict,
):
    """Shared run load + one ``analyze_spot_average`` forward."""
    run_dir = plot.resolve_run_dir(run_arg)
    session, z, _best_cost = plot.load_best(run_dir)
    train_opts = plot.load_train_opts(run_dir) or {}
    train_filter = str(train_opts.get("filter", "none"))
    session, z, _ms_changed = plot.override_session(
        run_dir=run_dir,
        session=session,
        z=z,
        **session_kwargs,
    )
    z = torch.tensor(
        np.asarray(z, dtype=np.float64), dtype=torch.float64, device=session.device,
    )
    schema = train.schema_copy(session.schema)
    param_vals = dict(active_config().get("param_vals") or {})
    z, schema = plot.override_params(
        z, schema, session, param_vals=param_vals,
    )
    session = session.with_schema(schema)
    params = train.override_val_from(
        train.assign_params(z, schema, session.connectome), session,
    )

    param_csv = os.path.join(run_dir, "param.csv")
    if not cells:
        with open(param_csv) as param_csv_file:
            cells = [row["cell"] for row in csv.DictReader(param_csv_file)]

    sess_one = plot.session_from_task(session, task, contrast)
    delta_ms = float(sess_one.delta_ms)
    opts = dict(
        (sess_one.train_opts or {}).get(f"{sess_one.primary_pack.task}_sti_opts")
        or {},
    )
    ms_pre, ms_sti, ms_response = _sti_ms(opts)
    analyze, baseline = _resolve_windows(
        check=check,
        ms_pre=ms_pre,
        ms_sti=ms_sti,
        ms_response=ms_response,
        ms_shown=FIGURE_PLOT.get("ms_shown"),
        baseline_ms_shown=ANALYZE_TRACE.get("baseline_ms_shown"),
        baseline_ms=float(ANALYZE_TRACE["trace_baseline_ms"]),
    )
    forward_stop = analyze[1]
    if baseline is not None:
        forward_stop = max(forward_stop, baseline[1])
    time_window = TimeWindow(kind="ms", start=0.0, stop=forward_stop)
    baseline_label = (
        f"{baseline[0]:g},{baseline[1]:g}" if baseline is not None else "none"
    )
    filter_label = session_kwargs.get("filter") or "run"
    print(
        f"check={check}  {task} {contrast} radius={radius}  "
        f"filter={filter_label}  "
        f"ms-shown={analyze[0]:g},{analyze[1]:g}  "
        f"baseline-ms-shown={baseline_label}  "
        f"forward TimeWindow(ms, 0, {forward_stop:g})  "
        f"ms_pre={ms_pre:g} ms_sti={ms_sti:g} ms_response={ms_response:g}  "
        f"param={_format_param(param_vals)}  "
        f"n_cell={len(cells)}",
        flush=True,
    )
    reports = analyze_spot_average(
        sess_one,
        params=params,
        cells=cells,
        task=task,
        contrast=contrast,
        time_window=time_window,
        radius=radius,
        train_filter=train_filter,
    )
    return cells, reports, delta_ms, analyze, baseline


def _print_oscillation(cells, reports, delta_ms, analyze) -> None:
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
            _trace_from_report(rep), analyze[0], analyze[1], delta_ms=delta_ms,
        )
        result = detect_oscillation(
            v,
            delta_ms=delta_ms,
            min_osc_f=ANALYZE_TRACE["trace_osc_min_f"],
            max_osc_f=ANALYZE_TRACE["trace_osc_max_f"],
            z_threshold=ANALYZE_TRACE["trace_osc_z_threshold"],
        )
        yes = "YES" if result["flag"] else "NO"
        print(
            f"{cell:<12} {yes:<6} {result['reason']:<12} "
            f"{result.get('v_peak_to_peak_over_std', 0):>8.2f} "
            f"{result.get('peak_f', 0):>8.2f} "
            f"{result.get('snr', 0):>8.2f} "
            f"{result.get('v_peak_to_peak', 0):>8.2f} "
            f"{result.get('std', 0):>8.2f} "
            f"{result.get('n_sample', 0):>6}",
            flush=True,
        )
        if result["flag"]:
            hit.append((cell, result))

    print(f"\nOscillating ({len(hit)}/{len(cells)}):", flush=True)
    for cell, report in hit:
        print(
            f"  {cell}: f={report['peak_f']:.2f}Hz  "
            f"v_peak_to_peak_over_std={report['v_peak_to_peak_over_std']:.2f}  "
            f"SNR={report['snr']:.2f}  v_peak_to_peak={report['v_peak_to_peak']:.2f}mV",
            flush=True,
        )


def _print_flat(cells, reports, delta_ms, analyze, baseline) -> None:
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
        trace = _trace_from_report(rep)
        base = _baseline_mean(trace, baseline, delta_ms=delta_ms)
        v = _slice_ms(trace, analyze[0], analyze[1], delta_ms=delta_ms)
        result = detect_flat(
            v,
            baseline=base,
            max_abs=ANALYZE_TRACE["trace_flat_max_abs"],
            v_peak_to_peak_max=ANALYZE_TRACE["trace_flat_v_peak_to_peak_max"],
            abs_mean=ANALYZE_TRACE["trace_flat_abs_mean"],
        )
        yes = "YES" if result["flag"] else "NO"
        print(
            f"{cell:<12} {yes:<6} {result['reason']:<12} "
            f"{result.get('baseline', 0):>8.2f} "
            f"{result.get('delta_mean', 0):>8.2f} "
            f"{result.get('abs_mean', 0):>8.3f} "
            f"{result.get('v_peak_to_peak', 0):>9.3f} "
            f"{result.get('max_abs', 0):>8.2f} "
            f"{result.get('n_sample', 0):>6}",
            flush=True,
        )
        if result["flag"]:
            hit.append((cell, result))

    print(f"\nFlat ({len(hit)}/{len(cells)}):", flush=True)
    for cell, report in hit:
        print(
            f"  {cell}: max|Δ|={report['max_abs']:.3f}  "
            f"v_peak_to_peak={report['v_peak_to_peak']:.3f}  "
            f"Δmean={report['delta_mean']:.3f}  base={report['baseline']:.2f}",
            flush=True,
        )


def _print_drift(cells, reports, delta_ms, analyze) -> None:
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
            _trace_from_report(rep), analyze[0], analyze[1], delta_ms=delta_ms,
        )
        result = detect_drift(
            v,
            delta_ms=delta_ms,
            min_slope_mv_over_s=ANALYZE_TRACE["trace_drift_min_slope_mv_over_s"],
            min_r=ANALYZE_TRACE["trace_drift_min_r"],
        )
        yes = "YES" if result["flag"] else "NO"
        print(
            f"{cell:<12} {yes:<6} {result['reason']:<12} "
            f"{result.get('slope_mv_over_s', 0):>10.3f} "
            f"{result.get('r', 0):>8.3f} "
            f"{result.get('delta_end_start', 0):>8.2f} "
            f"{result.get('n_sample', 0):>6}",
            flush=True,
        )
        if result["flag"]:
            hit.append((cell, result))

    print(f"\nDrifting ({len(hit)}/{len(cells)}):", flush=True)
    for cell, report in hit:
        print(
            f"  {cell}: {report['direction']}  slope={report['slope_mv_over_s']:.3f}mV/s  "
            f"r={report['r']:.3f}  Δend={report['delta_end_start']:.2f}mV",
            flush=True,
        )


def _print_stability(cells, reports, delta_ms, analyze, baseline) -> None:
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
        trace = _trace_from_report(rep)
        base = _baseline_mean(trace, baseline, delta_ms=delta_ms)
        v = _slice_ms(trace, analyze[0], analyze[1], delta_ms=delta_ms)
        result = detect_stability(
            v,
            baseline=base,
            delta_ms=delta_ms,
            min_osc_f=ANALYZE_TRACE["trace_osc_min_f"],
            max_osc_f=ANALYZE_TRACE["trace_osc_max_f"],
            z_threshold=ANALYZE_TRACE["trace_osc_z_threshold"],
            min_slope_mv_over_s=ANALYZE_TRACE["trace_drift_min_slope_mv_over_s"],
            min_r=ANALYZE_TRACE["trace_drift_min_r"],
            max_abs=ANALYZE_TRACE["trace_flat_max_abs"],
            v_peak_to_peak_max=ANALYZE_TRACE["trace_flat_v_peak_to_peak_max"],
            abs_mean=ANALYZE_TRACE["trace_flat_abs_mean"],
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
            f"{drift.get('slope_mv_over_s', 0):>9.3f} "
            f"{flat.get('max_abs', 0):>8.2f} "
            f"{result.get('n_sample', 0):>6}",
            flush=True,
        )

    print(f"\nStability summary ({sum(n_by_label.values())}/{len(cells)}):", flush=True)
    for label in (
        "stable_baseline", "oscillation", "rising", "falling", "unstable",
    ):
        if label in n_by_label:
            print(f"  {label}: {n_by_label[label]}", flush=True)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(hydra_config) -> None:
    apply_config(hydra_config)
    check = ANALYZE_TRACE.get("check")
    if check not in (
        CHECK_OSCILLATION, CHECK_FLAT, CHECK_DRIFT, CHECK_STABILITY,
    ):
        raise SystemExit(
            "check is required (oscillation|flat|drift|stability); "
            f"got {check!r}"
        )

    tasks = list(TRAIN_CONFIG["tasks"])
    contrasts = list(TRAIN_CONFIG["contrasts"])
    task = tasks[0]
    contrast = contrasts[0]
    radius = int(ANALYZE_CELL_DYNAMICS.get("radius") or 0)
    cells = parse_cells(ANALYZE_CELL_DYNAMICS.get("cells"))
    session_kwargs = session_kwargs_from_cli(hydra_config)

    for run_arg in ANALYZE_RUNS:
        cells, reports, delta_ms, analyze, baseline = _load_reports(
            run_arg,
            check=check,
            task=task,
            contrast=contrast,
            radius=radius,
            cells=cells,
            session_kwargs=session_kwargs,
        )
        if check == CHECK_OSCILLATION:
            _print_oscillation(cells, reports, delta_ms, analyze)
        elif check == CHECK_FLAT:
            _print_flat(cells, reports, delta_ms, analyze, baseline)
        elif check == CHECK_DRIFT:
            _print_drift(cells, reports, delta_ms, analyze)
        else:
            _print_stability(cells, reports, delta_ms, analyze, baseline)


if __name__ == "__main__":
    main()
