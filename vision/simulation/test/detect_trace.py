"""Classify spot R0-average ``v_post`` as oscillation (pre) or flat (post-onset).

Uses ``analyze.cell_dynamics.analyze_spot_average`` only — one forward, no
rewrite of core dynamics. Spot time axis: ``0`` = trial start; pre ends at
``ms_pre`` (onset). See ``analyze.cell_dynamics`` module doc.

Usage (from ``vision/simulation/``)::

    ../.venv/bin/python test/detect_trace.py \\
      hp_lp/28693664-... --check oscillation --ms-pre 1000

    ../.venv/bin/python test/detect_trace.py \\
      hp_lp/28702853-... --check flat
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

import figure.plot_run as plot_trained
import training
from analyze.cell_dynamics import TimeWindow, analyze_spot_average
from import_bootstrap import parse_comma_list

CHECK_OSCILLATION = "oscillation"
CHECK_FLAT = "flat"


def detect_oscillation(
    v_trace: np.ndarray,
    *,
    delta_ms: float,
    min_osc_freq_hz: float = 0.5,
    max_osc_freq_hz: float = 20.0,
    peak_threshold: float = 0.5,
    z_threshold: float = 2.0,
) -> dict:
    """FFT / pkpk metrics on one mean v_post trace (pre window already sliced)."""
    n = len(v_trace)
    if n < 10:
        return {"flag": False, "reason": "too_short", "n_samples": n}

    mean = float(np.mean(v_trace))
    std = float(np.std(v_trace))
    if std < 1e-6:
        return {"flag": False, "reason": "flat", "std": std, "n_samples": n}

    v_detrend = v_trace - mean
    dt = delta_ms / 1000.0
    freqs = np.fft.rfftfreq(n, dt)
    fft_mag = np.abs(np.fft.rfft(v_detrend))
    mask = (freqs >= min_osc_freq_hz) & (freqs <= max_osc_freq_hz)
    if not np.any(mask):
        return {
            "flag": False,
            "reason": "no_power_in_band",
            "peak_freq_hz": 0.0,
            "peak_power": 0.0,
            "n_samples": n,
        }

    peak_i = int(np.argmax(fft_mag[mask]))
    peak_freq = float(freqs[mask][peak_i])
    peak_power = float(fft_mag[mask][peak_i])
    total_power = float(np.sum(fft_mag[mask] ** 2))
    pkpk = float(np.ptp(v_trace))
    pkpk_z = pkpk / std if std > 0 else 0.0
    mean_band_power = total_power / float(np.sum(mask))
    snr = peak_power / np.sqrt(mean_band_power) if mean_band_power > 0 else 0.0
    flag = pkpk_z > z_threshold and peak_power > peak_threshold and snr > 2.0
    return {
        "flag": flag,
        "reason": "oscillation" if flag else "stable",
        "mean": mean,
        "std": std,
        "pkpk": pkpk,
        "pkpk_z": pkpk_z,
        "peak_freq_hz": peak_freq,
        "peak_power": peak_power,
        "snr": snr,
        "n_samples": n,
    }


def detect_flat(
    v_trace: np.ndarray,
    *,
    i_onset: int,
    i_pulse_end: int,
    baseline_n: int,
    max_abs: float,
    pkpk: float,
    abs_mean_pul: float,
) -> dict:
    """Post-onset flatness vs late-pre baseline (trace already full-trial sliced)."""
    n = len(v_trace)
    if n < 2 or i_onset <= 0 or i_onset >= n:
        return {"flag": False, "reason": "too_short", "n_samples": n}

    pre = v_trace[: i_onset + 1]
    pul_end = min(i_pulse_end, n - 1)
    pul = v_trace[i_onset : pul_end + 1]
    n_base = max(1, min(int(baseline_n), len(pre)))
    baseline = float(np.mean(pre[-n_base:]))
    d_pul = pul - baseline
    d_after = v_trace[i_onset:] - baseline
    abs_mean = float(np.mean(np.abs(d_pul))) if len(d_pul) else 0.0
    pkpk_pul = float(np.ptp(pul)) if len(pul) else 0.0
    pkpk_from_base = float(np.ptp(d_after)) if len(d_after) else 0.0
    max_abs_d = float(np.max(np.abs(d_after))) if len(d_after) else 0.0
    flag = (
        max_abs_d < max_abs
        and pkpk_from_base < pkpk
        and abs_mean < abs_mean_pul
    )
    return {
        "flag": flag,
        "reason": "flat" if flag else "responsive",
        "baseline": baseline,
        "delta_mean": float(np.mean(pul) - baseline) if len(pul) else 0.0,
        "abs_mean_pul": abs_mean,
        "pkpk_pul": pkpk_pul,
        "pkpk_from_base": pkpk_from_base,
        "max_abs": max_abs_d,
        "pkpk_all": float(np.ptp(v_trace)),
        "n_samples": n,
    }


def _stimulus_ms(args, opts: dict) -> tuple[float, float, float]:
    ms_pre = float(args.ms_pre) if args.ms_pre is not None else float(opts["ms_pre"])
    ms_pulse = (
        float(args.ms_pulse) if args.ms_pulse is not None else float(opts["ms_pulse"])
    )
    ms_response = (
        float(args.ms_response)
        if args.ms_response is not None
        else float(opts["ms_response"])
    )
    return ms_pre, ms_pulse, ms_response


def _time_window(check: str, ms_pre: float, ms_pulse: float, ms_response: float) -> TimeWindow:
    if check == CHECK_OSCILLATION:
        stop = ms_pre
    else:
        stop = ms_pre + ms_pulse + ms_response
    return TimeWindow(kind="ms", start=0.0, stop=stop)


def _load_reports(args):
    """Shared run load + one ``analyze_spot_average`` forward."""
    run_dir = plot_trained.resolve_run_dir(args.run)
    print(f"Loading run: {run_dir}", flush=True)
    session, z, _best_i, _best_cost = plot_trained.load_best(run_dir)
    z_t = torch.tensor(
        np.asarray(z, dtype=np.float64), dtype=torch.float64, device=session.device,
    )
    p = training.assign_params(z_t, list(session.schema), session.backend)

    param_csv = os.path.join(run_dir, "param.csv")
    if args.cells == "all":
        with open(param_csv) as f:
            cells = [row["cell"] for row in csv.DictReader(f)]
    else:
        cells = parse_comma_list(args.cells)

    sess_one = plot_trained.session_for_task(session, args.task)
    delta_ms = float(sess_one.delta_ms)
    opts = dict(
        (sess_one.train_opts or {}).get(f"{sess_one.primary_readout.name}_stimulus_opts")
        or {},
    )
    ms_pre, ms_pulse, ms_response = _stimulus_ms(args, opts)
    time_window = _time_window(args.check, ms_pre, ms_pulse, ms_response)
    print(
        f"check={args.check}  {args.task} radius={args.radius}  "
        f"TimeWindow(ms, 0, {time_window.stop:g})  "
        f"ms_pre={ms_pre:g} ms_pulse={ms_pulse:g} ms_response={ms_response:g}  "
        f"n_cells={len(cells)}",
        flush=True,
    )
    reports = analyze_spot_average(
        sess_one,
        p=p,
        cells=cells,
        task=args.task,
        time_window=time_window,
        radius=args.radius,
    )
    return cells, reports, delta_ms, ms_pre, ms_pulse, ms_response


def _print_oscillation(cells, reports, delta_ms, args) -> None:
    hdr = (
        f"{'Cell':<12} {'Osc?':<6} {'Reason':<12} "
        f"{'pkpk_z':>8} {'peak_f':>8} {'SNR':>8} {'pkpk':>8} {'std':>8} {'n':>6}"
    )
    print(f"\n{hdr}\n{'-' * len(hdr)}", flush=True)
    hit: list[tuple[str, dict]] = []
    for cell in cells:
        rep = reports.get(cell)
        if rep is None:
            print(f"{cell:<12} {'?':<6} {'no_report':<12}", flush=True)
            continue
        result = detect_oscillation(
            np.asarray(rep["v_post"], dtype=float),
            delta_ms=delta_ms,
            min_osc_freq_hz=args.min_freq,
            max_osc_freq_hz=args.max_freq,
            z_threshold=args.z_threshold,
        )
        yes = "YES" if result["flag"] else "NO"
        print(
            f"{cell:<12} {yes:<6} {result['reason']:<12} "
            f"{result.get('pkpk_z', 0):>8.2f} "
            f"{result.get('peak_freq_hz', 0):>8.2f} "
            f"{result.get('snr', 0):>8.2f} "
            f"{result.get('pkpk', 0):>8.2f} "
            f"{result.get('std', 0):>8.2f} "
            f"{result.get('n_samples', 0):>6}",
            flush=True,
        )
        if result["flag"]:
            hit.append((cell, result))

    print(f"\nOscillating ({len(hit)}/{len(cells)}):", flush=True)
    for cell, r in hit:
        print(
            f"  {cell}: freq={r['peak_freq_hz']:.2f}Hz  pkpk_z={r['pkpk_z']:.2f}  "
            f"SNR={r['snr']:.2f}  pkpk={r['pkpk']:.2f}mV",
            flush=True,
        )


def _print_flat(cells, reports, delta_ms, ms_pre, ms_pulse, args) -> None:
    i_onset = training.ms_to_t(ms_pre, delta_ms=delta_ms)
    i_pulse_end = training.ms_to_t(ms_pre + ms_pulse, delta_ms=delta_ms)
    baseline_n = max(1, int(float(args.baseline_ms) / delta_ms))
    hdr = (
        f"{'Cell':<12} {'Flat?':<6} {'Reason':<12} "
        f"{'base':>8} {'Δmean':>8} {'|μ|_pul':>8} {'pkpk_pul':>9} "
        f"{'max|Δ|':>8} {'pkpk_all':>9}"
    )
    print(f"\n{hdr}\n{'-' * len(hdr)}", flush=True)
    hit: list[tuple[str, dict]] = []
    for cell in cells:
        rep = reports.get(cell)
        if rep is None:
            print(f"{cell:<12} {'?':<6} {'no_report':<12}", flush=True)
            continue
        result = detect_flat(
            np.asarray(rep["v_post"], dtype=float),
            i_onset=i_onset,
            i_pulse_end=i_pulse_end,
            baseline_n=baseline_n,
            max_abs=args.max_abs,
            pkpk=args.pkpk,
            abs_mean_pul=args.abs_mean_pul,
        )
        yes = "YES" if result["flag"] else "NO"
        print(
            f"{cell:<12} {yes:<6} {result['reason']:<12} "
            f"{result.get('baseline', 0):>8.2f} "
            f"{result.get('delta_mean', 0):>8.2f} "
            f"{result.get('abs_mean_pul', 0):>8.3f} "
            f"{result.get('pkpk_pul', 0):>9.3f} "
            f"{result.get('max_abs', 0):>8.2f} "
            f"{result.get('pkpk_all', 0):>9.2f}",
            flush=True,
        )
        if result["flag"]:
            hit.append((cell, result))

    print(f"\nFlat ({len(hit)}/{len(cells)}):", flush=True)
    for cell, r in hit:
        print(
            f"  {cell}: max|Δ|={r['max_abs']:.3f}  pkpk_pul={r['pkpk_pul']:.3f}  "
            f"Δmean={r['delta_mean']:.3f}  base={r['baseline']:.2f}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run",
        help="run dir relative to PARAMETER_DIR (e.g. hp_lp/28693664-...) or absolute",
    )
    ap.add_argument(
        "--check",
        required=True,
        choices=(CHECK_OSCILLATION, CHECK_FLAT),
        help="oscillation: FFT in pre [0, ms_pre]; flat: post-onset vs late-pre",
    )
    ap.add_argument(
        "--cells",
        default="all",
        help="comma-separated cells or all (default: param.csv)",
    )
    ap.add_argument(
        "--ms-pre",
        type=float,
        default=None,
        help="pre length / onset ms; default: run spot_*_stimulus_opts.ms_pre",
    )
    ap.add_argument(
        "--ms-pulse",
        type=float,
        default=None,
        help="pulse length ms (flat); default: run spot_*_stimulus_opts.ms_pulse",
    )
    ap.add_argument(
        "--ms-response",
        type=float,
        default=None,
        help="response length ms (flat); default: run spot_*_stimulus_opts.ms_response",
    )
    ap.add_argument("--task", default="spot_bright")
    ap.add_argument("--radius", type=int, default=0, choices=(0, 1))
    # oscillation thresholds
    ap.add_argument("--z-threshold", type=float, default=2.0)
    ap.add_argument("--min-freq", type=float, default=0.5)
    ap.add_argument("--max-freq", type=float, default=20.0)
    # flat thresholds
    ap.add_argument(
        "--baseline-ms",
        type=float,
        default=200.0,
        help="late-pre baseline length ending at onset (flat)",
    )
    ap.add_argument("--max-abs", type=float, default=0.5, help="flat: max |Δ| after onset")
    ap.add_argument("--pkpk", type=float, default=1.0, help="flat: max pkpk after onset")
    ap.add_argument(
        "--abs-mean-pul",
        type=float,
        default=0.2,
        help="flat: max mean |Δ| during pulse",
    )
    args = ap.parse_args()

    cells, reports, delta_ms, ms_pre, ms_pulse, _ms_response = _load_reports(args)
    if args.check == CHECK_OSCILLATION:
        _print_oscillation(cells, reports, delta_ms, args)
    else:
        _print_flat(cells, reports, delta_ms, ms_pre, ms_pulse, args)


if __name__ == "__main__":
    main()
