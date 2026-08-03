"""FFT oscillation check on spot R0-average v_post in the pre window.

Uses ``analyze.cell_dynamics.analyze_spot_average`` only — no second forward.
Spot time axis: ``0`` = trial start; pre = ``TimeWindow("ms", 0, ms_pre)``
(not ``-ms_pre, 0``). See ``analyze.cell_dynamics`` module doc.

Usage (from ``vision/simulation/``)::

    ../.venv/bin/python test/detect_oscillation.py \\
      hp_lp/28693664-run-nofsteps-500-lrs-0.1-a-slow-init.L1,L2-0.6-ms-pre-1000-ms-pulse-100-ms-response-400 \\
      --ms-pre 1000
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
        return {"oscillating": False, "reason": "too_short", "n_samples": n}

    mean = float(np.mean(v_trace))
    std = float(np.std(v_trace))
    if std < 1e-6:
        return {"oscillating": False, "reason": "flat", "std": std, "n_samples": n}

    v_detrend = v_trace - mean
    dt = delta_ms / 1000.0
    freqs = np.fft.rfftfreq(n, dt)
    fft_mag = np.abs(np.fft.rfft(v_detrend))
    mask = (freqs >= min_osc_freq_hz) & (freqs <= max_osc_freq_hz)
    if not np.any(mask):
        return {
            "oscillating": False,
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
    oscillating = pkpk_z > z_threshold and peak_power > peak_threshold and snr > 2.0
    return {
        "oscillating": oscillating,
        "reason": "oscillation" if oscillating else "stable",
        "mean": mean,
        "std": std,
        "pkpk": pkpk,
        "pkpk_z": pkpk_z,
        "peak_freq_hz": peak_freq,
        "peak_power": peak_power,
        "snr": snr,
        "n_samples": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run",
        help="run dir relative to PARAMETER_DIR (e.g. hp_lp/28693664-...) or absolute",
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
        help=(
            "pre window STOP for TimeWindow ms [0, STOP]; default: run "
            "spot_*_stimulus_opts.ms_pre (spot absolute ms; 0=trial start)"
        ),
    )
    ap.add_argument("--task", default="spot_bright")
    ap.add_argument("--radius", type=int, default=0, choices=(0, 1))
    ap.add_argument("--z-threshold", type=float, default=2.0)
    ap.add_argument("--min-freq", type=float, default=0.5)
    ap.add_argument("--max-freq", type=float, default=20.0)
    args = ap.parse_args()

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
    ms_pre = float(args.ms_pre) if args.ms_pre is not None else float(opts["ms_pre"])
    # Spot absolute aligned ms: 0 = trial start; pre ends at ms_pre (onset).
    time_window = TimeWindow(kind="ms", start=0.0, stop=ms_pre)
    print(
        f"R0-average {args.task} radius={args.radius}  "
        f"TimeWindow(ms, 0, {ms_pre:g})  n_cells={len(cells)}",
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

    hdr = (
        f"{'Cell':<12} {'Osc?':<6} {'Reason':<12} "
        f"{'pkpk_z':>8} {'peak_f':>8} {'SNR':>8} {'pkpk':>8} {'std':>8} {'n':>6}"
    )
    print(f"\n{hdr}\n{'-' * len(hdr)}", flush=True)
    oscillating: list[tuple[str, dict]] = []
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
        osc = "YES" if result["oscillating"] else "NO"
        print(
            f"{cell:<12} {osc:<6} {result['reason']:<12} "
            f"{result.get('pkpk_z', 0):>8.2f} "
            f"{result.get('peak_freq_hz', 0):>8.2f} "
            f"{result.get('snr', 0):>8.2f} "
            f"{result.get('pkpk', 0):>8.2f} "
            f"{result.get('std', 0):>8.2f} "
            f"{result.get('n_samples', 0):>6}",
            flush=True,
        )
        if result["oscillating"]:
            oscillating.append((cell, result))

    print(f"\nOscillating ({len(oscillating)}/{len(cells)}):", flush=True)
    for cell, r in oscillating:
        print(
            f"  {cell}: freq={r['peak_freq_hz']:.2f}Hz  pkpk_z={r['pkpk_z']:.2f}  "
            f"SNR={r['snr']:.2f}  pkpk={r['pkpk']:.2f}mV",
            flush=True,
        )


if __name__ == "__main__":
    main()
