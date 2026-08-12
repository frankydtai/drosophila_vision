#!/usr/bin/env python3
"""Fit Gruntman et al. 2021 Fig. 2A T4 PD traces with one delayed LP.

The stimulus is a unit rectangular pulse (160 ms).  Each digitized T4 PC
(green) Vm trace is fit with

    Vm(t) = gain * LP_tau[pulse(t - delay)].

T5 and NC (black) traces are excluded.  Digitized traces are already
pre-stimulus baseline-subtracted, so the baseline is fixed at zero.
``gain_mv`` is the steady-state response to a unit input, not the observed
peak response.

Outputs:
  gruntman21_fit_lp.csv  one row per independently fitted T4-PC trace
  gruntman21_fit_lp.png  12 raw-data/fit panels
  gruntman21_fit_lp_shared_all.csv  per-position gains; one global tau and delay
  gruntman21_fit_lp_shared_all.png  global-tau/global-delay fit panels

Run:  ../.venv/bin/python gruntman21_fit_lp.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT_CSV = ROOT / "gruntman21" / "2a_digitized.csv"
OUT_CSV = HERE / "gruntman21_fit_lp.csv"
OUT_PNG = HERE / "gruntman21_fit_lp.png"
OUT_SHARED_ALL_CSV = HERE / "gruntman21_fit_lp_shared_all.csv"
OUT_SHARED_ALL_PNG = HERE / "gruntman21_fit_lp_shared_all.png"

DURATION_MS = 160.0
POSITIONS = list(range(-5, 7))
N_ROW, N_COL = 2, 6

TAU_STARTS_MS = (5.0, 15.0, 30.0, 60.0, 120.0, 250.0)
DELAY_STARTS_MS = (0.0, 10.0, 20.0, 30.0, 45.0, 65.0, 90.0)
LOWER_BOUNDS = np.array([0.5, 0.0, 0.0])  # tau, delay, gain
UPPER_BOUNDS = np.array([500.0, 120.0, 500.0])


def delayed_lp_pulse(
    time_ms: np.ndarray,
    duration_ms: float,
    tau_ms: float,
    delay_ms: float,
    gain_mv: float,
) -> np.ndarray:
    """Analytic first-order LP response to a delayed unit pulse."""
    elapsed = np.asarray(time_ms, dtype=float) - delay_ms
    response = np.zeros_like(elapsed)

    during = (elapsed >= 0.0) & (elapsed < duration_ms)
    response[during] = gain_mv * (1.0 - np.exp(-elapsed[during] / tau_ms))

    after = elapsed >= duration_ms
    response[after] = (
        gain_mv
        * (1.0 - np.exp(-duration_ms / tau_ms))
        * np.exp(-(elapsed[after] - duration_ms) / tau_ms)
    )
    return response


def load_t4_pd(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only T4 preferred-contrast (PC / green) traces."""
    return data[(data["cell_type"] == "T4") & (data["contrast"] == "PC")].copy()


def fit_trace(trace: pd.DataFrame) -> dict[str, float | np.ndarray]:
    """Fit tau, neural delay, and gain for one digitized curve."""
    trace = trace.sort_values("time_ms")
    time_ms = trace["time_ms"].to_numpy(dtype=float)
    vm_mv = trace["vm_mv"].to_numpy(dtype=float)
    duration_ms = DURATION_MS

    finite = np.isfinite(time_ms) & np.isfinite(vm_mv)
    time_ms = time_ms[finite]
    vm_mv = vm_mv[finite]
    gain0 = max(float(np.max(vm_mv)), 0.1)

    def residual(parameters: np.ndarray) -> np.ndarray:
        tau_ms, delay_ms, gain_mv = parameters
        prediction = delayed_lp_pulse(
            time_ms, duration_ms, tau_ms, delay_ms, gain_mv
        )
        return prediction - vm_mv

    best = None
    for tau0 in TAU_STARTS_MS:
        for delay0 in DELAY_STARTS_MS:
            result = least_squares(
                residual,
                x0=np.array([tau0, delay0, gain0]),
                bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
                loss="linear",
                max_nfev=20_000,
            )
            sse = float(np.dot(result.fun, result.fun))
            if best is None or sse < best[0]:
                best = (sse, result)

    assert best is not None
    sse, result = best
    tau_ms, delay_ms, gain_mv = map(float, result.x)
    prediction = delayed_lp_pulse(
        time_ms, duration_ms, tau_ms, delay_ms, gain_mv
    )
    ss_total = float(np.sum((vm_mv - np.mean(vm_mv)) ** 2))
    r_squared = 1.0 - sse / ss_total if ss_total > 0.0 else np.nan

    return {
        "tau_ms": tau_ms,
        "delay_ms": delay_ms,
        "gain_mv": gain_mv,
        "effective_offset_ms": delay_ms + duration_ms,
        "r_squared": r_squared,
        "sse": sse,
        "n_points": len(time_ms),
        "time_ms": time_ms,
        "vm_mv": vm_mv,
        "prediction_mv": prediction,
    }


def fit_all_shared(
    data: pd.DataFrame,
) -> tuple[list[dict[str, float | str]], dict[str, dict]]:
    """Fit per-position gains while sharing one tau and delay globally."""
    traces = []
    for position in POSITIONS:
        trace = data[data["position"] == position].sort_values("time_ms")
        if trace.empty:
            raise ValueError(f"missing T4 PC position {position:+d}")
        time_ms = trace["time_ms"].to_numpy(dtype=float)
        vm_mv = trace["vm_mv"].to_numpy(dtype=float)
        finite = np.isfinite(time_ms) & np.isfinite(vm_mv)
        traces.append(
            {
                "trace_id": str(trace["trace_id"].iloc[0]),
                "position": position,
                "time_ms": time_ms[finite],
                "vm_mv": vm_mv[finite],
            }
        )

    gain_starts = [
        max(float(np.max(trace["vm_mv"])), 0.1) for trace in traces
    ]
    lower_bounds = np.concatenate((LOWER_BOUNDS[:2], np.zeros(len(POSITIONS))))
    upper_bounds = np.concatenate(
        (UPPER_BOUNDS[:2], np.full(len(POSITIONS), 500.0))
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        tau_ms, delay_ms = parameters[:2]
        gains = dict(zip(POSITIONS, parameters[2:]))
        return np.concatenate(
            [
                delayed_lp_pulse(
                    trace["time_ms"],
                    DURATION_MS,
                    tau_ms,
                    delay_ms,
                    gains[trace["position"]],
                )
                - trace["vm_mv"]
                for trace in traces
            ]
        )

    best = None
    for tau0 in TAU_STARTS_MS:
        for delay0 in DELAY_STARTS_MS:
            result = least_squares(
                residual,
                x0=np.array([tau0, delay0, *gain_starts]),
                bounds=(lower_bounds, upper_bounds),
                loss="linear",
                max_nfev=20_000,
            )
            sse = float(np.dot(result.fun, result.fun))
            if best is None or sse < best[0]:
                best = (sse, result)

    assert best is not None
    global_sse, result = best
    tau_ms, delay_ms = map(float, result.x[:2])
    gains = dict(zip(POSITIONS, map(float, result.x[2:])))
    pooled_vm = np.concatenate([trace["vm_mv"] for trace in traces])
    pooled_ss_total = float(np.sum((pooled_vm - np.mean(pooled_vm)) ** 2))
    global_r_squared = (
        1.0 - global_sse / pooled_ss_total if pooled_ss_total > 0.0 else np.nan
    )

    rows = []
    fits = {}
    for trace in traces:
        prediction = delayed_lp_pulse(
            trace["time_ms"],
            DURATION_MS,
            tau_ms,
            delay_ms,
            gains[trace["position"]],
        )
        residual_trace = prediction - trace["vm_mv"]
        sse = float(np.dot(residual_trace, residual_trace))
        ss_total = float(
            np.sum((trace["vm_mv"] - np.mean(trace["vm_mv"])) ** 2)
        )
        r_squared = 1.0 - sse / ss_total if ss_total > 0.0 else np.nan
        trace_id = trace["trace_id"]
        rows.append(
            {
                "trace_id": trace_id,
                "position": trace["position"],
                "gain_mv": gains[trace["position"]],
                "delay_ms": delay_ms,
                "tau_ms": tau_ms,
                "effective_offset_ms": delay_ms + DURATION_MS,
                "r_squared": r_squared,
                "global_r_squared": global_r_squared,
                "sse": sse,
                "global_sse": global_sse,
                "n_points": len(trace["time_ms"]),
            }
        )
        fits[trace_id] = {
            "time_ms": trace["time_ms"],
            "vm_mv": trace["vm_mv"],
            "prediction_mv": prediction,
        }
    return rows, fits


def plot_fits(
    results: pd.DataFrame,
    fits: dict[str, dict],
    output_path: Path,
    title: str,
    fit_label: str,
) -> None:
    fig, axes = plt.subplots(
        N_ROW,
        N_COL,
        figsize=(18, 7),
        sharex=True,
        squeeze=False,
    )
    for panel_index, position in enumerate(POSITIONS):
        row_index, column_index = divmod(panel_index, N_COL)
        ax = axes[row_index, column_index]
        row = results[results.position == position].iloc[0]
        fit = fits[row.trace_id]

        ax.plot(
            fit["time_ms"],
            fit["vm_mv"],
            color="#2ca02c",
            lw=1.2,
            alpha=0.75,
            label="digitized data",
        )
        ax.plot(
            fit["time_ms"],
            fit["prediction_mv"],
            color="#2474b5",
            lw=2.0,
            ls="--",
            label=fit_label,
        )
        ax.axhline(0.0, color="0.75", lw=0.6)
        ax.axvline(0.0, color="0.6", lw=0.7, ls=":")
        ax.axvline(DURATION_MS, color="0.6", lw=0.7, ls=":")
        ax.axvline(row.delay_ms, color="#2474b5", lw=0.7, ls="-.")
        ax.axvline(row.effective_offset_ms, color="#2474b5", lw=0.7, ls="-.")
        ax.text(
            0.97,
            0.96,
            (
                f"gain={row.gain_mv:.2f} mV\n"
                f"delay={row.delay_ms:.1f} ms\n"
                f"τ={row.tau_ms:.1f} ms\n"
                f"R²={row.r_squared:.3f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )
        ax.set_title(f"{position:+d}", fontsize=11)
        if column_index == 0:
            ax.set_ylabel("Vm (mV)")
        if row_index == N_ROW - 1:
            ax.set_xlabel("time (ms)")
        if panel_index == len(POSITIONS) - 1:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    data = load_t4_pd(pd.read_csv(INPUT_CSV))
    rows = []
    fits: dict[str, dict] = {}

    for position in POSITIONS:
        trace = data[data["position"] == position]
        if trace.empty:
            raise SystemExit(f"missing T4 PC position {position:+d}")
        fit = fit_trace(trace)
        trace_id = str(trace["trace_id"].iloc[0])
        fits[trace_id] = fit
        rows.append(
            {
                "trace_id": trace_id,
                "position": position,
                "gain_mv": fit["gain_mv"],
                "delay_ms": fit["delay_ms"],
                "tau_ms": fit["tau_ms"],
                "effective_offset_ms": fit["effective_offset_ms"],
                "r_squared": fit["r_squared"],
                "sse": fit["sse"],
                "n_points": fit["n_points"],
            }
        )

    results = pd.DataFrame(rows).sort_values("position")
    results.to_csv(OUT_CSV, index=False)
    plot_fits(
        results,
        fits,
        OUT_PNG,
        "Gruntman 2021 Fig. 2A T4 PD — delayed single-time-constant LP fits",
        "single-τ LP fit",
    )

    all_rows, all_fits = fit_all_shared(data)
    all_results = pd.DataFrame(all_rows).sort_values("position")
    all_results.to_csv(OUT_SHARED_ALL_CSV, index=False)
    plot_fits(
        all_results,
        all_fits,
        OUT_SHARED_ALL_PNG,
        "Gruntman 2021 Fig. 2A T4 PD — position gains; one τ and delay shared by all",
        "global-τ/global-delay LP fit",
    )

    columns = ["trace_id", "gain_mv", "delay_ms", "tau_ms", "r_squared"]
    print(results[columns].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_PNG}")
    all_columns = [
        "trace_id",
        "gain_mv",
        "delay_ms",
        "tau_ms",
        "r_squared",
        "global_r_squared",
    ]
    print("\n=== position gains; tau and delay shared across all T4 PD traces ===")
    print(
        all_results[all_columns].to_string(
            index=False, float_format=lambda x: f"{x:.4g}"
        )
    )
    print(f"wrote {OUT_SHARED_ALL_CSV}")
    print(f"wrote {OUT_SHARED_ALL_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
