"""Fit each Gruntman et al. 2018 Fig. 2B trace with one delayed LP.

The stimulus is a unit rectangular pulse whose duration is known from the
experiment.  Each digitized Vm trace is fit independently with

    Vm(t) = gain * LP_tau[pulse(t - delay)].

The digitized traces are already pre-stimulus baseline-subtracted, so the
baseline is fixed at zero.  ``gain_mv`` is the steady-state response to a
unit input, not the observed peak response.

Outputs:
  gruntman18_fit_lp.csv  one row per fitted trace
  gruntman18_fit_lp.png  20 raw-data/fit panels
  gruntman18_fit_lp_shared_location.csv  all parameters shared per location
  gruntman18_fit_lp_shared_location.png  location-shared fit panels
  gruntman18_fit_lp_shared_all.csv  location gains; one global tau and delay
  gruntman18_fit_lp_shared_all.png  global-tau/global-delay fit panels

Run:  ../.venv/bin/python gruntman18_fit_lp.py
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
INPUT_CSV = ROOT / "gruntman18" / "2b_digitized.csv"
OUT_CSV = HERE / "gruntman18_fit_lp.csv"
OUT_PNG = HERE / "gruntman18_fit_lp.png"
OUT_SHARED_LOCATION_CSV = HERE / "gruntman18_fit_lp_shared_location.csv"
OUT_SHARED_LOCATION_PNG = HERE / "gruntman18_fit_lp_shared_location.png"
OUT_SHARED_ALL_CSV = HERE / "gruntman18_fit_lp_shared_all.csv"
OUT_SHARED_ALL_PNG = HERE / "gruntman18_fit_lp_shared_all.png"

DURATIONS_MS = [160, 80, 40, 20]
POSITION_IDXS = [-2, -1, 0, 1, 2]
COLORS = {
    -2: "#b8e06a",
    -1: "#2d6b1f",
    0: "#111111",
    1: "#b03078",
    2: "#e8b0d0",
}

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
    response[during] = gain_mv * (
        1.0 - np.exp(-elapsed[during] / tau_ms)
    )

    after = elapsed >= duration_ms
    response[after] = (
        gain_mv
        * (1.0 - np.exp(-duration_ms / tau_ms))
        * np.exp(-(elapsed[after] - duration_ms) / tau_ms)
    )
    return response


def fit_trace(trace: pd.DataFrame) -> dict[str, float | np.ndarray]:
    """Fit tau, neural delay, and gain for one digitized curve."""
    trace = trace.sort_values("time_ms")
    time_ms = trace["time_ms"].to_numpy(dtype=float)
    vm_mv = trace["vm_mv"].to_numpy(dtype=float)
    duration_ms = float(trace["flash_duration_ms"].iloc[0])

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


def fit_location_shared(
    location_data: pd.DataFrame,
) -> tuple[list[dict[str, float | str]], dict[str, dict]]:
    """Jointly fit one gain, delay, and tau across four pulse durations."""
    traces = []
    for duration_ms in DURATIONS_MS:
        trace = location_data[
            location_data["flash_duration_ms"] == duration_ms
        ].sort_values("time_ms")
        if trace.empty:
            raise ValueError(f"missing {duration_ms} ms trace")
        time_ms = trace["time_ms"].to_numpy(dtype=float)
        vm_mv = trace["vm_mv"].to_numpy(dtype=float)
        finite = np.isfinite(time_ms) & np.isfinite(vm_mv)
        traces.append(
            {
                "trace_id": str(trace["trace_id"].iloc[0]),
                "duration_ms": float(duration_ms),
                "time_ms": time_ms[finite],
                "vm_mv": vm_mv[finite],
            }
        )

    gain0 = max(float(max(np.max(t["vm_mv"]) for t in traces)), 0.1)

    def residual(parameters: np.ndarray) -> np.ndarray:
        tau_ms, delay_ms, gain_mv = parameters
        return np.concatenate(
            [
                delayed_lp_pulse(
                    trace["time_ms"],
                    trace["duration_ms"],
                    tau_ms,
                    delay_ms,
                    gain_mv,
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
                x0=np.array([tau0, delay0, gain0]),
                bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
                loss="linear",
                max_nfev=20_000,
            )
            sse = float(np.dot(result.fun, result.fun))
            if best is None or sse < best[0]:
                best = (sse, result)

    assert best is not None
    pooled_sse, result = best
    tau_ms, delay_ms, gain_mv = map(float, result.x)
    pooled_vm = np.concatenate([trace["vm_mv"] for trace in traces])
    pooled_ss_total = float(np.sum((pooled_vm - np.mean(pooled_vm)) ** 2))
    pooled_r_squared = (
        1.0 - pooled_sse / pooled_ss_total if pooled_ss_total > 0.0 else np.nan
    )

    rows = []
    fits = {}
    position_idx = int(location_data["position_idx"].iloc[0])
    for trace in traces:
        prediction = delayed_lp_pulse(
            trace["time_ms"],
            trace["duration_ms"],
            tau_ms,
            delay_ms,
            gain_mv,
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
                "flash_duration_ms": int(trace["duration_ms"]),
                "position_idx": position_idx,
                "gain_mv": gain_mv,
                "delay_ms": delay_ms,
                "tau_ms": tau_ms,
                "effective_offset_ms": delay_ms + trace["duration_ms"],
                "r_squared": r_squared,
                "location_r_squared": pooled_r_squared,
                "sse": sse,
                "location_sse": pooled_sse,
                "n_points": len(trace["time_ms"]),
            }
        )
        fits[trace_id] = {
            "time_ms": trace["time_ms"],
            "vm_mv": trace["vm_mv"],
            "prediction_mv": prediction,
        }
    return rows, fits


def fit_all_shared(
    data: pd.DataFrame,
) -> tuple[list[dict[str, float | str]], dict[str, dict]]:
    """Fit five location gains while sharing one tau and delay globally."""
    traces = []
    for position_idx in POSITION_IDXS:
        for duration_ms in DURATIONS_MS:
            trace = data[
                (data["position_idx"] == position_idx)
                & (data["flash_duration_ms"] == duration_ms)
            ].sort_values("time_ms")
            if trace.empty:
                raise ValueError(f"missing position {position_idx:+d}, {duration_ms} ms trace")
            time_ms = trace["time_ms"].to_numpy(dtype=float)
            vm_mv = trace["vm_mv"].to_numpy(dtype=float)
            finite = np.isfinite(time_ms) & np.isfinite(vm_mv)
            traces.append(
                {
                    "trace_id": str(trace["trace_id"].iloc[0]),
                    "position_idx": position_idx,
                    "duration_ms": float(duration_ms),
                    "time_ms": time_ms[finite],
                    "vm_mv": vm_mv[finite],
                }
            )

    gain_starts = [
        max(
            float(
                max(
                    np.max(trace["vm_mv"])
                    for trace in traces
                    if trace["position_idx"] == position_idx
                )
            ),
            0.1,
        )
        for position_idx in POSITION_IDXS
    ]
    lower_bounds = np.concatenate((LOWER_BOUNDS[:2], np.zeros(len(POSITION_IDXS))))
    upper_bounds = np.concatenate(
        (UPPER_BOUNDS[:2], np.full(len(POSITION_IDXS), 500.0))
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        tau_ms, delay_ms = parameters[:2]
        gains = dict(zip(POSITION_IDXS, parameters[2:]))
        return np.concatenate(
            [
                delayed_lp_pulse(
                    trace["time_ms"],
                    trace["duration_ms"],
                    tau_ms,
                    delay_ms,
                    gains[trace["position_idx"]],
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
    gains = dict(zip(POSITION_IDXS, map(float, result.x[2:])))
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
            trace["duration_ms"],
            tau_ms,
            delay_ms,
            gains[trace["position_idx"]],
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
                "flash_duration_ms": int(trace["duration_ms"]),
                "position_idx": trace["position_idx"],
                "gain_mv": gains[trace["position_idx"]],
                "delay_ms": delay_ms,
                "tau_ms": tau_ms,
                "effective_offset_ms": delay_ms + trace["duration_ms"],
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


def plot_fits(results: pd.DataFrame, fits: dict[str, dict]) -> None:
    fig, axes = plt.subplots(
        len(DURATIONS_MS),
        len(POSITION_IDXS),
        figsize=(17, 12),
        sharex=True,
        squeeze=False,
    )

    for row_index, duration_ms in enumerate(DURATIONS_MS):
        for column_index, position_idx in enumerate(POSITION_IDXS):
            ax = axes[row_index, column_index]
            row = results[
                (results.flash_duration_ms == duration_ms)
                & (results.position_idx == position_idx)
            ].iloc[0]
            fit = fits[row.trace_id]
            color = COLORS[position_idx]

            ax.plot(
                fit["time_ms"],
                fit["vm_mv"],
                color=color,
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
                label="single-τ LP fit",
            )
            ax.axhline(0.0, color="0.75", lw=0.6)
            ax.axvline(0.0, color="0.6", lw=0.7, ls=":")
            ax.axvline(duration_ms, color="0.6", lw=0.7, ls=":")
            ax.axvline(row.delay_ms, color="#2474b5", lw=0.7, ls="-.")
            ax.axvline(
                row.effective_offset_ms, color="#2474b5", lw=0.7, ls="-."
            )
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
            if row_index == 0:
                ax.set_title(f"{position_idx:+d}", fontsize=11)
            if column_index == 0:
                ax.set_ylabel(f"{duration_ms} ms\nVm (mV)")
            if row_index == len(DURATIONS_MS) - 1:
                ax.set_xlabel("time (ms)")
            if row_index == 0 and column_index == len(POSITION_IDXS) - 1:
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Gruntman 2018 Fig. 2B — delayed single-time-constant LP fits",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)


def plot_shared_fits(
    results: pd.DataFrame,
    fits: dict[str, dict],
    output_path: Path,
    title: str,
    fit_label: str,
) -> None:
    """Plot four-duration fits whose parameters are shared by location."""
    fig, axes = plt.subplots(
        len(DURATIONS_MS),
        len(POSITION_IDXS),
        figsize=(17, 12),
        sharex=True,
        squeeze=False,
    )
    for row_index, duration_ms in enumerate(DURATIONS_MS):
        for column_index, position_idx in enumerate(POSITION_IDXS):
            ax = axes[row_index, column_index]
            row = results[
                (results.flash_duration_ms == duration_ms)
                & (results.position_idx == position_idx)
            ].iloc[0]
            fit = fits[row.trace_id]
            ax.plot(
                fit["time_ms"],
                fit["vm_mv"],
                color=COLORS[position_idx],
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
            ax.axvline(duration_ms, color="0.6", lw=0.7, ls=":")
            ax.axvline(row.delay_ms, color="#2474b5", lw=0.7, ls="-.")
            ax.axvline(
                row.effective_offset_ms, color="#2474b5", lw=0.7, ls="-."
            )
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
            if row_index == 0:
                ax.set_title(f"{position_idx:+d}", fontsize=11)
            if column_index == 0:
                ax.set_ylabel(f"{duration_ms} ms\nVm (mV)")
            if row_index == len(DURATIONS_MS) - 1:
                ax.set_xlabel("time (ms)")
            if row_index == 0 and column_index == len(POSITION_IDXS) - 1:
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    data = pd.read_csv(INPUT_CSV)
    group_columns = [
        "trace_id",
        "flash_duration_ms",
        "position_idx",
    ]
    rows = []
    fits: dict[str, dict] = {}

    for keys, trace in data.groupby(group_columns, sort=False):
        trace_id, duration_ms, position_idx = keys
        fit = fit_trace(trace)
        fits[str(trace_id)] = fit
        rows.append(
            {
                "trace_id": trace_id,
                "flash_duration_ms": duration_ms,
                "position_idx": position_idx,
                "gain_mv": fit["gain_mv"],
                "delay_ms": fit["delay_ms"],
                "tau_ms": fit["tau_ms"],
                "effective_offset_ms": fit["effective_offset_ms"],
                "r_squared": fit["r_squared"],
                "sse": fit["sse"],
                "n_points": fit["n_points"],
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["flash_duration_ms", "position_idx"], ascending=[False, True]
    )
    results.to_csv(OUT_CSV, index=False)
    plot_fits(results, fits)

    shared_rows = []
    shared_fits: dict[str, dict] = {}
    for position_idx in POSITION_IDXS:
        location_rows, location_fits = fit_location_shared(
            data[data["position_idx"] == position_idx]
        )
        shared_rows.extend(location_rows)
        shared_fits.update(location_fits)
    shared_results = pd.DataFrame(shared_rows).sort_values(
        ["flash_duration_ms", "position_idx"], ascending=[False, True]
    )
    shared_results.to_csv(OUT_SHARED_LOCATION_CSV, index=False)
    plot_shared_fits(
        shared_results,
        shared_fits,
        OUT_SHARED_LOCATION_PNG,
        "Gruntman 2018 Fig. 2B — gain, delay, and τ shared across durations per location",
        "location-shared LP fit",
    )

    all_rows, all_fits = fit_all_shared(data)
    all_results = pd.DataFrame(all_rows).sort_values(
        ["flash_duration_ms", "position_idx"], ascending=[False, True]
    )
    all_results.to_csv(OUT_SHARED_ALL_CSV, index=False)
    plot_shared_fits(
        all_results,
        all_fits,
        OUT_SHARED_ALL_PNG,
        "Gruntman 2018 Fig. 2B — location gains; one τ and delay shared by all traces",
        "global-τ/global-delay LP fit",
    )

    columns = [
        "trace_id",
        "gain_mv",
        "delay_ms",
        "tau_ms",
        "r_squared",
    ]
    print(results[columns].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_PNG}")
    shared_columns = [
        "trace_id",
        "gain_mv",
        "delay_ms",
        "tau_ms",
        "r_squared",
        "location_r_squared",
    ]
    print("\n=== shared across durations within each location ===")
    print(
        shared_results[shared_columns].to_string(
            index=False, float_format=lambda x: f"{x:.4g}"
        )
    )
    print(f"wrote {OUT_SHARED_LOCATION_CSV}")
    print(f"wrote {OUT_SHARED_LOCATION_PNG}")
    all_columns = [
        "trace_id",
        "gain_mv",
        "delay_ms",
        "tau_ms",
        "r_squared",
        "global_r_squared",
    ]
    print("\n=== five location gains; tau and delay shared across all traces ===")
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
