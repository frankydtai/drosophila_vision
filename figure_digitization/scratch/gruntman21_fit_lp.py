#!/usr/bin/env python3
"""Fit Gruntman et al. 2021 Fig. 2A T4 PD traces with two delayed LPs.

The stimulus is a unit rectangular pulse (160 ms).  Each digitized T4 PC
(green) Vm trace is fit with

    Vm(t) = gain_pos * LP_tau_pos[pulse(t - delay_pos)]
          + gain_neg * LP_tau_neg[pulse(t - delay_neg)]

where gain_pos >= 0 and gain_neg <= 0 (enforced by bounds), so the model
has one positive and one negative channel with independent tau and delay.

T5 and NC (black) traces are excluded.  Digitized traces are already
pre-stimulus baseline-subtracted, so the baseline is fixed at zero.

Outputs:
  gruntman21_fit_lp.csv  one row per independently fitted T4-PC trace
  gruntman21_fit_lp.png  12 raw-data/fit panels
  gruntman21_fit_lp.csv  non-mirrored fit parameters
  gruntman21_fit_lp.png  non-mirrored fit panels

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
OUT_MIRRORED_D_CSV = HERE / "gruntman21_fit_lp_mirrored_d.csv"
OUT_MIRRORED_D_PNG = HERE / "gruntman21_fit_lp_mirrored_d.png"
OUT_MIRRORED_T_CSV = HERE / "gruntman21_fit_lp_mirrored_t.csv"
OUT_MIRRORED_T_PNG = HERE / "gruntman21_fit_lp_mirrored_t.png"
OUT_MIRRORED_G_CSV = HERE / "gruntman21_fit_lp_mirrored_g.csv"
OUT_MIRRORED_G_PNG = HERE / "gruntman21_fit_lp_mirrored_g.png"

DURATION_MS = 160.0
POSITIONS = list(range(-6, 6))
N_COL = 12

TAU_POS_STARTS_MS = (200.0,)
TAU_NEG_STARTS_MS = (300.0,)
DELAY_POS_STARTS_MS = (20.0,)
DELAY_NEG_STARTS_MS = (50.0,)


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


def two_pulse_lp(
    time_ms: np.ndarray,
    duration_ms: float,
    tau_pos_ms: float,
    tau_neg_ms: float,
    delay_pos_ms: float,
    gain_pos_mv: float,
    delay_neg_ms: float,
    gain_neg_mv: float,
) -> np.ndarray:
    """Sum of a positive and a negative LP response, each with its own tau and delay."""
    return delayed_lp_pulse(
        time_ms, duration_ms, tau_pos_ms, delay_pos_ms, gain_pos_mv
    ) + delayed_lp_pulse(
        time_ms, duration_ms, tau_neg_ms, delay_neg_ms, -abs(gain_neg_mv)
    )


def load_traces(data: pd.DataFrame, cell_type: str, contrast: str) -> pd.DataFrame:
    return data[(data["cell_type"] == cell_type) & (data["contrast"] == contrast)].copy()


def fit_non_mirrored(
    data: pd.DataFrame,
) -> tuple[list[dict[str, float | str]], dict[str, dict]]:
    """Fit per-position gain_pos and gain_neg; share tau_pos, tau_neg, delay_pos, delay_neg globally."""
    traces = []
    for position in POSITIONS:
        trace = data[data["position"] == position].sort_values("time_ms")
        if trace.empty:
            raise ValueError(f"missing position {position:+d}")
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

    n_pos = len(POSITIONS)
    gain_pos_starts = [max(float(np.max(t["vm_mv"])), 0.1) for t in traces]
    gain_neg_starts = [max(-float(np.min(t["vm_mv"])), 0.1) for t in traces]
    # parameters: [tau_pos, tau_neg, delay_pos, delay_neg, gain_pos_0..n, gain_neg_0..n]
    n_global = 4
    lower_bounds = np.concatenate((
        [0.5, 0.5, 0.0, 0.0],
        np.zeros(n_pos),
        np.zeros(n_pos),
    ))
    upper_bounds = np.concatenate((
        [1000.0, 1000.0, 120.0, 200.0],
        np.full(n_pos, 100.0),
        np.full(n_pos, 100.0),
    ))

    def residual(parameters: np.ndarray) -> np.ndarray:
        tau_pos_ms, tau_neg_ms, delay_pos_ms, delay_neg_ms = parameters[:n_global]
        gains_pos = dict(zip(POSITIONS, parameters[n_global: n_global + n_pos]))
        gains_neg = dict(zip(POSITIONS, parameters[n_global + n_pos:]))
        return np.concatenate(
            [
                two_pulse_lp(
                    trace["time_ms"], DURATION_MS, tau_pos_ms, tau_neg_ms,
                    delay_pos_ms, gains_pos[trace["position"]],
                    delay_neg_ms, gains_neg[trace["position"]],
                )
                - trace["vm_mv"]
                for trace in traces
            ]
        )

    best = None
    for tau_pos0 in TAU_POS_STARTS_MS:
        for tau_neg0 in TAU_NEG_STARTS_MS:
            for delay_pos0 in DELAY_POS_STARTS_MS:
                for delay_neg0 in DELAY_NEG_STARTS_MS:
                    result = least_squares(
                        residual,
                        x0=np.array([tau_pos0, tau_neg0, delay_pos0, delay_neg0,
                                     *gain_pos_starts, *gain_neg_starts]),
                        bounds=(lower_bounds, upper_bounds),
                        loss="linear",
                        max_nfev=20_000,
                    )
                    sse = float(np.dot(result.fun, result.fun))
                    if best is None or sse < best[0]:
                        best = (sse, result)

    assert best is not None
    global_sse, result = best
    tau_pos_ms, tau_neg_ms, delay_pos_ms, delay_neg_ms = map(float, result.x[:n_global])
    gains_pos = dict(zip(POSITIONS, map(float, result.x[n_global: n_global + n_pos])))
    gains_neg = dict(zip(POSITIONS, map(float, result.x[n_global + n_pos:])))
    pooled_vm = np.concatenate([t["vm_mv"] for t in traces])
    pooled_ss_total = float(np.sum((pooled_vm - np.mean(pooled_vm)) ** 2))
    global_r_squared = (
        1.0 - global_sse / pooled_ss_total if pooled_ss_total > 0.0 else np.nan
    )

    rows = []
    fits = {}
    for trace in traces:
        prediction = two_pulse_lp(
            trace["time_ms"], DURATION_MS, tau_pos_ms, tau_neg_ms,
            delay_pos_ms, gains_pos[trace["position"]],
            delay_neg_ms, gains_neg[trace["position"]],
        )
        residual_trace = prediction - trace["vm_mv"]
        sse = float(np.dot(residual_trace, residual_trace))
        ss_total = float(np.sum((trace["vm_mv"] - np.mean(trace["vm_mv"])) ** 2))
        r_squared = 1.0 - sse / ss_total if ss_total > 0.0 else np.nan
        trace_id = trace["trace_id"]
        rows.append(
            {
                "trace_id": trace_id,
                "position": trace["position"],
                "gain_pos_mv": gains_pos[trace["position"]],
                "gain_neg_mv": -abs(gains_neg[trace["position"]]),
                "delay_pos_ms": delay_pos_ms,
                "delay_neg_ms": delay_neg_ms,
                "tau_pos_ms": tau_pos_ms,
                "tau_neg_ms": tau_neg_ms,
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


def fit_mirrored_d(
    data_pc: pd.DataFrame,
    data_nc: pd.DataFrame,
) -> tuple[list[dict], dict[str, dict], list[dict], dict[str, dict]]:
    """Joint fit: PC delay_pos/neg = NC delay_neg/pos (mirrored delays)."""

    def _prep(data: pd.DataFrame) -> list[dict]:
        traces = []
        for position in POSITIONS:
            trace = data[data["position"] == position].sort_values("time_ms")
            if trace.empty:
                raise ValueError(f"missing position {position:+d}")
            time_ms = trace["time_ms"].to_numpy(dtype=float)
            vm_mv = trace["vm_mv"].to_numpy(dtype=float)
            finite = np.isfinite(time_ms) & np.isfinite(vm_mv)
            traces.append({
                "trace_id": str(trace["trace_id"].iloc[0]),
                "position": position,
                "time_ms": time_ms[finite],
                "vm_mv": vm_mv[finite],
            })
        return traces

    traces_pc = _prep(data_pc)
    traces_nc = _prep(data_nc)
    n_pos = len(POSITIONS)
    # parameters: [tau_pos, tau_neg, delay_pos, delay_neg,
    #              pc_gain_pos_0..n, pc_gain_neg_0..n,
    #              nc_gain_pos_0..n, nc_gain_neg_0..n]
    n_global = 4
    n_per_row = 2 * n_pos
    gain_pos_pc = [max(float(np.max(t["vm_mv"])), 0.1) for t in traces_pc]
    gain_neg_pc = [max(-float(np.min(t["vm_mv"])), 0.1) for t in traces_pc]
    gain_pos_nc = [max(float(np.max(t["vm_mv"])), 0.1) for t in traces_nc]
    gain_neg_nc = [max(-float(np.min(t["vm_mv"])), 0.1) for t in traces_nc]
    lower_bounds = np.concatenate((
        [0.5, 0.5, 0.0, 0.0],
        np.zeros(n_per_row), np.zeros(n_per_row),
    ))
    upper_bounds = np.concatenate((
        [1000.0, 1000.0, 120.0, 200.0],
        np.full(n_per_row, 100.0), np.full(n_per_row, 100.0),
    ))

    def residual(parameters: np.ndarray) -> np.ndarray:
        tau_pos_ms, tau_neg_ms, delay_pos_ms, delay_neg_ms = parameters[:n_global]
        pc_gp = dict(zip(POSITIONS, parameters[n_global: n_global + n_pos]))
        pc_gn = dict(zip(POSITIONS, parameters[n_global + n_pos: n_global + n_per_row]))
        nc_off = n_global + n_per_row
        nc_gp = dict(zip(POSITIONS, parameters[nc_off: nc_off + n_pos]))
        nc_gn = dict(zip(POSITIONS, parameters[nc_off + n_pos: nc_off + n_per_row]))
        parts = []
        for t in traces_pc:
            parts.append(two_pulse_lp(
                t["time_ms"], DURATION_MS, tau_pos_ms, tau_neg_ms,
                delay_pos_ms, pc_gp[t["position"]],
                delay_neg_ms, pc_gn[t["position"]],
            ) - t["vm_mv"])
        for t in traces_nc:
            # mirrored: NC delay_pos ← delay_neg, NC delay_neg ← delay_pos
            parts.append(two_pulse_lp(
                t["time_ms"], DURATION_MS, tau_pos_ms, tau_neg_ms,
                delay_neg_ms, nc_gp[t["position"]],
                delay_pos_ms, nc_gn[t["position"]],
            ) - t["vm_mv"])
        return np.concatenate(parts)

    best = None
    for tau_pos0 in TAU_POS_STARTS_MS:
        for tau_neg0 in TAU_NEG_STARTS_MS:
            for delay_pos0 in DELAY_POS_STARTS_MS:
                for delay_neg0 in DELAY_NEG_STARTS_MS:
                    result = least_squares(
                        residual,
                        x0=np.array([tau_pos0, tau_neg0, delay_pos0, delay_neg0,
                                     *gain_pos_pc, *gain_neg_pc,
                                     *gain_pos_nc, *gain_neg_nc]),
                        bounds=(lower_bounds, upper_bounds),
                        loss="linear",
                        max_nfev=40_000,
                    )
                    sse = float(np.dot(result.fun, result.fun))
                    if best is None or sse < best[0]:
                        best = (sse, result)

    assert best is not None
    global_sse, result = best
    tau_pos_ms, tau_neg_ms, delay_pos_ms, delay_neg_ms = map(float, result.x[:n_global])
    pc_gp = dict(zip(POSITIONS, map(float, result.x[n_global: n_global + n_pos])))
    pc_gn = dict(zip(POSITIONS, map(float, result.x[n_global + n_pos: n_global + n_per_row])))
    nc_off = n_global + n_per_row
    nc_gp = dict(zip(POSITIONS, map(float, result.x[nc_off: nc_off + n_pos])))
    nc_gn = dict(zip(POSITIONS, map(float, result.x[nc_off + n_pos: nc_off + n_per_row])))

    pooled_vm = np.concatenate(
        [t["vm_mv"] for t in traces_pc] + [t["vm_mv"] for t in traces_nc])
    pooled_ss_total = float(np.sum((pooled_vm - np.mean(pooled_vm)) ** 2))
    global_r_squared = (
        1.0 - global_sse / pooled_ss_total if pooled_ss_total > 0.0 else np.nan)

    def _emit(traces, gains_pos, gains_neg, d_pos, d_neg):
        rows = []
        fits = {}
        for t in traces:
            prediction = two_pulse_lp(
                t["time_ms"], DURATION_MS, tau_pos_ms, tau_neg_ms,
                d_pos, gains_pos[t["position"]],
                d_neg, gains_neg[t["position"]],
            )
            res = prediction - t["vm_mv"]
            sse = float(np.dot(res, res))
            ss_tot = float(np.sum((t["vm_mv"] - np.mean(t["vm_mv"])) ** 2))
            r_sq = 1.0 - sse / ss_tot if ss_tot > 0.0 else np.nan
            rows.append({
                "trace_id": t["trace_id"],
                "position": t["position"],
                "gain_pos_mv": gains_pos[t["position"]],
                "gain_neg_mv": -abs(gains_neg[t["position"]]),
                "delay_pos_ms": d_pos,
                "delay_neg_ms": d_neg,
                "tau_pos_ms": tau_pos_ms,
                "tau_neg_ms": tau_neg_ms,
                "r_squared": r_sq,
                "global_r_squared": global_r_squared,
                "sse": sse,
                "global_sse": global_sse,
                "n_points": len(t["time_ms"]),
            })
            fits[t["trace_id"]] = {
                "time_ms": t["time_ms"],
                "vm_mv": t["vm_mv"],
                "prediction_mv": prediction,
            }
        return rows, fits

    pc_rows, pc_fits = _emit(traces_pc, pc_gp, pc_gn, delay_pos_ms, delay_neg_ms)
    nc_rows, nc_fits = _emit(traces_nc, nc_gp, nc_gn, delay_neg_ms, delay_pos_ms)
    return pc_rows, pc_fits, nc_rows, nc_fits


def fit_mirrored_t(
    data_pc: pd.DataFrame,
    data_nc: pd.DataFrame,
) -> tuple[list[dict], dict[str, dict], list[dict], dict[str, dict]]:
    """Joint fit: PC tau_pos/neg = NC tau_neg/pos (mirrored taus); delays independent per row."""

    def _prep(data: pd.DataFrame) -> list[dict]:
        traces = []
        for position in POSITIONS:
            trace = data[data["position"] == position].sort_values("time_ms")
            if trace.empty:
                raise ValueError(f"missing position {position:+d}")
            time_ms = trace["time_ms"].to_numpy(dtype=float)
            vm_mv = trace["vm_mv"].to_numpy(dtype=float)
            finite = np.isfinite(time_ms) & np.isfinite(vm_mv)
            traces.append({
                "trace_id": str(trace["trace_id"].iloc[0]),
                "position": position,
                "time_ms": time_ms[finite],
                "vm_mv": vm_mv[finite],
            })
        return traces

    traces_pc = _prep(data_pc)
    traces_nc = _prep(data_nc)
    n_pos = len(POSITIONS)
    # parameters: [tau_pos, tau_neg, pc_delay_pos, pc_delay_neg, nc_delay_pos, nc_delay_neg,
    #              pc_gain_pos_0..n, pc_gain_neg_0..n,
    #              nc_gain_pos_0..n, nc_gain_neg_0..n]
    n_global = 6
    n_per_row = 2 * n_pos
    gain_pos_pc = [max(float(np.max(t["vm_mv"])), 0.1) for t in traces_pc]
    gain_neg_pc = [max(-float(np.min(t["vm_mv"])), 0.1) for t in traces_pc]
    gain_pos_nc = [max(float(np.max(t["vm_mv"])), 0.1) for t in traces_nc]
    gain_neg_nc = [max(-float(np.min(t["vm_mv"])), 0.1) for t in traces_nc]
    lower_bounds = np.concatenate((
        [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        np.zeros(n_per_row), np.zeros(n_per_row),
    ))
    upper_bounds = np.concatenate((
        [1000.0, 1000.0, 120.0, 200.0, 120.0, 200.0],
        np.full(n_per_row, 100.0), np.full(n_per_row, 100.0),
    ))

    def residual(parameters: np.ndarray) -> np.ndarray:
        (tau_pos_ms, tau_neg_ms,
         pc_d_pos, pc_d_neg, nc_d_pos, nc_d_neg) = parameters[:n_global]
        pc_gp = dict(zip(POSITIONS, parameters[n_global: n_global + n_pos]))
        pc_gn = dict(zip(POSITIONS, parameters[n_global + n_pos: n_global + n_per_row]))
        nc_off = n_global + n_per_row
        nc_gp = dict(zip(POSITIONS, parameters[nc_off: nc_off + n_pos]))
        nc_gn = dict(zip(POSITIONS, parameters[nc_off + n_pos: nc_off + n_per_row]))
        parts = []
        for t in traces_pc:
            parts.append(two_pulse_lp(
                t["time_ms"], DURATION_MS, tau_pos_ms, tau_neg_ms,
                pc_d_pos, pc_gp[t["position"]],
                pc_d_neg, pc_gn[t["position"]],
            ) - t["vm_mv"])
        for t in traces_nc:
            # mirrored tau: NC tau_pos ← tau_neg, NC tau_neg ← tau_pos
            parts.append(two_pulse_lp(
                t["time_ms"], DURATION_MS, tau_neg_ms, tau_pos_ms,
                nc_d_pos, nc_gp[t["position"]],
                nc_d_neg, nc_gn[t["position"]],
            ) - t["vm_mv"])
        return np.concatenate(parts)

    best = None
    for tau_pos0 in TAU_POS_STARTS_MS:
        for tau_neg0 in TAU_NEG_STARTS_MS:
            for pc_d_pos0 in DELAY_POS_STARTS_MS:
                for pc_d_neg0 in DELAY_NEG_STARTS_MS:
                    for nc_d_pos0 in DELAY_POS_STARTS_MS:
                        for nc_d_neg0 in DELAY_NEG_STARTS_MS:
                            result = least_squares(
                                residual,
                                x0=np.array([tau_pos0, tau_neg0,
                                             pc_d_pos0, pc_d_neg0,
                                             nc_d_pos0, nc_d_neg0,
                                             *gain_pos_pc, *gain_neg_pc,
                                             *gain_pos_nc, *gain_neg_nc]),
                                bounds=(lower_bounds, upper_bounds),
                                loss="linear",
                                max_nfev=40_000,
                            )
                            sse = float(np.dot(result.fun, result.fun))
                            if best is None or sse < best[0]:
                                best = (sse, result)

    assert best is not None
    global_sse, result = best
    (tau_pos_ms, tau_neg_ms,
     pc_d_pos, pc_d_neg, nc_d_pos, nc_d_neg) = map(float, result.x[:n_global])
    pc_gp = dict(zip(POSITIONS, map(float, result.x[n_global: n_global + n_pos])))
    pc_gn = dict(zip(POSITIONS, map(float, result.x[n_global + n_pos: n_global + n_per_row])))
    nc_off = n_global + n_per_row
    nc_gp = dict(zip(POSITIONS, map(float, result.x[nc_off: nc_off + n_pos])))
    nc_gn = dict(zip(POSITIONS, map(float, result.x[nc_off + n_pos: nc_off + n_per_row])))

    pooled_vm = np.concatenate(
        [t["vm_mv"] for t in traces_pc] + [t["vm_mv"] for t in traces_nc])
    pooled_ss_total = float(np.sum((pooled_vm - np.mean(pooled_vm)) ** 2))
    global_r_squared = (
        1.0 - global_sse / pooled_ss_total if pooled_ss_total > 0.0 else np.nan)

    def _emit(traces, gains_pos, gains_neg, tp, tn, d_pos, d_neg):
        rows = []
        fits = {}
        for t in traces:
            prediction = two_pulse_lp(
                t["time_ms"], DURATION_MS, tp, tn,
                d_pos, gains_pos[t["position"]],
                d_neg, gains_neg[t["position"]],
            )
            res = prediction - t["vm_mv"]
            sse = float(np.dot(res, res))
            ss_tot = float(np.sum((t["vm_mv"] - np.mean(t["vm_mv"])) ** 2))
            r_sq = 1.0 - sse / ss_tot if ss_tot > 0.0 else np.nan
            rows.append({
                "trace_id": t["trace_id"],
                "position": t["position"],
                "gain_pos_mv": gains_pos[t["position"]],
                "gain_neg_mv": -abs(gains_neg[t["position"]]),
                "delay_pos_ms": d_pos,
                "delay_neg_ms": d_neg,
                "tau_pos_ms": tp,
                "tau_neg_ms": tn,
                "r_squared": r_sq,
                "global_r_squared": global_r_squared,
                "sse": sse,
                "global_sse": global_sse,
                "n_points": len(t["time_ms"]),
            })
            fits[t["trace_id"]] = {
                "time_ms": t["time_ms"],
                "vm_mv": t["vm_mv"],
                "prediction_mv": prediction,
            }
        return rows, fits

    pc_rows, pc_fits = _emit(traces_pc, pc_gp, pc_gn,
                             tau_pos_ms, tau_neg_ms, pc_d_pos, pc_d_neg)
    nc_rows, nc_fits = _emit(traces_nc, nc_gp, nc_gn,
                             tau_neg_ms, tau_pos_ms, nc_d_pos, nc_d_neg)
    return pc_rows, pc_fits, nc_rows, nc_fits


def fit_mirrored_g(
    data_pc: pd.DataFrame,
    data_nc: pd.DataFrame,
) -> tuple[list[dict], dict[str, dict], list[dict], dict[str, dict], float]:
    """Joint fit with NC g- = a * PC g+ and NC g+ = a * PC g-."""

    def _prep(data: pd.DataFrame) -> list[dict]:
        traces = []
        for position in POSITIONS:
            trace = data[data["position"] == position].sort_values("time_ms")
            if trace.empty:
                raise ValueError(f"missing position {position:+d}")
            time_ms = trace["time_ms"].to_numpy(dtype=float)
            vm_mv = trace["vm_mv"].to_numpy(dtype=float)
            finite = np.isfinite(time_ms) & np.isfinite(vm_mv)
            traces.append({
                "trace_id": str(trace["trace_id"].iloc[0]),
                "position": position,
                "time_ms": time_ms[finite],
                "vm_mv": vm_mv[finite],
            })
        return traces

    traces_pc = _prep(data_pc)
    traces_nc = _prep(data_nc)
    n_pos = len(POSITIONS)
    # Eight PC/NC temporal parameters, one global gain scale a, and 24 PC gains.
    n_global = 9
    gain_pos_pc = [max(float(np.max(t["vm_mv"])), 0.1) for t in traces_pc]
    gain_neg_pc = [max(-float(np.min(t["vm_mv"])), 0.1) for t in traces_pc]
    lower_bounds = np.concatenate((
        [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        np.zeros(2 * n_pos),
    ))
    upper_bounds = np.concatenate((
        [1000.0, 1000.0, 120.0, 200.0,
         1000.0, 1000.0, 120.0, 200.0, 10.0],
        np.full(2 * n_pos, 100.0),
    ))

    def residual(parameters: np.ndarray) -> np.ndarray:
        (pc_tp, pc_tn, pc_dp, pc_dn,
         nc_tp, nc_tn, nc_dp, nc_dn, gain_scale_a) = parameters[:n_global]
        pc_gp = dict(zip(POSITIONS, parameters[n_global:n_global + n_pos]))
        pc_gn = dict(zip(POSITIONS, parameters[n_global + n_pos:]))
        parts = []
        for t in traces_pc:
            position = t["position"]
            parts.append(two_pulse_lp(
                t["time_ms"], DURATION_MS, pc_tp, pc_tn,
                pc_dp, pc_gp[position], pc_dn, pc_gn[position],
            ) - t["vm_mv"])
        for t in traces_nc:
            position = t["position"]
            parts.append(two_pulse_lp(
                t["time_ms"], DURATION_MS, nc_tp, nc_tn,
                nc_dp, gain_scale_a * pc_gn[position],
                nc_dn, gain_scale_a * pc_gp[position],
            ) - t["vm_mv"])
        return np.concatenate(parts)

    best = None
    for pc_tp0 in TAU_POS_STARTS_MS:
        for pc_tn0 in TAU_NEG_STARTS_MS:
            for pc_dp0 in DELAY_POS_STARTS_MS:
                for pc_dn0 in DELAY_NEG_STARTS_MS:
                    for nc_tp0 in TAU_POS_STARTS_MS:
                        for nc_tn0 in TAU_NEG_STARTS_MS:
                            for nc_dp0 in DELAY_POS_STARTS_MS:
                                for nc_dn0 in DELAY_NEG_STARTS_MS:
                                    result = least_squares(
                                        residual,
                                        x0=np.array([
                                            pc_tp0, pc_tn0, pc_dp0, pc_dn0,
                                            nc_tp0, nc_tn0, nc_dp0, nc_dn0, 1.0,
                                            *gain_pos_pc, *gain_neg_pc,
                                        ]),
                                        bounds=(lower_bounds, upper_bounds),
                                        loss="linear",
                                        max_nfev=40_000,
                                    )
                                    sse = float(np.dot(result.fun, result.fun))
                                    if best is None or sse < best[0]:
                                        best = (sse, result)

    assert best is not None
    global_sse, result = best
    (pc_tp, pc_tn, pc_dp, pc_dn,
     nc_tp, nc_tn, nc_dp, nc_dn, gain_scale_a) = map(
        float, result.x[:n_global])
    pc_gp = dict(zip(POSITIONS, map(float, result.x[n_global:n_global + n_pos])))
    pc_gn = dict(zip(POSITIONS, map(float, result.x[n_global + n_pos:])))
    nc_gp = {position: gain_scale_a * pc_gn[position] for position in POSITIONS}
    nc_gn = {position: gain_scale_a * pc_gp[position] for position in POSITIONS}

    pooled_vm = np.concatenate(
        [t["vm_mv"] for t in traces_pc] + [t["vm_mv"] for t in traces_nc])
    pooled_ss_total = float(np.sum((pooled_vm - np.mean(pooled_vm)) ** 2))
    global_r_squared = (
        1.0 - global_sse / pooled_ss_total if pooled_ss_total > 0.0 else np.nan)

    def _emit(traces, gains_pos, gains_neg, tp, tn, d_pos, d_neg):
        rows = []
        fits = {}
        for t in traces:
            position = t["position"]
            prediction = two_pulse_lp(
                t["time_ms"], DURATION_MS, tp, tn,
                d_pos, gains_pos[position], d_neg, gains_neg[position],
            )
            res = prediction - t["vm_mv"]
            sse = float(np.dot(res, res))
            ss_tot = float(np.sum((t["vm_mv"] - np.mean(t["vm_mv"])) ** 2))
            r_sq = 1.0 - sse / ss_tot if ss_tot > 0.0 else np.nan
            rows.append({
                "trace_id": t["trace_id"],
                "position": position,
                "gain_pos_mv": gains_pos[position],
                "gain_neg_mv": -abs(gains_neg[position]),
                "gain_scale_a": gain_scale_a,
                "delay_pos_ms": d_pos,
                "delay_neg_ms": d_neg,
                "tau_pos_ms": tp,
                "tau_neg_ms": tn,
                "r_squared": r_sq,
                "global_r_squared": global_r_squared,
                "sse": sse,
                "global_sse": global_sse,
                "n_points": len(t["time_ms"]),
            })
            fits[t["trace_id"]] = {
                "time_ms": t["time_ms"],
                "vm_mv": t["vm_mv"],
                "prediction_mv": prediction,
            }
        return rows, fits

    pc_rows, pc_fits = _emit(
        traces_pc, pc_gp, pc_gn, pc_tp, pc_tn, pc_dp, pc_dn)
    nc_rows, nc_fits = _emit(
        traces_nc, nc_gp, nc_gn, nc_tp, nc_tn, nc_dp, nc_dn)
    return pc_rows, pc_fits, nc_rows, nc_fits, gain_scale_a


def plot_fits_multi(
    row_groups: list[tuple[str, str, pd.DataFrame, dict[str, dict]]],
    output_path: Path,
    title: str,
    fit_label: str,
) -> None:
    """Plot multiple row groups (each: row_label, data_color, results, fits)."""
    n_row = len(row_groups)
    fig, axes = plt.subplots(
        n_row,
        N_COL,
        figsize=(30, 3.5 * n_row),
        sharex=True,
        squeeze=False,
    )
    for row_index, (row_label, data_color, results, fits) in enumerate(row_groups):
        for column_index, position in enumerate(POSITIONS):
            ax = axes[row_index, column_index]
            row = results[results.position == position].iloc[0]
            fit = fits[row.trace_id]

            ax.plot(fit["time_ms"], fit["vm_mv"],
                    color=data_color, lw=1.2, alpha=0.75, label="digitized data")
            ax.plot(fit["time_ms"], fit["prediction_mv"],
                    color="#2474b5", lw=2.0, ls="--", label=fit_label)
            t = fit["time_ms"]
            lp_pos = delayed_lp_pulse(t, DURATION_MS, row.tau_pos_ms,
                                      row.delay_pos_ms, row.gain_pos_mv)
            lp_neg = delayed_lp_pulse(t, DURATION_MS, row.tau_neg_ms,
                                      row.delay_neg_ms, row.gain_neg_mv)
            ax.plot(t, lp_pos, color="#9467bd", lw=0.9, ls="--", alpha=0.7,
                    label="pos LP")
            ax.plot(t, lp_neg, color="#9467bd", lw=0.9, ls=":", alpha=0.7,
                    label="neg LP")
            bipolar = np.where(
                (t >= row.delay_pos_ms) & (t < row.delay_pos_ms + DURATION_MS),
                row.gain_pos_mv, 0.0,
            ) + np.where(
                (t >= row.delay_neg_ms) & (t < row.delay_neg_ms + DURATION_MS),
                row.gain_neg_mv, 0.0,
            )
            ax_pulse = ax.twinx()
            ax_pulse.plot(t, bipolar, color="0.55", lw=1.0, ls="--",
                          label="bipolar input", zorder=1)
            ax_pulse.tick_params(labelright=True, right=True, labelsize=6, colors="0.5")
            if column_index == N_COL - 1:
                ax_pulse.set_ylabel("input (mV)", fontsize=8, color="0.5")
            else:
                ax_pulse.set_ylabel("")

            ax.axhline(0.0, color="0.75", lw=0.6)
            ax.axvline(0.0, color="0.6", lw=0.7, ls=":")
            ax.axvline(DURATION_MS, color="0.6", lw=0.7, ls=":")
            annotation = (
                f"g+={row.gain_pos_mv:.2f}\n"
                f"g-={row.gain_neg_mv:.2f}\n"
                f"R²={row.r_squared:.3f}"
            )
            if column_index == 0:
                annotation = (
                    f"d+={row.delay_pos_ms:.1f} ms\n"
                    f"τ+={row.tau_pos_ms:.1f} ms\n"
                    f"d-={row.delay_neg_ms:.1f} ms\n"
                    f"τ-={row.tau_neg_ms:.1f} ms\n"
                ) + annotation
            ax.text(
                0.97, 0.96, annotation,
                transform=ax.transAxes, va="top", ha="right", fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )
            if row_index == 0:
                ax.set_title(f"{position:+d}", fontsize=11)
            if column_index == 0:
                ax.set_ylabel(f"{row_label}\nVm (mV)")
            if row_index == n_row - 1:
                ax.set_xlabel("time (ms)")
            if row_index == 0 and column_index == N_COL - 1:
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    raw = pd.read_csv(INPUT_CSV)
    all_columns = [
        "trace_id", "gain_pos_mv", "gain_neg_mv",
        "delay_pos_ms", "delay_neg_ms", "tau_pos_ms", "tau_neg_ms",
        "r_squared", "global_r_squared",
    ]
    row_specs = [
        ("T4 PC", "#2ca02c", "T4", "PC"),
        ("T4 NC", "#303030", "T4", "NC"),
        ("T5 PC", "#d62728", "T5", "PC"),
        ("T5 NC", "#7f7f7f", "T5", "NC"),
    ]

    row_groups = []
    result_frames = []
    for row_label, data_color, cell_type, contrast in row_specs:
        data = load_traces(raw, cell_type, contrast)
        fit_rows, fits = fit_non_mirrored(data)
        results = pd.DataFrame(fit_rows).sort_values("position")
        result_frames.append(results)
        row_groups.append((row_label, data_color, results, fits))
        print(f"\n=== {row_label} — non-mirrored ===")
        print(results[all_columns].to_string(
            index=False, float_format=lambda x: f"{x:.4g}"))

    pd.concat(result_frames, ignore_index=True).to_csv(OUT_CSV, index=False)
    plot_fits_multi(
        row_groups,
        OUT_PNG,
        "Gruntman 2021 Fig. 2A T4/T5 — non-mirrored",
        "non-mirrored LP fit",
    )
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_PNG}")

    d_row_groups = []
    d_result_frames = []
    for cell_type, pc_color, nc_color in [
        ("T4", "#2ca02c", "#303030"),
        ("T5", "#d62728", "#7f7f7f"),
    ]:
        data_pc = load_traces(raw, cell_type, "PC")
        data_nc = load_traces(raw, cell_type, "NC")
        pc_rows, pc_fits, nc_rows, nc_fits = fit_mirrored_d(data_pc, data_nc)
        for contrast, color, rows, fits in [
            ("PC", pc_color, pc_rows, pc_fits),
            ("NC", nc_color, nc_rows, nc_fits),
        ]:
            results = pd.DataFrame(rows).sort_values("position")
            d_result_frames.append(results)
            d_row_groups.append((f"{cell_type} {contrast}", color, results, fits))
            print(f"\n=== {cell_type} {contrast} — mirrored delays ===")
            print(results[all_columns].to_string(
                index=False, float_format=lambda x: f"{x:.4g}"))

    pd.concat(d_result_frames, ignore_index=True).to_csv(
        OUT_MIRRORED_D_CSV, index=False)
    plot_fits_multi(
        d_row_groups,
        OUT_MIRRORED_D_PNG,
        "Gruntman 2021 Fig. 2A T4/T5 — mirrored delays",
        "mirrored-delay LP fit",
    )
    print(f"\nwrote {OUT_MIRRORED_D_CSV}")
    print(f"wrote {OUT_MIRRORED_D_PNG}")

    t_row_groups = []
    t_result_frames = []
    for cell_type, pc_color, nc_color in [
        ("T4", "#2ca02c", "#303030"),
        ("T5", "#d62728", "#7f7f7f"),
    ]:
        data_pc = load_traces(raw, cell_type, "PC")
        data_nc = load_traces(raw, cell_type, "NC")
        pc_rows, pc_fits, nc_rows, nc_fits = fit_mirrored_t(data_pc, data_nc)
        for contrast, color, rows, fits in [
            ("PC", pc_color, pc_rows, pc_fits),
            ("NC", nc_color, nc_rows, nc_fits),
        ]:
            results = pd.DataFrame(rows).sort_values("position")
            t_result_frames.append(results)
            t_row_groups.append((f"{cell_type} {contrast}", color, results, fits))
            print(f"\n=== {cell_type} {contrast} — mirrored taus ===")
            print(results[all_columns].to_string(
                index=False, float_format=lambda x: f"{x:.4g}"))

    pd.concat(t_result_frames, ignore_index=True).to_csv(
        OUT_MIRRORED_T_CSV, index=False)
    plot_fits_multi(
        t_row_groups,
        OUT_MIRRORED_T_PNG,
        "Gruntman 2021 Fig. 2A T4/T5 — mirrored taus",
        "mirrored-τ LP fit",
    )
    print(f"\nwrote {OUT_MIRRORED_T_CSV}")
    print(f"wrote {OUT_MIRRORED_T_PNG}")

    g_row_groups = []
    g_result_frames = []
    gain_columns = [*all_columns, "gain_scale_a"]
    gain_scales = {}
    for cell_type, pc_color, nc_color in [
        ("T4", "#2ca02c", "#303030"),
        ("T5", "#d62728", "#7f7f7f"),
    ]:
        data_pc = load_traces(raw, cell_type, "PC")
        data_nc = load_traces(raw, cell_type, "NC")
        pc_rows, pc_fits, nc_rows, nc_fits, gain_scale_a = fit_mirrored_g(
            data_pc, data_nc)
        gain_scales[cell_type] = gain_scale_a
        for contrast, color, rows, fits in [
            ("PC", pc_color, pc_rows, pc_fits),
            ("NC", nc_color, nc_rows, nc_fits),
        ]:
            results = pd.DataFrame(rows).sort_values("position")
            g_result_frames.append(results)
            g_row_groups.append((f"{cell_type} {contrast}", color, results, fits))
            print(
                f"\n=== {cell_type} {contrast} — mirrored gains "
                f"(a={gain_scale_a:.4g}) ===")
            print(results[gain_columns].to_string(
                index=False, float_format=lambda x: f"{x:.4g}"))

    pd.concat(g_result_frames, ignore_index=True).to_csv(
        OUT_MIRRORED_G_CSV, index=False)
    plot_fits_multi(
        g_row_groups,
        OUT_MIRRORED_G_PNG,
        ("Gruntman 2021 Fig. 2A T4/T5 — mirrored gains "
         f"(a_T4={gain_scales['T4']:.3g}; a_T5={gain_scales['T5']:.3g})"),
        "mirrored-gain LP fit",
    )
    print(f"\nwrote {OUT_MIRRORED_G_CSV}")
    print(f"wrote {OUT_MIRRORED_G_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
