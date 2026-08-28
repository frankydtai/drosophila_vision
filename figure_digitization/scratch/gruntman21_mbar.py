#!/usr/bin/env python3
"""Continuous moving-bar responses from the fitted T4 PC and NC filters.

The bar follows a continuous trajectory through the fitted spatial gain
curves.  There is no per-position dwell or pulse duration.  Width 4 uses the
fitted g4 curves directly.  Width 1 folds a regularized inverse of the
four-point spatial sum into the same frequency-domain temporal convolution,
so no discrete g1 table is constructed.

Outputs:
  gruntman21_mbar.csv
  gruntman21_mbar.png
  gruntman21_mbar_w1_56.csv
  gruntman21_mbar_w1_56.png  width-1 bar-convolved response at 56.25 deg/s
  gruntman21_mbar_w1_56_fig1.png  Fig. 1 Ci w1 vs per-trace gain/shift fit
  gruntman21_mbar_w1_56_fig1.csv  per-trace gain and shift_ms fit
  gruntman21_mbar_w1_56_fig1_shared.png  Fig. 1 Ci w1 vs shared gain/shift fit
  gruntman21_mbar_w1_56_fig1_shared.csv  shared gain, shift_ms, and per-trace R²
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import next_fast_len

from gruntman21_t4_g import POSITIONS, fit_g4, gaussian_curve


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT_CSV = HERE / "gruntman21_fit_lp.csv"
INPUT_W1_CSV = ROOT / "gruntman21" / "2ax2bc_digitized.csv"
INPUT_1CI_CSV = ROOT / "gruntman21" / "1ci_digitized.csv"
OUTPUT_CSV = HERE / "gruntman21_mbar.csv"
OUTPUT_PNG = HERE / "gruntman21_mbar.png"
OUTPUT_W1_CSV = HERE / "gruntman21_mbar_w1_56.csv"
OUTPUT_W1_PNG = HERE / "gruntman21_mbar_w1_56.png"
OUTPUT_W1_FIG1_PNG = HERE / "gruntman21_mbar_w1_56_fig1.png"
OUTPUT_W1_FIG1_CSV = HERE / "gruntman21_mbar_w1_56_fig1.csv"
OUTPUT_W1_FIG1_SHARED_PNG = HERE / "gruntman21_mbar_w1_56_fig1_shared.png"
OUTPUT_W1_FIG1_SHARED_CSV = HERE / "gruntman21_mbar_w1_56_fig1_shared.csv"
W1_FIG1_SHIFT_MS = np.arange(-200.0, 200.0 + 0.5, 1.0)

W1_TARGET_WIDTH_LED = 1
W1_POSITION_STEP_DEG = 2.25
W1_SPEED_DEG_PER_S = 56.25
W1_SCAN_DIRECTIONS = {
    "PD": tuple(np.arange(-2.0, 2.0 + 0.5 / 2.0, 0.5)),
    "ND": tuple(np.arange(2.0, -2.0 - 0.5 / 2.0, -0.5)),
}
# One W1_SCAN position step is 2.25 deg; integer position units span 4.5 deg.
W1_POSITION_STEP_POS = 0.5

POSITION_STEP_DEG = 4.5
W1_BAR_WIDTH_POS = W1_POSITION_STEP_DEG / POSITION_STEP_DEG
SPEEDS_DEG_PER_S = (1.40625, 14.0625, 28.125, 56.25, 112.5)
DT_MS = 1.0
PRE_MS = 1000.0
POST_MS = 2500.0
SPATIAL_REGULARIZATION = 1.0

DIRECTIONS = ("PD", "ND")
BAR_WIDTHS = (4, 1)
CONTRASTS = ("PC", "NC")
DIRECTION_COLORS = {"PD": "tab:red", "ND": "tab:blue"}
WIDTH_LINESTYLES = {4: "-", 1: "--"}


def load_t4_condition(
    fits: pd.DataFrame,
    contrast: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Load one T4 contrast and its four globally fitted parameters."""
    condition = fits[
        fits["trace_id"].str.startswith(f"T4_{contrast}_")
    ].copy()
    condition = condition.sort_values("position")
    if condition["position"].astype(int).tolist() != POSITIONS:
        raise ValueError(f"expected T4 {contrast} positions {POSITIONS}")

    parameter_columns = (
        "delay_pos_ms",
        "delay_neg_ms",
        "tau_pos_ms",
        "tau_neg_ms",
    )
    parameters: dict[str, float] = {}
    for column in parameter_columns:
        values = condition[column].to_numpy(dtype=float)
        if not np.allclose(values, values[0], atol=1e-9, rtol=0.0):
            raise ValueError(f"T4 {contrast} {column} is not globally shared")
        parameters[column] = float(values[0])
    return condition, parameters


def fitted_g4(
    condition: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive- and negative-channel Gaussian parameters."""
    gains = condition.set_index("position")
    _, positive = fit_g4(gains["gain_pos_mv"], sign=1.0)
    _, negative = fit_g4(gains["gain_neg_mv"], sign=-1.0)
    return positive, negative


def trajectory(
    time_ms: np.ndarray,
    direction: str,
    position_time_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the continuous bar position and in-range mask."""
    scan_duration_ms = (POSITIONS[-1] - POSITIONS[0]) * position_time_ms
    active = (time_ms >= 0.0) & (time_ms <= scan_duration_ms)
    if direction == "PD":
        position = POSITIONS[0] + time_ms / position_time_ms
    elif direction == "ND":
        position = POSITIONS[-1] - time_ms / position_time_ms
    else:
        raise ValueError(f"unknown direction {direction!r}")
    return position, active


def temporal_transfer(
    angular_frequency: np.ndarray,
    tau_ms: float,
    delay_ms: float,
) -> np.ndarray:
    """Frequency response of a delayed, unit-area first-order LP kernel."""
    return np.exp(-1j * angular_frequency * delay_ms) / (
        1.0 + 1j * angular_frequency * tau_ms
    )


def inverse_four_point_transfer(
    angular_frequency: np.ndarray,
    direction: str,
    position_time_ms: float,
) -> np.ndarray:
    """Regularized inverse of g4(x)=sum(g1(x+j), j=-1..2)."""
    direction_sign = 1.0 if direction == "PD" else -1.0
    phase = angular_frequency * position_time_ms * direction_sign
    four_point = sum(np.exp(1j * phase * shift) for shift in (-1, 0, 1, 2))

    # A second-difference penalty removes the three oscillatory null modes.
    second_difference_power = (2.0 - 2.0 * np.cos(phase)) ** 2
    denominator = (
        np.abs(four_point) ** 2
        + SPATIAL_REGULARIZATION * second_difference_power
    )
    return np.conjugate(four_point) / denominator


def convolved_component(
    drive: np.ndarray,
    time_ms: np.ndarray,
    tau_ms: float,
    delay_ms: float,
    direction: str,
    bar_width: int,
    position_time_ms: float,
) -> np.ndarray:
    """Apply the spatial-width operator and temporal LP in one transform."""
    tail_ms = delay_ms + 12.0 * tau_ms
    n_tail = int(np.ceil(tail_ms / DT_MS))
    n_fft = next_fast_len(len(time_ms) + n_tail)
    angular_frequency = 2.0 * np.pi * np.fft.rfftfreq(n_fft, d=DT_MS)

    transfer = temporal_transfer(angular_frequency, tau_ms, delay_ms)
    if bar_width == 1:
        transfer *= inverse_four_point_transfer(
            angular_frequency,
            direction,
            position_time_ms,
        )
    elif bar_width != 4:
        raise ValueError(f"unsupported bar width {bar_width}")

    response = np.fft.irfft(
        np.fft.rfft(drive, n=n_fft) * transfer,
        n=n_fft,
    )
    return response[: len(time_ms)]


def moving_response(
    time_ms: np.ndarray,
    speed_deg_per_s: float,
    contrast: str,
    direction: str,
    bar_width: int,
    positive_parameters: np.ndarray,
    negative_parameters: np.ndarray,
    temporal_parameters: dict[str, float],
) -> pd.DataFrame:
    """Calculate one continuous moving-bar response."""
    position_time_ms = POSITION_STEP_DEG / speed_deg_per_s * 1000.0
    position, active = trajectory(
        time_ms,
        direction,
        position_time_ms,
    )
    positive_drive = np.zeros_like(time_ms)
    negative_drive = np.zeros_like(time_ms)
    positive_drive[active] = gaussian_curve(
        position[active],
        *positive_parameters,
    )
    negative_drive[active] = -gaussian_curve(
        position[active],
        *negative_parameters,
    )

    positive_response = convolved_component(
        positive_drive,
        time_ms,
        temporal_parameters["tau_pos_ms"],
        temporal_parameters["delay_pos_ms"],
        direction,
        bar_width,
        position_time_ms,
    )
    negative_response = convolved_component(
        negative_drive,
        time_ms,
        temporal_parameters["tau_neg_ms"],
        temporal_parameters["delay_neg_ms"],
        direction,
        bar_width,
        position_time_ms,
    )

    reported_position = position.copy()
    reported_position[~active] = np.nan
    return pd.DataFrame(
        {
            "time_ms": time_ms,
            "speed_deg_per_s": speed_deg_per_s,
            "contrast": contrast,
            "direction": direction,
            "bar_width": bar_width,
            "position": reported_position,
            "positive_mv": positive_response,
            "negative_mv": negative_response,
            "total_mv": positive_response + negative_response,
        }
    )


def plot_responses(responses: pd.DataFrame) -> None:
    """Plot PC/NC in rows, speeds in columns, and widths by line style."""
    fig, axes = plt.subplots(
        len(CONTRASTS),
        len(SPEEDS_DEG_PER_S),
        figsize=(22.5, 7.0),
        sharex="col",
        sharey=True,
        squeeze=False,
    )
    for row, contrast in enumerate(CONTRASTS):
        for column, speed_deg_per_s in enumerate(SPEEDS_DEG_PER_S):
            axis = axes[row, column]
            for bar_width in BAR_WIDTHS:
                for direction in DIRECTIONS:
                    trace = responses[
                        (responses["contrast"] == contrast)
                        & (responses["bar_width"] == bar_width)
                        & (
                            responses["speed_deg_per_s"]
                            == speed_deg_per_s
                        )
                        & (responses["direction"] == direction)
                    ]
                    axis.plot(
                        trace["time_ms"],
                        trace["total_mv"],
                        color=DIRECTION_COLORS[direction],
                        linestyle=WIDTH_LINESTYLES[bar_width],
                        linewidth=2.0,
                        label=f"{direction}, width {bar_width}",
                    )
            axis.axhline(0.0, color="0.65", linewidth=0.8)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(f"speed = {speed_deg_per_s:g} deg/s")

    for row, contrast in enumerate(CONTRASTS):
        axes[row, 0].set_ylabel(f"T4 {contrast}\nresponse (mV)")

    for column, speed_deg_per_s in enumerate(SPEEDS_DEG_PER_S):
        position_time_ms = POSITION_STEP_DEG / speed_deg_per_s * 1000.0
        scan_duration_ms = (
            (POSITIONS[-1] - POSITIONS[0]) * position_time_ms
        )
        axes[-1, column].set_xlim(
            -250.0,
            scan_duration_ms + 2000.0,
        )
        axes[-1, column].set_xlabel(
            "time from entry into fitted spatial range (ms)"
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        ncol=len(DIRECTIONS),
    )
    fig.suptitle(
        "T4 continuous moving-bar response; no per-position dwell"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)


def temporal_convolved_component(
    drive: np.ndarray,
    time_ms: np.ndarray,
    tau_ms: float,
    delay_ms: float,
) -> np.ndarray:
    """Apply only the delayed temporal LP, with no width conversion."""
    return convolved_component(
        drive,
        time_ms,
        tau_ms,
        delay_ms,
        "PD",
        4,
        1.0,
    )


def load_w1_gains(
    fits: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]]:
    """Scale fitted W4 gains by the position-specific Fig. 2C W1/W4 ratios."""
    data = pd.read_csv(INPUT_W1_CSV)
    data = data[
        (data["cell_type"] == "T4")
        & (data["target_width_led"] == W1_TARGET_WIDTH_LED)
    ].copy()
    gains = {}
    for contrast in CONTRASTS:
        condition, temporal_parameters = load_t4_condition(fits, contrast)
        condition = condition.set_index("position")
        positions = np.asarray(W1_SCAN_DIRECTIONS["PD"], dtype=float)
        positive = np.empty_like(positions)
        negative = np.empty_like(positions)
        for index, position in enumerate(positions):
            scaled = data[
                (data["contrast"] == contrast)
                & np.isclose(data["position"], position)
            ]
            scale_values = scaled["scale_factor"].drop_duplicates().to_numpy(
                dtype=float
            )
            if len(scale_values) != 1:
                raise ValueError(
                    f"expected one T4 {contrast} W1/W4 scale at {position:g}"
                )
            source_position = int(round(2.0 * position))
            scale = float(scale_values[0])
            positive[index] = (
                scale * float(condition.loc[source_position, "gain_pos_mv"])
            )
            negative[index] = (
                scale * float(condition.loc[source_position, "gain_neg_mv"])
            )
        gains[contrast] = (
            positions,
            positive,
            negative,
            temporal_parameters,
        )
    return gains


def w1_leading_edge(
    time_ms: np.ndarray,
    direction: str,
    *,
    scan_start_pos: float,
    position_delay_ms: float,
) -> np.ndarray:
    """Leading bar edge; ``t=0`` is when the edge reaches ``scan_start_pos``."""
    step = W1_POSITION_STEP_POS * time_ms / position_delay_ms
    if direction == "PD":
        return scan_start_pos + step
    if direction == "ND":
        return scan_start_pos - step
    raise ValueError(f"unknown direction {direction!r}")


def w1_bar_spatial_drive(
    leading_edge: np.ndarray,
    positions: np.ndarray,
    gain: np.ndarray,
    direction: str,
    *,
    bar_width_pos: float = W1_BAR_WIDTH_POS,
    quad_step: float = 0.05,
) -> np.ndarray:
    """Integrate ``gain`` under a uniform W1 bar behind the leading edge."""
    offsets = np.arange(0.0, bar_width_pos + quad_step / 2.0, quad_step)
    if direction == "PD":
        sample_pos = leading_edge[:, None] - offsets[None, :]
    elif direction == "ND":
        sample_pos = leading_edge[:, None] + offsets[None, :]
    else:
        raise ValueError(f"unknown direction {direction!r}")
    gain_under_bar = np.interp(
        sample_pos, positions, gain, left=0.0, right=0.0,
    )
    trapz = getattr(np, "trapezoid", np.trapz)
    return trapz(gain_under_bar, x=offsets, axis=1)


def convolved_w1_responses(
    fits: pd.DataFrame,
    position_delay_ms: float,
) -> pd.DataFrame:
    """W1 moving-bar drive: bar-width spatial integral × fitted temporal LPs."""
    gain_tables = load_w1_gains(fits)
    scan_positions = W1_SCAN_DIRECTIONS["PD"]
    scan_duration_ms = (len(scan_positions) - 1) * position_delay_ms
    time_ms = np.arange(
        -PRE_MS,
        scan_duration_ms + POST_MS + DT_MS / 2.0,
        DT_MS,
    )
    frames: list[pd.DataFrame] = []
    for contrast in CONTRASTS:
        positions, gain_pos, gain_neg, temporal = gain_tables[contrast]
        for direction in DIRECTIONS:
            scan_start_pos = (
                scan_positions[0] if direction == "PD" else scan_positions[-1]
            )
            leading_edge = w1_leading_edge(
                time_ms,
                direction,
                scan_start_pos=scan_start_pos,
                position_delay_ms=position_delay_ms,
            )
            positive_drive = w1_bar_spatial_drive(
                leading_edge, positions, gain_pos, direction,
            )
            negative_drive = w1_bar_spatial_drive(
                leading_edge, positions, gain_neg, direction,
            )
            positive_response = temporal_convolved_component(
                positive_drive,
                time_ms,
                temporal["tau_pos_ms"],
                temporal["delay_pos_ms"],
            )
            negative_response = temporal_convolved_component(
                negative_drive,
                time_ms,
                temporal["tau_neg_ms"],
                temporal["delay_neg_ms"],
            )
            frames.append(
                pd.DataFrame(
                    {
                        "time_ms": time_ms,
                        "speed_deg_per_s": W1_SPEED_DEG_PER_S,
                        "contrast": contrast,
                        "direction": direction,
                        "bar_width": W1_TARGET_WIDTH_LED,
                        "positive_mv": positive_response,
                        "negative_mv": negative_response,
                        "total_mv": positive_response + negative_response,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def w1_scan_duration_ms(position_delay_ms: float) -> float:
    return (len(W1_SCAN_DIRECTIONS["PD"]) - 1) * position_delay_ms


def w1_model_fig1_pairs(
    responses: pd.DataFrame,
    fig1_traces: dict[tuple[str, str], pd.DataFrame],
):
    for contrast in CONTRASTS:
        for direction in DIRECTIONS:
            key = (contrast, direction)
            yield (
                contrast,
                direction,
                responses[
                    (responses["contrast"] == contrast)
                    & (responses["direction"] == direction)
                ].sort_values("time_ms"),
                fig1_traces[key].sort_values("time_ms"),
            )


def plot_w1_56(responses: pd.DataFrame, position_delay_ms: float) -> None:
    """Plot width-1 bar-convolved T4 responses at 56.25 deg/s."""
    fig, axes = plt.subplots(
        len(CONTRASTS),
        1,
        figsize=(8.0, 6.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    scan_duration_ms = w1_scan_duration_ms(position_delay_ms)
    for row, contrast in enumerate(CONTRASTS):
        axis = axes[row, 0]
        for direction in DIRECTIONS:
            trace = responses[
                (responses["contrast"] == contrast)
                & (responses["direction"] == direction)
            ]
            axis.plot(
                trace["time_ms"],
                trace["total_mv"],
                color=DIRECTION_COLORS[direction],
                linewidth=2.0,
                label=direction,
            )
        axis.axhline(0.0, color="0.65", linewidth=0.8)
        axis.axvline(0.0, color="0.65", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.set_ylabel(f"T4 {contrast}\nresponse (mV)")
    axes[-1, 0].set_xlim(-100.0, scan_duration_ms + 500.0)
    axes[-1, 0].set_xlabel(
        "time from leading edge at scan start (ms)",
    )
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "T4 width-1 moving-bar convolution from Fig. 2A×2C gains\n"
        f"bar-width spatial integral at {W1_SPEED_DEG_PER_S:g} deg/s "
        f"({position_delay_ms:g} ms per position step)"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(OUTPUT_W1_PNG, dpi=180)
    plt.close(fig)


def load_1ci_w1_traces() -> dict[tuple[str, str], pd.DataFrame]:
    """Load the four T4 width-1 Fig. 1 Ci population traces."""
    data = pd.read_csv(INPUT_1CI_CSV)
    traces: dict[tuple[str, str], pd.DataFrame] = {}
    for contrast in CONTRASTS:
        for direction in DIRECTIONS:
            trace = data[
                (data["cell_type"] == "T4")
                & (data["width_led"] == W1_TARGET_WIDTH_LED)
                & (data["contrast"] == contrast)
                & (data["direction"] == direction)
            ].sort_values("time_ms")
            if trace.empty:
                raise ValueError(f"missing Fig. 1 Ci T4 w1 {contrast} {direction}")
            traces[(contrast, direction)] = trace
    return traces


def shifted_model(
    model_time_ms: np.ndarray,
    model_mv: np.ndarray,
    target_time_ms: np.ndarray,
    shift_ms: float,
) -> np.ndarray:
    """Evaluate ``model(time - shift_ms)`` on ``target_time_ms``."""
    return np.interp(
        target_time_ms - shift_ms,
        model_time_ms,
        model_mv,
        left=np.nan,
        right=np.nan,
    )


def r_squared(target_mv: np.ndarray, fitted_mv: np.ndarray) -> float:
    mask = np.isfinite(fitted_mv) & np.isfinite(target_mv)
    target_masked = target_mv[mask]
    fitted_masked = fitted_mv[mask]
    ss_tot = float(np.sum((target_masked - target_masked.mean()) ** 2))
    if ss_tot <= 0.0:
        return np.nan
    return 1.0 - float(np.sum((target_masked - fitted_masked) ** 2)) / ss_tot


def optimal_gain(model_mv: np.ndarray, target_mv: np.ndarray) -> float | None:
    denom = float(np.dot(model_mv, model_mv))
    if denom <= 0.0:
        return None
    return float(np.dot(model_mv, target_mv) / denom)


def fit_gain_shift(
    model_time_ms: np.ndarray,
    model_mv: np.ndarray,
    target_time_ms: np.ndarray,
    target_mv: np.ndarray,
) -> tuple[float, float, float]:
    """Return gain, shift_ms, and R² for ``gain * model(t - shift_ms)``."""
    best_gain = np.nan
    best_shift_ms = 0.0
    best_sse = np.inf
    target_mv = np.asarray(target_mv, dtype=float)
    for shift_ms in W1_FIG1_SHIFT_MS:
        model_shifted = shifted_model(
            model_time_ms,
            model_mv,
            target_time_ms,
            shift_ms,
        )
        mask = np.isfinite(model_shifted) & np.isfinite(target_mv)
        if mask.sum() < 5:
            continue
        gain = optimal_gain(model_shifted[mask], target_mv[mask])
        if gain is None:
            continue
        resid = target_mv[mask] - gain * model_shifted[mask]
        sse = float(np.dot(resid, resid))
        if sse < best_sse:
            best_sse = sse
            best_gain = gain
            best_shift_ms = float(shift_ms)
    if not np.isfinite(best_gain):
        raise ValueError("gain/shift fit failed on all shift candidates")
    fitted = best_gain * shifted_model(
        model_time_ms,
        model_mv,
        target_time_ms,
        best_shift_ms,
    )
    return best_gain, best_shift_ms, r_squared(target_mv, fitted)


def fit_shared_gain_shift(
    responses: pd.DataFrame,
    fig1_traces: dict[tuple[str, str], pd.DataFrame],
) -> tuple[float, float, float, list[dict[str, float | str]]]:
    """Return shared gain, shift_ms, global R², and per-trace fit rows."""
    best_gain = np.nan
    best_shift_ms = 0.0
    best_sse = np.inf
    for shift_ms in W1_FIG1_SHIFT_MS:
        model_stack: list[float] = []
        target_stack: list[float] = []
        for _, _, model, fig1 in w1_model_fig1_pairs(responses, fig1_traces):
            model_shifted = shifted_model(
                model["time_ms"].to_numpy(dtype=float),
                model["total_mv"].to_numpy(dtype=float),
                fig1["time_ms"].to_numpy(dtype=float),
                shift_ms,
            )
            target_mv = fig1["vm_mv"].to_numpy(dtype=float)
            mask = np.isfinite(model_shifted) & np.isfinite(target_mv)
            model_stack.extend(model_shifted[mask].tolist())
            target_stack.extend(target_mv[mask].tolist())
        if len(model_stack) < 5:
            continue
        model_arr = np.asarray(model_stack, dtype=float)
        target_arr = np.asarray(target_stack, dtype=float)
        gain = optimal_gain(model_arr, target_arr)
        if gain is None:
            continue
        resid = target_arr - gain * model_arr
        sse = float(np.dot(resid, resid))
        if sse < best_sse:
            best_sse = sse
            best_gain = gain
            best_shift_ms = float(shift_ms)
    if not np.isfinite(best_gain):
        raise ValueError("shared gain/shift fit failed on all shift candidates")

    all_target: list[float] = []
    all_fitted: list[float] = []
    fit_rows: list[dict[str, float | str]] = []
    for contrast, direction, model, fig1 in w1_model_fig1_pairs(
        responses,
        fig1_traces,
    ):
        fitted = best_gain * shifted_model(
            model["time_ms"].to_numpy(dtype=float),
            model["total_mv"].to_numpy(dtype=float),
            fig1["time_ms"].to_numpy(dtype=float),
            best_shift_ms,
        )
        target_mv = fig1["vm_mv"].to_numpy(dtype=float)
        mask = np.isfinite(fitted) & np.isfinite(target_mv)
        fit_rows.append(
            {
                "contrast": contrast,
                "direction": direction,
                "shared_gain": best_gain,
                "shared_shift_ms": best_shift_ms,
                "trace_r_squared": r_squared(target_mv, fitted),
            }
        )
        all_target.extend(target_mv[mask].tolist())
        all_fitted.extend(fitted[mask].tolist())
    global_r_squared = r_squared(
        np.asarray(all_target, dtype=float),
        np.asarray(all_fitted, dtype=float),
    )
    for row in fit_rows:
        row["global_r_squared"] = global_r_squared
    return best_gain, best_shift_ms, global_r_squared, fit_rows


def plot_w1_56_fig1(
    responses: pd.DataFrame,
    fig1_traces: dict[tuple[str, str], pd.DataFrame],
    position_delay_ms: float,
    fits: pd.DataFrame,
    *,
    output_png: Path = OUTPUT_W1_FIG1_PNG,
    shared: bool = False,
) -> None:
    """Overlay Fig. 1 Ci w1 traces with per-trace or shared gain/shift fits."""
    fig, axes = plt.subplots(
        len(CONTRASTS),
        1,
        figsize=(9.0, 6.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    scan_duration_ms = w1_scan_duration_ms(position_delay_ms)
    for row, contrast in enumerate(CONTRASTS):
        axis = axes[row, 0]
        for pair_contrast, direction, model, fig1 in w1_model_fig1_pairs(
            responses,
            fig1_traces,
        ):
            if pair_contrast != contrast:
                continue
            fit_row = fits[
                (fits["contrast"] == contrast)
                & (fits["direction"] == direction)
            ].iloc[0]
            fig1_time_ms = fig1["time_ms"].to_numpy(dtype=float)
            if shared:
                gain = float(fit_row["shared_gain"])
                shift_ms = float(fit_row["shared_shift_ms"])
                dash_label = f"{direction} shared fit"
            else:
                gain = float(fit_row["gain"])
                shift_ms = float(fit_row["shift_ms"])
                dash_label = (
                    f"{direction} bar-convolved "
                    f"(g={gain:.3g}, Δt={shift_ms:+.0f} ms)"
                )
            fitted_mv = gain * shifted_model(
                model["time_ms"].to_numpy(dtype=float),
                model["total_mv"].to_numpy(dtype=float),
                fig1_time_ms,
                shift_ms,
            )
            color = DIRECTION_COLORS[direction]
            axis.plot(
                fig1_time_ms,
                fig1["vm_mv"].to_numpy(dtype=float),
                color=color,
                linewidth=2.0,
                linestyle="-",
                label=f"{direction} Fig.1 Ci",
            )
            axis.plot(
                fig1_time_ms,
                fitted_mv,
                color=color,
                linewidth=2.0,
                linestyle="--",
                label=dash_label,
            )
        axis.axhline(0.0, color="0.65", linewidth=0.8)
        axis.axvline(0.0, color="0.65", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.set_ylabel(f"T4 {contrast}\nresponse (mV)")
    axes[-1, 0].set_xlim(-50.0, min(900.0, scan_duration_ms + 500.0))
    axes[-1, 0].set_xlabel("time from leading edge at scan start (ms)")
    axes[0, 0].legend(frameon=False, loc="upper right", fontsize=8)
    if shared:
        fit_row = fits.iloc[0]
        fig.suptitle(
            "T4 width-1 @ 56.25 deg/s: Fig. 1 Ci vs shared bar-convolved fit\n"
            f"shared gain={float(fit_row['shared_gain']):.3g}, "
            f"Δt={float(fit_row['shared_shift_ms']):+.0f} ms, "
            f"global R²={float(fit_row['global_r_squared']):.3f}"
        )
    else:
        fig.suptitle(
            "T4 width-1 @ 56.25 deg/s: Fig. 1 Ci vs bar-convolved model\n"
            "solid = Fig. 1 Ci; dashed = gain × model(t − Δt)"
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def write_w1_56_fig1(
    responses: pd.DataFrame,
    position_delay_ms: float,
    *,
    shared: bool = False,
) -> None:
    """Fit gain/shift and write the Fig. 1 Ci overlay PNG/CSV."""
    fig1_traces = load_1ci_w1_traces()
    fit_rows: list[dict[str, float | str]] = []
    if shared:
        shared_gain, shared_shift_ms, global_r_squared, fit_rows = (
            fit_shared_gain_shift(responses, fig1_traces)
        )
        for row in fit_rows:
            print(
                f"Fig.1 shared {row['contrast']} {row['direction']}: "
                f"trace R²={float(row['trace_r_squared']):.4f}"
            )
        print(
            f"Fig.1 shared fit: gain={shared_gain:.4g}, "
            f"shift_ms={shared_shift_ms:+.1f}, global R²={global_r_squared:.4f}"
        )
        output_csv = OUTPUT_W1_FIG1_SHARED_CSV
        output_png = OUTPUT_W1_FIG1_SHARED_PNG
    else:
        for contrast, direction, model, fig1 in w1_model_fig1_pairs(
            responses,
            fig1_traces,
        ):
            gain, shift_ms, trace_r_squared = fit_gain_shift(
                model["time_ms"].to_numpy(dtype=float),
                model["total_mv"].to_numpy(dtype=float),
                fig1["time_ms"].to_numpy(dtype=float),
                fig1["vm_mv"].to_numpy(dtype=float),
            )
            fit_rows.append(
                {
                    "contrast": contrast,
                    "direction": direction,
                    "gain": gain,
                    "shift_ms": shift_ms,
                    "r_squared": trace_r_squared,
                }
            )
            print(
                f"Fig.1 fit {contrast} {direction}: "
                f"gain={gain:.4g}, shift_ms={shift_ms:+.1f}, "
                f"R²={trace_r_squared:.4f}"
            )
        output_csv = OUTPUT_W1_FIG1_CSV
        output_png = OUTPUT_W1_FIG1_PNG
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(output_csv, index=False, float_format="%.9f")
    plot_w1_56_fig1(
        responses,
        fig1_traces,
        position_delay_ms,
        fits,
        output_png=output_png,
        shared=shared,
    )
    print(f"wrote {output_csv}")
    print(f"wrote {output_png}")


def write_w1_56(fits: pd.DataFrame) -> None:
    """Build bar-width convolved width-1 traces and write CSV/PNG."""
    position_delay_ms = (
        W1_POSITION_STEP_DEG / W1_SPEED_DEG_PER_S * 1000.0
    )
    responses = convolved_w1_responses(fits, position_delay_ms)
    responses.to_csv(OUTPUT_W1_CSV, index=False, float_format="%.9f")
    plot_w1_56(responses, position_delay_ms)
    write_w1_56_fig1(responses, position_delay_ms)
    write_w1_56_fig1(responses, position_delay_ms, shared=True)
    print(f"wrote {OUTPUT_W1_CSV}")
    print(f"wrote {OUTPUT_W1_PNG}")


def main() -> int:
    fits = pd.read_csv(INPUT_CSV)
    conditions = {}
    for contrast in CONTRASTS:
        condition, temporal_parameters = load_t4_condition(
            fits,
            contrast,
        )
        positive_parameters, negative_parameters = fitted_g4(condition)
        conditions[contrast] = (
            positive_parameters,
            negative_parameters,
            temporal_parameters,
        )

    responses = pd.concat(
        [
            moving_response(
                np.arange(
                    -PRE_MS,
                    (
                        (POSITIONS[-1] - POSITIONS[0])
                        * POSITION_STEP_DEG
                        / speed_deg_per_s
                        * 1000.0
                    )
                    + POST_MS
                    + DT_MS / 2.0,
                    DT_MS,
                ),
                speed_deg_per_s,
                contrast,
                direction,
                bar_width,
                *conditions[contrast],
            )
            for speed_deg_per_s in SPEEDS_DEG_PER_S
            for contrast in CONTRASTS
            for bar_width in BAR_WIDTHS
            for direction in DIRECTIONS
        ],
        ignore_index=True,
    )
    responses.to_csv(OUTPUT_CSV, index=False, float_format="%.9f")
    plot_responses(responses)

    for contrast in CONTRASTS:
        temporal_parameters = conditions[contrast][2]
        print(
            f"T4 {contrast} temporal parameters:",
            ", ".join(
                f"{name}={value:.6f}"
                for name, value in temporal_parameters.items()
            ),
        )
        for speed_deg_per_s in SPEEDS_DEG_PER_S:
            for bar_width in BAR_WIDTHS:
                for direction in DIRECTIONS:
                    total = responses.loc[
                        (responses["contrast"] == contrast)
                        & (responses["bar_width"] == bar_width)
                        & (
                            responses["speed_deg_per_s"]
                            == speed_deg_per_s
                        )
                        & (responses["direction"] == direction),
                        "total_mv",
                    ]
                    print(
                        f"{contrast} speed {speed_deg_per_s:g} "
                        f"width {bar_width} {direction}: "
                        f"min={total.min():.6f}, "
                        f"max={total.max():.6f} mV"
                    )
    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {OUTPUT_PNG}")
    write_w1_56(fits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
