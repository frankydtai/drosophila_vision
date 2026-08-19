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
INPUT_CSV = HERE / "gruntman21_fit_lp.csv"
OUTPUT_CSV = HERE / "gruntman21_mbar.csv"
OUTPUT_PNG = HERE / "gruntman21_mbar.png"

POSITION_STEP_DEG = 4.5
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
