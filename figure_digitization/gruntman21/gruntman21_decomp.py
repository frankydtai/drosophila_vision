#!/usr/bin/env python3
"""Plot Gaussian three-source spatial fit for Gruntman21 T4 width 1/4.

Input:
  ../gruntman21/2ax2bc_digitized.csv

Output:
  gruntman21_decomp.png
  gruntman21_mi1.csv
  gruntman21_mi4.csv
  gruntman21_mi1_mi4_fit_lp.csv
  gruntman21_mi1_mi4_fit_lp.png

Run:
  ../.venv/bin/python gruntman21_decomp.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares, minimize_scalar


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE.parent / "gruntman21" / "2ax2bc_digitized.csv"
GAUSSIAN_OUTPUT_PNG = HERE / "gruntman21_decomp.png"
GAUSSIAN_COMPARISON_CSV = HERE / "gruntman21_gaussian_comparison.csv"
MI9_CSV = HERE / "gruntman21_mi9.csv"
MI1_CSV = HERE / "gruntman21_mi1.csv"
MI4_CSV = HERE / "gruntman21_mi4.csv"
LP_OUTPUT_CSV = HERE / "gruntman21_mi1_mi4_fit_lp.csv"
LP_OUTPUT_PNG = HERE / "gruntman21_mi1_mi4_fit_lp.png"

CELL_TYPE = "T4"
CONTRASTS = ("PC", "NC")
COLORS = {"PC": "#549f5c", "NC": "#303030"}
WIDTH_LED = 1
SPATIAL_POSITIONS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
SOURCE_NAMES = ("Mi9", "Mi1", "Mi4")
SOURCE_CENTERS = (-1.0, 0.0, 1.0)
# Intrinsic source widths, in the same position units as SOURCE_CENTERS.
# Mi4 remains a point input, represented by None rather than a Gaussian.
SOURCE_SPATIAL_SIGMAS = (1.0, 0.5, None)
EXTENDED_ATTENUATION_DISTANCES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
STIMULUS_DURATION_MS = 160.0
TAU_STARTS_MS = (5.0, 15.0, 30.0, 60.0, 120.0, 250.0)
DELAY_STARTS_MS = (0.0, 10.0, 20.0, 30.0, 45.0, 65.0, 90.0)
LP_LOWER_BOUNDS = np.array([0.5, 0.0, -500.0])
LP_UPPER_BOUNDS = np.array([500.0, 120.0, 500.0])


def load_trace(
    data: pd.DataFrame,
    contrast: str,
    position: float,
) -> pd.DataFrame:
    """Return the single requested trace at one receptive-field position."""
    trace = data[
        (data["cell_type"] == CELL_TYPE)
        & (data["contrast"] == contrast)
        & (data["target_width_led"] == WIDTH_LED)
        & np.isclose(data["position"], position)
    ][["time_ms", "vm_mv"]].sort_values("time_ms")

    if trace.empty:
        raise ValueError(f"missing {contrast} trace at position {position:+g}")
    if trace["time_ms"].duplicated().any():
        raise ValueError(
            f"duplicate time sample in {contrast} at position {position:+g}"
        )
    return trace.rename(columns={"vm_mv": f"vm_{position:+g}"})


def align_traces(
    data: pd.DataFrame,
    contrast: str,
    positions: tuple[float, ...],
) -> pd.DataFrame:
    """Align requested traces on their common native time grid."""
    traces = [load_trace(data, contrast, position) for position in positions]
    aligned = traces[0]
    for trace in traces[1:]:
        aligned = aligned.merge(
            trace,
            on="time_ms",
            how="inner",
            validate="one_to_one",
        )
    if len(aligned) != min(len(trace) for trace in traces):
        raise ValueError(
            f"the requested {contrast} traces do not share the same time grid"
        )
    return aligned


def spatial_weight_matrix(
    attenuations: np.ndarray,
    attenuation_distances: tuple[float, ...],
) -> np.ndarray:
    """Return position-by-source weights for the three-source model."""
    if attenuations.shape != (len(attenuation_distances),):
        raise ValueError(
            f"expected {len(attenuation_distances)} attenuation values"
        )
    attenuation_by_distance = dict(zip(attenuation_distances, attenuations))
    weights = np.zeros((len(SPATIAL_POSITIONS), len(SOURCE_CENTERS)))
    for row, position in enumerate(SPATIAL_POSITIONS):
        for column, center in enumerate(SOURCE_CENTERS):
            distance = abs(position - center)
            if np.isclose(distance, 0.0):
                weights[row, column] = 1.0
            elif distance <= attenuation_distances[-1]:
                weights[row, column] = attenuation_by_distance[distance]
    return weights


def layered_spatial_weight_matrix(
    light_attenuations: np.ndarray,
    attenuation_distances: tuple[float, ...],
) -> np.ndarray:
    """Return source/light overlap gains after summing over internal mids.

    At each stimulus position, the source profile and shifted light Gaussian
    are multiplied over the internal spatial coordinate and then summed.  For
    Gaussians this cross-correlation has variance equal to the sum of their
    variances.  Each source curve is peak-normalized to one at its center.
    """
    if not np.isclose(attenuation_distances[0], 0.5):
        raise ValueError("the first attenuation distance must be 0.5")
    a_half = float(light_attenuations[0])
    if not 0.0 < a_half < 1.0:
        raise ValueError("cannot infer light sigma from A0.5")
    light_sigma = 0.5 / np.sqrt(-2.0 * np.log(a_half))

    weights = np.zeros((len(SPATIAL_POSITIONS), len(SOURCE_CENTERS)))
    for row, position in enumerate(SPATIAL_POSITIONS):
        for column, (center, source_sigma) in enumerate(
            zip(SOURCE_CENTERS, SOURCE_SPATIAL_SIGMAS)
        ):
            effective_sigma = (
                light_sigma
                if source_sigma is None
                else np.sqrt(light_sigma**2 + source_sigma**2)
            )
            distance = position - center
            weights[row, column] = np.exp(
                -0.5 * (distance / effective_sigma) ** 2
            )
    return weights


def solve_spatial_sources(
    attenuations: np.ndarray,
    observed: np.ndarray,
    attenuation_distances: tuple[float, ...],
    *,
    layered: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve optimal Mi9/Mi1/Mi4 time series and return their fitted traces."""
    weight_function = (
        layered_spatial_weight_matrix if layered else spatial_weight_matrix
    )
    weights = weight_function(attenuations, attenuation_distances)
    sources = np.linalg.lstsq(weights, observed, rcond=None)[0]
    return sources, weights @ sources


def fit_spatial_model(
    data: pd.DataFrame,
    attenuation_distances: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Fit shared A0.5/A1/A1.5/A2 to all nine PC and NC panels."""
    aligned_by_contrast = {
        contrast: align_traces(data, contrast, SPATIAL_POSITIONS)
        for contrast in CONTRASTS
    }
    observed_by_contrast = {
        contrast: np.vstack(
            [aligned[f"vm_{position:+g}"].to_numpy() for position in SPATIAL_POSITIONS]
        )
        for contrast, aligned in aligned_by_contrast.items()
    }

    def residual(attenuations: np.ndarray) -> np.ndarray:
        errors = []
        for observed in observed_by_contrast.values():
            _, fitted = solve_spatial_sources(
                attenuations,
                observed,
                attenuation_distances,
            )
            errors.append((fitted - observed).ravel())
        return np.concatenate(errors)

    def objective(attenuations: np.ndarray) -> float:
        errors = residual(attenuations)
        return float(errors @ errors)

    global_result = differential_evolution(
        objective,
        bounds=[(0.0, 1.0)] * len(attenuation_distances),
        seed=21,
        tol=1e-9,
        maxiter=300,
        popsize=15,
        polish=False,
        workers=1,
    )
    local_result = least_squares(
        residual,
        x0=global_result.x,
        bounds=(0.0, 1.0),
        max_nfev=20_000,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    if not local_result.success:
        raise RuntimeError(f"spatial attenuation fit failed: {local_result.message}")
    return local_result.x, aligned_by_contrast, observed_by_contrast


def gaussian_attenuations(sigma: float) -> np.ndarray:
    """Return normalized Gaussian A0.5..A3 values for one positive sigma."""
    if sigma <= 0.0:
        raise ValueError("Gaussian sigma must be positive")
    distances = np.asarray(EXTENDED_ATTENUATION_DISTANCES)
    return np.exp(-0.5 * (distances / sigma) ** 2)


def fit_gaussian_spatial_model(
    observed_by_contrast: dict[str, np.ndarray],
    *,
    layered: bool = False,
) -> tuple[float, np.ndarray]:
    """Fit the light Gaussian sigma to the same nine PC and NC panels."""
    def objective(log_sigma: float) -> float:
        attenuations = gaussian_attenuations(float(np.exp(log_sigma)))
        squared_error = 0.0
        for observed in observed_by_contrast.values():
            _, fitted = solve_spatial_sources(
                attenuations,
                observed,
                EXTENDED_ATTENUATION_DISTANCES,
                layered=layered,
            )
            error = fitted - observed
            squared_error += float(np.sum(error**2))
        return squared_error

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-13, "maxiter": 10_000},
    )
    if not result.success:
        raise RuntimeError(f"Gaussian spatial fit failed: {result.message}")
    sigma = float(np.exp(result.x))
    return sigma, gaussian_attenuations(sigma)


def spatial_output_tables(
    attenuations: np.ndarray,
    attenuation_distances: tuple[float, ...],
    aligned_by_contrast: dict[str, pd.DataFrame],
    observed_by_contrast: dict[str, np.ndarray],
    *,
    layered: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build tidy observed/fit and inferred-source output tables."""
    fit_frames = []
    source_frames = []
    for contrast in CONTRASTS:
        aligned = aligned_by_contrast[contrast]
        observed = observed_by_contrast[contrast]
        sources, fitted = solve_spatial_sources(
            attenuations,
            observed,
            attenuation_distances,
            layered=layered,
        )
        time_ms = aligned["time_ms"].to_numpy()
        for row, position in enumerate(SPATIAL_POSITIONS):
            fit_frames.append(
                pd.DataFrame(
                    {
                        "contrast": contrast,
                        "position": position,
                        "time_ms": time_ms,
                        "observed_vm_mv": observed[row],
                        "fitted_vm_mv": fitted[row],
                        "residual_mv": observed[row] - fitted[row],
                    }
                )
            )
        for row, source_name in enumerate(SOURCE_NAMES):
            source_frames.append(
                pd.DataFrame(
                    {
                        "contrast": contrast,
                        "source": source_name,
                        "center_position": SOURCE_CENTERS[row],
                        "time_ms": time_ms,
                        "source_vm_mv": sources[row],
                    }
                )
            )
    return (
        pd.concat(fit_frames, ignore_index=True),
        pd.concat(source_frames, ignore_index=True),
    )


def build_gaussian_comparison(
    old_sigma: float,
    new_sigma: float,
) -> pd.DataFrame:
    """Return old/new common-Gaussian values at requested neighbour distances."""
    distances = np.asarray((0.5, 1.0, 1.5, 2.0))
    old_gain = np.exp(-0.5 * (distances / old_sigma) ** 2)
    new_gain = np.exp(-0.5 * (distances / new_sigma) ** 2)
    return pd.DataFrame(
        {
            "neighbor_distance": distances,
            "old_sigma": old_sigma,
            "old_gaussian_gain": old_gain,
            "new_sigma": new_sigma,
            "new_gaussian_gain": new_gain,
            "new_minus_old": new_gain - old_gain,
        }
    )


def plot_spatial_fit(
    old_output: pd.DataFrame,
    old_sources: pd.DataFrame,
    new_output: pd.DataFrame,
    new_sources: pd.DataFrame,
    output_path: Path,
    gaussian_comparison: pd.DataFrame,
) -> None:
    """Plot the complete 12-panel old and 12-panel new decompositions."""
    fig = plt.figure(figsize=(22.5, 15.0))
    grid = fig.add_gridspec(4, 9, height_ratios=(1.0, 1.0, 1.0, 1.0))

    def source_row(row: int) -> list[plt.Axes]:
        return [
            fig.add_subplot(grid[row, 0:3]),
            fig.add_subplot(grid[row, 3:6]),
            fig.add_subplot(grid[row, 6:9]),
        ]

    def position_row(row: int) -> list[plt.Axes]:
        return [fig.add_subplot(grid[row, column]) for column in range(9)]

    old_source_axes = source_row(0)
    old_position_axes = position_row(1)
    new_source_axes = source_row(2)
    new_position_axes = position_row(3)

    def draw_model(
        model_name: str,
        output: pd.DataFrame,
        sources: pd.DataFrame,
        source_axes: list[plt.Axes],
        position_axes: list[plt.Axes],
    ) -> None:
        for ax, source_name in zip(source_axes, SOURCE_NAMES):
            for contrast in CONTRASTS:
                trace = sources[
                    (sources["contrast"] == contrast)
                    & (sources["source"] == source_name)
                ]
                ax.plot(
                    trace["time_ms"],
                    trace["source_vm_mv"],
                    color=COLORS[contrast],
                    linewidth=1.8,
                    label=contrast,
                )
            ax.axvspan(0.0, 160.0, color="0.92", zorder=-2)
            ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=-1)
            ax.axvline(0.0, color="0.75", linewidth=0.7, zorder=-1)
            center = SOURCE_CENTERS[SOURCE_NAMES.index(source_name)]
            ax.set_title(f"{model_name} input {source_name}, center {center:+g}")
            ax.set_xlabel("time (ms)")
        source_axes[0].set_ylabel(f"{model_name} source Vm (mV)")
        source_axes[-1].legend(frameon=False, fontsize=8)

        for ax, position in zip(position_axes, SPATIAL_POSITIONS):
            for contrast in CONTRASTS:
                trace = output[
                    (output["contrast"] == contrast)
                    & np.isclose(output["position"], position)
                ]
                ax.plot(
                    trace["time_ms"],
                    trace["observed_vm_mv"],
                    color=COLORS[contrast],
                    linestyle="-",
                    linewidth=1.7,
                    label=f"{contrast} original",
                )
                ax.plot(
                    trace["time_ms"],
                    trace["fitted_vm_mv"],
                    color=COLORS[contrast],
                    linestyle="--",
                    linewidth=1.7,
                    label=f"{contrast} {model_name} fit",
                )
            ax.axvspan(0.0, 160.0, color="0.92", zorder=-2)
            ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=-1)
            ax.axvline(0.0, color="0.75", linewidth=0.7, zorder=-1)
            ax.set_title(f"{model_name} position {position:+g}")
            ax.set_xlabel("time (ms)")
            ax.tick_params(labelsize=7)
        position_axes[0].set_ylabel(f"{model_name} Vm (mV)")
        position_axes[-1].legend(frameon=False, fontsize=7)

    draw_model(
        "OLD",
        old_output,
        old_sources,
        old_source_axes,
        old_position_axes,
    )
    draw_model(
        "NEW",
        new_output,
        new_sources,
        new_source_axes,
        new_position_axes,
    )

    all_source_axes = old_source_axes + new_source_axes
    source_y_min = min(ax.get_ylim()[0] for ax in all_source_axes)
    source_y_max = max(ax.get_ylim()[1] for ax in all_source_axes)
    for ax in all_source_axes:
        ax.set_ylim(source_y_min, source_y_max)

    all_position_axes = old_position_axes + new_position_axes
    position_y_min = min(ax.get_ylim()[0] for ax in all_position_axes)
    position_y_max = max(ax.get_ylim()[1] for ax in all_position_axes)
    for ax in all_position_axes:
        ax.set_ylim(position_y_min, position_y_max)
    old_values = ", ".join(
        f"A{row.neighbor_distance:g}={row.old_gaussian_gain:.6f}"
        for row in gaussian_comparison.itertuples(index=False)
    )
    new_values = ", ".join(
        f"A{row.neighbor_distance:g}={row.new_gaussian_gain:.6f}"
        for row in gaussian_comparison.itertuples(index=False)
    )
    old_sigma = gaussian_comparison["old_sigma"].iloc[0]
    new_sigma = gaussian_comparison["new_sigma"].iloc[0]
    fig.suptitle(
        "T4 width 1/4 — complete OLD (12 panels) and NEW (12 panels) decomposition\n"
        f"OLD common Gaussian $\\sigma={old_sigma:.6f}$: {old_values}\n"
        f"NEW common Gaussian $\\sigma={new_sigma:.6f}$: {new_values}; "
        "Mi9 source $\\sigma=1$, Mi1 source $\\sigma=0.5$, Mi4 point"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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


def fit_lp_trace(
    time_ms: np.ndarray,
    vm_mv: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Fit one decomposed trace using the Gruntman18 delayed-LP method."""
    finite = np.isfinite(time_ms) & np.isfinite(vm_mv)
    time_ms = np.asarray(time_ms, dtype=float)[finite]
    vm_mv = np.asarray(vm_mv, dtype=float)[finite]
    if len(time_ms) == 0:
        raise ValueError("cannot fit an empty LP trace")

    extreme = int(np.argmax(np.abs(vm_mv)))
    gain0 = float(vm_mv[extreme])
    if np.isclose(gain0, 0.0):
        gain0 = 0.1

    def residual(parameters: np.ndarray) -> np.ndarray:
        tau_ms, delay_ms, gain_mv = parameters
        return delayed_lp_pulse(
            time_ms,
            STIMULUS_DURATION_MS,
            tau_ms,
            delay_ms,
            gain_mv,
        ) - vm_mv

    best = None
    for tau0 in TAU_STARTS_MS:
        for delay0 in DELAY_STARTS_MS:
            result = least_squares(
                residual,
                x0=np.array([tau0, delay0, gain0]),
                bounds=(LP_LOWER_BOUNDS, LP_UPPER_BOUNDS),
                loss="linear",
                max_nfev=20_000,
            )
            sse = float(result.fun @ result.fun)
            if best is None or sse < best[0]:
                best = (sse, result)

    assert best is not None
    sse, result = best
    if not result.success:
        raise RuntimeError(f"LP fit failed: {result.message}")
    tau_ms, delay_ms, gain_mv = map(float, result.x)
    prediction = delayed_lp_pulse(
        time_ms,
        STIMULUS_DURATION_MS,
        tau_ms,
        delay_ms,
        gain_mv,
    )
    ss_total = float(np.sum((vm_mv - np.mean(vm_mv)) ** 2))
    r_squared = 1.0 - sse / ss_total if ss_total > 0.0 else np.nan
    return {
        "tau_ms": tau_ms,
        "delay_ms": delay_ms,
        "gain_mv": gain_mv,
        "effective_offset_ms": delay_ms + STIMULUS_DURATION_MS,
        "r_squared": r_squared,
        "sse": sse,
        "n_points": len(time_ms),
        "time_ms": time_ms,
        "vm_mv": vm_mv,
        "prediction_mv": prediction,
    }


def fit_and_plot_sources(
    source_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Fit Mi1/Mi4 traces and plot one overlaid PC/NC panel per source."""
    rows = []
    fits = {}
    for source_name, source_data in source_tables.items():
        for contrast in CONTRASTS:
            fit = fit_lp_trace(
                source_data["time_ms"].to_numpy(),
                source_data[f"vm_{contrast}"].to_numpy(),
            )
            trace_id = f"{source_name}_{contrast}"
            fits[trace_id] = fit
            rows.append(
                {
                    "trace_id": trace_id,
                    "source": source_name,
                    "contrast": contrast,
                    "flash_duration_ms": STIMULUS_DURATION_MS,
                    "gain_mv": fit["gain_mv"],
                    "delay_ms": fit["delay_ms"],
                    "tau_ms": fit["tau_ms"],
                    "effective_offset_ms": fit["effective_offset_ms"],
                    "r_squared": fit["r_squared"],
                    "sse": fit["sse"],
                    "n_points": fit["n_points"],
                }
            )

    results = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharex=True, squeeze=False)
    for column_index, source_name in enumerate(source_tables):
        ax = axes[0, column_index]
        for contrast in CONTRASTS:
            trace_id = f"{source_name}_{contrast}"
            fit = fits[trace_id]
            ax.plot(
                fit["time_ms"],
                fit["vm_mv"],
                color=COLORS[contrast],
                linewidth=1.5,
                label=f"{contrast} data",
            )
            ax.plot(
                fit["time_ms"],
                fit["prediction_mv"],
                color=COLORS[contrast],
                linestyle="--",
                linewidth=2.0,
                label=f"{contrast} LP fit",
            )
        source_results = results[results["source"] == source_name].set_index("contrast")
        annotation_lines = []
        for contrast in CONTRASTS:
            row = source_results.loc[contrast]
            annotation_lines.append(
                f"{contrast}: tau={row['tau_ms']:.2f} ms, delay={row['delay_ms']:.2f} ms"
            )
            annotation_lines.append(
                f"    gain={row['gain_mv']:.2f} mV, R2={row['r_squared']:.4f}"
            )
        ax.text(
            0.02,
            0.98,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        ax.axhline(0.0, color="0.75", linewidth=0.7)
        ax.axvline(0.0, color="0.6", linewidth=0.7, linestyle=":")
        ax.axvline(
            STIMULUS_DURATION_MS,
            color="0.6",
            linewidth=0.7,
            linestyle=":",
        )
        ax.set_title(source_name)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("source Vm (mV)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Gruntman 2021 decomposed Mi1/Mi4 — delayed single-τ LP fits")
    fig.tight_layout()
    fig.savefig(LP_OUTPUT_PNG, dpi=180)
    plt.close(fig)
    return results


def main() -> int:
    data = pd.read_csv(INPUT_CSV)

    _, extended_aligned, extended_observed = fit_spatial_model(
        data,
        EXTENDED_ATTENUATION_DISTANCES,
    )

    old_gaussian_sigma, old_gaussian = fit_gaussian_spatial_model(
        extended_observed,
        layered=False,
    )
    # With the intrinsic source widths fixed, jointly refit one common light
    # Gaussian.  Source/light overlap gives each source an effective variance
    # equal to the sum of the common-light and intrinsic-source variances.
    new_light_sigma, new_light_gaussian = fit_gaussian_spatial_model(
        extended_observed,
        layered=True,
    )

    gaussian_comparison = build_gaussian_comparison(
        old_gaussian_sigma,
        new_light_sigma,
    )
    gaussian_comparison.to_csv(GAUSSIAN_COMPARISON_CSV, index=False)

    old_output, old_sources = spatial_output_tables(
        old_gaussian,
        EXTENDED_ATTENUATION_DISTANCES,
        extended_aligned,
        extended_observed,
        layered=False,
    )
    gaussian_output, gaussian_sources = spatial_output_tables(
        new_light_gaussian,
        EXTENDED_ATTENUATION_DISTANCES,
        extended_aligned,
        extended_observed,
        layered=True,
    )

    plot_spatial_fit(
        old_output,
        old_sources,
        gaussian_output,
        gaussian_sources,
        GAUSSIAN_OUTPUT_PNG,
        gaussian_comparison,
    )

    source_outputs = {}
    for source_name in SOURCE_NAMES:
        source_output = None
        for contrast in CONTRASTS:
            mask = (
                (gaussian_sources["contrast"] == contrast)
                & (gaussian_sources["source"] == source_name)
            )
            contrast_output = gaussian_sources[mask][
                ["time_ms", "source_vm_mv"]
            ].rename(columns={"source_vm_mv": f"vm_{contrast}"})
            if source_output is None:
                source_output = contrast_output.copy()
            else:
                source_output = source_output.merge(
                    contrast_output,
                    on="time_ms",
                    how="inner",
                )
        assert source_output is not None
        source_outputs[source_name] = source_output.sort_values("time_ms")

    source_outputs["Mi9"].to_csv(MI9_CSV, index=False)
    source_outputs["Mi1"].to_csv(MI1_CSV, index=False)
    source_outputs["Mi4"].to_csv(MI4_CSV, index=False)
    lp_results = fit_and_plot_sources(
        {"Mi1": source_outputs["Mi1"], "Mi4": source_outputs["Mi4"]}
    )
    lp_results.to_csv(LP_OUTPUT_CSV, index=False)

    print(f"old shared Gaussian sigma: {old_gaussian_sigma:.9g}")
    print(f"new fitted common light Gaussian sigma: {new_light_sigma:.9g}")
    print(gaussian_comparison.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"wrote {GAUSSIAN_COMPARISON_CSV}")
    print(f"wrote {GAUSSIAN_OUTPUT_PNG}")
    print(f"wrote {MI9_CSV}")
    print(f"wrote {MI1_CSV}")
    print(f"wrote {MI4_CSV}")
    print(lp_results.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"wrote {LP_OUTPUT_CSV}")
    print(f"wrote {LP_OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
