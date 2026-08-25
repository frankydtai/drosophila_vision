#!/usr/bin/env python3
"""Plot V(+0) - V(+2) - V(-2) for Gruntman21 T4 width 1/4.

Input:
  ../gruntman21/2ax2bc_digitized.csv

Output:
  gruntman21_decomp.png
  gruntman21_pm1_fit.png
  gruntman21_spatial_fit.png

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
OUTPUT_PNG = HERE / "gruntman21_decomp.png"
FIT_OUTPUT_PNG = HERE / "gruntman21_pm1_fit.png"
SPATIAL_OUTPUT_PNG = HERE / "gruntman21_spatial_fit.png"
EXTENDED_OUTPUT_PNG = HERE / "gruntman21_spatial_fit_a3.png"
ATTENUATION_OUTPUT_PNG = HERE / "gruntman21_attenuation_a0_a3.png"
GAUSSIAN_OUTPUT_PNG = HERE / "gruntman21_spatial_fit_gaussian.png"
GAUSSIAN_COMPARE_PNG = HERE / "gruntman21_attenuation_gaussian_compare.png"

CELL_TYPE = "T4"
CONTRASTS = ("PC", "NC")
COLORS = {"PC": "#549f5c", "NC": "#303030"}
WIDTH_LED = 1
POSITIONS = (0.0, 2.0, -2.0)
FIT_SIDES = (-1.0, 1.0)
SPATIAL_POSITIONS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
SOURCE_NAMES = ("M9", "M1", "M4")
SOURCE_CENTERS = (-1.0, 0.0, 1.0)
ATTENUATION_DISTANCES = (0.5, 1.0, 1.5, 2.0)
EXTENDED_ATTENUATION_DISTANCES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


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


def fit_shared_a(data: pd.DataFrame) -> tuple[float, dict[str, pd.DataFrame]]:
    """Fit one positive a jointly to PC/NC traces at positions +1 and -1."""
    fit_data = {
        contrast: align_traces(data, contrast, (0.0, 2.0, -2.0, 1.0, -1.0))
        for contrast in CONTRASTS
    }

    def residual(log_a: np.ndarray) -> np.ndarray:
        a = float(np.exp(log_a[0]))
        errors = []
        for aligned in fit_data.values():
            errors.append(
                aligned["vm_+0"].to_numpy() * a
                + aligned["vm_+2"].to_numpy() / a
                - aligned["vm_+1"].to_numpy()
            )
            errors.append(
                aligned["vm_+0"].to_numpy() * a
                + aligned["vm_-2"].to_numpy() / a
                - aligned["vm_-1"].to_numpy()
            )
        return np.concatenate(errors)

    result = least_squares(residual, x0=np.array([0.0]), max_nfev=10_000)
    if not result.success:
        raise RuntimeError(f"a fit failed: {result.message}")
    return float(np.exp(result.x[0])), fit_data


def plot_pm1_fit(
    a: float,
    fit_data: dict[str, pd.DataFrame],
) -> None:
    """Plot measured ±1 traces solid and fitted traces dashed."""
    fig, axes = plt.subplots(
        1,
        len(FIT_SIDES),
        figsize=(12.0, 4.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for column, side in enumerate(FIT_SIDES):
        ax = axes[0, column]
        for contrast in CONTRASTS:
            aligned = fit_data[contrast]
            source_position = 2.0 * side
            measured = aligned[f"vm_{side:+g}"]
            fitted = aligned["vm_+0"] * a + aligned[
                f"vm_{source_position:+g}"
            ] / a
            ax.plot(
                aligned["time_ms"],
                measured,
                color=COLORS[contrast],
                linestyle="-",
                linewidth=2.0,
                label=f"{contrast} original",
            )
            ax.plot(
                aligned["time_ms"],
                fitted,
                color=COLORS[contrast],
                linestyle="--",
                linewidth=2.0,
                label=f"{contrast} fit",
            )
        ax.axvspan(0.0, 160.0, color="0.92", zorder=-2)
        ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=-1)
        ax.axvline(0.0, color="0.75", linewidth=0.7, zorder=-1)
        ax.set_title(f"position {side:+g}")
        ax.set_xlabel("time (ms)")
        ax.legend(frameon=False, fontsize=9)
    axes[0, 0].set_ylabel("Vm (mV)")
    fig.suptitle(
        "T4 width 1/4: "
        rf"$V_{{\pm1}}=aV_0+V_{{\pm2}}/a$, shared $a={a:.6f}$"
    )
    fig.tight_layout()
    fig.savefig(FIT_OUTPUT_PNG, dpi=180)
    plt.close(fig)


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


def solve_spatial_sources(
    attenuations: np.ndarray,
    observed: np.ndarray,
    attenuation_distances: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve optimal M9/M1/M4 time series and return their fitted traces."""
    weights = spatial_weight_matrix(attenuations, attenuation_distances)
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
) -> tuple[float, np.ndarray]:
    """Fit one Gaussian sigma to the same nine PC and NC panels."""
    def objective(log_sigma: float) -> float:
        attenuations = gaussian_attenuations(float(np.exp(log_sigma)))
        squared_error = 0.0
        for observed in observed_by_contrast.values():
            _, fitted = solve_spatial_sources(
                attenuations,
                observed,
                EXTENDED_ATTENUATION_DISTANCES,
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


def plot_spatial_fit(
    attenuations: np.ndarray,
    attenuation_distances: tuple[float, ...],
    output: pd.DataFrame,
    sources: pd.DataFrame,
    output_path: Path,
    model_label: str = "three-source spatial fit",
) -> None:
    """Plot three source panels above nine observed/fit position panels."""
    fig = plt.figure(figsize=(22.5, 7.5))
    grid = fig.add_gridspec(2, 9, height_ratios=(1.0, 1.0))
    source_axes = [
        fig.add_subplot(grid[0, 0:3]),
        fig.add_subplot(grid[0, 3:6]),
        fig.add_subplot(grid[0, 6:9]),
    ]
    position_axes = [fig.add_subplot(grid[1, column]) for column in range(9)]

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
        ax.set_title(f"input {source_name}, center {center:+g}")
        ax.set_xlabel("time (ms)")
    source_axes[0].set_ylabel("source Vm (mV)")
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
                label=f"{contrast} fit",
            )
        ax.axvspan(0.0, 160.0, color="0.92", zorder=-2)
        ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=-1)
        ax.axvline(0.0, color="0.75", linewidth=0.7, zorder=-1)
        ax.set_title(f"position {position:+g}")
        ax.set_xlabel("time (ms)")
        ax.tick_params(labelsize=7)
    position_axes[0].set_ylabel("Vm (mV)")
    position_axes[-1].legend(frameon=False, fontsize=7)

    for row_axes in (source_axes, position_axes):
        y_min = min(ax.get_ylim()[0] for ax in row_axes)
        y_max = max(ax.get_ylim()[1] for ax in row_axes)
        for ax in row_axes:
            ax.set_ylim(y_min, y_max)
    values = ", ".join(
        f"A{distance:g}={value:.6f}"
        for distance, value in zip(attenuation_distances, attenuations)
    )
    fig.suptitle(f"T4 width 1/4 {model_label}\n{values}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_attenuation_a0_a3(attenuations: np.ndarray) -> None:
    """Plot the extended spatial attenuation profile from A0=1 to A3."""
    distances = np.asarray((0.0, *EXTENDED_ATTENUATION_DISTANCES))
    values = np.asarray((1.0, *attenuations))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        distances,
        values,
        color="#4c72b0",
        marker="o",
        linewidth=2.0,
    )
    for distance, value in zip(distances, values):
        ax.annotate(
            f"{value:.3f}",
            (distance, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(distances)
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("distance")
    ax.set_ylabel("attenuation A")
    ax.set_title("T4 width 1/4 spatial attenuation: A0 to A3")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ATTENUATION_OUTPUT_PNG, dpi=180)
    plt.close(fig)


def plot_gaussian_attenuation_comparison(
    unconstrained: np.ndarray,
    gaussian: np.ndarray,
    sigma: float,
) -> None:
    """Compare unconstrained and Gaussian-constrained A0..A3 curves."""
    distances = np.asarray((0.0, *EXTENDED_ATTENUATION_DISTANCES))
    unconstrained_values = np.asarray((1.0, *unconstrained))
    gaussian_values = np.asarray((1.0, *gaussian))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        distances,
        unconstrained_values,
        color="#4c72b0",
        marker="o",
        linewidth=2.0,
        label="unconstrained",
    )
    dense_distance = np.linspace(0.0, 3.0, 500)
    ax.plot(
        dense_distance,
        np.exp(-0.5 * (dense_distance / sigma) ** 2),
        color="#dd8452",
        linewidth=2.0,
        label=rf"Gaussian ($\sigma={sigma:.6f}$)",
    )
    ax.plot(
        distances,
        gaussian_values,
        color="#dd8452",
        marker="o",
        linewidth=0,
    )
    ax.set_xticks(distances)
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("distance")
    ax.set_ylabel("attenuation A")
    ax.set_title("T4 width 1/4: unconstrained vs Gaussian attenuation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(GAUSSIAN_COMPARE_PNG, dpi=180)
    plt.close(fig)


def main() -> int:
    data = pd.read_csv(INPUT_CSV)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    summaries = []
    for contrast in CONTRASTS:
        aligned = align_traces(data, contrast, POSITIONS)

        result_mv = aligned["vm_+0"] - aligned["vm_+2"] - aligned["vm_-2"]
        if not np.isfinite(result_mv.to_numpy()).all():
            raise ValueError(f"non-finite value in decomposed {contrast} trace")

        ax.plot(
            aligned["time_ms"],
            result_mv,
            color=COLORS[contrast],
            linewidth=2.0,
            label=contrast,
        )
        summaries.append(
            f"{contrast}: n={len(aligned)}, "
            f"time={aligned.time_ms.min():.1f}..{aligned.time_ms.max():.1f} ms, "
            f"Vm={result_mv.min():.3f}..{result_mv.max():.3f} mV"
        )

    ax.axvspan(0.0, 160.0, color="0.92", zorder=-2)
    ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=-1)
    ax.axvline(0.0, color="0.75", linewidth=0.7, zorder=-1)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("Vm (mV)")
    ax.set_title("T4 width 1/4: V(+0) - V(+2) - V(-2)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    a, fit_data = fit_shared_a(data)
    plot_pm1_fit(a, fit_data)
    attenuations, spatial_aligned, spatial_observed = fit_spatial_model(
        data,
        ATTENUATION_DISTANCES,
    )
    spatial_output, spatial_sources = spatial_output_tables(
        attenuations,
        ATTENUATION_DISTANCES,
        spatial_aligned,
        spatial_observed,
    )
    plot_spatial_fit(
        attenuations,
        ATTENUATION_DISTANCES,
        spatial_output,
        spatial_sources,
        SPATIAL_OUTPUT_PNG,
    )

    extended, extended_aligned, extended_observed = fit_spatial_model(
        data,
        EXTENDED_ATTENUATION_DISTANCES,
    )
    extended_output, extended_sources = spatial_output_tables(
        extended,
        EXTENDED_ATTENUATION_DISTANCES,
        extended_aligned,
        extended_observed,
    )
    plot_spatial_fit(
        extended,
        EXTENDED_ATTENUATION_DISTANCES,
        extended_output,
        extended_sources,
        EXTENDED_OUTPUT_PNG,
    )
    plot_attenuation_a0_a3(extended)

    gaussian_sigma, gaussian = fit_gaussian_spatial_model(
        extended_observed,
    )
    gaussian_output, gaussian_sources = spatial_output_tables(
        gaussian,
        EXTENDED_ATTENUATION_DISTANCES,
        extended_aligned,
        extended_observed,
    )
    plot_spatial_fit(
        gaussian,
        EXTENDED_ATTENUATION_DISTANCES,
        gaussian_output,
        gaussian_sources,
        GAUSSIAN_OUTPUT_PNG,
        model_label=rf"Gaussian spatial fit ($\sigma={gaussian_sigma:.6f}$)",
    )
    plot_gaussian_attenuation_comparison(
        extended,
        gaussian,
        gaussian_sigma,
    )
    print("\n".join(summaries))
    print(f"best shared a={a:.9f}")
    print(
        "best spatial attenuations: "
        + ", ".join(
            f"A{distance:g}={value:.9f}"
            for distance, value in zip(ATTENUATION_DISTANCES, attenuations)
        )
    )
    print(
        "best extended spatial attenuations: "
        + ", ".join(
            f"A{distance:g}={value:.9f}"
            for distance, value in zip(
                EXTENDED_ATTENUATION_DISTANCES,
                extended,
            )
        )
    )
    print(f"best Gaussian sigma={gaussian_sigma:.9f}")
    print(
        "Gaussian spatial attenuations: "
        + ", ".join(
            f"A{distance:g}={value:.9f}"
            for distance, value in zip(
                EXTENDED_ATTENUATION_DISTANCES,
                gaussian,
            )
        )
    )
    print("attenuation differences (Gaussian - unconstrained):")
    for distance, free_value, gaussian_value in zip(
        EXTENDED_ATTENUATION_DISTANCES,
        extended,
        gaussian,
    ):
        print(
            f"  A{distance:g}: {gaussian_value - free_value:+.9f} "
            f"({free_value:.9f} -> {gaussian_value:.9f})"
        )
    for contrast in CONTRASTS:
        residual = spatial_output.loc[
            spatial_output["contrast"] == contrast,
            "residual_mv",
        ].to_numpy()
        print(f"{contrast} spatial RMSE={np.sqrt(np.mean(residual**2)):.6f} mV")
        free_trace = extended_output[
            extended_output["contrast"] == contrast
        ].reset_index(drop=True)
        gaussian_trace = gaussian_output[
            gaussian_output["contrast"] == contrast
        ].reset_index(drop=True)
        if not free_trace[["position", "time_ms"]].equals(
            gaussian_trace[["position", "time_ms"]]
        ):
            raise RuntimeError("Gaussian and unconstrained output grids differ")
        free_rmse = float(np.sqrt(np.mean(free_trace["residual_mv"] ** 2)))
        gaussian_rmse = float(
            np.sqrt(np.mean(gaussian_trace["residual_mv"] ** 2))
        )
        fitted_difference = (
            gaussian_trace["fitted_vm_mv"] - free_trace["fitted_vm_mv"]
        ).to_numpy()
        print(
            f"{contrast} Gaussian comparison: "
            f"RMSE {free_rmse:.9f} -> {gaussian_rmse:.9f} mV "
            f"(delta {gaussian_rmse - free_rmse:+.9f}); "
            f"fit-difference RMS={np.sqrt(np.mean(fitted_difference**2)):.9f} mV, "
            f"max_abs={np.max(np.abs(fitted_difference)):.9f} mV"
        )
    print(f"wrote {OUTPUT_PNG}")
    print(f"wrote {FIT_OUTPUT_PNG}")
    print(f"wrote {SPATIAL_OUTPUT_PNG}")
    print(f"wrote {EXTENDED_OUTPUT_PNG}")
    print(f"wrote {ATTENUATION_OUTPUT_PNG}")
    print(f"wrote {GAUSSIAN_OUTPUT_PNG}")
    print(f"wrote {GAUSSIAN_COMPARE_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
