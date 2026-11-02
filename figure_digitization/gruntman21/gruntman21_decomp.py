#!/home/yitai/drosophila_vision/figure_digitization/.venv/bin/python
"""Fit one shared spatial Gaussian for Gruntman21 T4's three sources.

Mi9, Mi1, and Mi4 use the same fitted sigma, with centers at -1, 0, and +1.
The Mi4 position-trace output is recentered to 0 after decomposition.

Input:
  ../gruntman21/2ax2bc_digitized.csv

Output:
  gruntman21_gaussian_fit.csv
  gruntman21_decomp.png
  gruntman21_mi1_position_traces.csv/.png
  gruntman21_mi4_position_traces.csv/.png
  gruntman21_mi1_mi4_fit_lp.csv
  gruntman21_mi1_mi4_fit_lp.png

Run:
  ../.venv/bin/python gruntman21_decomp.py

Constrained variant (Mi4 PC <= 0 throughout the 0..160 ms stimulus):
  ../.venv/bin/python gruntman21_decomp.py --mi4-pc-nonpositive
  Writes gruntman21_decomp2.png and matching gaussian_fit, fitted_traces,
  and sources CSVs with the gruntman21_decomp2_ prefix.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_VENV_PYTHON = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
if Path(sys.executable).parent.resolve() != _VENV_PYTHON.parent.resolve():
    os.execv(_VENV_PYTHON, [str(_VENV_PYTHON), *sys.argv])

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE.parent / "gruntman21" / "2ax2bc_digitized.csv"
GAUSSIAN_OUTPUT_PNG = HERE / "gruntman21_decomp.png"
GAUSSIAN_FIT_CSV = HERE / "gruntman21_gaussian_fit.csv"
MI1_POSITION_CSV = HERE / "gruntman21_mi1_position_traces.csv"
MI1_POSITION_PNG = HERE / "gruntman21_mi1_position_traces.png"
MI4_POSITION_CSV = HERE / "gruntman21_mi4_position_traces.csv"
MI4_POSITION_PNG = HERE / "gruntman21_mi4_position_traces.png"
LP_OUTPUT_CSV = HERE / "gruntman21_mi1_mi4_fit_lp.csv"
LP_OUTPUT_PNG = HERE / "gruntman21_mi1_mi4_fit_lp.png"

CELL_TYPE = "T4"
CONTRASTS = ("PC", "NC")
COLORS = {"PC": "#549f5c", "NC": "#303030"}
WIDTH_LED = 1
SPATIAL_POSITIONS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
SOURCE_NAMES = ("Mi9", "Mi1", "Mi4")
SOURCE_CENTERS = (-1.0, 0.0, 1.0)
ATTENUATION_DISTANCES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
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
    sigma: float,
    *,
    centers: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Use one fitted, peak-normalized Gaussian width for every source."""
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Gaussian sigma must be finite and positive")
    source_centers = SOURCE_CENTERS if centers is None else centers
    if len(source_centers) != len(SOURCE_NAMES):
        raise ValueError("centers length must match SOURCE_NAMES")
    distances = (
        np.asarray(SPATIAL_POSITIONS)[:, None]
        - np.asarray(source_centers)[None, :]
    )
    return np.exp(-0.5 * (distances / sigma) ** 2)


def solve_spatial_sources(
    sigma: float,
    observed: np.ndarray,
    *,
    mi4_nonpositive_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve optimal Mi9/Mi1/Mi4 time series and return their fitted traces."""
    weights = spatial_weight_matrix(sigma)
    sources = np.linalg.lstsq(weights, observed, rcond=None)[0]
    if mi4_nonpositive_mask is not None:
        mask = np.asarray(mi4_nonpositive_mask, dtype=bool)
        if mask.shape != (observed.shape[1],):
            raise ValueError("Mi4 constraint mask must match the observed time grid")
        mi4_index = SOURCE_NAMES.index("Mi4")
        active = np.flatnonzero(mask & (sources[mi4_index] > 0.0))
        if active.size:
            # One upper bound gives an exact active-set solution: when the
            # unconstrained Mi4 value is positive, put it on the zero boundary
            # and refit Mi9/Mi1 jointly rather than clipping the fitted trace.
            free = [i for i in range(len(SOURCE_NAMES)) if i != mi4_index]
            constrained = np.zeros((len(SOURCE_NAMES), len(active)))
            constrained[free] = np.linalg.lstsq(
                weights[:, free], observed[:, active], rcond=None
            )[0]
            sources[:, active] = constrained
    return sources, weights @ sources


def fit_gaussian_spatial_model(
    observed_by_contrast: dict[str, np.ndarray],
    *,
    mi4_pc_nonpositive_mask: np.ndarray | None = None,
) -> float:
    """Fit one sigma shared by all three sources and all PC/NC panels."""
    def objective(log_sigma: float) -> float:
        sigma = float(np.exp(log_sigma))
        squared_error = 0.0
        for contrast, observed in observed_by_contrast.items():
            _, fitted = solve_spatial_sources(
                sigma,
                observed,
                mi4_nonpositive_mask=(
                    mi4_pc_nonpositive_mask if contrast == "PC" else None
                ),
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
    return float(np.exp(result.x))


def spatial_output_tables(
    sigma: float,
    aligned_by_contrast: dict[str, pd.DataFrame],
    observed_by_contrast: dict[str, np.ndarray],
    *,
    mi4_pc_nonpositive_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build tidy observed/fit and inferred-source output tables."""
    fit_frames = []
    source_frames = []
    for contrast in CONTRASTS:
        aligned = aligned_by_contrast[contrast]
        observed = observed_by_contrast[contrast]
        sources, fitted = solve_spatial_sources(
            sigma,
            observed,
            mi4_nonpositive_mask=(
                mi4_pc_nonpositive_mask if contrast == "PC" else None
            ),
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


def build_gaussian_fit(sigma: float) -> pd.DataFrame:
    """Record the shared fitted sigma and its gains at neighbour distances."""
    distances = np.asarray(ATTENUATION_DISTANCES)
    return pd.DataFrame(
        {
            "neighbor_distance": distances,
            "shared_sigma": sigma,
            "gaussian_gain": np.exp(-0.5 * (distances / sigma) ** 2),
        }
    )


def plot_spatial_fit(
    output: pd.DataFrame,
    sources: pd.DataFrame,
    output_path: Path,
    gaussian_fit: pd.DataFrame,
    *,
    constraint_note: str | None = None,
) -> None:
    """Plot three inferred sources and all nine fitted position panels."""
    fig = plt.figure(figsize=(22.5, 8.0))
    grid = fig.add_gridspec(2, 9)
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
        ax.axvspan(0.0, STIMULUS_DURATION_MS, color="0.92", zorder=-2)
        ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=-1)
        ax.axvline(0.0, color="0.75", linewidth=0.7, zorder=-1)
        center = SOURCE_CENTERS[SOURCE_NAMES.index(source_name)]
        ax.set_title(f"Input {source_name}, center {center:+g}")
        ax.set_xlabel("time (ms)")
    source_axes[0].set_ylabel("Source Vm (mV)")
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
        ax.axvspan(0.0, STIMULUS_DURATION_MS, color="0.92", zorder=-2)
        ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=-1)
        ax.axvline(0.0, color="0.75", linewidth=0.7, zorder=-1)
        ax.set_title(f"Position {position:+g}")
        ax.set_xlabel("time (ms)")
        ax.tick_params(labelsize=7)
    position_axes[0].set_ylabel("Vm (mV)")
    position_axes[-1].legend(frameon=False, fontsize=7)

    source_y_min = min(ax.get_ylim()[0] for ax in source_axes)
    source_y_max = max(ax.get_ylim()[1] for ax in source_axes)
    for ax in source_axes:
        ax.set_ylim(source_y_min, source_y_max)

    position_y_min = min(ax.get_ylim()[0] for ax in position_axes)
    position_y_max = max(ax.get_ylim()[1] for ax in position_axes)
    for ax in position_axes:
        ax.set_ylim(position_y_min, position_y_max)
    gains = ", ".join(
        f"A{row.neighbor_distance:g}={row.gaussian_gain:.6f}"
        for row in gaussian_fit.itertuples(index=False)
    )
    sigma = gaussian_fit["shared_sigma"].iloc[0]
    title = (
        "T4 width 1/4 — three-source decomposition\n"
        rf"Mi9, Mi1, Mi4 share fitted Gaussian $\sigma={sigma:.6f}$: {gains}"
    )
    if constraint_note is not None:
        title += f"\n{constraint_note}"
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92 if constraint_note else 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_source_position_traces(
    sigma: float,
    sources: pd.DataFrame,
    source_name: str,
    *,
    center_position: float | None = None,
) -> pd.DataFrame:
    """Scale one source trace using the shared fitted spatial Gaussian."""
    source_column = SOURCE_NAMES.index(source_name)
    centers = list(SOURCE_CENTERS)
    if center_position is not None:
        centers[source_column] = center_position
    weights = spatial_weight_matrix(sigma, centers=tuple(centers))
    center = centers[source_column]
    frames = []
    for contrast in CONTRASTS:
        source = sources[
            (sources["contrast"] == contrast)
            & (sources["source"] == source_name)
        ][["time_ms", "source_vm_mv"]]
        for row, position in enumerate(SPATIAL_POSITIONS):
            weight = weights[row, source_column]
            frames.append(
                pd.DataFrame(
                    {
                        "source": source_name,
                        "contrast": contrast,
                        "center_position": center,
                        "position": position,
                        "spatial_gain": weight,
                        "time_ms": source["time_ms"].to_numpy(),
                        "contribution_vm_mv": (
                            weight * source["source_vm_mv"].to_numpy()
                        ),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def plot_source_position_traces(
    traces: pd.DataFrame,
    common_sigma: float,
    source_name: str,
    output_png: Path,
) -> None:
    """Plot source PC/NC contribution traces from position -2 through +2."""
    center = float(traces["center_position"].iloc[0])
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 10.0), sharex=True, sharey=True)
    for ax, position in zip(axes.flat, SPATIAL_POSITIONS):
        position_data = traces[np.isclose(traces["position"], position)]
        for contrast in CONTRASTS:
            trace = position_data[position_data["contrast"] == contrast]
            gain = trace["spatial_gain"].iloc[0]
            ax.plot(
                trace["time_ms"],
                trace["contribution_vm_mv"],
                color=COLORS[contrast],
                linewidth=1.8,
                label=f"{contrast} (gain={gain:.6f})",
            )
        ax.axvspan(0.0, STIMULUS_DURATION_MS, color="0.92", zorder=-2)
        ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=-1)
        ax.axvline(0.0, color="0.75", linewidth=0.7, zorder=-1)
        ax.set_title(f"{source_name} position {position:+g}")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(f"{source_name} contribution (mV)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        rf"{source_name} contribution traces, position -2 to +2 "
        rf"(center={center:+g}; shared fitted Gaussian $\sigma={common_sigma:.6f}$)"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_png, dpi=180)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mi4-pc-nonpositive",
        action="store_true",
        help="Fit Mi4 PC <= 0 at 0..160 ms and save the separate decomp2 outputs.",
    )
    args = parser.parse_args()
    data = pd.read_csv(INPUT_CSV)
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
    mi4_pc_nonpositive_mask = None
    if args.mi4_pc_nonpositive:
        pc_time_ms = aligned_by_contrast["PC"]["time_ms"].to_numpy()
        mi4_pc_nonpositive_mask = (
            (pc_time_ms >= 0.0) & (pc_time_ms <= STIMULUS_DURATION_MS)
        )
    shared_sigma = fit_gaussian_spatial_model(
        observed_by_contrast,
        mi4_pc_nonpositive_mask=mi4_pc_nonpositive_mask,
    )
    gaussian_fit = build_gaussian_fit(shared_sigma)
    gaussian_output, gaussian_sources = spatial_output_tables(
        shared_sigma,
        aligned_by_contrast,
        observed_by_contrast,
        mi4_pc_nonpositive_mask=mi4_pc_nonpositive_mask,
    )

    if args.mi4_pc_nonpositive:
        output_png = HERE / "gruntman21_decomp2.png"
        fit_csv = HERE / "gruntman21_decomp2_gaussian_fit.csv"
        traces_csv = HERE / "gruntman21_decomp2_fitted_traces.csv"
        sources_csv = HERE / "gruntman21_decomp2_sources.csv"
        mi4_pc_stimulus = gaussian_sources[
            (gaussian_sources["source"] == "Mi4")
            & (gaussian_sources["contrast"] == "PC")
            & gaussian_sources["time_ms"].between(0.0, STIMULUS_DURATION_MS)
        ]["source_vm_mv"]
        if mi4_pc_stimulus.empty or (mi4_pc_stimulus > 0.0).any():
            raise RuntimeError("Mi4 PC must be <= 0 throughout the stimulus")
        gaussian_fit.to_csv(fit_csv, index=False)
        gaussian_output.to_csv(traces_csv, index=False)
        gaussian_sources.to_csv(sources_csv, index=False)
        plot_spatial_fit(
            gaussian_output,
            gaussian_sources,
            output_png,
            gaussian_fit,
            constraint_note=(
                r"Constraint: Mi4 PC $\leq 0$ for $0 \leq t \leq 160$ ms"
            ),
        )
        print(f"shared fitted Gaussian sigma: {shared_sigma:.9g}")
        print(f"Mi4 PC maximum at 0..160 ms: {mi4_pc_stimulus.max():.9g} mV")
        print(f"wrote {output_png}")
        print(f"wrote {fit_csv}")
        print(f"wrote {traces_csv}")
        print(f"wrote {sources_csv}")
        return 0

    gaussian_fit.to_csv(GAUSSIAN_FIT_CSV, index=False)
    plot_spatial_fit(
        gaussian_output,
        gaussian_sources,
        GAUSSIAN_OUTPUT_PNG,
        gaussian_fit,
    )

    for source_name, position_csv, position_png, center_position in (
        ("Mi1", MI1_POSITION_CSV, MI1_POSITION_PNG, None),
        ("Mi4", MI4_POSITION_CSV, MI4_POSITION_PNG, 0.0),
    ):
        position_traces = build_source_position_traces(
            shared_sigma,
            gaussian_sources,
            source_name,
            center_position=center_position,
        )
        position_traces.to_csv(position_csv, index=False)
        plot_source_position_traces(
            position_traces,
            shared_sigma,
            source_name,
            position_png,
        )

    source_outputs = {}
    for source_name in ("Mi1", "Mi4"):
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

    lp_results = fit_and_plot_sources(source_outputs)
    lp_results.to_csv(LP_OUTPUT_CSV, index=False)

    print(f"shared fitted Gaussian sigma (Mi9/Mi1/Mi4, PC/NC): {shared_sigma:.9g}")
    print(gaussian_fit.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"wrote {GAUSSIAN_FIT_CSV}")
    print(f"wrote {GAUSSIAN_OUTPUT_PNG}")
    print(f"wrote {MI1_POSITION_CSV}")
    print(f"wrote {MI1_POSITION_PNG}")
    print(f"wrote {MI4_POSITION_CSV}")
    print(f"wrote {MI4_POSITION_PNG}")
    print(lp_results.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"wrote {LP_OUTPUT_CSV}")
    print(f"wrote {LP_OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
