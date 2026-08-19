"""Plot fitted T4 preferred-direction gains across stimulus positions.

Reads ``gruntman21_fit_lp.csv`` and writes:

* ``gruntman21_t4_g.csv`` with observed g4 and smooth recovered G values
* ``gruntman21_t4_g.png`` with all four gain curves
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "gruntman21_fit_lp.csv"
OUTPUT_CSV = HERE / "gruntman21_t4_g.csv"
OUTPUT_PNG = HERE / "gruntman21_t4_g.png"
POSITIONS = list(range(-6, 6))
G_POSITIONS = list(range(-7, 8))
POSITIVE_COLOR = "tab:blue"
NEGATIVE_COLOR = "tab:orange"


def gaussian_curve(
    position: np.ndarray,
    baseline: float,
    amplitude: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    """Gaussian plus a constant baseline; inflections are center ± sigma."""
    return baseline + amplitude * np.exp(
        -0.5 * ((np.asarray(position, dtype=float) - center) / sigma) ** 2
    )


def fit_g4(g4: pd.Series, sign: float) -> tuple[pd.Series, np.ndarray]:
    """Fit the signed g4 magnitude with a two-inflection Gaussian curve."""
    position = np.asarray(POSITIONS, dtype=float)
    magnitude = sign * g4.loc[POSITIONS].to_numpy(dtype=float)
    baseline0 = max(float(np.min(magnitude)), 0.0)
    amplitude0 = max(float(np.max(magnitude) - baseline0), 0.1)
    center0 = float(position[np.argmax(magnitude)])

    result = least_squares(
        lambda parameters: gaussian_curve(position, *parameters) - magnitude,
        x0=np.array([baseline0, amplitude0, center0, 2.5]),
        bounds=(
            np.array([0.0, 0.0, -8.0, 0.25]),
            np.array([100.0, 200.0, 8.0, 20.0]),
        ),
        max_nfev=20_000,
    )
    fitted = sign * gaussian_curve(position, *result.x)
    return pd.Series(fitted, index=POSITIONS), result.x


def solve_g(g4: pd.Series) -> np.ndarray:
    """Return the minimum-curvature G that reproduces fitted g4 exactly."""
    position_to_index = {
        position: index for index, position in enumerate(G_POSITIONS)
    }
    matrix = np.zeros((len(POSITIONS), len(G_POSITIONS)), dtype=float)
    for row, position in enumerate(POSITIONS):
        for source_position in range(position - 1, position + 3):
            matrix[row, position_to_index[source_position]] = 1.0

    target = g4.loc[POSITIONS].to_numpy(dtype=float)
    particular = np.linalg.lstsq(matrix, target, rcond=None)[0]

    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=True)
    tolerance = singular_values[0] * max(matrix.shape) * np.finfo(float).eps
    rank = int(np.sum(singular_values > tolerance))
    null_basis = right_vectors[rank:].T

    second_difference = np.zeros(
        (len(G_POSITIONS) - 2, len(G_POSITIONS)), dtype=float
    )
    for row in range(len(G_POSITIONS) - 2):
        second_difference[row, row:row + 3] = (1.0, -2.0, 1.0)

    null_weights = np.linalg.lstsq(
        second_difference @ null_basis,
        -(second_difference @ particular),
        rcond=None,
    )[0]
    solution = particular + null_basis @ null_weights
    if not np.allclose(matrix @ solution, target, atol=1e-9, rtol=0.0):
        raise RuntimeError("smooth G solution does not reproduce g4 exactly")
    return solution


def main() -> int:
    fits = pd.read_csv(INPUT_CSV)
    t4_pc = fits[fits["trace_id"].str.startswith("T4_PC_")].copy()
    t4_pc = t4_pc.sort_values("position")

    observed_positions = t4_pc["position"].astype(int).tolist()
    if observed_positions != POSITIONS:
        raise ValueError(
            f"expected T4 PC positions {POSITIONS}, got {observed_positions}"
        )

    observed_gains = t4_pc[["position", "gain_pos_mv", "gain_neg_mv"]].rename(
        columns={"gain_pos_mv": "g4+", "gain_neg_mv": "g4-"}
    )
    observed_gains = observed_gains.set_index("position")
    fitted_pos, parameters_pos = fit_g4(observed_gains["g4+"], sign=1.0)
    fitted_neg, parameters_neg = fit_g4(observed_gains["g4-"], sign=-1.0)
    fitted_gains = pd.DataFrame({
        "g4+ fit": fitted_pos,
        "g4- fit": fitted_neg,
    })
    gains = pd.DataFrame({
        "position": G_POSITIONS,
        "G+": solve_g(fitted_pos),
        "G-": solve_g(fitted_neg),
    }).set_index("position")
    gains = observed_gains.join(fitted_gains).join(gains, how="outer").reset_index()
    gains = gains[[
        "position", "g4+", "g4-", "g4+ fit", "g4- fit", "G+", "G-",
    ]]
    gains.to_csv(OUTPUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        gains["position"], gains["g4+"],
        color=POSITIVE_COLOR, marker="o", linewidth=0, label="g4+ data",
    )
    ax.plot(
        gains["position"], gains["g4-"],
        color=NEGATIVE_COLOR, marker="o", linewidth=0, label="g4- data",
    )
    dense_position = np.linspace(-7.0, 7.0, 500)
    ax.plot(
        dense_position, gaussian_curve(dense_position, *parameters_pos),
        color=POSITIVE_COLOR, linewidth=2, label="g4+ two-inflection fit",
    )
    ax.plot(
        dense_position, -gaussian_curve(dense_position, *parameters_neg),
        color=NEGATIVE_COLOR, linewidth=2, label="g4- two-inflection fit",
    )
    ax.plot(
        gains["position"], gains["G+"],
        color=POSITIVE_COLOR, marker="s", linewidth=1.7,
        linestyle="--", label="G+",
    )
    ax.plot(
        gains["position"], gains["G-"],
        color=NEGATIVE_COLOR, marker="s", linewidth=1.7,
        linestyle="--", label="G-",
    )
    ax.axhline(0.0, color="0.65", linewidth=0.8)
    ax.set_xticks(G_POSITIONS)
    ax.set_xlim(-7, 7)
    ax.set_xlabel("position")
    ax.set_ylabel("gain (mV)")
    ax.set_title("Gruntman 2021 T4 preferred-direction gains")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    print(gains.to_string(index=False))
    print(
        "g4+ inflections:",
        f"{parameters_pos[2] - parameters_pos[3]:.6f},",
        f"{parameters_pos[2] + parameters_pos[3]:.6f}",
    )
    print(
        "g4- inflections:",
        f"{parameters_neg[2] - parameters_neg[3]:.6f},",
        f"{parameters_neg[2] + parameters_neg[3]:.6f}",
    )
    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
