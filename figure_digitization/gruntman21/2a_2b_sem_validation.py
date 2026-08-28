#!/usr/bin/env python3
"""Extract Figure 2A SEM at trace extrema and compare it with Figure 2B SEM."""

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_2A = HERE / "2a_data_160ms.csv"
INPUT_2B = HERE / "2b_data.csv"
OUTPUT = HERE / "2a_2b_sem_validation.csv"
WIDTH4_CENTER_OFFSET_LED = 2


def extract_2a_extrema(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["cell_type", "contrast", "position"]
    for (cell_type, contrast, position), trace in data.groupby(keys, sort=False):
        trace = trace[np.isfinite(trace.vm_mv)]
        if trace.empty:
            continue
        for extremum, index in (
            ("depolarization", trace.vm_mv.idxmax()),
            ("hyperpolarization", trace.vm_mv.idxmin()),
        ):
            point = trace.loc[index]
            rows.append(
                {
                    "cell_type": cell_type,
                    "contrast": contrast,
                    "position_led": position - WIDTH4_CENTER_OFFSET_LED,
                    "extremum": extremum,
                    "time_ms_2a_extremum": point.time_ms,
                    "response_mv_2a": point.vm_mv,
                    "sem_mv_2a": point.vm_sem_mv,
                    "n_cells_2a": int(point.n_cells),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    extrema_2a = extract_2a_extrema(pd.read_csv(INPUT_2A))
    means_2b = pd.read_csv(INPUT_2B).query("statistic == 'mean'").rename(
        columns={
            "response_mv": "response_mv_2b",
            "response_sem_mv": "sem_mv_2b",
            "n_cells": "n_cells_2b",
        }
    )
    keys = ["cell_type", "contrast", "position_led", "extremum"]
    columns_2b = keys + ["response_mv_2b", "sem_mv_2b", "n_cells_2b"]
    comparison = extrema_2a.merge(means_2b[columns_2b], on=keys, how="inner")
    comparison["sem_difference_mv"] = comparison.sem_mv_2a - comparison.sem_mv_2b
    comparison["sem_abs_difference_mv"] = comparison.sem_difference_mv.abs()
    comparison["sem_equal"] = np.isclose(
        comparison.sem_mv_2a,
        comparison.sem_mv_2b,
        rtol=1e-6,
        atol=1e-6,
    )
    comparison["n_cells_equal"] = comparison.n_cells_2a.eq(comparison.n_cells_2b)
    comparison.to_csv(OUTPUT, index=False, float_format="%.6f")

    print(f"Wrote {OUTPUT} ({len(comparison)} matched extrema)")
    print(f"Exact SEM matches: {comparison.sem_equal.sum()}/{len(comparison)}")
    print(f"Equal cell counts: {comparison.n_cells_equal.sum()}/{len(comparison)}")
    print(f"SEM MAE: {comparison.sem_abs_difference_mv.mean():.6f} mV")
    print(f"SEM max error: {comparison.sem_abs_difference_mv.max():.6f} mV")
    print(f"SEM correlation: {comparison.sem_mv_2a.corr(comparison.sem_mv_2b):.6f}")


if __name__ == "__main__":
    main()
