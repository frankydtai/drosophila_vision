"""
BANC Dataset Utility Functions
==============================
Loader, analysis, and plotting helpers for the BANC connectome dataset.

Datasets
--------
neurons.csv.gz       : 115,151 neurons with region, NT type, body part, etc.
connections_princeton.csv.gz : 3,777,046 synaptic connections with neuropil & syn_count.

Usage
-----
    from banc_utils import load_neurons, load_connections, plot_bar
    neurons_df, conn_df = load_both()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Default paths (relative to this file's directory) ─────────────────────────
# The datasets now live directly under Connectome Dataset/Banc/.
DATA_DIR = Path(__file__).parent
NEURONS_PATH     = DATA_DIR / "neurons.csv.gz"
CONNECTIONS_PATH = DATA_DIR / "connections_princeton.csv.gz"

# ── Shared plot style ─────────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-whitegrid"


# =============================================================================
# Data Loading
# =============================================================================

def load_neurons(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load neurons.csv.gz.

    Columns
    -------
    Root ID, Top in/out region, Community labels,
    Predicted NT type, Predicted NT confidence,
    Verified NT type, Verified Neuropeptide, Body Part

    Returns
    -------
    pd.DataFrame  shape ~ (115151, 8)
    """
    p = Path(path) if path else NEURONS_PATH
    df = pd.read_csv(p, compression="gzip")
    # strip accidental whitespace in column names
    df.columns = df.columns.str.strip()
    return df


def load_connections(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load connections_princeton.csv.gz.

    Columns
    -------
    pre_root_id, post_root_id, neuropil, syn_count, nt_type

    Returns
    -------
    pd.DataFrame  shape ~ (3777046, 5)
    """
    p = Path(path) if path else CONNECTIONS_PATH
    df = pd.read_csv(p, compression="gzip")
    return df


def load_both(
    neurons_path: Optional[str | Path] = None,
    connections_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: returns (neurons_df, connections_df)."""
    return load_neurons(neurons_path), load_connections(connections_path)


# =============================================================================
# Neuron Analysis
# =============================================================================

def neuron_region_counts(neurons_df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """Top-N 'Top in/out region' value counts."""
    return neurons_df["Top in/out region"].value_counts().head(top_n)


def neuron_nt_type_counts(neurons_df: pd.DataFrame) -> pd.Series:
    """Predicted NT type distribution (NaN excluded)."""
    return neurons_df["Predicted NT type"].value_counts(dropna=True)


def neuron_body_part_counts(neurons_df: pd.DataFrame) -> pd.Series:
    """Body Part distribution."""
    return neurons_df["Body Part"].value_counts(dropna=True)


def neuron_community_label_counts(
    neurons_df: pd.DataFrame, top_n: int = 20
) -> pd.Series:
    """Top-N Community labels value counts."""
    return neurons_df["Community labels"].value_counts(dropna=True).head(top_n)


def neuron_nt_confidence_stats(neurons_df: pd.DataFrame) -> pd.Series:
    """Summary statistics for 'Predicted NT confidence'."""
    return neurons_df["Predicted NT confidence"].describe()


def missing_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with missing value count and percentage per column,
    sorted by descending missing percentage.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return (
        pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        .sort_values("missing_pct", ascending=False)
    )

# =============================================================================
# Connection Analysis
# =============================================================================

def connection_neuropil_counts(
    conn_df: pd.DataFrame, top_n: int = 20
) -> pd.Series:
    """Top-N neuropil regions by total connection count."""
    return conn_df["neuropil"].value_counts().head(top_n)


def connection_syn_count_stats(conn_df: pd.DataFrame) -> pd.Series:
    """Summary statistics for syn_count."""
    return conn_df["syn_count"].describe()


def connection_nt_type_counts(conn_df: pd.DataFrame) -> pd.Series:
    """NT type distribution in connection table.

    Missing ``nt_type`` is counted as ``(missing)``. Using only
    ``value_counts(dropna=True)`` yields an empty Series when every row is NaN,
    which breaks bar plots.
    """
    if conn_df.empty:
        return pd.Series(dtype=np.int64)
    return conn_df["nt_type"].fillna("(missing)").value_counts()


def top_upstream_neurons(
    conn_df: pd.DataFrame, top_n: int = 20
) -> pd.Series:
    """Neurons ranked by total outgoing synapse count."""
    return (
        conn_df.groupby("pre_root_id")["syn_count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )


def top_downstream_neurons(
    conn_df: pd.DataFrame, top_n: int = 20
) -> pd.Series:
    """Neurons ranked by total incoming synapse count."""
    return (
        conn_df.groupby("post_root_id")["syn_count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )


def syn_count_per_neuropil_nt(
    conn_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot table: sum of syn_count grouped by (neuropil x nt_type).
    Rows = neuropil, columns = nt_type.

    Missing ``nt_type`` is grouped as ``(missing)``. Dropping NaNs only
    (previous behaviour) yields an empty pivot when every row has NaN, which
    breaks heatmaps.
    """
    if conn_df.empty:
        return pd.DataFrame()
    sub = conn_df.copy()
    sub["nt_type"] = sub["nt_type"].fillna("(missing)")
    return (
        sub.groupby(["neuropil", "nt_type"])["syn_count"]
        .sum()
        .unstack(fill_value=0)
    )


def degree_distribution(
    conn_df: pd.DataFrame,
    kind: str = "out",
) -> pd.Series:
    """
    Compute out-degree (pre) or in-degree (post) distribution.

    Parameters
    ----------
    kind : 'out' or 'in'

    Returns
    -------
    pd.Series  index = degree, values = number of neurons
    """
    col = "pre_root_id" if kind == "out" else "post_root_id"
    degrees = conn_df.groupby(col).size()
    return degrees.value_counts().sort_index()


def connections_for_neuron(
    conn_df: pd.DataFrame,
    root_id: int,
    direction: str = "both",
) -> pd.DataFrame:
    """
    Return all connections for a single neuron.

    Parameters
    ----------
    direction : 'pre', 'post', or 'both'
    """
    if direction == "pre":
        return conn_df[conn_df["pre_root_id"] == root_id]
    if direction == "post":
        return conn_df[conn_df["post_root_id"] == root_id]
    return conn_df[
        (conn_df["pre_root_id"] == root_id) | (conn_df["post_root_id"] == root_id)
    ]


# =============================================================================
# Generic Plotting Helpers
# =============================================================================

def plot_bar(
    series: pd.Series,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Count",
    color: str = "steelblue",
    figsize: tuple = (10, 5),
    rotate_xticks: int = 45,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Horizontal or vertical bar chart from a pd.Series."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.style.use(PLOT_STYLE)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if series is None or len(series) == 0:
        ax.text(
            0.5,
            0.5,
            "No data to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax
    series.plot(kind="bar", ax=ax, color=color, edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate_xticks)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    return ax


def plot_barh(
    series: pd.Series,
    title: str = "",
    xlabel: str = "Count",
    ylabel: str = "",
    color: str = "steelblue",
    figsize: tuple = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Horizontal bar chart."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.style.use(PLOT_STYLE)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    series.sort_values().plot(kind="barh", ax=ax, color=color, edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    return ax


def plot_histogram(
    series: pd.Series,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Frequency",
    bins: int = 50,
    color: str = "steelblue",
    log_scale: bool = False,
    figsize: tuple = (9, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Histogram for a numeric series."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.style.use(PLOT_STYLE)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    series.dropna().plot(kind="hist", bins=bins, ax=ax, color=color, edgecolor="white",
                         logy=log_scale)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return ax


def plot_pie(
    series: pd.Series,
    title: str = "",
    figsize: tuple = (7, 7),
    ax: Optional[plt.Axes] = None,
    max_slices: int = 10,
) -> plt.Axes:
    """
    Pie chart. Groups all entries beyond max_slices as 'Other'.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.style.use(PLOT_STYLE)
    if len(series) > max_slices:
        top = series.head(max_slices)
        other = pd.Series({"Other": series.iloc[max_slices:].sum()})
        series = pd.concat([top, other])
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.pie(series.values, labels=series.index, autopct="%1.1f%%", startangle=140)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return ax


def plot_heatmap(
    df: pd.DataFrame,
    title: str = "",
    figsize: tuple = (12, 8),
    cmap: str = "YlOrRd",
    fmt: str = ".0f",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Heatmap via seaborn."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if df is None or df.size == 0 or df.shape[0] == 0 or df.shape[1] == 0:
        ax.text(
            0.5,
            0.5,
            "No data to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        return ax
    sns.heatmap(df, ax=ax, cmap=cmap, fmt=fmt, annot=(df.shape[0] <= 20),
                linewidths=0.3, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return ax


def plot_degree_distribution(
    conn_df: pd.DataFrame,
    kind: str = "out",
    log_log: bool = True,
    figsize: tuple = (8, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot degree distribution (log-log by default to check power-law).

    Parameters
    ----------
    kind : 'out' or 'in'
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.style.use(PLOT_STYLE)
    deg_dist = degree_distribution(conn_df, kind=kind)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.scatter(deg_dist.index, deg_dist.values, s=10, alpha=0.7, color="steelblue")
    if log_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    label = "Out-degree" if kind == "out" else "In-degree"
    ax.set_xlabel(f"{label} (# connections per neuron)")
    ax.set_ylabel("Number of neurons")
    ax.set_title(f"{label} Distribution" + (" (log-log)" if log_log else ""),
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return ax
