"""
FAFB Dataset Utility Functions
==============================
Loader, analysis, and plotting helpers for the FAFB connectome dataset.

This module uses strict FAFB-native schema (no BANC compatibility columns).
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
DATA_DIR = Path(__file__).parent
CELL_TYPES_PATH = DATA_DIR / "consolidated_cell_types.csv.gz"
CLASSIFICATION_PATH = DATA_DIR / "classification.csv.gz"
NT_PATH = DATA_DIR / "neurons.csv.gz"  # optional
CONNECTIONS_PATH = DATA_DIR / "connections_princeton.csv.gz"

# ── Shared plot style ─────────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-whitegrid"


def _read_csv_gz(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip")
    df.columns = df.columns.str.strip()
    return df


def _resolve_data_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (DATA_DIR / p)


def _infer_nt_from_connections() -> pd.DataFrame:
    """
    Infer per-neuron NT type from outgoing FAFB connections.
    Returns columns: root_id, nt_type, nt_score.
    """
    conn = load_connections()
    if conn.empty or "nt_type" not in conn.columns:
        return pd.DataFrame(columns=["root_id", "nt_type", "nt_score"])

    wcol = "syn_count" if "syn_count" in conn.columns else None
    sub = conn[["pre_root_id", "nt_type"] + ([wcol] if wcol else [])].copy()
    sub = sub.rename(columns={"pre_root_id": "root_id"})
    sub["nt_type"] = sub["nt_type"].fillna("(missing)")
    if wcol is None:
        sub["weight"] = 1.0
    else:
        sub["weight"] = pd.to_numeric(sub[wcol], errors="coerce").fillna(1.0).astype(float)

    agg = (
        sub.groupby(["root_id", "nt_type"])["weight"]
        .sum()
        .reset_index()
        .sort_values(["root_id", "weight"], ascending=[True, False])
    )
    if agg.empty:
        return pd.DataFrame(columns=["root_id", "nt_type", "nt_score"])

    top = agg.groupby("root_id", as_index=False).first()
    total = agg.groupby("root_id", as_index=False)["weight"].sum().rename(columns={"weight": "weight_total"})
    top = top.merge(total, on="root_id", how="left")
    top["nt_score"] = (top["weight"] / top["weight_total"]).astype(float)
    top["nt_type"] = top["nt_type"].replace("(missing)", np.nan)
    return top[["root_id", "nt_type", "nt_score"]]


# =============================================================================
# Data Loading
# =============================================================================

def load_neurons(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load FAFB neurons in native FAFB schema.

    If ``path`` is provided, reads that file directly. Otherwise merges:
      - consolidated_cell_types.csv.gz (required)
      - classification.csv.gz (required)
      - neurons.csv.gz (optional)
    """
    if path:
        p = _resolve_data_path(path)
        if p.exists():
            candidate = _read_csv_gz(p)
            # Guard against accidentally loading non-FAFB files (e.g. BANC neurons.csv.gz)
            if "root_id" in candidate.columns:
                base = candidate
            else:
                warnings.warn(
                    f"Ignoring non-FAFB neurons file: {p}. Falling back to FAFB merged sources."
                )
                path = None
        else:
            # Keep notebook compatibility with legacy calls like
            # load_neurons("neurons.csv.gz") when this optional file is absent.
            path = None

    if not path:
        if not CELL_TYPES_PATH.exists():
            raise FileNotFoundError(f"Required file not found: {CELL_TYPES_PATH}")
        if not CLASSIFICATION_PATH.exists():
            raise FileNotFoundError(f"Required file not found: {CLASSIFICATION_PATH}")

        base = _read_csv_gz(CELL_TYPES_PATH)
        cl = _read_csv_gz(CLASSIFICATION_PATH)
        base = base.merge(cl, on="root_id", how="left")

        if NT_PATH.exists():
            nt = _read_csv_gz(NT_PATH)
            if "nt_type_score" in nt.columns and "nt_score" not in nt.columns:
                nt = nt.rename(columns={"nt_type_score": "nt_score"})
            keep = ["root_id"] + [
                c for c in [
                    "nt_type", "nt_score",
                    "da_avg", "ser_avg", "gaba_avg", "glut_avg", "ach_avg", "oct_avg"
                ] if c in nt.columns
            ]
            base = base.merge(nt[keep], on="root_id", how="left")
        else:
            # Fallback: infer NT type from outgoing connection labels.
            nt_infer = _infer_nt_from_connections()
            if not nt_infer.empty:
                base = base.merge(nt_infer, on="root_id", how="left")

    # If caller passed a partial FAFB table (e.g. neurons.csv.gz), enrich with
    # core hierarchy columns so downstream notebook code can rely on them.
    if "root_id" in base.columns:
        if "super_class" not in base.columns and CLASSIFICATION_PATH.exists():
            cl = _read_csv_gz(CLASSIFICATION_PATH)
            keep = ["root_id"] + [c for c in ["flow", "super_class", "class", "sub_class", "hemilineage", "side", "nerve"] if c in cl.columns]
            base = base.merge(cl[keep], on="root_id", how="left")
        if "cell_type" not in base.columns and CELL_TYPES_PATH.exists():
            ct = _read_csv_gz(CELL_TYPES_PATH)
            keep = ["root_id"] + [c for c in ["primary_type", "additional_type(s)"] if c in ct.columns]
            base = base.merge(ct[keep], on="root_id", how="left")

    # Canonical naming for downstream usage.
    if "primary_type" in base.columns and "cell_type" not in base.columns:
        base = base.rename(columns={"primary_type": "cell_type"})
    if "nt_type_score" in base.columns and "nt_score" not in base.columns:
        base = base.rename(columns={"nt_type_score": "nt_score"})
    return base


def load_connections(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load FAFB connections. Defaults to connections_princeton.csv.gz and
    falls back to available no-threshold alternatives.
    """
    if path:
        p = _resolve_data_path(path)
        return _read_csv_gz(p)

    candidates = [
        CONNECTIONS_PATH,
        DATA_DIR / "connections_princeton_no_threshold.csv.gz",
        DATA_DIR / "connections_buhmann_no_threshold.csv.gz",
    ]
    for p in candidates:
        if p.exists():
            return _read_csv_gz(p)
    raise FileNotFoundError(f"No FAFB connections file found under {DATA_DIR}")


def load_both(
    neurons_path: Optional[str | Path] = None,
    connections_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_neurons(neurons_path), load_connections(connections_path)


# =============================================================================
# Neuron Analysis (FAFB-native columns)
# =============================================================================

def neuron_region_counts(neurons_df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    return neurons_df["super_class"].value_counts(dropna=True).head(top_n)


def neuron_nt_type_counts(neurons_df: pd.DataFrame) -> pd.Series:
    return neurons_df["nt_type"].value_counts(dropna=True)


def neuron_body_part_counts(neurons_df: pd.DataFrame) -> pd.Series:
    if "side" in neurons_df.columns:
        return neurons_df["side"].value_counts(dropna=True)
    return pd.Series(dtype=np.int64)


def neuron_community_label_counts(
    neurons_df: pd.DataFrame, top_n: int = 20
) -> pd.Series:
    if "sub_class" in neurons_df.columns:
        return neurons_df["sub_class"].value_counts(dropna=True).head(top_n)
    return pd.Series(dtype=np.int64)


def neuron_nt_confidence_stats(neurons_df: pd.DataFrame) -> pd.Series:
    if "nt_score" not in neurons_df.columns:
        return pd.Series(dtype=float)
    return neurons_df["nt_score"].describe()


def missing_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return (
        pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        .sort_values("missing_pct", ascending=False)
    )


# =============================================================================
# Connection Analysis
# =============================================================================

def connection_neuropil_counts(conn_df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    return conn_df["neuropil"].value_counts().head(top_n)


def connection_syn_count_stats(conn_df: pd.DataFrame) -> pd.Series:
    return conn_df["syn_count"].describe()


def connection_nt_type_counts(conn_df: pd.DataFrame) -> pd.Series:
    if conn_df.empty:
        return pd.Series(dtype=np.int64)
    return conn_df["nt_type"].fillna("(missing)").value_counts()


def top_upstream_neurons(conn_df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    return (
        conn_df.groupby("pre_root_id")["syn_count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )


def top_downstream_neurons(conn_df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    return (
        conn_df.groupby("post_root_id")["syn_count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )


def syn_count_per_neuropil_nt(conn_df: pd.DataFrame) -> pd.DataFrame:
    if conn_df.empty:
        return pd.DataFrame()
    sub = conn_df.copy()
    sub["nt_type"] = sub["nt_type"].fillna("(missing)")
    return (
        sub.groupby(["neuropil", "nt_type"])["syn_count"]
        .sum()
        .unstack(fill_value=0)
    )


def degree_distribution(conn_df: pd.DataFrame, kind: str = "out") -> pd.Series:
    col = "pre_root_id" if kind == "out" else "post_root_id"
    degrees = conn_df.groupby(col).size()
    return degrees.value_counts().sort_index()


def connections_for_neuron(
    conn_df: pd.DataFrame,
    root_id: int,
    direction: str = "both",
) -> pd.DataFrame:
    if direction == "pre":
        return conn_df[conn_df["pre_root_id"] == root_id]
    if direction == "post":
        return conn_df[conn_df["post_root_id"] == root_id]
    return conn_df[
        (conn_df["pre_root_id"] == root_id) | (conn_df["post_root_id"] == root_id)
    ]


# =============================================================================
# Generic Plotting Helpers (same style/signature)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.style.use(PLOT_STYLE)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if series is None or len(series) == 0:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", transform=ax.transAxes, fontsize=12)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.style.use(PLOT_STYLE)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    series.dropna().plot(kind="hist", bins=bins, ax=ax, color=color, edgecolor="white", logy=log_scale)
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
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if df is None or df.size == 0 or df.shape[0] == 0 or df.shape[1] == 0:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        return ax
    sns.heatmap(df, ax=ax, cmap=cmap, fmt=fmt, annot=(df.shape[0] <= 20), linewidths=0.3, cbar_kws={"shrink": 0.8})
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
    ax.set_title(f"{label} Distribution" + (" (log-log)" if log_log else ""), fontsize=13, fontweight="bold")
    plt.tight_layout()
    return ax
