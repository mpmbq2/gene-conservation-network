"""Standardized plotting functions for exploring analysis results.

All functions accept an optional matplotlib Axes for composing multi-panel figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from gene_conservation_network.analysis.hypotheses import HypothesisResult


def plot_feature_scatter(
    merged: pl.DataFrame,
    x_col: str,
    y_col: str,
    species_label: str = "",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Scatter plot of two features with regression line and correlation annotation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    valid = merged.select(x_col, y_col).drop_nulls()
    x = valid[x_col].to_numpy()
    y = valid[y_col].to_numpy()

    ax.scatter(x, y, alpha=0.3, s=5, **kwargs)

    # Regression line
    if len(x) >= 3:
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=1.5)

        # Spearman correlation annotation
        spearman_r = stats.spearmanr(x, y).statistic
        ax.annotate(
            f"Spearman r = {spearman_r:.3f}\nn = {len(x):,}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            verticalalignment="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if species_label:
        ax.set_title(species_label)

    return ax


def plot_correlation_heatmap(
    correlations: pl.DataFrame,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of feature-feature correlations.

    Args:
        correlations: DataFrame from compute_pairwise_correlations() with columns
                      [network_feature, ortholog_feature, correlation].
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Pivot to matrix form
    net_features = correlations["network_feature"].unique().sort().to_list()
    orth_features = correlations["ortholog_feature"].unique().sort().to_list()

    matrix = np.full((len(net_features), len(orth_features)), np.nan)
    for row in correlations.iter_rows(named=True):
        i = net_features.index(row["network_feature"])
        j = orth_features.index(row["ortholog_feature"])
        matrix[i, j] = row["correlation"]

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(orth_features)))
    ax.set_xticklabels(orth_features, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(net_features)))
    ax.set_yticklabels(net_features, fontsize=8)

    # Annotate cells with correlation values
    for i in range(len(net_features)):
        for j in range(len(orth_features)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    plt.colorbar(im, ax=ax, label="Spearman r")
    if title:
        ax.set_title(title)

    return ax


def plot_species_comparison(
    results: list[HypothesisResult],
    metric_name: str = "",
) -> plt.Figure:
    """Forest-plot style comparison of a hypothesis result across species pairs."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(results) * 0.4)))

    labels = []
    values = []
    for r in results:
        label = f"{r.species}->{r.target_species}" if r.target_species else r.species
        labels.append(f"{label} (n={r.n_genes:,})")
        values.append(r.statistic_value)

    y_pos = range(len(results))
    ax.barh(y_pos, values, align="center", height=0.6, color="steelblue", alpha=0.7)
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"{results[0].statistic_name}" if results else "Statistic")
    ax.set_title(metric_name or "Species comparison")
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_degree_distribution(
    features: pl.DataFrame,
    species_label: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Log-log degree distribution plot."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    degrees = features["degree"].to_numpy()
    # Count frequency of each degree
    unique, counts = np.unique(degrees, return_counts=True)

    ax.scatter(unique, counts, s=10, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (log)")
    ax.set_ylabel("Frequency (log)")
    if species_label:
        ax.set_title(f"Degree distribution: {species_label}")

    return ax
