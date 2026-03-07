"""Structured descriptive hypothesis exploration.

This is explicitly a descriptive analysis -- we compute effect sizes (correlation
coefficients) to characterize relationships, not to perform null hypothesis
significance testing. P-values are not the focus; the goal is to describe the
direction, magnitude, and consistency of relationships across species and thresholds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl
from scipy import stats

from gene_conservation_network.data.species import Species


@dataclass
class HypothesisResult:
    """Result of a descriptive hypothesis analysis."""

    name: str
    species: str
    target_species: str | None  # For cross-species comparisons
    threshold: float  # Association threshold used
    statistic_name: str  # e.g., "spearman_r"
    statistic_value: float
    n_genes: int
    summary: str  # Human-readable summary


def describe_hub_conservation(
    merged_features: pl.DataFrame,
    species: Species,
    target_species: Species,
    threshold: float,
    hub_metric: str = "degree",
    conservation_metric: str = "max_wormhole_score",
) -> HypothesisResult:
    """Hypothesis 1: Hub genes are more conserved.

    Describes the correlation between a network centrality measure and ortholog confidence.
    """
    valid = merged_features.select(hub_metric, conservation_metric).drop_nulls()
    n = len(valid)

    if n < 3:
        return HypothesisResult(
            name="hub_conservation",
            species=species.common_name,
            target_species=target_species.common_name,
            threshold=threshold,
            statistic_name="spearman_r",
            statistic_value=float("nan"),
            n_genes=n,
            summary=f"Insufficient data (n={n}) to compute correlation.",
        )

    x = valid[hub_metric].to_numpy()
    y = valid[conservation_metric].to_numpy()
    result = stats.spearmanr(x, y)
    r = float(result.statistic)

    direction = "positively" if r > 0 else "negatively"
    strength = "strongly" if abs(r) > 0.5 else "moderately" if abs(r) > 0.2 else "weakly"

    summary = (
        f"{species.common_name}->{target_species.common_name} (threshold={threshold}): "
        f"{hub_metric} is {strength} {direction} correlated with {conservation_metric} "
        f"(Spearman r={r:.3f}, n={n:,})"
    )

    return HypothesisResult(
        name="hub_conservation",
        species=species.common_name,
        target_species=target_species.common_name,
        threshold=threshold,
        statistic_name="spearman_r",
        statistic_value=r,
        n_genes=n,
        summary=summary,
    )


def describe_hub_ambiguity(
    merged_features: pl.DataFrame,
    species: Species,
    target_species: Species,
    threshold: float,
    hub_metric: str = "degree",
    ambiguity_metric: str = "ortholog_count",
) -> HypothesisResult:
    """Hypothesis 2: Hub genes have more ambiguous orthologs.

    Describes the correlation between a network centrality measure and ortholog count/entropy.
    """
    valid = merged_features.select(hub_metric, ambiguity_metric).drop_nulls()
    n = len(valid)

    if n < 3:
        return HypothesisResult(
            name="hub_ambiguity",
            species=species.common_name,
            target_species=target_species.common_name,
            threshold=threshold,
            statistic_name="spearman_r",
            statistic_value=float("nan"),
            n_genes=n,
            summary=f"Insufficient data (n={n}) to compute correlation.",
        )

    x = valid[hub_metric].to_numpy()
    y = valid[ambiguity_metric].to_numpy()
    result = stats.spearmanr(x, y)
    r = float(result.statistic)

    direction = "positively" if r > 0 else "negatively"
    strength = "strongly" if abs(r) > 0.5 else "moderately" if abs(r) > 0.2 else "weakly"

    summary = (
        f"{species.common_name}->{target_species.common_name} (threshold={threshold}): "
        f"{hub_metric} is {strength} {direction} correlated with {ambiguity_metric} "
        f"(Spearman r={r:.3f}, n={n:,})"
    )

    return HypothesisResult(
        name="hub_ambiguity",
        species=species.common_name,
        target_species=target_species.common_name,
        threshold=threshold,
        statistic_name="spearman_r",
        statistic_value=r,
        n_genes=n,
        summary=summary,
    )


def describe_all_hypotheses(
    merged_features: pl.DataFrame,
    species: Species,
    target_species: Species,
    threshold: float,
) -> list[HypothesisResult]:
    """Run all descriptive analyses for a species pair at a given threshold.

    Explores multiple combinations of hub metrics x conservation/ambiguity metrics.
    """
    hub_metrics = ["degree", "weighted_degree", "betweenness", "eigenvector", "pagerank"]
    conservation_metrics = ["max_wormhole_score", "mean_wormhole_score", "max_votes"]
    ambiguity_metrics = ["ortholog_count", "vote_entropy"]

    results = []

    # Available hub metrics (only include those present in the DataFrame)
    available_hub = [m for m in hub_metrics if m in merged_features.columns]
    available_conservation = [m for m in conservation_metrics if m in merged_features.columns]
    available_ambiguity = [m for m in ambiguity_metrics if m in merged_features.columns]

    for hub in available_hub:
        for cons in available_conservation:
            results.append(
                describe_hub_conservation(
                    merged_features,
                    species,
                    target_species,
                    threshold,
                    hub_metric=hub,
                    conservation_metric=cons,
                )
            )
        for amb in available_ambiguity:
            results.append(
                describe_hub_ambiguity(
                    merged_features,
                    species,
                    target_species,
                    threshold,
                    hub_metric=hub,
                    ambiguity_metric=amb,
                )
            )

    return results


def describe_threshold_sensitivity(
    species: Species,
    target_species: Species,
    thresholds: list[float],
    compute_fn: Callable[[float], pl.DataFrame],
) -> pl.DataFrame:
    """Run a hypothesis across multiple thresholds to assess robustness.

    Args:
        species: Query species.
        target_species: Target species.
        thresholds: List of association thresholds to test.
        compute_fn: Function that takes a threshold and returns merged features DataFrame.

    Returns: DataFrame with columns [threshold, statistic_name, statistic_value, n_genes]
    """
    rows = []
    for t in thresholds:
        merged = compute_fn(t)
        result = describe_hub_conservation(merged, species, target_species, t)
        rows.append(
            {
                "threshold": t,
                "statistic_name": result.statistic_name,
                "statistic_value": result.statistic_value,
                "n_genes": result.n_genes,
            }
        )

    return pl.DataFrame(rows)
