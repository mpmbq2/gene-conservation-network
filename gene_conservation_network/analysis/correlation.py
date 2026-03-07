"""Feature-feature correlation analysis.

Join network and ortholog features and compute descriptive correlations.
Focus is on effect sizes (correlation coefficients), not significance testing.
"""

from __future__ import annotations

from loguru import logger
import polars as pl
from scipy import stats

from gene_conservation_network.data.gene_ids import GeneIDResolver


def merge_features(
    network_features: pl.DataFrame,
    ortholog_features: pl.DataFrame,
    id_resolver: GeneIDResolver,
) -> pl.DataFrame:
    """Join network features (keyed by NCBI GeneID) with ortholog features (keyed by canonical ID).

    Uses the GeneIDResolver to bridge the ID gap. The network features have gene_id
    as NCBI GeneIDs (int), while ortholog features have gene_id as canonical IDs (str).

    Returns: DataFrame with all network + ortholog feature columns, keyed by gene_id (NCBI).
    """
    # Build a mapping frame: canonical_id -> ncbi_id
    mapping = pl.DataFrame(
        {
            "canonical_id": list(id_resolver._canonical_to_ncbi.keys()),
            "ncbi_id": list(id_resolver._canonical_to_ncbi.values()),
        },
        schema={"canonical_id": pl.Utf8, "ncbi_id": pl.Int64},
    )

    # Add NCBI IDs to ortholog features (which are keyed by canonical IDs)
    ortholog_with_ncbi = ortholog_features.join(
        mapping.rename({"canonical_id": "gene_id"}),
        on="gene_id",
        how="inner",
    )

    original_ortholog_count = len(ortholog_features)
    mapped_count = len(ortholog_with_ncbi)
    if mapped_count < original_ortholog_count:
        logger.warning(
            f"Mapped {mapped_count}/{original_ortholog_count} ortholog feature rows "
            f"to NCBI IDs ({100 * mapped_count / max(original_ortholog_count, 1):.1f}%)"
        )

    # Drop the canonical gene_id and rename ncbi_id to gene_id for the join
    ortholog_cols = [c for c in ortholog_with_ncbi.columns if c not in ("gene_id",)]
    ortholog_for_join = ortholog_with_ncbi.select(ortholog_cols).rename({"ncbi_id": "gene_id"})

    # Join on gene_id (both now NCBI int)
    merged = network_features.join(ortholog_for_join, on="gene_id", how="inner")

    logger.info(
        f"Merged features: {len(merged)} genes "
        f"(from {len(network_features)} network, {original_ortholog_count} ortholog)"
    )

    return merged


def compute_pairwise_correlations(
    merged: pl.DataFrame,
    network_cols: list[str],
    ortholog_cols: list[str],
    method: str = "spearman",
) -> pl.DataFrame:
    """Compute correlation between each network feature and each ortholog feature.

    Args:
        merged: DataFrame with both network and ortholog feature columns.
        network_cols: Column names for network features (e.g., ["degree", "betweenness"]).
        ortholog_cols: Column names for ortholog features (e.g., ["ortholog_count", "max_wormhole_score"]).
        method: Correlation method, either "spearman" or "pearson".

    Returns: DataFrame with columns [network_feature, ortholog_feature, correlation, n]
    """
    corr_fn = stats.spearmanr if method == "spearman" else stats.pearsonr

    results = []
    for net_col in network_cols:
        for orth_col in ortholog_cols:
            # Drop rows with NaN in either column
            valid = merged.select(net_col, orth_col).drop_nulls()
            n = len(valid)
            if n < 3:
                results.append(
                    {
                        "network_feature": net_col,
                        "ortholog_feature": orth_col,
                        "correlation": float("nan"),
                        "n": n,
                    }
                )
                continue

            x = valid[net_col].to_numpy()
            y = valid[orth_col].to_numpy()
            result = corr_fn(x, y)
            corr_value = result.statistic if hasattr(result, "statistic") else result[0]

            results.append(
                {
                    "network_feature": net_col,
                    "ortholog_feature": orth_col,
                    "correlation": float(corr_value),
                    "n": n,
                }
            )

    return pl.DataFrame(results)


def compute_correlation_matrix(
    merged: pl.DataFrame,
    feature_cols: list[str],
    method: str = "spearman",
) -> pl.DataFrame:
    """Full correlation matrix across all features.

    Args:
        merged: DataFrame with feature columns.
        feature_cols: Column names to include in the matrix.
        method: Correlation method, either "spearman" or "pearson".

    Returns: DataFrame with columns [feature] + feature_cols, forming a square matrix.
    """
    corr_fn = stats.spearmanr if method == "spearman" else stats.pearsonr

    matrix: list[dict[str, object]] = []
    for i, col_i in enumerate(feature_cols):
        row: dict[str, object] = {"feature": col_i}
        for j, col_j in enumerate(feature_cols):
            if i == j:
                row[col_j] = 1.0
            else:
                valid = merged.select(col_i, col_j).drop_nulls()
                if len(valid) < 3:
                    row[col_j] = float("nan")
                else:
                    x = valid[col_i].to_numpy()
                    y = valid[col_j].to_numpy()
                    result = corr_fn(x, y)
                    corr_value = result.statistic if hasattr(result, "statistic") else result[0]
                    row[col_j] = float(corr_value)
        matrix.append(row)

    return pl.DataFrame(matrix)
