"""Ortholog-derived per-gene features.

Compute per-gene features from WORMHOLE ortholog data that capture
"conservation" (how confident are ortholog calls?) and "ambiguity"
(how many ortholog candidates does a gene have?).
"""

from __future__ import annotations

import math

import polars as pl


def compute_ortholog_count(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Number of ortholog targets per query gene.

    Higher count = more ortholog candidates = potentially more ambiguity.

    Returns: DataFrame with columns [gene_id, ortholog_count]
    """
    return (
        orthologs.group_by(gene_col)
        .agg(pl.len().alias("ortholog_count"))
        .rename({gene_col: "gene_id"})
    )


def compute_rbh_count(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Number of reciprocal best hit targets per query gene.

    Typically 0 or 1 for most genes. >1 indicates duplication events.

    Returns: DataFrame with columns [gene_id, rbh_count]
    """
    return (
        orthologs.group_by(gene_col)
        .agg(pl.col("rbh").sum().alias("rbh_count"))
        .rename({gene_col: "gene_id"})
    )


def compute_max_ortholog_score(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Maximum WORMHOLE score across all ortholog targets.

    Captures the strength of the best ortholog relationship.

    Returns: DataFrame with columns [gene_id, max_wormhole_score]
    """
    return (
        orthologs.group_by(gene_col)
        .agg(pl.col("wormhole_score").max().alias("max_wormhole_score"))
        .rename({gene_col: "gene_id"})
    )


def compute_mean_ortholog_score(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Mean WORMHOLE score across all ortholog targets.

    Returns: DataFrame with columns [gene_id, mean_wormhole_score]
    """
    return (
        orthologs.group_by(gene_col)
        .agg(pl.col("wormhole_score").mean().alias("mean_wormhole_score"))
        .rename({gene_col: "gene_id"})
    )


def compute_max_votes(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Maximum vote count across all ortholog targets.

    Higher votes = more algorithm agreement = stronger evidence.

    Returns: DataFrame with columns [gene_id, max_votes]
    """
    return (
        orthologs.group_by(gene_col)
        .agg(pl.col("votes").max().alias("max_votes"))
        .rename({gene_col: "gene_id"})
    )


def _shannon_entropy(values: list[int]) -> float:
    """Compute Shannon entropy of a list of integer counts."""
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log2(p) for p in probs)


def compute_vote_entropy(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Shannon entropy of the vote distribution across ortholog targets.

    High entropy = orthologs have diverse support patterns = more ambiguity.

    Returns: DataFrame with columns [gene_id, vote_entropy]
    """
    # Collect vote lists per gene, then compute entropy
    grouped = orthologs.group_by(gene_col).agg(pl.col("votes").alias("vote_list"))

    gene_ids = []
    entropies = []
    for row in grouped.iter_rows(named=True):
        gene_ids.append(row[gene_col])
        entropies.append(_shannon_entropy(row["vote_list"]))

    return pl.DataFrame(
        {"gene_id": gene_ids, "vote_entropy": entropies},
        schema={"gene_id": pl.Utf8, "vote_entropy": pl.Float64},
    )


def compute_has_rbh(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Binary: does this gene have at least one RBH?

    Returns: DataFrame with columns [gene_id, has_rbh]
    """
    return (
        orthologs.group_by(gene_col)
        .agg((pl.col("rbh").sum() > 0).cast(pl.Int8).alias("has_rbh"))
        .rename({gene_col: "gene_id"})
    )


def compute_all_ortholog_features(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Combine all ortholog features into a single DataFrame.

    Returns: DataFrame with columns [gene_id, ortholog_count, rbh_count,
             max_wormhole_score, mean_wormhole_score, max_votes,
             vote_entropy, has_rbh]
    """
    count = compute_ortholog_count(orthologs, gene_col)
    rbh = compute_rbh_count(orthologs, gene_col)
    max_score = compute_max_ortholog_score(orthologs, gene_col)
    mean_score = compute_mean_ortholog_score(orthologs, gene_col)
    max_v = compute_max_votes(orthologs, gene_col)
    entropy = compute_vote_entropy(orthologs, gene_col)
    has_r = compute_has_rbh(orthologs, gene_col)

    # Join all on gene_id
    result = count
    for df in [rbh, max_score, mean_score, max_v, entropy, has_r]:
        result = result.join(df, on="gene_id", how="left")

    return result
