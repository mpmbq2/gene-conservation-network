"""Tests for gene_conservation_network.features.ortholog_features.

Uses synthetic ortholog data with hand-calculated expected values.
"""

import math

import polars as pl
import pytest

from gene_conservation_network.features.ortholog_features import (
    compute_all_ortholog_features,
    compute_has_rbh,
    compute_max_ortholog_score,
    compute_max_votes,
    compute_mean_ortholog_score,
    compute_ortholog_count,
    compute_rbh_count,
    compute_vote_entropy,
)


@pytest.fixture
def synthetic_orthologs():
    """Synthetic ortholog data:
    - Gene A: 5 orthologs (1 RBH), scores [0.9, 0.7, 0.5, 0.3, 0.1], votes [13, 8, 5, 3, 1]
    - Gene B: 1 ortholog (1 RBH), score [0.95], votes [15]
    - Gene C: 3 orthologs (0 RBH), scores [0.4, 0.3, 0.2], votes [4, 4, 4]
    """
    return pl.DataFrame(
        {
            "query_id": ["A", "A", "A", "A", "A", "B", "C", "C", "C"],
            "target_id": ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"],
            "wormhole_score": [0.9, 0.7, 0.5, 0.3, 0.1, 0.95, 0.4, 0.3, 0.2],
            "votes": [13, 8, 5, 3, 1, 15, 4, 4, 4],
            "rbh": [1, 0, 0, 0, 0, 1, 0, 0, 0],
        }
    )


class TestOrthologCount:
    def test_counts(self, synthetic_orthologs):
        result = compute_ortholog_count(synthetic_orthologs)
        a = result.filter(pl.col("gene_id") == "A")["ortholog_count"].item()
        b = result.filter(pl.col("gene_id") == "B")["ortholog_count"].item()
        c = result.filter(pl.col("gene_id") == "C")["ortholog_count"].item()
        assert a == 5
        assert b == 1
        assert c == 3


class TestRBHCount:
    def test_rbh_counts(self, synthetic_orthologs):
        result = compute_rbh_count(synthetic_orthologs)
        a = result.filter(pl.col("gene_id") == "A")["rbh_count"].item()
        b = result.filter(pl.col("gene_id") == "B")["rbh_count"].item()
        c = result.filter(pl.col("gene_id") == "C")["rbh_count"].item()
        assert a == 1
        assert b == 1
        assert c == 0


class TestMaxOrthologScore:
    def test_max_scores(self, synthetic_orthologs):
        result = compute_max_ortholog_score(synthetic_orthologs)
        a = result.filter(pl.col("gene_id") == "A")["max_wormhole_score"].item()
        b = result.filter(pl.col("gene_id") == "B")["max_wormhole_score"].item()
        c = result.filter(pl.col("gene_id") == "C")["max_wormhole_score"].item()
        assert a == pytest.approx(0.9)
        assert b == pytest.approx(0.95)
        assert c == pytest.approx(0.4)


class TestMeanOrthologScore:
    def test_mean_scores(self, synthetic_orthologs):
        result = compute_mean_ortholog_score(synthetic_orthologs)
        a = result.filter(pl.col("gene_id") == "A")["mean_wormhole_score"].item()
        assert a == pytest.approx(0.5)  # (0.9+0.7+0.5+0.3+0.1)/5


class TestMaxVotes:
    def test_max_votes(self, synthetic_orthologs):
        result = compute_max_votes(synthetic_orthologs)
        a = result.filter(pl.col("gene_id") == "A")["max_votes"].item()
        b = result.filter(pl.col("gene_id") == "B")["max_votes"].item()
        assert a == 13
        assert b == 15


class TestVoteEntropy:
    def test_single_ortholog_zero_entropy(self, synthetic_orthologs):
        result = compute_vote_entropy(synthetic_orthologs)
        b = result.filter(pl.col("gene_id") == "B")["vote_entropy"].item()
        # Only one ortholog -> entropy = 0 (single element, -1*log2(1) = 0)
        assert b == pytest.approx(0.0)

    def test_equal_votes_max_entropy(self, synthetic_orthologs):
        result = compute_vote_entropy(synthetic_orthologs)
        c = result.filter(pl.col("gene_id") == "C")["vote_entropy"].item()
        # Gene C has 3 orthologs all with votes=4 -> max entropy for 3 items
        expected = -3 * (1 / 3) * math.log2(1 / 3)  # log2(3)
        assert c == pytest.approx(expected)

    def test_unequal_votes_intermediate_entropy(self, synthetic_orthologs):
        result = compute_vote_entropy(synthetic_orthologs)
        a = result.filter(pl.col("gene_id") == "A")["vote_entropy"].item()
        # Gene A has votes [13, 8, 5, 3, 1] -> entropy should be > 0 but < log2(5)
        assert 0 < a < math.log2(5)


class TestHasRBH:
    def test_has_rbh(self, synthetic_orthologs):
        result = compute_has_rbh(synthetic_orthologs)
        a = result.filter(pl.col("gene_id") == "A")["has_rbh"].item()
        b = result.filter(pl.col("gene_id") == "B")["has_rbh"].item()
        c = result.filter(pl.col("gene_id") == "C")["has_rbh"].item()
        assert a == 1
        assert b == 1
        assert c == 0


class TestComputeAll:
    def test_all_columns(self, synthetic_orthologs):
        result = compute_all_ortholog_features(synthetic_orthologs)
        expected_cols = {
            "gene_id", "ortholog_count", "rbh_count",
            "max_wormhole_score", "mean_wormhole_score",
            "max_votes", "vote_entropy", "has_rbh",
        }
        assert set(result.columns) == expected_cols

    def test_all_genes_present(self, synthetic_orthologs):
        result = compute_all_ortholog_features(synthetic_orthologs)
        assert len(result) == 3  # A, B, C
