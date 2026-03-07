"""Tests for gene_conservation_network.analysis.correlation."""

import polars as pl
import pytest

from gene_conservation_network.analysis.correlation import (
    compute_correlation_matrix,
    compute_pairwise_correlations,
    merge_features,
)


class TestComputePairwiseCorrelations:
    def test_perfect_positive_correlation(self):
        """x = [1,2,3,4,5], y = [2,4,6,8,10] -> r = 1.0"""
        df = pl.DataFrame({
            "degree": [1, 2, 3, 4, 5],
            "max_wormhole_score": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        result = compute_pairwise_correlations(
            df,
            network_cols=["degree"],
            ortholog_cols=["max_wormhole_score"],
            method="spearman",
        )
        assert len(result) == 1
        r = result["correlation"].item()
        assert r == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        df = pl.DataFrame({
            "degree": [1, 2, 3, 4, 5],
            "max_wormhole_score": [10.0, 8.0, 6.0, 4.0, 2.0],
        })
        result = compute_pairwise_correlations(
            df,
            network_cols=["degree"],
            ortholog_cols=["max_wormhole_score"],
            method="spearman",
        )
        r = result["correlation"].item()
        assert r == pytest.approx(-1.0)

    def test_near_zero_correlation(self):
        """Roughly uncorrelated data."""
        df = pl.DataFrame({
            "degree": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "ortholog_count": [5, 3, 7, 1, 9, 2, 8, 4, 6, 10],
        })
        result = compute_pairwise_correlations(
            df,
            network_cols=["degree"],
            ortholog_cols=["ortholog_count"],
            method="spearman",
        )
        r = result["correlation"].item()
        # Should be close to zero but not necessarily exact
        assert abs(r) < 0.5

    def test_multiple_features(self):
        df = pl.DataFrame({
            "degree": [1, 2, 3, 4, 5],
            "betweenness": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_wormhole_score": [2.0, 4.0, 6.0, 8.0, 10.0],
            "ortholog_count": [1, 1, 2, 2, 3],
        })
        result = compute_pairwise_correlations(
            df,
            network_cols=["degree", "betweenness"],
            ortholog_cols=["max_wormhole_score", "ortholog_count"],
            method="spearman",
        )
        # Should have 2 x 2 = 4 correlation pairs
        assert len(result) == 4
        assert set(result.columns) == {"network_feature", "ortholog_feature", "correlation", "n"}

    def test_insufficient_data(self):
        df = pl.DataFrame({
            "degree": [1, 2],
            "score": [3.0, 4.0],
        })
        result = compute_pairwise_correlations(
            df,
            network_cols=["degree"],
            ortholog_cols=["score"],
        )
        # With only 2 data points, correlation should be NaN
        assert result["correlation"].item() != result["correlation"].item()  # NaN != NaN


class TestComputeCorrelationMatrix:
    def test_diagonal_is_one(self):
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [1, 3, 5, 7, 9],
        })
        result = compute_correlation_matrix(df, feature_cols=["a", "b", "c"])
        # Diagonal should all be 1.0
        for i, col in enumerate(["a", "b", "c"]):
            assert result[col][i] == pytest.approx(1.0)

    def test_symmetric(self):
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })
        result = compute_correlation_matrix(df, feature_cols=["a", "b"])
        ab = result["b"][0]
        ba = result["a"][1]
        assert ab == pytest.approx(ba)


class TestMergeFeatures:
    """Test merge with a mock resolver-like setup."""

    def test_merge_with_synthetic_data(self):
        """Use a mock approach: test the join logic without real files."""
        # Simulate network features keyed by NCBI ID
        network_features = pl.DataFrame({
            "gene_id": [100, 200, 300],
            "degree": [10, 20, 30],
        }, schema={"gene_id": pl.Int64, "degree": pl.Int64})

        # Simulate ortholog features keyed by canonical ID
        ortholog_features = pl.DataFrame({
            "gene_id": ["GENE_A", "GENE_B", "GENE_C"],
            "ortholog_count": [5, 3, 7],
        })

        # Create a mock resolver with the mapping
        class MockResolver:
            _canonical_to_ncbi = {
                "GENE_A": 100,
                "GENE_B": 200,
                "GENE_C": 300,
            }

        merged = merge_features(network_features, ortholog_features, MockResolver())
        assert len(merged) == 3
        assert "degree" in merged.columns
        assert "ortholog_count" in merged.columns
