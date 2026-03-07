"""Tests for gene_conservation_network.analysis.hypotheses."""

import polars as pl
import pytest

from gene_conservation_network.analysis.hypotheses import (
    HypothesisResult,
    describe_all_hypotheses,
    describe_hub_ambiguity,
    describe_hub_conservation,
    describe_threshold_sensitivity,
)
from gene_conservation_network.data.species import FLY, HUMAN


@pytest.fixture
def synthetic_merged():
    """Synthetic merged features with known relationships."""
    return pl.DataFrame({
        "gene_id": list(range(100)),
        "degree": list(range(100)),
        "weighted_degree": [float(x) * 1.5 for x in range(100)],
        "betweenness": [float(x) / 100 for x in range(100)],
        "eigenvector": [float(x) / 100 for x in range(100)],
        "pagerank": [1.0 / 100] * 100,
        # Conservation: positively correlated with degree
        "max_wormhole_score": [0.01 * x + 0.1 for x in range(100)],
        "mean_wormhole_score": [0.005 * x + 0.05 for x in range(100)],
        "max_votes": list(range(100)),
        # Ambiguity: also positively correlated with degree
        "ortholog_count": [1 + x // 10 for x in range(100)],
        "vote_entropy": [0.01 * x for x in range(100)],
    })


class TestDescribeHubConservation:
    def test_result_structure(self, synthetic_merged):
        result = describe_hub_conservation(
            synthetic_merged, FLY, HUMAN, threshold=5.0,
        )
        assert isinstance(result, HypothesisResult)
        assert result.name == "hub_conservation"
        assert result.species == "fly"
        assert result.target_species == "human"
        assert result.threshold == 5.0
        assert result.statistic_name == "spearman_r"
        assert result.n_genes == 100

    def test_positive_correlation(self, synthetic_merged):
        result = describe_hub_conservation(
            synthetic_merged, FLY, HUMAN, threshold=5.0,
        )
        # degree and max_wormhole_score are positively correlated in our synthetic data
        assert result.statistic_value > 0

    def test_summary_readable(self, synthetic_merged):
        result = describe_hub_conservation(
            synthetic_merged, FLY, HUMAN, threshold=5.0,
        )
        assert "fly" in result.summary
        assert "human" in result.summary
        assert "r=" in result.summary

    def test_insufficient_data(self):
        tiny = pl.DataFrame({
            "degree": [1],
            "max_wormhole_score": [0.5],
        })
        result = describe_hub_conservation(tiny, FLY, HUMAN, threshold=5.0)
        assert result.n_genes == 1
        assert "Insufficient" in result.summary


class TestDescribeHubAmbiguity:
    def test_result_structure(self, synthetic_merged):
        result = describe_hub_ambiguity(
            synthetic_merged, FLY, HUMAN, threshold=5.0,
        )
        assert result.name == "hub_ambiguity"
        assert isinstance(result.statistic_value, float)

    def test_positive_correlation(self, synthetic_merged):
        result = describe_hub_ambiguity(
            synthetic_merged, FLY, HUMAN, threshold=5.0,
        )
        # degree and ortholog_count are positively correlated
        assert result.statistic_value > 0


class TestDescribeAllHypotheses:
    def test_returns_multiple_results(self, synthetic_merged):
        results = describe_all_hypotheses(synthetic_merged, FLY, HUMAN, threshold=5.0)
        assert len(results) > 0
        assert all(isinstance(r, HypothesisResult) for r in results)

    def test_covers_both_hypotheses(self, synthetic_merged):
        results = describe_all_hypotheses(synthetic_merged, FLY, HUMAN, threshold=5.0)
        names = {r.name for r in results}
        assert "hub_conservation" in names
        assert "hub_ambiguity" in names


class TestThresholdSensitivity:
    def test_produces_one_row_per_threshold(self, synthetic_merged):
        thresholds = [3.0, 4.0, 5.0]
        result = describe_threshold_sensitivity(
            FLY, HUMAN, thresholds,
            compute_fn=lambda t: synthetic_merged,  # Same data regardless of threshold
        )
        assert len(result) == 3
        assert set(result.columns) == {"threshold", "statistic_name", "statistic_value", "n_genes"}
