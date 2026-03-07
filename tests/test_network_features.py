"""Tests for gene_conservation_network.features.network.

Tests use well-known graph topologies (star, complete, path) with
hand-calculated expected values.
"""

import polars as pl
import pytest

from gene_conservation_network.features.network import (
    build_graph,
    compute_all_network_features,
    compute_betweenness_centrality,
    compute_closeness_centrality,
    compute_clustering_coefficient,
    compute_degree,
    compute_eigenvector_centrality,
    compute_pagerank,
    compute_weighted_degree,
)


def _make_edges(edge_list: list[tuple[int, int, float]]) -> pl.DataFrame:
    """Helper to create an edge DataFrame from a list of (g1, g2, weight) tuples."""
    return pl.DataFrame(
        edge_list,
        schema={"gene_id_1": pl.Int64, "gene_id_2": pl.Int64, "association": pl.Float64},
        orient="row",
    )


class TestStarGraph:
    """Star graph: center node 0 connected to 1, 2, 3, 4."""

    @pytest.fixture
    def star_edges(self):
        return _make_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
        ])

    @pytest.fixture
    def star_graph(self, star_edges):
        return build_graph(star_edges)

    def test_build_graph(self, star_graph):
        graph, node_map = star_graph
        assert graph.num_nodes() == 5
        assert graph.num_edges() == 4

    def test_degree(self, star_graph):
        graph, node_map = star_graph
        df = compute_degree(graph, node_map)
        center = df.filter(pl.col("gene_id") == 0)
        assert center["degree"].item() == 4
        leaves = df.filter(pl.col("gene_id") != 0)
        assert (leaves["degree"] == 1).all()

    def test_weighted_degree(self, star_graph):
        graph, node_map = star_graph
        df = compute_weighted_degree(graph, node_map)
        center = df.filter(pl.col("gene_id") == 0)
        assert center["weighted_degree"].item() == pytest.approx(4.0)

    def test_betweenness(self, star_graph):
        graph, node_map = star_graph
        df = compute_betweenness_centrality(graph, node_map)
        center = df.filter(pl.col("gene_id") == 0)
        # Center has highest betweenness (normalized = 1.0 for star)
        assert center["betweenness"].item() == pytest.approx(1.0)
        leaves = df.filter(pl.col("gene_id") != 0)
        assert (leaves["betweenness"] == 0.0).all()

    def test_closeness(self, star_graph):
        graph, node_map = star_graph
        df = compute_closeness_centrality(graph, node_map)
        center = df.filter(pl.col("gene_id") == 0)
        # Center has closeness = 1.0 (distance 1 to all others)
        assert center["closeness"].item() == pytest.approx(1.0)

    def test_clustering_coefficient(self, star_graph):
        graph, node_map = star_graph
        df = compute_clustering_coefficient(graph, node_map)
        # No triangles in a star graph -> all clustering coefficients = 0
        assert (df["clustering_coeff"] == 0.0).all()

    def test_eigenvector(self, star_graph):
        graph, node_map = star_graph
        df = compute_eigenvector_centrality(graph, node_map)
        center = df.filter(pl.col("gene_id") == 0)
        # Center should have highest eigenvector centrality
        max_eigenvector = df["eigenvector"].max()
        assert center["eigenvector"].item() == pytest.approx(max_eigenvector)

    def test_pagerank(self, star_graph):
        graph, node_map = star_graph
        df = compute_pagerank(graph, node_map)
        center = df.filter(pl.col("gene_id") == 0)
        # Center should have highest pagerank
        max_pr = df["pagerank"].max()
        assert center["pagerank"].item() == pytest.approx(max_pr)


class TestCompleteGraph:
    """Complete graph K4: all 4 nodes connected to each other."""

    @pytest.fixture
    def complete_edges(self):
        return _make_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
        ])

    @pytest.fixture
    def complete_graph(self, complete_edges):
        return build_graph(complete_edges)

    def test_degree(self, complete_graph):
        graph, node_map = complete_graph
        df = compute_degree(graph, node_map)
        # All nodes have degree 3
        assert (df["degree"] == 3).all()

    def test_betweenness_zero(self, complete_graph):
        graph, node_map = complete_graph
        df = compute_betweenness_centrality(graph, node_map)
        # All betweenness = 0 in complete graph
        assert (df["betweenness"] == 0.0).all()

    def test_clustering_one(self, complete_graph):
        graph, node_map = complete_graph
        df = compute_clustering_coefficient(graph, node_map)
        # Clustering coefficient = 1.0 for all nodes in complete graph
        for val in df["clustering_coeff"].to_list():
            assert val == pytest.approx(1.0)


class TestPathGraph:
    """Path graph: 0 -- 1 -- 2 -- 3."""

    @pytest.fixture
    def path_edges(self):
        return _make_edges([
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
        ])

    @pytest.fixture
    def path_graph(self, path_edges):
        return build_graph(path_edges)

    def test_degree(self, path_graph):
        graph, node_map = path_graph
        df = compute_degree(graph, node_map)
        # End nodes have degree 1, middle nodes have degree 2
        end_0 = df.filter(pl.col("gene_id") == 0)["degree"].item()
        end_3 = df.filter(pl.col("gene_id") == 3)["degree"].item()
        mid_1 = df.filter(pl.col("gene_id") == 1)["degree"].item()
        mid_2 = df.filter(pl.col("gene_id") == 2)["degree"].item()
        assert end_0 == 1
        assert end_3 == 1
        assert mid_1 == 2
        assert mid_2 == 2

    def test_closeness(self, path_graph):
        graph, node_map = path_graph
        df = compute_closeness_centrality(graph, node_map)
        # Middle nodes should have higher closeness than endpoints
        mid_1 = df.filter(pl.col("gene_id") == 1)["closeness"].item()
        end_0 = df.filter(pl.col("gene_id") == 0)["closeness"].item()
        assert mid_1 > end_0


class TestComputeAll:
    def test_all_features_schema(self):
        edges = _make_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (2, 3, 2.0),
        ])
        result = compute_all_network_features(edges)
        expected_cols = {
            "gene_id", "degree", "weighted_degree", "betweenness",
            "closeness", "eigenvector", "pagerank", "clustering_coeff",
        }
        assert set(result.columns) == expected_cols
        assert len(result) == 4

    def test_no_nan_in_features(self):
        edges = _make_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
        ])
        result = compute_all_network_features(edges)
        for col in result.columns:
            if col != "gene_id":
                assert result[col].null_count() == 0
