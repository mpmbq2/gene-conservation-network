"""Network feature extraction using rustworkx.

Build coexpression graphs and compute per-gene network features.
Each feature function is independently testable and can be run selectively.
"""

from __future__ import annotations

from loguru import logger
import polars as pl
import rustworkx as rx


def build_graph(edges: pl.DataFrame) -> tuple[rx.PyGraph, dict[int, int]]:
    """Build a rustworkx undirected graph from a coexpression edge list.

    Args:
        edges: DataFrame with columns [gene_id_1, gene_id_2, association]

    Returns:
        graph: rustworkx PyGraph with association as edge weights
        node_map: dict mapping gene_id (int) -> rustworkx node index (int)
    """
    graph = rx.PyGraph()
    node_map: dict[int, int] = {}

    # Collect unique gene IDs
    gene_ids_1 = edges["gene_id_1"].unique().to_list()
    gene_ids_2 = edges["gene_id_2"].unique().to_list()
    all_gene_ids = sorted(set(gene_ids_1) | set(gene_ids_2))

    # Add nodes
    for gene_id in all_gene_ids:
        idx = graph.add_node(gene_id)
        node_map[gene_id] = idx

    # Add edges
    edge_list = edges.select("gene_id_1", "gene_id_2", "association").iter_rows()
    for g1, g2, assoc in edge_list:
        if g1 in node_map and g2 in node_map:
            graph.add_edge(node_map[g1], node_map[g2], assoc)

    logger.info(f"Built graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    return graph, node_map


def _reverse_node_map(node_map: dict[int, int]) -> dict[int, int]:
    """Reverse node_map: rustworkx index -> gene_id."""
    return {v: k for k, v in node_map.items()}


def compute_degree(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Compute degree for each node.

    Returns: DataFrame with columns [gene_id, degree]
    """
    reverse = _reverse_node_map(node_map)
    data = [(reverse[idx], graph.degree(idx)) for idx in graph.node_indices()]
    return pl.DataFrame(data, schema={"gene_id": pl.Int64, "degree": pl.Int64}, orient="row")


def compute_weighted_degree(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Sum of edge weights per node.

    Returns: DataFrame with columns [gene_id, weighted_degree]
    """
    reverse = _reverse_node_map(node_map)
    result = []
    for idx in graph.node_indices():
        edges = graph.incident_edges(idx)
        weight_sum = sum(graph.get_edge_data_by_index(e) for e in edges)
        result.append((reverse[idx], weight_sum))
    return pl.DataFrame(
        result, schema={"gene_id": pl.Int64, "weighted_degree": pl.Float64}, orient="row"
    )


def compute_betweenness_centrality(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Betweenness centrality (multithreaded via Rayon).

    Returns: DataFrame with columns [gene_id, betweenness]
    """
    reverse = _reverse_node_map(node_map)
    centrality = rx.betweenness_centrality(graph)
    data = [(reverse[idx], centrality[idx]) for idx in graph.node_indices()]
    return pl.DataFrame(
        data, schema={"gene_id": pl.Int64, "betweenness": pl.Float64}, orient="row"
    )


def compute_closeness_centrality(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Closeness centrality.

    Returns: DataFrame with columns [gene_id, closeness]
    """
    reverse = _reverse_node_map(node_map)
    centrality = rx.closeness_centrality(graph)
    data = [(reverse[idx], centrality[idx]) for idx in graph.node_indices()]
    return pl.DataFrame(data, schema={"gene_id": pl.Int64, "closeness": pl.Float64}, orient="row")


def compute_eigenvector_centrality(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Eigenvector centrality.

    Returns: DataFrame with columns [gene_id, eigenvector]
    """
    reverse = _reverse_node_map(node_map)
    try:
        centrality = rx.eigenvector_centrality(graph)
    except rx.FailedToConverge:
        logger.warning("Eigenvector centrality failed to converge; returning NaN")
        data = [(reverse[idx], float("nan")) for idx in graph.node_indices()]
        return pl.DataFrame(
            data, schema={"gene_id": pl.Int64, "eigenvector": pl.Float64}, orient="row"
        )
    data = [(reverse[idx], centrality[idx]) for idx in graph.node_indices()]
    return pl.DataFrame(
        data, schema={"gene_id": pl.Int64, "eigenvector": pl.Float64}, orient="row"
    )


def compute_pagerank(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """PageRank centrality.

    PageRank requires a directed graph. We create a DiGraph with edges in
    both directions (equivalent to the undirected graph) and run pagerank on that.

    Returns: DataFrame with columns [gene_id, pagerank]
    """
    reverse = _reverse_node_map(node_map)

    # Build a directed graph with bidirectional edges
    digraph = rx.PyDiGraph()
    di_node_map: dict[int, int] = {}
    for idx in graph.node_indices():
        di_idx = digraph.add_node(graph[idx])
        di_node_map[idx] = di_idx

    for edge_idx in graph.edge_indices():
        endpoints = graph.get_edge_endpoints_by_index(edge_idx)
        weight = graph.get_edge_data_by_index(edge_idx)
        u, v = endpoints
        digraph.add_edge(di_node_map[u], di_node_map[v], weight)
        digraph.add_edge(di_node_map[v], di_node_map[u], weight)

    pr = rx.pagerank(digraph)
    data = [(reverse[idx], pr[di_node_map[idx]]) for idx in graph.node_indices()]
    return pl.DataFrame(data, schema={"gene_id": pl.Int64, "pagerank": pl.Float64}, orient="row")


def compute_clustering_coefficient(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Local clustering coefficient per node.

    Computed manually since rustworkx doesn't provide per-node clustering.
    For a node with degree < 2, the clustering coefficient is 0.

    Returns: DataFrame with columns [gene_id, clustering_coeff]
    """
    reverse = _reverse_node_map(node_map)
    result = []

    for idx in graph.node_indices():
        neighbors = list(graph.neighbors(idx))
        k = len(neighbors)
        if k < 2:
            result.append((reverse[idx], 0.0))
            continue

        # Count edges between neighbors
        triangles = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1 :]:
                if graph.has_edge(n1, n2):
                    triangles += 1

        max_edges = k * (k - 1) / 2
        cc = triangles / max_edges
        result.append((reverse[idx], cc))

    return pl.DataFrame(
        result, schema={"gene_id": pl.Int64, "clustering_coeff": pl.Float64}, orient="row"
    )


def compute_all_network_features(edges: pl.DataFrame) -> pl.DataFrame:
    """Convenience function: build graph and compute all features.

    Returns: DataFrame with columns [gene_id, degree, weighted_degree, betweenness,
             closeness, eigenvector, pagerank, clustering_coeff]
    """
    graph, node_map = build_graph(edges)

    logger.info("Computing degree...")
    degree = compute_degree(graph, node_map)
    logger.info("Computing weighted degree...")
    weighted = compute_weighted_degree(graph, node_map)
    logger.info("Computing betweenness centrality...")
    betweenness = compute_betweenness_centrality(graph, node_map)
    logger.info("Computing closeness centrality...")
    closeness = compute_closeness_centrality(graph, node_map)
    logger.info("Computing eigenvector centrality...")
    eigenvector = compute_eigenvector_centrality(graph, node_map)
    logger.info("Computing PageRank...")
    pagerank = compute_pagerank(graph, node_map)
    logger.info("Computing clustering coefficient...")
    clustering = compute_clustering_coefficient(graph, node_map)

    # Join all on gene_id
    result = degree
    for df in [weighted, betweenness, closeness, eigenvector, pagerank, clustering]:
        result = result.join(df, on="gene_id", how="left")

    logger.info(f"Computed all network features for {len(result)} genes")
    return result
