"""Tests for gene_conservation_network.data.coexpression."""

from pathlib import Path

import polars as pl
import pytest

from gene_conservation_network.data.coexpression import CoexpressionDataset
from gene_conservation_network.data.species import FLY, YEAST

COXPRESDB_DIR = Path("data/02_transformed/coxpresdb")

pytestmark = pytest.mark.skipif(
    not COXPRESDB_DIR.exists() or not any(COXPRESDB_DIR.iterdir()),
    reason="COXPRESdb transformed data not available (run transform_coxpresdb_data first)",
)


class TestCoexpressionDataset:
    @pytest.fixture
    def fly_dataset(self):
        return CoexpressionDataset(FLY, variant="u")

    def test_parquet_glob(self, fly_dataset):
        glob = fly_dataset.parquet_glob
        assert glob.endswith("*.parquet")
        assert "Dme-u" in glob

    def test_num_genes(self, fly_dataset):
        n = fly_dataset.num_genes
        assert n > 0
        # Fly unified dataset should have ~12,209 genes
        assert n > 10000

    def test_gene_ids(self, fly_dataset):
        ids = fly_dataset.gene_ids()
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)
        # IDs should be sorted
        assert ids == sorted(ids)

    def test_query_edges_schema(self, fly_dataset):
        edges = fly_dataset.query_edges(threshold=7.0)
        assert set(edges.columns) == {"gene_id_1", "gene_id_2", "association"}
        assert edges["gene_id_1"].dtype == pl.Int64 or edges["gene_id_1"].dtype == pl.Int32
        assert edges["association"].dtype == pl.Float64 or edges["association"].dtype == pl.Float32
        assert len(edges) > 0

    def test_threshold_filtering_reduces_edges(self, fly_dataset):
        """Higher thresholds should produce fewer edges."""
        edges_5 = fly_dataset.query_edges(threshold=5.0)
        edges_7 = fly_dataset.query_edges(threshold=7.0)
        assert len(edges_5) > len(edges_7)

    def test_query_single_gene(self, fly_dataset):
        gene_ids = fly_dataset.gene_ids()
        first_gene = gene_ids[0]
        df = fly_dataset.query_gene(first_gene)
        assert len(df) > 0
        assert set(df.columns) == {"gene_id_1", "gene_id_2", "association"}

    def test_query_single_gene_with_threshold(self, fly_dataset):
        gene_ids = fly_dataset.gene_ids()
        first_gene = gene_ids[0]
        df_all = fly_dataset.query_gene(first_gene)
        df_filtered = fly_dataset.query_gene(first_gene, threshold=5.0)
        assert len(df_filtered) <= len(df_all)

    def test_invalid_gene_raises(self, fly_dataset):
        with pytest.raises(FileNotFoundError):
            fly_dataset.query_gene(999999999)

    def test_repr(self, fly_dataset):
        r = repr(fly_dataset)
        assert "fly" in r
        assert "Dme-u" in r


class TestCoexpressionDatasetYeast:
    """Test with yeast (smaller dataset, faster)."""

    @pytest.fixture
    def yeast_dataset(self):
        return CoexpressionDataset(YEAST, variant="u")

    def test_yeast_edges(self, yeast_dataset):
        edges = yeast_dataset.query_edges(threshold=5.0)
        assert len(edges) > 0
        # Yeast should have ~22K edges at z>=5 per the plan
        assert len(edges) > 10000
