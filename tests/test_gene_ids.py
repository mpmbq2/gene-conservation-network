"""Tests for gene_conservation_network.data.gene_ids."""

from pathlib import Path

import polars as pl
import pytest

from gene_conservation_network.data.gene_ids import GeneIDResolver
from gene_conservation_network.data.species import FLY, HUMAN, YEAST

# These tests read actual alias files from data/01_raw/wormhole_extracts/
ALIASES_DIR = Path("data/01_raw/wormhole_extracts")

pytestmark = pytest.mark.skipif(
    not (ALIASES_DIR / "dm-aliases.txt").exists(),
    reason="WORMHOLE alias files not available (run extract_wormhole_data first)",
)


class TestFlyResolver:
    @pytest.fixture
    def resolver(self):
        return GeneIDResolver(FLY, aliases_dir=ALIASES_DIR)

    def test_ncbi_to_canonical_known(self, resolver):
        # NCBI 43852 -> FBgn0000008 (the 'a' gene in fly)
        result = resolver.ncbi_to_canonical(43852)
        assert result == "FBgn0000008"

    def test_canonical_to_ncbi_known(self, resolver):
        result = resolver.canonical_to_ncbi("FBgn0000008")
        assert result == 43852

    def test_unknown_ncbi_returns_none(self, resolver):
        assert resolver.ncbi_to_canonical(999999999) is None

    def test_unknown_canonical_returns_none(self, resolver):
        assert resolver.canonical_to_ncbi("FAKE_GENE_ID") is None

    def test_coverage_stats(self, resolver):
        cov = resolver.coverage
        assert cov["species"] == "fly"
        assert cov["total_canonical_ids"] > 0
        assert cov["mapped_to_ncbi"] > 0
        assert 0 < cov["coverage_pct"] <= 100

    def test_repr(self, resolver):
        r = repr(resolver)
        assert "fly" in r
        assert "mapped" in r


class TestHumanResolver:
    @pytest.fixture
    def resolver(self):
        return GeneIDResolver(HUMAN, aliases_dir=ALIASES_DIR)

    def test_coverage_stats(self, resolver):
        cov = resolver.coverage
        assert cov["species"] == "human"
        assert cov["total_canonical_ids"] > 0
        assert cov["mapped_to_ncbi"] > 0


class TestResolveCoexpressionIds:
    @pytest.fixture
    def resolver(self):
        return GeneIDResolver(FLY, aliases_dir=ALIASES_DIR)

    def test_resolve_synthetic_coexpression(self, resolver):
        """Test resolving a small synthetic coexpression DataFrame."""
        # Use known fly NCBI IDs
        coex = pl.DataFrame(
            {
                "gene_id_1": [43852, 43852],
                "gene_id_2": [31209, 36248],
                "association": [5.0, 3.0],
            },
            schema={"gene_id_1": pl.Int64, "gene_id_2": pl.Int64, "association": pl.Float64},
        )

        result = resolver.resolve_coexpression_ids(coex)
        assert "canonical_id_1" in result.columns
        assert "canonical_id_2" in result.columns
        # Should have at most as many rows as input
        assert len(result) <= len(coex)

    def test_unmappable_ids_dropped(self, resolver):
        """Test that rows with unmappable IDs are dropped."""
        coex = pl.DataFrame(
            {
                "gene_id_1": [43852, 999999999],
                "gene_id_2": [31209, 31209],
                "association": [5.0, 3.0],
            },
            schema={"gene_id_1": pl.Int64, "gene_id_2": pl.Int64, "association": pl.Float64},
        )

        result = resolver.resolve_coexpression_ids(coex)
        # The row with 999999999 should be dropped
        assert len(result) <= 1


class TestResolveOrthologIds:
    @pytest.fixture
    def resolver(self):
        return GeneIDResolver(FLY, aliases_dir=ALIASES_DIR)

    def test_resolve_synthetic_orthologs(self, resolver):
        """Test resolving a small synthetic ortholog DataFrame."""
        ortho = pl.DataFrame(
            {
                "query_id": ["FBgn0000008", "FBgn0000008"],
                "target_id": ["FBgn0000008", "FBgn0000008"],
                "wormhole_score": [0.9, 0.5],
            }
        )

        result = resolver.resolve_ortholog_ids(ortho)
        assert "query_ncbi_id" in result.columns
        assert "target_ncbi_id" in result.columns
