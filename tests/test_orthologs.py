"""Tests for gene_conservation_network.data.orthologs."""

from pathlib import Path

import polars as pl
import pytest

from gene_conservation_network.data.orthologs import ORTHOLOG_COLUMNS, OrthologDataset
from gene_conservation_network.data.species import FLY, HUMAN, YEAST, SpeciesPair

WORMHOLE_DIR = Path("data/01_raw/wormhole_extracts")

pytestmark = pytest.mark.skipif(
    not (WORMHOLE_DIR / "dmhs-WORMHOLE-orthologs.txt").exists(),
    reason="WORMHOLE ortholog files not available (run extract_wormhole_data first)",
)


class TestOrthologDataset:
    @pytest.fixture
    def fly_human(self):
        pair = SpeciesPair(query=FLY, target=HUMAN)
        return OrthologDataset(pair, data_dir=WORMHOLE_DIR)

    def test_file_path(self, fly_human):
        assert fly_human.file_path.name == "dmhs-WORMHOLE-orthologs.txt"

    def test_column_names(self, fly_human):
        df = fly_human.all_pairs()
        assert set(df.columns) == set(ORTHOLOG_COLUMNS)

    def test_all_pairs_not_empty(self, fly_human):
        df = fly_human.all_pairs()
        assert len(df) > 0

    def test_best_hits_subset(self, fly_human):
        all_df = fly_human.all_pairs()
        best = fly_human.best_hits()
        assert len(best) > 0
        assert len(best) <= len(all_df)
        assert (best["best_hit"] == 1).all()

    def test_reciprocal_best_hits_subset(self, fly_human):
        all_df = fly_human.all_pairs()
        rbh = fly_human.reciprocal_best_hits()
        assert len(rbh) >= 0
        assert len(rbh) <= len(all_df)
        if len(rbh) > 0:
            assert (rbh["rbh"] == 1).all()

    def test_rbh_subset_of_best_hits(self, fly_human):
        best = fly_human.best_hits()
        rbh = fly_human.reciprocal_best_hits()
        # All RBH should also be best hits
        assert len(rbh) <= len(best)

    def test_filter_by_score(self, fly_human):
        all_df = fly_human.all_pairs()
        filtered = fly_human.filter_by_score(0.5)
        assert len(filtered) <= len(all_df)
        assert (filtered["wormhole_score"] >= 0.5).all()

    def test_filter_by_votes(self, fly_human):
        all_df = fly_human.all_pairs()
        filtered = fly_human.filter_by_votes(5)
        assert len(filtered) <= len(all_df)
        assert (filtered["votes"] >= 5).all()


class TestOrthologDatasetMissing:
    def test_missing_file_raises(self):
        pair = SpeciesPair(query=FLY, target=YEAST)
        with pytest.raises(FileNotFoundError):
            OrthologDataset(pair, data_dir=Path("/nonexistent/dir"))


class TestSpeciesPairPrefix:
    def test_fly_human_prefix(self):
        pair = SpeciesPair(query=FLY, target=HUMAN)
        assert pair.wormhole_prefix == "dmhs"

    def test_yeast_fly_prefix(self):
        pair = SpeciesPair(query=YEAST, target=FLY)
        assert pair.wormhole_prefix == "scdm"
