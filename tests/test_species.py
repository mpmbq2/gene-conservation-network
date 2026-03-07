"""Tests for gene_conservation_network.data.species."""

import pytest

from gene_conservation_network.data.species import (
    ALL_SPECIES,
    FLY,
    HUMAN,
    MOUSE,
    WORM,
    YEAST,
    ZEBRAFISH,
    Species,
    SpeciesPair,
    all_species_pairs,
    species_by_coxpresdb_code,
    species_by_taxonomy_id,
    species_by_wormhole_code,
)


class TestSpecies:
    def test_all_species_count(self):
        assert len(ALL_SPECIES) == 6

    def test_unique_coxpresdb_codes(self):
        codes = [s.coxpresdb_code for s in ALL_SPECIES]
        assert len(codes) == len(set(codes))

    def test_unique_wormhole_codes(self):
        codes = [s.wormhole_code for s in ALL_SPECIES]
        assert len(codes) == len(set(codes))

    def test_unique_taxonomy_ids(self):
        ids = [s.ncbi_taxonomy_id for s in ALL_SPECIES]
        assert len(ids) == len(set(ids))

    def test_species_str(self):
        assert "fly" in str(FLY)
        assert "Drosophila" in str(FLY)

    def test_species_frozen(self):
        with pytest.raises(AttributeError):
            FLY.common_name = "bee"  # type: ignore[misc]


class TestSpeciesPair:
    def test_wormhole_prefix(self):
        pair = SpeciesPair(query=FLY, target=ZEBRAFISH)
        assert pair.wormhole_prefix == "dmdr"

    def test_wormhole_prefix_reverse(self):
        pair = SpeciesPair(query=ZEBRAFISH, target=FLY)
        assert pair.wormhole_prefix == "drdm"

    def test_all_species_pairs_count(self):
        pairs = all_species_pairs()
        # 6 species, each paired with 5 others = 30 directed pairs
        assert len(pairs) == 30

    def test_all_species_pairs_no_self_pairs(self):
        pairs = all_species_pairs()
        for pair in pairs:
            assert pair.query != pair.target

    def test_pair_str(self):
        pair = SpeciesPair(query=FLY, target=HUMAN)
        assert "fly" in str(pair)
        assert "human" in str(pair)


class TestLookups:
    def test_by_coxpresdb_code(self):
        assert species_by_coxpresdb_code("Dme") == FLY
        assert species_by_coxpresdb_code("Hsa") == HUMAN
        assert species_by_coxpresdb_code("Cel") == WORM

    def test_by_wormhole_code(self):
        assert species_by_wormhole_code("dm") == FLY
        assert species_by_wormhole_code("hs") == HUMAN
        assert species_by_wormhole_code("sc") == YEAST

    def test_by_taxonomy_id(self):
        assert species_by_taxonomy_id(7227) == FLY
        assert species_by_taxonomy_id(9606) == HUMAN
        assert species_by_taxonomy_id(10090) == MOUSE

    def test_invalid_coxpresdb_code(self):
        with pytest.raises(KeyError, match="Unknown COXPRESdb code"):
            species_by_coxpresdb_code("Xyz")

    def test_invalid_wormhole_code(self):
        with pytest.raises(KeyError, match="Unknown WORMHOLE code"):
            species_by_wormhole_code("zz")

    def test_invalid_taxonomy_id(self):
        with pytest.raises(KeyError, match="Unknown taxonomy ID"):
            species_by_taxonomy_id(99999)
