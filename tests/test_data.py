"""Tests for data module basic functionality."""

from gene_conservation_network.data.species import ALL_SPECIES, all_species_pairs


def test_code_is_tested():
    """Verify the data module is importable and has expected content."""
    assert len(ALL_SPECIES) == 6
    assert len(all_species_pairs()) == 30
