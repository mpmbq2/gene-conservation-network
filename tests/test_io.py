from pathlib import Path

import pytest

from gene_conservation_network.io import (
    CoexpressionSchema,
    load_coxpresdb_coexpression,
)


def test_load_coxpresdb_small_dataset():
    """Test loading Spo-u dataset (smallest available)"""
    df = load_coxpresdb_coexpression(
        species="Spo",
        modality="union",
        data_dir=Path("data/01_raw/coxpresdb_extracts"),
    )
    # Pandera validation happens automatically
    assert len(df) > 0
    assert list(df.columns) == ["gene_id_1", "gene_id_2", "association"]
    assert df["gene_id_1"].dtype == int
    assert df["gene_id_2"].dtype == int
    assert df["association"].dtype == float


def test_species_case_insensitive():
    """Test that species code is case-insensitive"""
    df = load_coxpresdb_coexpression(
        species="spo",  # lowercase
        modality="union",
        data_dir=Path("data/01_raw/coxpresdb_extracts"),
    )
    assert len(df) > 0


def test_invalid_modality():
    """Test that invalid modality raises ValueError"""
    with pytest.raises(ValueError, match="modality"):
        load_coxpresdb_coexpression(
            species="Spo",
            modality="invalid",
            data_dir=Path("data/01_raw/coxpresdb_extracts"),
        )


def test_file_not_found():
    """Test that missing file raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        load_coxpresdb_coexpression(
            species="Xxx",  # non-existent species
            modality="union",
            data_dir=Path("data/01_raw/coxpresdb_extracts"),
        )


def test_schema_validation():
    """Test that output matches expected schema"""
    df = load_coxpresdb_coexpression(
        species="Spo",
        modality="union",
        data_dir=Path("data/01_raw/coxpresdb_extracts"),
    )
    # This will raise if schema doesn't match
    CoexpressionSchema.validate(df)
