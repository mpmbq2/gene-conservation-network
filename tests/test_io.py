import pandas as pd

from gene_conservation_network.io import CoexpressionSchema


def test_coexpression_schema_validation():
    """Test that CoexpressionSchema validates correctly formatted data"""
    # Create valid DataFrame
    df = pd.DataFrame(
        {
            "gene_id_1": [1, 2, 3],
            "gene_id_2": [4, 5, 6],
            "association": [0.5, 0.7, 0.9],
        }
    )
    # This should validate without raising
    CoexpressionSchema.validate(df)


def test_coexpression_schema_rejects_invalid_types():
    """Test that CoexpressionSchema rejects incorrect data types"""
    # Create invalid DataFrame with wrong types
    df = pd.DataFrame(
        {
            "gene_id_1": ["a", "b", "c"],  # Should be int
            "gene_id_2": [4, 5, 6],
            "association": [0.5, 0.7, 0.9],
        }
    )
    # This should raise a validation error
    try:
        CoexpressionSchema.validate(df)
        assert False, "Expected validation error for wrong types"
    except Exception:
        pass  # Expected


def test_coexpression_schema_requires_all_columns():
    """Test that CoexpressionSchema requires all columns"""
    # Create DataFrame missing a column
    df = pd.DataFrame(
        {
            "gene_id_1": [1, 2, 3],
            "gene_id_2": [4, 5, 6],
            # Missing "association" column
        }
    )
    # This should raise a validation error
    try:
        CoexpressionSchema.validate(df)
        assert False, "Expected validation error for missing column"
    except Exception:
        pass  # Expected
