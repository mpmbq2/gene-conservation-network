"""WORMHOLE ortholog data access.

Clean interface to load and filter WORMHOLE ortholog data for species pairs.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import polars as pl

from gene_conservation_network.config import RAW_DATA_DIR
from gene_conservation_network.data.species import SpeciesPair

# Column name normalization: raw WORMHOLE column names -> snake_case
_COLUMN_RENAME = {
    "Query.TaxID": "query_tax_id",
    "Query.ID": "query_id",
    "Target.TaxID": "target_tax_id",
    "Target.ID": "target_id",
    "Pattern": "pattern",
    "Votes": "votes",
    "Vote.Score": "vote_score",
    "WORMHOLE.Score": "wormhole_score",
    "Best.Hit": "best_hit",
    "RBH": "rbh",
}

ORTHOLOG_COLUMNS = list(_COLUMN_RENAME.values())


class OrthologDataset:
    """Access WORMHOLE ortholog data for a species pair."""

    def __init__(
        self,
        pair: SpeciesPair,
        data_dir: Path = RAW_DATA_DIR / "wormhole_extracts",
    ):
        self._pair = pair
        self._data_dir = data_dir
        self._file_path = data_dir / f"{pair.wormhole_prefix}-WORMHOLE-orthologs.txt"

        if not self._file_path.exists():
            raise FileNotFoundError(
                f"Ortholog file not found: {self._file_path}. "
                f"Run the extract_wormhole_data pipeline stage first."
            )

        logger.info(f"OrthologDataset({pair}): using {self._file_path.name}")

    def _load(self) -> pl.DataFrame:
        """Load the full ortholog file with normalized column names."""
        df = pl.read_csv(
            self._file_path,
            separator="\t",
            schema_overrides={
                "Query.TaxID": pl.Int64,
                "Query.ID": pl.Utf8,
                "Target.TaxID": pl.Int64,
                "Target.ID": pl.Utf8,
                "Pattern": pl.Utf8,
                "Votes": pl.Int64,
                "Vote.Score": pl.Float64,
                "WORMHOLE.Score": pl.Float64,
                "Best.Hit": pl.Int64,
                "RBH": pl.Int64,
            },
        )
        return df.rename(_COLUMN_RENAME)

    def all_pairs(self) -> pl.DataFrame:
        """Return all ortholog pairs."""
        return self._load()

    def best_hits(self) -> pl.DataFrame:
        """Return only best-hit ortholog pairs (best_hit == 1)."""
        return self._load().filter(pl.col("best_hit") == 1)

    def reciprocal_best_hits(self) -> pl.DataFrame:
        """Return only reciprocal best hit pairs (rbh == 1)."""
        return self._load().filter(pl.col("rbh") == 1)

    def filter_by_score(self, min_score: float) -> pl.DataFrame:
        """Return pairs with wormhole_score >= min_score."""
        return self._load().filter(pl.col("wormhole_score") >= min_score)

    def filter_by_votes(self, min_votes: int) -> pl.DataFrame:
        """Return pairs with votes >= min_votes."""
        return self._load().filter(pl.col("votes") >= min_votes)

    @property
    def file_path(self) -> Path:
        """Path to the underlying ortholog file."""
        return self._file_path

    def __repr__(self) -> str:
        return f"OrthologDataset({self._pair}, file={self._file_path.name})"
