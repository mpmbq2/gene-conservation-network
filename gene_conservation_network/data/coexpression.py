"""Coexpression data access via DuckDB.

Clean interface to query COXPRESdb parquet coexpression data using DuckDB
for efficient scanning and Polars for downstream manipulation.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from loguru import logger
import polars as pl

from gene_conservation_network.config import COXPRESDB_TRANSFORMED_DIR
from gene_conservation_network.data.species import Species


class CoexpressionDataset:
    """Access coexpression data for a single species via DuckDB.

    Each species' coexpression data is stored as many small parquet files
    (one per gene) in a directory like:
        data/02_transformed/coxpresdb/Dme-u.v22-05.G12209-S15610.../*.parquet

    This class auto-discovers the correct directory and provides methods
    to query edges by threshold.
    """

    def __init__(
        self,
        species: Species,
        variant: str = "u",
        data_dir: Path = COXPRESDB_TRANSFORMED_DIR,
    ):
        self._species = species
        self._variant = variant
        self._data_dir = data_dir
        self._dataset_dir = self._discover_dataset_dir()

        logger.info(
            f"CoexpressionDataset({species.common_name}, variant={variant}): "
            f"using {self._dataset_dir.name}"
        )

    def _discover_dataset_dir(self) -> Path:
        """Find the dataset directory matching species code and variant."""
        prefix = f"{self._species.coxpresdb_code}-{self._variant}."
        candidates = [
            d for d in self._data_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
        ]

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No dataset directory found for {self._species.common_name} "
                f"variant='{self._variant}' in {self._data_dir}. "
                f"Expected a directory starting with '{prefix}'"
            )
        if len(candidates) > 1:
            # Pick the one with the latest version (sort lexicographically)
            candidates.sort(key=lambda d: d.name)
            logger.warning(
                f"Multiple directories found for {prefix}*: "
                f"{[c.name for c in candidates]}. Using {candidates[-1].name}"
            )
        return candidates[-1]

    @property
    def parquet_glob(self) -> str:
        """The glob pattern for this dataset's parquet files."""
        return str(self._dataset_dir / "*.parquet")

    @property
    def num_genes(self) -> int:
        """Number of genes (parquet files) in the dataset."""
        return len(list(self._dataset_dir.glob("*.parquet")))

    def query_edges(self, threshold: float) -> pl.DataFrame:
        """Return edges with association >= threshold.

        Args:
            threshold: Minimum association z-score to include an edge.
                       No default -- callers must be explicit about their threshold choice.

        Returns:
            Polars DataFrame with columns [gene_id_1, gene_id_2, association]
        """
        sql = f"""
            SELECT gene_id_1, gene_id_2, association
            FROM read_parquet('{self.parquet_glob}')
            WHERE association >= {threshold}
        """
        logger.info(f"Querying {self._species.common_name} edges with threshold >= {threshold}...")
        result = duckdb.sql(sql).pl()
        logger.info(f"Found {len(result):,} edges")
        return result

    def gene_ids(self) -> list[int]:
        """Return all gene IDs in the dataset.

        Gene IDs are extracted from the parquet file names (each file = one gene).
        """
        return sorted(int(p.stem) for p in self._dataset_dir.glob("*.parquet"))

    def query_gene(self, gene_id: int, threshold: float | None = None) -> pl.DataFrame:
        """Return all edges for a specific gene.

        Args:
            gene_id: The NCBI GeneID to query.
            threshold: Optional minimum association z-score filter.

        Returns:
            Polars DataFrame with columns [gene_id_1, gene_id_2, association]
        """
        parquet_path = self._dataset_dir / f"{gene_id}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"No parquet file for gene {gene_id} in {self._dataset_dir.name}"
            )

        df = pl.read_parquet(parquet_path)
        if threshold is not None:
            df = df.filter(pl.col("association") >= threshold)
        return df

    def __repr__(self) -> str:
        return (
            f"CoexpressionDataset({self._species.common_name}, "
            f"variant={self._variant!r}, dir={self._dataset_dir.name})"
        )
