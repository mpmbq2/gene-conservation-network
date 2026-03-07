"""Gene ID resolution between COXPRESdb (NCBI GeneID) and WORMHOLE (canonical ID) systems.

The two main data sources use different gene identifier systems:
- COXPRESdb coexpression data uses NCBI GeneIDs (integers), e.g., 43852
- WORMHOLE ortholog data uses species-native canonical IDs, e.g., FBgn0000008 (FlyBase)

The WORMHOLE alias files provide the bridge: each maps canonical IDs to their aliases,
which include the NCBI GeneIDs used by COXPRESdb.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import polars as pl

from gene_conservation_network.config import RAW_DATA_DIR
from gene_conservation_network.data.species import Species


class GeneIDResolver:
    """Resolves gene IDs between COXPRESdb (NCBI GeneID) and WORMHOLE (canonical ID) systems.

    The alias files have three columns (tab-separated, no header):
        tax_id  canonical_id  alias

    Each canonical ID has multiple aliases. One of those aliases is typically the NCBI
    GeneID (a pure integer). This class builds bidirectional lookup tables between
    NCBI GeneIDs and canonical IDs.
    """

    def __init__(
        self,
        species: Species,
        aliases_dir: Path = RAW_DATA_DIR / "wormhole_extracts",
    ):
        self._species = species
        self._aliases_path = aliases_dir / f"{species.wormhole_code}-aliases.txt"

        if not self._aliases_path.exists():
            raise FileNotFoundError(
                f"Alias file not found: {self._aliases_path}. "
                f"Run the extract_wormhole_data pipeline stage first."
            )

        self._ncbi_to_canonical: dict[int, str] = {}
        self._canonical_to_ncbi: dict[str, int] = {}
        self._total_canonical_ids = 0
        self._load_aliases()

    def _load_aliases(self) -> None:
        """Parse the alias file and build lookup tables."""
        df = pl.read_csv(
            self._aliases_path,
            separator="\t",
            has_header=False,
            new_columns=["tax_id", "canonical", "alias"],
            schema_overrides={
                "tax_id": pl.Utf8,
                "canonical": pl.Utf8,
                "alias": pl.Utf8,
            },
        )

        self._total_canonical_ids = df["canonical"].n_unique()

        # Find aliases that are pure integers (candidate NCBI GeneIDs).
        # Strategy: filter aliases matching ^\d+$ and cast to int.
        integer_aliases = df.filter(pl.col("alias").str.contains(r"^\d+$")).with_columns(
            pl.col("alias").cast(pl.Int64).alias("ncbi_id")
        )

        # For each canonical ID, pick the integer alias as the NCBI GeneID.
        # Some canonical IDs may have multiple integer aliases; we pick the first one.
        # Some canonical IDs ARE the NCBI GeneID (e.g., human/mouse where canonical = integer).
        canonical_ncbi = (
            integer_aliases.group_by("canonical")
            .agg(pl.col("ncbi_id").first())
            .select(["canonical", "ncbi_id"])
        )

        for row in canonical_ncbi.iter_rows(named=True):
            canonical = row["canonical"]
            ncbi_id = row["ncbi_id"]
            self._ncbi_to_canonical[ncbi_id] = canonical
            self._canonical_to_ncbi[canonical] = ncbi_id

        mapped = len(self._canonical_to_ncbi)
        logger.info(
            f"GeneIDResolver({self._species.common_name}): "
            f"mapped {mapped}/{self._total_canonical_ids} canonical IDs "
            f"to NCBI GeneIDs ({100 * mapped / max(self._total_canonical_ids, 1):.1f}%)"
        )

    def ncbi_to_canonical(self, ncbi_id: int) -> str | None:
        """Convert an NCBI GeneID to the WORMHOLE canonical ID.

        Returns None if the mapping is not found.
        """
        return self._ncbi_to_canonical.get(ncbi_id)

    def canonical_to_ncbi(self, canonical_id: str) -> int | None:
        """Convert a WORMHOLE canonical ID to an NCBI GeneID.

        Returns None if the mapping is not found.
        """
        return self._canonical_to_ncbi.get(canonical_id)

    def resolve_coexpression_ids(self, coex_df: pl.DataFrame) -> pl.DataFrame:
        """Add canonical ID columns to a coexpression DataFrame.

        Input columns: gene_id_1 (int), gene_id_2 (int), association (float)
        Output adds: canonical_id_1 (str), canonical_id_2 (str)
        Rows where either ID cannot be resolved are dropped (with a warning log).
        """
        # Build a Polars-native mapping frame for efficient joins
        mapping = pl.DataFrame(
            {
                "ncbi_id": list(self._ncbi_to_canonical.keys()),
                "canonical_id": list(self._ncbi_to_canonical.values()),
            },
            schema={"ncbi_id": pl.Int64, "canonical_id": pl.Utf8},
        )

        original_count = len(coex_df)

        result = coex_df.join(
            mapping.rename({"ncbi_id": "gene_id_1", "canonical_id": "canonical_id_1"}),
            on="gene_id_1",
            how="inner",
        ).join(
            mapping.rename({"ncbi_id": "gene_id_2", "canonical_id": "canonical_id_2"}),
            on="gene_id_2",
            how="inner",
        )

        dropped = original_count - len(result)
        if dropped > 0:
            logger.warning(
                f"Dropped {dropped}/{original_count} coexpression rows "
                f"due to unmappable gene IDs ({100 * dropped / original_count:.1f}%)"
            )

        return result

    def resolve_ortholog_ids(self, ortholog_df: pl.DataFrame) -> pl.DataFrame:
        """Add NCBI GeneID columns to an ortholog DataFrame.

        Input columns: query_id (str), target_id (str), ...
        Output adds: query_ncbi_id (int), target_ncbi_id (int)
        Rows where resolution fails are dropped (with a warning log).
        """
        mapping = pl.DataFrame(
            {
                "canonical_id": list(self._canonical_to_ncbi.keys()),
                "ncbi_id": list(self._canonical_to_ncbi.values()),
            },
            schema={"canonical_id": pl.Utf8, "ncbi_id": pl.Int64},
        )

        original_count = len(ortholog_df)

        result = ortholog_df.join(
            mapping.rename({"canonical_id": "query_id", "ncbi_id": "query_ncbi_id"}),
            on="query_id",
            how="inner",
        ).join(
            mapping.rename({"canonical_id": "target_id", "ncbi_id": "target_ncbi_id"}),
            on="target_id",
            how="inner",
        )

        dropped = original_count - len(result)
        if dropped > 0:
            logger.warning(
                f"Dropped {dropped}/{original_count} ortholog rows "
                f"due to unmappable gene IDs ({100 * dropped / original_count:.1f}%)"
            )

        return result

    @property
    def coverage(self) -> dict:
        """Return mapping coverage stats."""
        return {
            "species": self._species.common_name,
            "total_canonical_ids": self._total_canonical_ids,
            "mapped_to_ncbi": len(self._canonical_to_ncbi),
            "mapped_from_ncbi": len(self._ncbi_to_canonical),
            "coverage_pct": round(
                100 * len(self._canonical_to_ncbi) / max(self._total_canonical_ids, 1), 1
            ),
        }

    def __repr__(self) -> str:
        cov = self.coverage
        return (
            f"GeneIDResolver({self._species.common_name}: "
            f"{cov['mapped_to_ncbi']}/{cov['total_canonical_ids']} mapped)"
        )
