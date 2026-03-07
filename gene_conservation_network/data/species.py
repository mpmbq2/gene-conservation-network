"""Species registry and metadata.

Single source of truth for species metadata used across the project.
Eliminates hardcoded strings and makes it easy to iterate over species.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Species:
    """A species with its identifiers across data sources."""

    common_name: str  # "fly", "human", "yeast"
    scientific_name: str  # "Drosophila melanogaster"
    coxpresdb_code: str  # "Dme" (3-letter code used in COXPRESdb file names)
    wormhole_code: str  # "dm" (2-letter code used in WORMHOLE file names)
    ncbi_taxonomy_id: int  # 7227

    def __str__(self) -> str:
        return f"{self.common_name} ({self.scientific_name})"


# Pre-defined instances for the 6 WORMHOLE species
WORM = Species("worm", "Caenorhabditis elegans", "Cel", "ce", 6239)
FLY = Species("fly", "Drosophila melanogaster", "Dme", "dm", 7227)
ZEBRAFISH = Species("zebrafish", "Danio rerio", "Dre", "dr", 7955)
HUMAN = Species("human", "Homo sapiens", "Hsa", "hs", 9606)
MOUSE = Species("mouse", "Mus musculus", "Mmu", "mm", 10090)
YEAST = Species("yeast", "Saccharomyces cerevisiae", "Sce", "sc", 4932)

ALL_SPECIES: list[Species] = [WORM, FLY, ZEBRAFISH, HUMAN, MOUSE, YEAST]

# Lookup tables (built once at import time)
_BY_COXPRESDB_CODE: dict[str, Species] = {s.coxpresdb_code: s for s in ALL_SPECIES}
_BY_WORMHOLE_CODE: dict[str, Species] = {s.wormhole_code: s for s in ALL_SPECIES}
_BY_TAXONOMY_ID: dict[int, Species] = {s.ncbi_taxonomy_id: s for s in ALL_SPECIES}


@dataclass(frozen=True)
class SpeciesPair:
    """A directed pair of species (query -> target)."""

    query: Species
    target: Species

    @property
    def wormhole_prefix(self) -> str:
        """WORMHOLE file prefix, e.g., 'dmdr' for fly->zebrafish."""
        return f"{self.query.wormhole_code}{self.target.wormhole_code}"

    def __str__(self) -> str:
        return f"{self.query.common_name}->{self.target.common_name}"


def all_species_pairs() -> list[SpeciesPair]:
    """Return all 30 directed pairs of the 6 WORMHOLE species."""
    return [SpeciesPair(query=q, target=t) for q in ALL_SPECIES for t in ALL_SPECIES if q != t]


def species_by_coxpresdb_code(code: str) -> Species:
    """Look up a Species by its COXPRESdb 3-letter code (e.g., 'Dme')."""
    try:
        return _BY_COXPRESDB_CODE[code]
    except KeyError:
        valid = sorted(_BY_COXPRESDB_CODE.keys())
        raise KeyError(f"Unknown COXPRESdb code '{code}'. Valid codes: {valid}") from None


def species_by_wormhole_code(code: str) -> Species:
    """Look up a Species by its WORMHOLE 2-letter code (e.g., 'dm')."""
    try:
        return _BY_WORMHOLE_CODE[code]
    except KeyError:
        valid = sorted(_BY_WORMHOLE_CODE.keys())
        raise KeyError(f"Unknown WORMHOLE code '{code}'. Valid codes: {valid}") from None


def species_by_taxonomy_id(tax_id: int) -> Species:
    """Look up a Species by its NCBI taxonomy ID (e.g., 7227)."""
    try:
        return _BY_TAXONOMY_ID[tax_id]
    except KeyError:
        valid = sorted(_BY_TAXONOMY_ID.keys())
        raise KeyError(f"Unknown taxonomy ID {tax_id}. Valid IDs: {valid}") from None
