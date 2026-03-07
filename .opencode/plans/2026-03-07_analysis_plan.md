# Gene Conservation Network: Analysis Plan

**Date:** 2026-03-07
**Status:** Approved (v2) - ready for implementation

## Goal

Explore the relationship between gene coexpression network properties and cross-species conservation. Three core hypotheses:

1. **Hub genes are more conserved**: High-degree genes in coexpression networks have higher ortholog confidence scores across species.
2. **Hub genes have more ambiguous orthologs**: High-degree genes have more ortholog candidates (many-to-many mappings), reflecting evolutionary redundancy for critical genes.
3. **The hub-conservation relationship varies with evolutionary distance**: The strength/direction of the network-ortholog relationship depends on how far apart the species are.

> **Note:** Hypothesis 3 is deferred. The framework will support it, but we won't implement evolutionary distance metrics in this iteration.

---

## Feasibility Analysis

### Data Scale (Unified Variants, 6 WORMHOLE Species)

| Species | Code | Genes | Raw Records | Disk Size |
|---------|------|------:|------------:|----------:|
| C. elegans (worm) | Cel-u | 14,532 | 211M | 1.3 GB |
| D. melanogaster (fly) | Dme-u | 12,209 | 149M | 951 MB |
| D. rerio (zebrafish) | Dre-u | 20,344 | 414M | 2.7 GB |
| H. sapiens (human) | Hsa-u | 16,651 | 277M | 1.8 GB |
| M. musculus (mouse) | Mmu-u | 17,008 | 289M | 1.9 GB |
| S. cerevisiae (yeast) | Sce-u | 5,718 | 33M | 224 MB |
| **Total** | | **86,462** | **1.37B** | **~8.9 GB** |

### After Thresholding (Association z-score >= 5)

| Species | Edges at z>=5 |
|---------|-------------:|
| Cel-u | ~119K |
| Dme-u | ~147K |
| Dre-u | ~179K |
| Hsa-u | ~184K |
| Mmu-u | ~318K |
| Sce-u | ~22K |
| **Total** | **~970K** |

**Key insight:** Thresholding reduces 1.37 billion raw edges to ~970K. After filtering, all graphs fit easily in memory.

### Threshold Sensitivity

The association threshold is a critical parameter that shapes the network topology and could influence downstream results. **All analysis functions will accept the threshold as a parameter** so results can be computed across multiple thresholds (e.g., z >= 3, 4, 5, 6) to assess robustness. If a finding only holds at one threshold, it is likely an artifact of that threshold choice rather than a genuine biological signal.

### Ortholog Data (WORMHOLE)

- 30 pairwise files (6 species x 5 targets) totaling ~2.6 GB, ~11.4M rows
- Largest pairs: Dre-Hsa (1.31M rows), Mmu-Dre (1.24M rows) -- zebrafish whole-genome duplication creates many-to-many mappings
- All files are tab-separated with consistent 10-column format

### Memory Requirements

| Operation | 16 GB RAM | 32 GB RAM |
|-----------|:---------:|:---------:|
| Stream + filter parquet files (DuckDB) | OK (~2.5 GB peak) | OK |
| Hold all filtered graphs in memory | OK (~1 GB total) | OK |
| Graph algorithms (rustworkx, multithreaded) | OK | OK |
| Load ALL raw records at once | NO (33 GB needed) | Tight |

**Verdict:** Entirely feasible on a single machine with 16+ GB RAM. The bottleneck is I/O (reading 86K small parquet files), not memory. DuckDB handles this efficiently.

### Out-of-Memory Strategy

| Approach | Peak Memory | Speed | Best For |
|----------|:-----------:|:-----:|----------|
| DuckDB (SQL on parquet) | ~2.5 GB | Fast | Heavy scan+filter on raw parquet |
| Polars lazy scan | ~5-17 GB (configurable) | Fast | Transform-heavy pipelines |
| Hybrid DuckDB + Polars | ~3 GB | Fast | Our use case |

**Decision: DuckDB for scanning/filtering raw parquet, Polars for downstream manipulation.** This matches the pattern already established in the exploratory notebook.

---

## ID Mapping Challenge

This is the core data engineering problem. The two data sources use different gene ID systems:

| Data Source | ID System | Example |
|-------------|-----------|---------|
| COXPRESdb coexpression data | NCBI GeneIDs (integers) | `43852` (fly gene) |
| WORMHOLE ortholog data | Species-native IDs | `FBgn0000008` (FlyBase), `WBGene00000001` (WormBase), `ENSG00000092148` (Ensembl) |

The WORMHOLE alias files provide the bridge: each maps NCBI GeneIDs (among other aliases) to the canonical IDs used in ortholog files. The `GeneIDResolver` class (below) will handle this translation.

**Complication for human/mouse:** The alias files for hs and mm use NCBI GeneIDs as the canonical form, while the ortholog files use Ensembl IDs. The resolver must handle bidirectional lookups.

---

## Architecture

### Package Structure

```
gene_conservation_network/
|-- __init__.py                    # Existing
|-- config.py                      # Existing (path constants)
|-- schemas.py                     # Renamed from io.py; expanded with schemas for all data types
|
|-- data/                          # NEW: Data access layer
|   |-- __init__.py
|   |-- species.py                 # Species registry and metadata
|   |-- gene_ids.py                # Gene ID resolution across naming systems
|   |-- coexpression.py            # Coexpression dataset access via DuckDB
|   |-- orthologs.py               # WORMHOLE ortholog data access
|
|-- features/                      # NEW: Feature extraction
|   |-- __init__.py
|   |-- network.py                 # Graph construction + network features (rustworkx)
|   |-- ortholog_features.py       # Ortholog-derived per-gene features
|
|-- analysis/                      # NEW: Relationship exploration
|   |-- __init__.py
|   |-- correlation.py             # Feature-feature correlation analysis
|   |-- hypotheses.py              # Structured hypothesis tests
|   |-- visualization.py           # Plotting functions
```

### New Dependencies

| Package | Purpose | Notes |
|---------|---------|-------|
| `rustworkx` | Graph library (replaces NetworkX) | Rust-backed, 3-100x faster, multithreaded centrality |
| `scipy` | Statistical tests | Spearman/Pearson correlation p-values |

---

## Module Specifications

### 1. `data/species.py` -- Species Registry

**Purpose:** Single source of truth for species metadata. Eliminates hardcoded strings and makes it easy to iterate over species.

```python
@dataclass(frozen=True)
class Species:
    common_name: str          # "fly", "human", "yeast"
    scientific_name: str      # "Drosophila melanogaster"
    coxpresdb_code: str       # "Dme" (3-letter code used in COXPRESdb file names)
    wormhole_code: str        # "dm" (2-letter code used in WORMHOLE file names)
    ncbi_taxonomy_id: int     # 7227

# Pre-defined instances
WORM = Species("worm", "Caenorhabditis elegans", "Cel", "ce", 6239)
FLY = Species("fly", "Drosophila melanogaster", "Dme", "dm", 7227)
ZEBRAFISH = Species("zebrafish", "Danio rerio", "Dre", "dr", 7955)
HUMAN = Species("human", "Homo sapiens", "Hsa", "hs", 9606)
MOUSE = Species("mouse", "Mus musculus", "Mmu", "mm", 10090)
YEAST = Species("yeast", "Saccharomyces cerevisiae", "Sce", "sc", 4932)

ALL_SPECIES: list[Species] = [WORM, FLY, ZEBRAFISH, HUMAN, MOUSE, YEAST]

@dataclass(frozen=True)
class SpeciesPair:
    query: Species
    target: Species

    @property
    def wormhole_prefix(self) -> str:
        """e.g., 'dmdr' for fly->zebrafish"""
        return f"{self.query.wormhole_code}{self.target.wormhole_code}"

def all_species_pairs() -> list[SpeciesPair]:
    """All 30 directed pairs."""
    ...

# Lookup helpers
def species_by_coxpresdb_code(code: str) -> Species: ...
def species_by_wormhole_code(code: str) -> Species: ...
def species_by_taxonomy_id(tax_id: int) -> Species: ...
```

**Tests (`tests/test_species.py`):**
- Verify all 6 species have unique codes/taxonomy IDs
- Verify `all_species_pairs()` returns 30 pairs
- Verify lookup functions work correctly
- Verify `SpeciesPair.wormhole_prefix` produces correct strings

---

### 2. `data/gene_ids.py` -- Gene ID Resolution

**Purpose:** Bridge between NCBI GeneIDs (used by COXPRESdb) and species-native canonical IDs (used by WORMHOLE orthologs).

```python
class GeneIDResolver:
    """Resolves gene IDs between COXPRESdb (NCBI GeneID) and WORMHOLE (canonical ID) systems."""

    def __init__(self, species: Species, aliases_dir: Path = RAW_DATA_DIR / "wormhole_extracts"):
        # Loads {species.wormhole_code}-aliases.txt
        # Builds bidirectional lookup tables:
        #   _ncbi_to_canonical: dict[int, str]   (many-to-one, but we pick the canonical)
        #   _canonical_to_ncbi: dict[str, int]   (may be one-to-many for some species)
        ...

    def ncbi_to_canonical(self, ncbi_id: int) -> str | None:
        """Convert an NCBI GeneID to the WORMHOLE canonical ID."""
        ...

    def canonical_to_ncbi(self, canonical_id: str) -> int | None:
        """Convert a WORMHOLE canonical ID to an NCBI GeneID."""
        ...

    def resolve_coexpression_ids(self, coex_df: pl.DataFrame) -> pl.DataFrame:
        """Add canonical ID columns to a coexpression DataFrame.

        Input columns: gene_id_1 (int), gene_id_2 (int), association (float)
        Output adds: canonical_id_1 (str), canonical_id_2 (str)
        Rows where resolution fails are dropped (with a warning log).
        """
        ...

    def resolve_ortholog_ids(self, ortholog_df: pl.DataFrame) -> pl.DataFrame:
        """Add NCBI GeneID columns to an ortholog DataFrame.

        Input columns: Query.ID (str), Target.ID (str), ...
        Output adds: query_ncbi_id (int), target_ncbi_id (int)
        """
        ...

    @property
    def coverage(self) -> dict:
        """Return mapping coverage stats: how many IDs can be resolved."""
        ...
```

**Key design decisions:**
- The alias files contain multiple aliases per gene. We need to identify which alias is the NCBI GeneID (it's the one that's a pure integer and matches the taxonomy's GeneID range).
- For yeast (sc), canonical IDs are SGD systematic names (e.g., `YAL003W`), not Ensembl or NCBI.
- The resolver is species-specific -- you create one per species.

**Tests (`tests/test_gene_ids.py`):**
- Test with known fly gene: NCBI `43852` <-> FlyBase `FBgn0000008`
- Test with known human gene: verify Ensembl <-> NCBI mapping
- Test resolution of a small synthetic coexpression DataFrame
- Test handling of unmappable IDs (returns None, logs warning)
- Test coverage stats

---

### 3. `data/coexpression.py` -- Coexpression Data Access

**Purpose:** Clean interface to query the parquet coexpression data via DuckDB.

```python
class CoexpressionDataset:
    """Access coexpression data for a single species via DuckDB."""

    def __init__(
        self,
        species: Species,
        variant: str = "u",
        data_dir: Path = COXPRESDB_TRANSFORMED_DIR,
    ):
        # Resolves the parquet glob pattern, e.g.:
        # data/02_transformed/coxpresdb/Dme-u.v22-05.G12209-S15610.../*.parquet
        # Validates the directory exists
        ...

    @property
    def parquet_glob(self) -> str:
        """The glob pattern for this dataset's parquet files."""
        ...

    @property
    def num_genes(self) -> int:
        """Number of genes (parquet files) in the dataset."""
        ...

    def query_edges(self, threshold: float) -> pl.DataFrame:
        """Return edges with association >= threshold.

        Args:
            threshold: Minimum association z-score to include an edge.
                       No default -- callers must be explicit about their threshold choice.

        Returns: Polars DataFrame with columns [gene_id_1, gene_id_2, association]
        """
        ...

    def gene_ids(self) -> list[int]:
        """Return all gene IDs in the dataset."""
        ...

    def query_gene(self, gene_id: int, threshold: float | None = None) -> pl.DataFrame:
        """Return all edges for a specific gene."""
        ...
```

**Implementation detail:** Uses `duckdb.sql()` with parquet glob scanning, converts to Polars via `.pl()`. The glob pattern is auto-discovered from the data directory by matching the species' `coxpresdb_code` and the variant letter.

**Tests (`tests/test_coexpression.py`):**
- Test glob pattern construction
- Test that `query_edges` returns correct schema
- Test threshold filtering reduces edge count
- Test with a mock/fixture parquet dataset (small synthetic data created in conftest.py)

---

### 4. `data/orthologs.py` -- Ortholog Data Access

**Purpose:** Clean interface to WORMHOLE ortholog data.

```python
class OrthologDataset:
    """Access WORMHOLE ortholog data for a species pair."""

    def __init__(
        self,
        pair: SpeciesPair,
        data_dir: Path = RAW_DATA_DIR / "wormhole_extracts",
    ):
        # Locates the file: {pair.wormhole_prefix}-WORMHOLE-orthologs.txt
        ...

    def all_pairs(self) -> pl.DataFrame:
        """Return all ortholog pairs."""
        ...

    def best_hits(self) -> pl.DataFrame:
        """Return only best-hit ortholog pairs (Best.Hit == 1)."""
        ...

    def reciprocal_best_hits(self) -> pl.DataFrame:
        """Return only reciprocal best hit pairs (RBH == 1)."""
        ...

    def filter_by_score(self, min_score: float) -> pl.DataFrame:
        """Return pairs with WORMHOLE.Score >= min_score."""
        ...

    def filter_by_votes(self, min_votes: int) -> pl.DataFrame:
        """Return pairs with Votes >= min_votes."""
        ...
```

**Column naming convention:** The raw files use `Query.TaxID`, `Query.ID`, etc. These will be normalized to snake_case on load: `query_tax_id`, `query_id`, `target_tax_id`, `target_id`, `pattern`, `votes`, `vote_score`, `wormhole_score`, `best_hit`, `rbh`.

**Tests (`tests/test_orthologs.py`):**
- Test file path construction from SpeciesPair
- Test column names after loading
- Test that `best_hits()` only returns rows where `best_hit == 1`
- Test that `reciprocal_best_hits()` only returns rows where `rbh == 1`
- Test score/vote filtering

---

### 5. `features/network.py` -- Network Feature Extraction

**Purpose:** Build coexpression graphs and compute per-gene network features using rustworkx.

```python
def build_graph(edges: pl.DataFrame) -> tuple[rx.PyGraph, dict[int, int]]:
    """Build a rustworkx undirected graph from a coexpression edge list.

    Args:
        edges: DataFrame with columns [gene_id_1, gene_id_2, association]

    Returns:
        graph: rustworkx PyGraph with association as edge weights
        node_map: dict mapping gene_id (int) -> rustworkx node index (int)
    """
    ...

def compute_degree(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Compute degree for each node.

    Returns: DataFrame with columns [gene_id, degree]
    """
    ...

def compute_weighted_degree(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Sum of edge weights per node.

    Returns: DataFrame with columns [gene_id, weighted_degree]
    """
    ...

def compute_betweenness_centrality(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Betweenness centrality (multithreaded via Rayon).

    Returns: DataFrame with columns [gene_id, betweenness]
    """
    ...

def compute_closeness_centrality(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Closeness centrality.

    Returns: DataFrame with columns [gene_id, closeness]
    """
    ...

def compute_eigenvector_centrality(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Eigenvector centrality.

    Returns: DataFrame with columns [gene_id, eigenvector]
    """
    ...

def compute_pagerank(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """PageRank centrality.

    Returns: DataFrame with columns [gene_id, pagerank]
    """
    ...

def compute_clustering_coefficient(graph: rx.PyGraph, node_map: dict[int, int]) -> pl.DataFrame:
    """Local clustering coefficient per node.

    Returns: DataFrame with columns [gene_id, clustering_coeff]
    """
    ...

def compute_all_network_features(edges: pl.DataFrame) -> pl.DataFrame:
    """Convenience function: build graph and compute all features.

    Returns: DataFrame with columns [gene_id, degree, weighted_degree, betweenness,
             closeness, eigenvector, pagerank, clustering_coeff]
    """
    graph, node_map = build_graph(edges)
    ...  # join all individual feature DataFrames on gene_id
```

**Why separate functions?** Each is independently testable and can be run selectively (e.g., skip expensive betweenness for quick exploration). The `compute_all_network_features` convenience function is the main entry point for pipeline use.

**Tests (`tests/test_network_features.py`):**
- **Star graph (5 nodes):** center node has degree 4, leaf nodes have degree 1. Center has highest betweenness (1.0 for normalized). Known PageRank values.
- **Complete graph (4 nodes):** all nodes have degree 3, betweenness 0, clustering coefficient 1.0.
- **Path graph (4 nodes):** known centrality values for a simple chain.
- Test that output DataFrames have correct schema and no NaN values.

---

### 6. `features/ortholog_features.py` -- Ortholog-Derived Features

**Purpose:** Compute per-gene features derived from ortholog data that capture "conservation" and "ambiguity."

```python
def compute_ortholog_count(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Number of ortholog targets per query gene.

    Returns: DataFrame with columns [gene_id, ortholog_count]
    Higher count = more ortholog candidates = potentially more ambiguity.
    """
    ...

def compute_rbh_count(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Number of reciprocal best hit targets per query gene.

    Returns: DataFrame with columns [gene_id, rbh_count]
    Typically 0 or 1 for most genes. >1 indicates duplication events.
    """
    ...

def compute_max_ortholog_score(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Maximum WORMHOLE score across all ortholog targets.

    Returns: DataFrame with columns [gene_id, max_wormhole_score]
    Captures the strength of the best ortholog relationship.
    """
    ...

def compute_mean_ortholog_score(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Mean WORMHOLE score across all ortholog targets.

    Returns: DataFrame with columns [gene_id, mean_wormhole_score]
    """
    ...

def compute_max_votes(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Maximum vote count across all ortholog targets.

    Returns: DataFrame with columns [gene_id, max_votes]
    Higher votes = more algorithm agreement = stronger evidence.
    """
    ...

def compute_vote_entropy(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Shannon entropy of the vote distribution across ortholog targets.

    Returns: DataFrame with columns [gene_id, vote_entropy]
    High entropy = orthologs have diverse support patterns = more ambiguity.
    """
    ...

def compute_has_rbh(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Binary: does this gene have at least one RBH?

    Returns: DataFrame with columns [gene_id, has_rbh]
    """
    ...

def compute_all_ortholog_features(
    orthologs: pl.DataFrame,
    gene_col: str = "query_id",
) -> pl.DataFrame:
    """Combine all ortholog features into a single DataFrame.

    Returns: DataFrame with columns [gene_id, ortholog_count, rbh_count,
             max_wormhole_score, mean_wormhole_score, max_votes,
             vote_entropy, has_rbh]
    """
    ...
```

**Tests (`tests/test_ortholog_features.py`):**
- Synthetic data: gene A has 5 orthologs (1 RBH), gene B has 1 ortholog (1 RBH), gene C has 3 orthologs (0 RBH)
- Verify counts, scores, entropy values against hand-calculated expected results
- Edge cases: gene with no orthologs, gene with all RBHs

---

### 7. `analysis/correlation.py` -- Feature Correlation

**Purpose:** Join network and ortholog features and compute descriptive correlations. Focus is on effect sizes (correlation coefficients), not significance testing.

```python
def merge_features(
    network_features: pl.DataFrame,
    ortholog_features: pl.DataFrame,
    id_resolver: GeneIDResolver,
) -> pl.DataFrame:
    """Join network features (keyed by NCBI GeneID) with ortholog features (keyed by canonical ID).

    Uses the GeneIDResolver to bridge the ID gap.

    Returns: DataFrame with all network + ortholog feature columns, keyed by gene_id.
    """
    ...

def compute_pairwise_correlations(
    merged: pl.DataFrame,
    network_cols: list[str],
    ortholog_cols: list[str],
    method: str = "spearman",
) -> pl.DataFrame:
    """Compute correlation between each network feature and each ortholog feature.

    Returns: DataFrame with columns [network_feature, ortholog_feature, correlation, n]
    """
    ...

def compute_correlation_matrix(
    merged: pl.DataFrame,
    feature_cols: list[str],
    method: str = "spearman",
) -> pl.DataFrame:
    """Full correlation matrix across all features.

    Returns: Square DataFrame (feature x feature).
    """
    ...
```

**Tests (`tests/test_correlation.py`):**
- Synthetic data with known correlation (e.g., x = [1,2,3,4,5], y = [2,4,6,8,10] -> r = 1.0)
- Test that uncorrelated data produces near-zero coefficients
- Test that merge correctly bridges IDs

---

### 8. `analysis/hypotheses.py` -- Descriptive Hypothesis Exploration

**Purpose:** Structured functions for each hypothesis, returning standardized descriptive result objects. This is explicitly a **descriptive analysis** -- we compute effect sizes and correlations to characterize relationships, not to perform null hypothesis significance testing. P-values are not the focus; the goal is to describe the direction, magnitude, and consistency of relationships across species and thresholds.

```python
@dataclass
class HypothesisResult:
    name: str
    species: str
    target_species: str | None   # For cross-species comparisons
    threshold: float             # Association threshold used
    statistic_name: str          # e.g., "spearman_r"
    statistic_value: float
    n_genes: int
    summary: str                 # Human-readable summary

def describe_hub_conservation(
    merged_features: pl.DataFrame,
    species: Species,
    target_species: Species,
    threshold: float,
    hub_metric: str = "degree",
    conservation_metric: str = "max_wormhole_score",
) -> HypothesisResult:
    """Hypothesis 1: Hub genes are more conserved.

    Describes the correlation between a network centrality measure and ortholog confidence.
    """
    ...

def describe_hub_ambiguity(
    merged_features: pl.DataFrame,
    species: Species,
    target_species: Species,
    threshold: float,
    hub_metric: str = "degree",
    ambiguity_metric: str = "ortholog_count",
) -> HypothesisResult:
    """Hypothesis 2: Hub genes have more ambiguous orthologs.

    Describes the correlation between a network centrality measure and ortholog count/entropy.
    """
    ...

def describe_all_hypotheses(
    merged_features: pl.DataFrame,
    species: Species,
    target_species: Species,
    threshold: float,
) -> list[HypothesisResult]:
    """Run all descriptive analyses for a species pair at a given threshold.

    Explores multiple combinations of hub metrics x conservation/ambiguity metrics.
    """
    ...

def describe_threshold_sensitivity(
    species: Species,
    target_species: Species,
    thresholds: list[float],
    compute_fn: Callable,  # Function that produces merged features for a given threshold
) -> pl.DataFrame:
    """Run a hypothesis across multiple thresholds to assess robustness.

    Returns: DataFrame with columns [threshold, statistic_name, statistic_value, n_genes]
    """
    ...
```

**Tests (`tests/test_hypotheses.py`):**
- Synthetic data with known relationships
- Verify result structure (all fields populated)
- Verify summary string is human-readable
- Verify threshold sensitivity produces results for each threshold

---

### 9. `analysis/visualization.py` -- Plotting

**Purpose:** Standardized, publication-quality plots for exploring results.

```python
def plot_feature_scatter(
    merged: pl.DataFrame,
    x_col: str,
    y_col: str,
    species_label: str = "",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Scatter plot of two features with regression line and correlation annotation."""
    ...

def plot_correlation_heatmap(
    correlations: pl.DataFrame,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of feature-feature correlations."""
    ...

def plot_species_comparison(
    results: list[HypothesisResult],
    metric_name: str = "",
) -> plt.Figure:
    """Forest-plot style comparison of a hypothesis result across species pairs."""
    ...

def plot_degree_distribution(
    features: pl.DataFrame,
    species_label: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Log-log degree distribution plot."""
    ...
```

---

## DVC Pipeline Extensions

New data layers and pipeline stages:

```
data/
|-- 01_raw/              # Existing (DVC-tracked)
|-- 02_transformed/      # Existing (DVC-tracked)
|-- 03_features/         # NEW
|   |-- network/         # Per-species, per-threshold network feature tables
|   |   |-- cel_u_t3.parquet    # threshold=3
|   |   |-- cel_u_t4.parquet    # threshold=4
|   |   |-- cel_u_t5.parquet    # threshold=5
|   |   |-- cel_u_t6.parquet    # threshold=6
|   |   |-- dme_u_t3.parquet
|   |   |-- ...
|   |-- orthologs/       # Per-pair ortholog feature tables (threshold-independent)
|       |-- dmdr.parquet
|       |-- ...
|-- 04_merged/           # NEW
    |-- dmdr_t3.parquet  # Network + ortholog features merged, threshold=3
    |-- dmdr_t4.parquet
    |-- dmdr_t5.parquet
    |-- dmdr_t6.parquet
    |-- ...
```

New `dvc.yaml` stages:

```yaml
compute_network_features:
  cmd: pixi run python scripts/compute_network_features.py
  deps:
    - scripts/compute_network_features.py
    - gene_conservation_network/features/network.py
    - data/02_transformed/coxpresdb/
  params:
    - params/compute_features.yaml:
      - species
      - variant
      - thresholds          # List of thresholds, e.g. [3, 4, 5, 6]
  outs:
    - data/03_features/network/

compute_ortholog_features:
  cmd: pixi run python scripts/compute_ortholog_features.py
  deps:
    - scripts/compute_ortholog_features.py
    - gene_conservation_network/features/ortholog_features.py
    - data/01_raw/wormhole_extracts/
  params:
    - params/compute_features.yaml:
      - species_pairs
  outs:
    - data/03_features/orthologs/

merge_features:
  cmd: pixi run python scripts/merge_features.py
  deps:
    - scripts/merge_features.py
    - gene_conservation_network/analysis/correlation.py
    - data/03_features/
    - data/01_raw/wormhole_extracts/  # For alias resolution
  outs:
    - data/04_merged/
```

---

## Testing Strategy

Each module gets a corresponding test file:

| Test File | Tests | Data |
|-----------|-------|------|
| `tests/test_species.py` | Species mapping, pair generation, lookups | No data needed |
| `tests/test_gene_ids.py` | Alias resolution, coverage stats | Reads actual alias files |
| `tests/test_coexpression.py` | Glob construction, query schema, filtering | Synthetic parquet fixtures |
| `tests/test_orthologs.py` | File path construction, column names, filtering | Synthetic TSV fixtures |
| `tests/test_network_features.py` | Features on star/complete/path graphs | Synthetic edge DataFrames |
| `tests/test_ortholog_features.py` | Counts, scores, entropy on synthetic data | Synthetic ortholog DataFrames |
| `tests/test_correlation.py` | Correlation with known values | Synthetic feature DataFrames |
| `tests/test_hypotheses.py` | Result structure, summary output | Synthetic merged DataFrames |

---

## Implementation Order

| Phase | Modules | Rationale |
|-------|---------|-----------|
| **1. Foundation** | `data/species.py`, `data/gene_ids.py`, tests | Prerequisites for everything. Gets ID mapping right first. |
| **2. Data Access** | `data/coexpression.py`, `data/orthologs.py`, tests | With species and IDs in place, build the data layer. |
| **3. Network Features** | `features/network.py`, tests | Build graphs and extract per-gene features. |
| **4. Ortholog Features** | `features/ortholog_features.py`, tests | Extract per-gene features from ortholog data. |
| **5. Analysis** | `analysis/correlation.py`, `analysis/hypotheses.py`, tests | Join features and test hypotheses. |
| **6. Visualization** | `analysis/visualization.py` | Plotting functions. |
| **7. Pipeline** | `scripts/`, `dvc.yaml`, `params/` | DVC pipeline stages. |
| **8. Notebook** | `notebooks/` | Full workflow demonstration. |

Each phase is independently testable. Tests should pass before moving to the next phase.

---

## New Dependencies to Add

```toml
# In pyproject.toml [tool.pixi.dependencies]
rustworkx = ">=0.15,<1"

# In pyproject.toml [tool.pixi.pypi-dependencies] or [tool.pixi.dependencies]
scipy = ">=1.11,<2"
```

---

## Resolved Decisions

1. **Threshold sensitivity:** Yes -- all analysis functions accept threshold as a parameter. The DVC pipeline computes features at multiple thresholds (e.g., z >= 3, 4, 5, 6) and the `describe_threshold_sensitivity` function enables direct comparison of results across thresholds.
2. **Descriptive analysis, not NHST:** This is a descriptive/exploratory analysis. We compute effect sizes (correlation coefficients) to characterize relationships, not p-values to reject null hypotheses. The `HypothesisResult` dataclass omits p-values; the focus is on the direction, magnitude, and consistency of correlations across species and thresholds.
3. **Rename `io.py` to `schemas.py`:** Confirmed. Rename and expand with schemas for all data types.

## Future Directions

1. **Confounders:** Gene length, expression level, and functional category could confound the hub-conservation relationship. A future iteration should control for these by including them as covariates or stratifying analyses.
2. **Evolutionary distance (Hypothesis 3):** When ready, add divergence times from TimeTree.org as a species-pair-level covariate to test whether the hub-conservation relationship varies with evolutionary distance.
3. **Cross-species network comparison:** Instead of just correlating within one species' network, compare network properties of orthologous gene pairs across species (e.g., does a hub gene in fly also tend to be a hub gene in human?).
