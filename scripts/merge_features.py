"""Merge network and ortholog features for all species pairs at all thresholds.

DVC pipeline stage: reads network features and ortholog features,
resolves gene IDs, and writes merged parquet files.

Usage:
    pixi run python scripts/merge_features.py
"""

from pathlib import Path

import polars as pl
import yaml
from loguru import logger
from tqdm import tqdm

from gene_conservation_network.analysis.correlation import merge_features
from gene_conservation_network.config import (
    MERGED_DIR,
    NETWORK_FEATURES_DIR,
    ORTHOLOG_FEATURES_DIR,
    PROJ_ROOT,
)
from gene_conservation_network.data.gene_ids import GeneIDResolver
from gene_conservation_network.data.species import species_by_wormhole_code


def main():
    params_path = PROJ_ROOT / "params" / "compute_features.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    species_pairs = params["species_pairs"]
    thresholds = params["thresholds"]
    variant = params["variant"]

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    # Cache resolvers by species code
    resolvers: dict[str, GeneIDResolver] = {}

    for pair_code in tqdm(species_pairs, desc="Species pairs"):
        query_code = pair_code[:2]
        target_code = pair_code[2:]
        query = species_by_wormhole_code(query_code)
        target = species_by_wormhole_code(target_code)

        # Load ortholog features for this pair
        ortholog_path = ORTHOLOG_FEATURES_DIR / f"{pair_code}.parquet"
        if not ortholog_path.exists():
            logger.warning(f"Ortholog features not found: {ortholog_path}")
            continue
        ortholog_features = pl.read_parquet(ortholog_path)

        # Get or create resolver for the query species
        if query_code not in resolvers:
            resolvers[query_code] = GeneIDResolver(query)

        resolver = resolvers[query_code]

        for threshold in thresholds:
            out_path = MERGED_DIR / f"{pair_code}_t{threshold}.parquet"
            if out_path.exists():
                logger.info(f"Skipping {out_path.name} (already exists)")
                continue

            # Load network features for query species at this threshold
            network_name = f"{query.coxpresdb_code.lower()}_{variant}_t{threshold}.parquet"
            network_path = NETWORK_FEATURES_DIR / network_name
            if not network_path.exists():
                logger.warning(f"Network features not found: {network_path}")
                continue

            network_features = pl.read_parquet(network_path)

            logger.info(f"Merging {pair_code} threshold={threshold}")
            merged = merge_features(network_features, ortholog_features, resolver)

            if len(merged) == 0:
                logger.warning(f"No merged data for {pair_code} at threshold={threshold}")
                continue

            merged.write_parquet(out_path)
            logger.success(f"Wrote {out_path.name}: {len(merged)} genes")

    logger.success("Feature merging complete")


if __name__ == "__main__":
    main()
