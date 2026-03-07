"""Compute network features for all species at all thresholds.

DVC pipeline stage: reads coexpression parquet data, builds graphs,
computes per-gene network features, and writes output parquet files.

Usage:
    pixi run python scripts/compute_network_features.py
"""

from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

from gene_conservation_network.config import NETWORK_FEATURES_DIR, PROJ_ROOT
from gene_conservation_network.data.coexpression import CoexpressionDataset
from gene_conservation_network.data.species import species_by_coxpresdb_code
from gene_conservation_network.features.network import compute_all_network_features


def main():
    params_path = PROJ_ROOT / "params" / "compute_features.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    species_codes = params["species"]
    variant = params["variant"]
    thresholds = params["thresholds"]

    NETWORK_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for code in tqdm(species_codes, desc="Species"):
        species = species_by_coxpresdb_code(code)
        dataset = CoexpressionDataset(species, variant=variant)

        for threshold in tqdm(thresholds, desc=f"  {species.common_name} thresholds", leave=False):
            out_name = f"{code.lower()}_{variant}_t{threshold}.parquet"
            out_path = NETWORK_FEATURES_DIR / out_name

            if out_path.exists():
                logger.info(f"Skipping {out_name} (already exists)")
                continue

            logger.info(f"Computing network features: {species.common_name}, threshold={threshold}")
            edges = dataset.query_edges(threshold=float(threshold))

            if len(edges) == 0:
                logger.warning(f"No edges at threshold={threshold} for {species.common_name}")
                continue

            features = compute_all_network_features(edges)
            features.write_parquet(out_path)
            logger.success(
                f"Wrote {out_name}: {len(features)} genes, {len(edges):,} edges"
            )

    logger.success("Network feature computation complete")


if __name__ == "__main__":
    main()
