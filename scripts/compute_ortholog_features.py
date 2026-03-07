"""Compute ortholog features for all species pairs.

DVC pipeline stage: reads WORMHOLE ortholog data, computes per-gene
ortholog features, and writes output parquet files.

Usage:
    pixi run python scripts/compute_ortholog_features.py
"""

from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

from gene_conservation_network.config import ORTHOLOG_FEATURES_DIR, PROJ_ROOT
from gene_conservation_network.data.orthologs import OrthologDataset
from gene_conservation_network.data.species import SpeciesPair, species_by_wormhole_code
from gene_conservation_network.features.ortholog_features import compute_all_ortholog_features


def main():
    params_path = PROJ_ROOT / "params" / "compute_features.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    species_pairs = params["species_pairs"]

    ORTHOLOG_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for pair_code in tqdm(species_pairs, desc="Species pairs"):
        # Parse the 4-letter pair code (e.g., "dmdr" -> query="dm", target="dr")
        query_code = pair_code[:2]
        target_code = pair_code[2:]
        query = species_by_wormhole_code(query_code)
        target = species_by_wormhole_code(target_code)
        pair = SpeciesPair(query=query, target=target)

        out_path = ORTHOLOG_FEATURES_DIR / f"{pair_code}.parquet"
        if out_path.exists():
            logger.info(f"Skipping {pair_code} (already exists)")
            continue

        logger.info(f"Computing ortholog features: {pair}")
        dataset = OrthologDataset(pair)
        orthologs = dataset.all_pairs()

        if len(orthologs) == 0:
            logger.warning(f"No ortholog data for {pair}")
            continue

        features = compute_all_ortholog_features(orthologs)
        features.write_parquet(out_path)
        logger.success(f"Wrote {pair_code}.parquet: {len(features)} genes")

    logger.success("Ortholog feature computation complete")


if __name__ == "__main__":
    main()
