from pathlib import Path
from zipfile import ZipFile

from loguru import logger
import pandas as pd
import pandera as pa
from tqdm import tqdm


class CoexpressionSchema(pa.DataFrameModel):
    gene_id_1: int = pa.Field()
    gene_id_2: int = pa.Field()
    association: float = pa.Field()

    class Config:
        strict = True


def load_coxpresdb_coexpression(
    species: str,
    modality: str,
    data_dir: Path,
    version: str | None = None,
) -> pd.DataFrame:
    """
    Load COXPRESdb gene coexpression data.

    Args:
        species: Species code (case-insensitive), e.g., "Hsa", "Mmu", "Sce"
        modality: One of "microarray", "rna-seq", or "union"
        data_dir: Path to directory containing ZIP files
        version: Optional version for datasets with multiple versions (e.g., "Hsa2")

    Returns:
        DataFrame with columns: gene_id_1, gene_id_2, association

    Raises:
        ValueError: Invalid species/modality or multiple versions found
        FileNotFoundError: ZIP file not found
    """
    # Normalize species to title case
    species = species.title()

    # Validate and map modality
    modality_map = {"microarray": "m", "rna-seq": "r", "union": "u"}
    if modality not in modality_map:
        raise ValueError(
            f"Invalid modality: {modality}. Must be one of: microarray, rna-seq, union"
        )
    modality_code = modality_map[modality]

    # Construct glob pattern
    pattern = f"{version or species}-{modality_code}.*.zip"
    matching_files = sorted(Path(data_dir).glob(pattern))

    # Handle file matches
    if not matching_files:
        raise FileNotFoundError(f"No ZIP file found matching pattern '{pattern}' in {data_dir}")

    if len(matching_files) > 1 and version is None:
        files_str = ", ".join(f.name for f in matching_files)
        raise ValueError(
            f"Multiple versions found: {files_str}. Please specify version parameter."
        )

    # Use most recent if multiple matches with version specified
    zip_path = matching_files[-1]
    logger.info(f"Loading COXPRESdb data from: {zip_path.name}")

    records = []

    with ZipFile(zip_path, "r") as zip_file:
        file_list = [
            f for f in zip_file.namelist() if not f.endswith("/") and not f.startswith(".")
        ]

        for filename in tqdm(file_list, desc="Loading gene pairs"):
            try:
                gene_id_1 = int(filename)
            except ValueError:
                continue

            # Read file content
            content = zip_file.read(filename).decode("utf-8")
            lines = content.strip().split("\n")

            for line in lines:
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                try:
                    gene_id_2 = int(parts[0])
                    association = float(parts[1])
                    records.append(
                        {
                            "gene_id_1": gene_id_1,
                            "gene_id_2": gene_id_2,
                            "association": association,
                        }
                    )
                except (ValueError, IndexError):
                    continue

    # Create DataFrame
    df = pd.DataFrame(records)

    # Validate with pandera
    CoexpressionSchema.validate(df)

    return df


def list_available_coxpresdb_datasets(data_dir: Path) -> pd.DataFrame:
    """
    List all available COXPRESdb datasets in directory.

    Returns:
        DataFrame with columns: species, modality, version, filename
    """
    data_dir = Path(data_dir)
    datasets = []

    for zip_file in data_dir.glob("*.zip"):
        filename = zip_file.name
        # Parse filename: {Species}-{Modality}.v{Version}.G{NumGenes}-S{NumSamples}.{Method}.zip
        parts = filename.replace(".zip", "").split(".")
        if len(parts) < 2:
            continue

        name_part = parts[0]  # Species-Modality
        if "-" not in name_part:
            continue

        species, modality_code = name_part.split("-")
        modality_map_inv = {"m": "microarray", "r": "rna-seq", "u": "union"}
        modality = modality_map_inv.get(modality_code, modality_code)

        version = parts[1] if len(parts) > 1 else None

        datasets.append(
            {
                "species": species,
                "modality": modality,
                "version": version,
                "filename": filename,
            }
        )

    return pd.DataFrame(datasets)
