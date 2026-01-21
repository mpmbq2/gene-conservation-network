"""Transform raw COXPRESdb ZIP files into parquet format for efficient querying."""

import json
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm

from gene_conservation_network.config import PROJ_ROOT
from gene_conservation_network.io import CoexpressionSchema


def load_config(config_path: Path) -> dict:
    """Load transformation configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate DataFrame against CoexpressionSchema."""
    try:
        CoexpressionSchema.validate(df)
        return True
    except Exception as e:
        logger.warning(f"Schema validation failed: {e}")
        return False


def process_zip_file(
    zip_path: Path,
    output_dir: Path,
    compression: str,
) -> dict:
    """
    Process a single COXPRESdb ZIP file and convert to parquet format.

    Args:
        zip_path: Path to input ZIP file
        output_dir: Directory to save parquet files
        compression: Compression algorithm (e.g., 'snappy')

    Returns:
        Dictionary with processing statistics
    """
    dataset_name = zip_path.stem
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "dataset": dataset_name,
        "files_processed": 0,
        "files_failed": 0,
        "records_total": 0,
        "errors": [],
    }

    try:
        with ZipFile(zip_path, "r") as zip_file:
            file_list = [
                f for f in zip_file.namelist() if not f.endswith("/") and not f.startswith(".")
            ]

            for filename in tqdm(
                file_list,
                desc=f"Processing {dataset_name}",
                leave=False,
            ):
                try:
                    # Parse gene_id_1 from filename
                    gene_id_1 = int(filename)
                except ValueError:
                    continue

                try:
                    # Skip if already processed
                    output_file = dataset_output_dir / f"{gene_id_1}.parquet"
                    if output_file.exists():
                        logger.debug(f"Skipping already processed: {gene_id_1}")
                        stats["files_processed"] += 1
                        continue

                    # Read file content
                    content = zip_file.read(filename).decode("utf-8")
                    lines = content.strip().split("\n")

                    records = []
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

                    if records:
                        # Create DataFrame and validate
                        df = pd.DataFrame(records)

                        if validate_dataframe(df):
                            # Save as parquet with snappy compression
                            df.to_parquet(
                                output_file,
                                compression=compression,
                                index=False,
                            )
                            stats["files_processed"] += 1
                            stats["records_total"] += len(df)
                            #logger.debug(f"Processed {gene_id_1}: {len(df)} records")
                        else:
                            stats["files_failed"] += 1
                            stats["errors"].append(f"Schema validation failed for {gene_id_1}")
                    else:
                        stats["files_processed"] += 1

                except Exception as e:
                    stats["files_failed"] += 1
                    error_msg = f"Error processing gene {gene_id_1}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

    except Exception as e:
        logger.error(f"Error processing ZIP file {zip_path}: {str(e)}")
        stats["errors"].append(f"ZIP processing error: {str(e)}")

    return stats


def main():
    """Main transformation pipeline."""
    # Load configuration
    config_path = PROJ_ROOT / "params" / "transform_coxpresdb_data.yaml"
    config = load_config(config_path)

    input_dir = PROJ_ROOT / config["input_dir"]
    output_dir = PROJ_ROOT / config["output_dir"]
    compression = config["compression"]

    logger.info(f"Starting COXPRESdb transformation")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Compression: {compression}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all ZIP files
    zip_files = sorted(input_dir.glob("*.zip"))

    if not zip_files:
        logger.warning(f"No ZIP files found in {input_dir}")
        return

    logger.info(f"Found {len(zip_files)} ZIP files to process")

    # Process each ZIP file
    all_stats = []
    start_time = datetime.now()

    for zip_path in tqdm(zip_files, desc="Processing datasets"):
        logger.info(f"Processing: {zip_path.name}")
        stats = process_zip_file(zip_path, output_dir, compression)
        all_stats.append(stats)
        logger.info(
            f"Completed {stats['dataset']}: "
            f"{stats['files_processed']} files, "
            f"{stats['records_total']} records"
        )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Create summary log
    summary = {
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "datasets_processed": len(all_stats),
        "total_files": sum(s["files_processed"] for s in all_stats),
        "total_failed": sum(s["files_failed"] for s in all_stats),
        "total_records": sum(s["records_total"] for s in all_stats),
        "details": all_stats,
    }

    # Write log file to parent directory to avoid DVC output conflicts
    log_file = PROJ_ROOT / "data" / "02_transformed" / "transform_log.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.success(f"Transformation complete!")
    logger.info(f"Log file saved to: {log_file}")
    logger.info(f"Total datasets processed: {summary['datasets_processed']}")
    logger.info(f"Total files processed: {summary['total_files']}")
    logger.info(f"Total records: {summary['total_records']}")
    logger.info(f"Duration: {duration:.2f} seconds")

    if summary["total_failed"] > 0:
        logger.warning(f"Failed files: {summary['total_failed']}")


if __name__ == "__main__":
    main()
