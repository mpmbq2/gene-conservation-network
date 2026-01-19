#!/usr/bin/env python3
"""
Extract COXPRESdb gene correlation tables.

This script reads configuration from params.yaml and downloads
the specified gene correlation tables to the configured output directory.
"""

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from loguru import logger
from tqdm import tqdm
import yaml

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJ_ROOT / "params/extract_coxpresdb_data.yaml"

def load_config():
    """Load configuration from params/extract_coxpresdb_data.yaml."""
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        logger.error("Config file not found: {}", PARAMS_PATH)
        return {}

def get_filename_from_url(url: str) -> str:
    """Extract filename from URL."""
    path = urlparse(url).path
    return Path(path).name

def download_file(url: str, destination: Path) -> bool:
    """Download a file with a progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Check if file already exists and has the same size
        if destination.exists() and total_size > 0:
            if destination.stat().st_size == total_size:
                logger.info(
                    "Skipping {} (already exists and size matches)",
                    destination.name,
                )
                return True

        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        logger.error("Failed to download {}: {}", url, e)
        if destination.exists():
            destination.unlink()  # Remove partial file
        return False

def main():
    """Main extraction function."""
    logger.info("Starting COXPRESdb Data Extraction")
    
    config = load_config()
    if not config:
        logger.error("No configuration found for extract_coxpresdb_data")
        return

    output_dir = Path(PROJ_ROOT) / config.get("output_dir", "data/01_raw/coxpresdb_extracts")
    files_to_download = config.get("files_to_download", [])

    logger.info("Output directory: {}", output_dir)
    logger.info("Found {} files to download", len(files_to_download))

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize download log
    log_file = output_dir / "download_log.json"
    download_log = {
        "timestamp": datetime.now().isoformat(),
        "files": []
    }
    
    successful_downloads = 0
    
    for url in files_to_download:
        filename = get_filename_from_url(url)
        file_path = output_dir / filename
        
        logger.info("Processing: {}", filename)
        
        if download_file(url, file_path):
            successful_downloads += 1
            download_log["files"].append({
                "url": url,
                "local_path": str(file_path),
                "status": "success",
                "timestamp": datetime.now().isoformat()
            })
        else:
            download_log["files"].append({
                "url": url,
                "status": "failed",
                "error": "Download failed",
                "timestamp": datetime.now().isoformat()
            })

    # Save download log
    with open(log_file, 'w') as f:
        json.dump(download_log, f, indent=2)
    
    logger.info("Extraction Complete!")
    logger.info(
        "Successfully processed: {}/{} files",
        successful_downloads,
        len(files_to_download),
    )
    logger.info("Log file: {}", log_file)

if __name__ == "__main__":
    main()
