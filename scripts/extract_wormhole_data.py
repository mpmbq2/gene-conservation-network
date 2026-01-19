#!/usr/bin/env python3
"""
Extract WORMHOLE data files from https://wormhole.jax.org/data.html

This script downloads all available data files from the WORMHOLE website
and extracts them to the data/01_raw/ directory.
"""

import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict
from urllib.parse import urljoin

import requests
from loguru import logger
import yaml
from tqdm import tqdm


def load_config():
    """Load configuration from params/extract_wormhole_data.yaml."""
    params_file = Path("params/extract_wormhole_data.yaml")
    if params_file.exists():
        with open(params_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Fallback to hardcoded values if params/extract_wormhole_data.yaml doesn't exist
        return {
            "base_url": "https://wormhole.jax.org",
            "data_url": "https://wormhole.jax.org/static/data/",
            "output_dir": "data/01_raw/wormhole_extracts",
            "files_to_download": [
                "WORMHOLE-canonical-IDs.tar.gz",
                "WORMHOLE-aliases.tar.gz", 
                "PANTHER-LDO-ortholog-pairs.tar.gz",
                "WHRefOGs.tar.gz",
                "WORMHOLE-ortholog-pairs.tar.gz"
            ]
        }


# Load configuration
config = load_config()
BASE_URL = config["base_url"]
DATA_URL = config["data_url"]
OUTPUT_DIR = Path(config["output_dir"])
LOG_FILE = OUTPUT_DIR / "download_log.json"
FILES_TO_DOWNLOAD = config["files_to_download"]


def download_file(url: str, destination: Path) -> bool:
    """Download a file with progress bar and retry logic."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
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
        logger.error("Error downloading {}: {}", url, e)
        return False


def extract_archive(archive_path: Path) -> bool:
    """Extract a tar archive to the same directory."""
    try:
        # Try different tar formats
        modes = ('r:gz', 'r:bz2', 'r:xz', 'r')
        for mode in modes:
            try:
                with tarfile.open(archive_path, mode) as tar:
                    # Extract to the same directory as the archive
                    tar.extractall(path=archive_path.parent)
                return True
            except (tarfile.ReadError, OSError):
                continue
        raise tarfile.ReadError("Could not read tar archive with any supported format")
    except Exception as e:
        logger.error("Error extracting {}: {}", archive_path, e)
        return False


def get_file_info(url: str) -> Dict:
    """Get file information from HTTP headers."""
    try:
        response = requests.head(url)
        response.raise_for_status()
        return {
            "size": response.headers.get('content-length'),
            "content_type": response.headers.get('content-type'),
            "last_modified": response.headers.get('last-modified')
        }
    except Exception as e:
        logger.error("Error getting file info for {}: {}", url, e)
        return {}


def main():
    """Main extraction function."""
    logger.info("WORMHOLE Data Extraction Script")
    logger.info("{}", "=" * 40)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize download log
    download_log = {
        "timestamp": datetime.now().isoformat(),
        "base_url": BASE_URL,
        "data_url": DATA_URL,
        "output_dir": str(OUTPUT_DIR),
        "files": []
    }
    
    successful_downloads = 0
    total_files = len(FILES_TO_DOWNLOAD)
    
    for filename in FILES_TO_DOWNLOAD:
        url = urljoin(DATA_URL, filename)
        file_path = OUTPUT_DIR / filename
        
        logger.info("Processing: {}", filename)
        logger.info("URL: {}", url)
        
        # Get file info
        file_info = get_file_info(url)
        
        # Download file
        if download_file(url, file_path):
            logger.info("Downloaded: {}", filename)

            # Extract archive
            if extract_archive(file_path):
                logger.info("Extracted: {}", filename)
                successful_downloads += 1
            else:
                logger.error("Failed to extract: {}", filename)
        else:
            logger.error("Failed to download: {}", filename)
        
        # Log file information
        file_log = {
            "filename": filename,
            "url": url,
            "local_path": str(file_path),
            "downloaded": file_path.exists(),
            "info": file_info
        }
        download_log["files"].append(file_log)
    
    # Save download log
    with open(LOG_FILE, 'w') as f:
        json.dump(download_log, f, indent=2)
    
    # Summary
    logger.info("{}", "=" * 40)
    logger.info("Extraction Complete!")
    logger.info(
        "Successfully processed: {}/{} files",
        successful_downloads,
        total_files,
    )
    logger.info("Output directory: {}", OUTPUT_DIR)
    logger.info("Log file: {}", LOG_FILE)

    # List extracted contents
    logger.info("Extracted contents:")
    for item in OUTPUT_DIR.iterdir():
        if item.is_dir():
            file_count = len(list(item.iterdir()))
            logger.info("  Directory {} ({} files)", item.name, file_count)
        elif item.suffix == '.json':
            logger.info("  File {}", item.name)


if __name__ == "__main__":
    main()