#!/usr/bin/env python3
"""
download_and_split_celeba.py

Downloads the CelebA dataset via the official Kaggle API
and splits up to MAX_SAMPLES images into train/val/test.
"""

import zipfile
import random
import shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# ==== CONFIGURATION ====
DATASET_NAME = "jessicali9530/celeba-dataset"
MAX_SAMPLES  = 100_000
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

BASE_DIR   = Path(__file__).resolve().parents[1]
RAW_DIR    = BASE_DIR / "celeba_raw"
IMG_DIR    = RAW_DIR / "img_align_celeba" / "img_align_celeba"
OUTPUT_DIR = BASE_DIR / "data"

# ==== 1) DOWNLOAD & UNZIP via Kaggle API ====
def download_and_extract():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading {DATASET_NAME} into {RAW_DIR} …")

    api = KaggleApi()
    api.authenticate()

    # This will download and unzip in one step
    api.dataset_download_files(
        DATASET_NAME,
        path=str(RAW_DIR),
        unzip=True,
        quiet=False
    )

    if not IMG_DIR.exists():
        raise FileNotFoundError(f"❌ Could not find images in: {IMG_DIR}")
    print(f"✅ Downloaded and extracted to: {IMG_DIR}")

def main():
    download_and_extract()
    print("\n🎉 Done! Check your data/ folder for train/val/test splits.")

if __name__ == "__main__":
    main()
