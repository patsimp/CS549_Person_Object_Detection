#!/usr/bin/env python3
"""
preprocess_celeba.py

Preprocesses CelebA dataset images:
1. Resizes all images to 64x64 pixels
2. Converts to RGB format (if needed)
3. Saves in consistent JPEG format
"""

import os
import sys
import random
from pathlib import Path
from PIL import Image
import concurrent.futures
from tqdm import tqdm
import argparse

# ==== CONFIGURATION ====
IMAGE_SIZE = 64  # Target size (64x64 pixels)
MAX_IMAGES = None  # Process all images by default
NUM_WORKERS = os.cpu_count()  # Use all available CPU cores

# Get the correct base directory (3 levels up from script location)
BASE_DIR = Path(__file__).resolve().parents[2]
# Raw data directory
RAW_DIR = BASE_DIR / "celeba_raw"
# Original images directory
IMG_DIR = RAW_DIR / "img_align_celeba" / "img_align_celeba"
# Processed images output directory
PROCESSED_DIR = BASE_DIR / "processed"


def normalize_images(max_images=None):
    """Preprocess and normalize images"""
    print("Preprocessing and normalizing images...")

    # Create processed directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Get all image paths
    image_paths = list(IMG_DIR.glob("*.jpg"))

    if not image_paths:
        print(f"Error: No images found in {IMG_DIR}")
        return

    print(f"Found {len(image_paths)} images")

    # Limit number of images if requested
    if max_images is not None and max_images < len(image_paths):
        print(f"Limiting to {max_images} random images")
        random.shuffle(image_paths)
        image_paths = image_paths[:max_images]

    print(f"Processing {len(image_paths)} images")

    # Process images in parallel
    successful = 0
    failed = 0

    # Create list of (source, destination) pairs
    process_pairs = []
    for src_path in image_paths:
        dest_path = PROCESSED_DIR / src_path.name
        process_pairs.append((src_path, dest_path))

    print(f"Resizing and normalizing images to {IMAGE_SIZE}x{IMAGE_SIZE} pixels...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_image, src_path, dest_path): src_path
            for src_path, dest_path in process_pairs
        }

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(process_pairs)):
            src_path = future_to_path[future]
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
                failed += 1

    print(f"Processing complete: {successful} successful, {failed} failed")
    print(f"Normalized images saved to: {PROCESSED_DIR}")


def process_image(src_path, dest_path):
    """
    Process a single image:
    - Resize to 64x64
    - Convert to RGB format
    - Save as JPEG
    """
    try:
        # Open image and convert to RGB
        img = Image.open(src_path).convert('RGB')

        # Resize to target size (64x64)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Save as JPEG
        img.save(dest_path, format='JPEG', quality=95)
        return True

    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess CelebA images')
    parser.add_argument('--max', type=int, default=None) # done to limit number of images processed in testing
    args = parser.parse_args()

    # Set max images from command line if provided
    max_images = args.max or MAX_IMAGES

    print(f"Base directory: {BASE_DIR}")
    print(f"Looking for images in: {IMG_DIR}")

    # Create processed directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Check if images directory exists
    if not IMG_DIR.exists():
        print(f"Error: Image directory not found: {IMG_DIR}")
        print("Make sure to run kaggledatadownload.py first to download the dataset.")
        return

    # Get all image paths
    image_paths = list(IMG_DIR.glob("*.jpg"))

    if not image_paths:
        print(f"Error: No images found in {IMG_DIR}")
        return

    print(f"Found {len(image_paths)} images")

    # Limit number of images if requested
    if max_images is not None and max_images < len(image_paths):
        print(f"Limiting to {max_images} random images")
        random.shuffle(image_paths)
        image_paths = image_paths[:max_images]

    print(f"Processing {len(image_paths)} images")

    # Process images in parallel
    successful = 0
    failed = 0

    normalize_images(args.max) #calls the normalize image function

    # Create list of (source, destination) pairs
    process_pairs = []
    for src_path in image_paths:
        dest_path = PROCESSED_DIR / src_path.name
        process_pairs.append((src_path, dest_path))

    print(f"Resizing images to {IMAGE_SIZE}x{IMAGE_SIZE} pixels...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_image, src_path, dest_path): src_path
            for src_path, dest_path in process_pairs
        }

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(process_pairs)):
            src_path = future_to_path[future]
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
                failed += 1

    print(f"Processing complete: {successful} successful, {failed} failed")
    print(f"Processed images saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()