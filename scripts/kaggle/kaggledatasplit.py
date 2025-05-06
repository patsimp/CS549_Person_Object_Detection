import zipfile
import random
import shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# ==== CONFIGURATION ====
DATASET_NAME = "jessicali9530/celeba-dataset"
MAX_SAMPLES  = 300 #TODO: Adjust values for actual training
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

BASE_DIR   = Path(__file__).resolve().parents[2] #traverses file heirarchy backwards
RAW_DIR    = BASE_DIR / "celeba_raw"
IMG_DIR    = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "data" 


def sample_and_split():
    images = list(IMG_DIR.glob("*.jpg"))
    print(f"🔍 Found {len(images)} total images.")

    # Shuffle & limit to MAX_SAMPLES
    random.shuffle(images)
    images = images[: min(len(images), MAX_SAMPLES)]
    print(f"✂️  Using {len(images)} images for splitting.")

    # Compute split counts
    n = len(images)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val   = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": images[:n_train],
        "val":   images[n_train : n_train + n_val],
        "test":  images[n_train + n_val :],
    }

    # Copy into data/{split}/person/
    for split, files in splits.items():
        dest = OUTPUT_DIR / split / "person"
        dest.mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy(img, dest / img.name)
        print(f"✅ {split.upper()}: copied {len(files)} → {dest.relative_to(BASE_DIR)}")


def main():
    sample_and_split()
    print("\n🎉 Done! Check your data/ folder for train/val/test splits.")

if __name__ == "__main__":
    main()
