from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz

# ====== Configuration ======
CLASSES = [
    "Car", "Chair", "Bottle", "Dog", "Cat", "Cup", "Sofa", "Backpack",
    "Laptop", "Bicycle", "Keyboard", "Toilet", "Television", "Microwave oven"
]

MAX_SAMPLES = 1000  # Adjust as needed

# Output directory (relative to project root)
BASE_DIR = Path(__file__).resolve().parents[1]
EXPORT_DIR = BASE_DIR / "openimages_v7_subset"

# ====== Download from FiftyOne Zoo ======
print("📦 Downloading OpenImages V7 subset...")
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    max_samples=MAX_SAMPLES,
    label_types=["detections"],
    classes=CLASSES
)

# ====== Export to ImageDirectory format ======
dataset.export(
    export_dir=str(EXPORT_DIR),
    dataset_type=fo.types.ImageDirectory,
    label_field=None  # We're just exporting images
)

print(f"✅ Downloaded and exported to:\n{EXPORT_DIR}")
