from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# ==== Paths ====
BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = BASE_DIR / "openimages_v7_subset"
DEST_DIR = BASE_DIR / "data"

TRAIN_DIR = DEST_DIR / "train" / "object"
VAL_DIR   = DEST_DIR / "val" / "object"
TEST_DIR  = DEST_DIR / "test" / "object"

# ==== Load .jpg files ====
image_files = sorted(SOURCE_DIR.glob("*.jpg"))  # Sort for consistent ordering

if not SOURCE_DIR.exists() or not image_files:
    raise RuntimeError(f"❌ No .jpg files found in {SOURCE_DIR}")

print(f"📸 Found {len(image_files)} images in source folder.")

# ==== Train/Val/Test Split ====
train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

splits = [
    ("train", train_files, TRAIN_DIR),
    ("val",   val_files,   VAL_DIR),
    ("test",  test_files,  TEST_DIR),
]

# ==== Copy images ====
def copy_images(files, target_dir: Path, label: str):
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_path in files:
        dest_path = target_dir / src_path.name
        shutil.copy2(src_path, dest_path)
        copied += 1
    print(f"✅ {label.upper()}: {copied} images copied to {target_dir.relative_to(BASE_DIR)}")

for label, filelist, target in splits:
    copy_images(filelist, target, label)

print("\n🎉 All images successfully split into train/val/test sets.")
