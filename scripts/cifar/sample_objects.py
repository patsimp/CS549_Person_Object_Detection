import os
import random
from pathlib import Path
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image

#TODO: Resize to 64?
def save_images(dataset, dest_dir, num_samples, resize=(218, 218)):
    os.makedirs(dest_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), num_samples)
    for i, idx in enumerate(indices):
        img, _ = dataset[idx]  # ignore label; treat all as generic "object"
        img = transforms.Resize(resize)(img)  # resize to match CelebA
        img.save(os.path.join(dest_dir, f"object_{i}.jpg"))
    print(f"✅ Saved {num_samples} images to {dest_dir}")

# Load CIFAR-10 datasets
train_set = CIFAR10(root='cifar_data', train=True, download=True)
test_set  = CIFAR10(root='cifar_data', train=False, download=True)

BASE_DIR   = Path(__file__).resolve().parents[2]

# Save object images to match your face data split
TRAIN_OBJ_DIR = BASE_DIR / "data" / "train" / "object"
VAL_OBJ_DIR = BASE_DIR / "data" / "val" / "object"
TEST_OBJ_DIR = BASE_DIR / "data" / "test" / "object"

# Make sure the directories exist
TRAIN_OBJ_DIR.mkdir(parents=True, exist_ok=True)
VAL_OBJ_DIR.mkdir(parents=True, exist_ok=True)
TEST_OBJ_DIR.mkdir(parents=True, exist_ok=True)

# Save object images to match your face data split
save_images(train_set, TRAIN_OBJ_DIR, 200)  # 2000
save_images(train_set, VAL_OBJ_DIR, 50)  # 500
save_images(test_set, TEST_OBJ_DIR, 50)  # 500

print("🎉 CIFAR-10 object images downloaded, resized, and saved.")
