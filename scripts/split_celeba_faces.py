import os
import random
import shutil
from pathlib import Path

# CONFIG
source_dir = "celeba/img_align_celeba"
dest_base = "data"
splits = {"train": 0.7, "val": 0.15, "test": 0.15}
num_samples = 3000  # total images to sample

# SETUP
random.seed(42)
all_images = os.listdir(source_dir)
sampled_images = random.sample(all_images, num_samples)

split_counts = {
    k: int(v * num_samples) for k, v in splits.items()
}
split_counts["train"] += num_samples - sum(split_counts.values())  # fix rounding

# Ensure destination folders exist
for split in splits:
    Path(f"{dest_base}/{split}/person").mkdir(parents=True, exist_ok=True)

# Distribute images
idx = 0
for split, count in split_counts.items():
    for _ in range(count):
        filename = sampled_images[idx]
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_base, split, "person", filename)
        shutil.copy2(src, dst)
        idx += 1

print("✅ Finished sampling and copying CelebA images.")
