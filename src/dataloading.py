from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# ==== Paths ====
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# ==== Image Transformations ====
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ==== Dataset & DataLoader ====
train_dataset = datasets.ImageFolder(root=DATA_DIR / "train", transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# ==== Inspect First Batch ====
images, labels = next(iter(train_loader))
print(f"✅ Image batch shape: {images.shape}")  # (B, 3, 224, 224)
print(f"✅ Labels: {labels}")
print(f"✅ Tensor dtype: {images.dtype}")
print(f"✅ Pixel range: min={images.min().item():.3f}, max={images.max().item():.3f}")
print(f"✅ Class mapping: {train_dataset.class_to_idx}")

# ==== De-normalize for Visualization ====
def denormalize(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return img_tensor * std + mean

# ==== Show Images ====
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    img = denormalize(images[i]).permute(1, 2, 0).clamp(0, 1).numpy()
    axs[i].imshow(img)
    axs[i].set_title(f"Label: {labels[i].item()}")
    axs[i].axis("off")

plt.suptitle("🔍 Preprocessed Images (Denormalized for Viewing)")
plt.tight_layout()
plt.show()
