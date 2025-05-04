from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Define the 128×128 preprocessing transforms
preprocess_128 = transforms.Compose([
    transforms.Resize((128, 128)),      # upscale from 28×28 → 128×128
    transforms.ToTensor(),              # [0,255] PIL → [0.0,1.0] Tensor, shape [1,H,W]
    transforms.Normalize((0.5,), (0.5,))# center at 0: mean=0.5, std=0.5 for single channel
])

# 2. Load FashionMNIST with these transforms
train_dataset = datasets.FashionMNIST(
    root="data/",
    train=True,
    download=True,
    transform=preprocess_128
)
test_dataset = datasets.FashionMNIST(
    root="data/",
    train=False,
    download=True,
    transform=preprocess_128
)

# 3. Create DataLoaders
batch_size = 32
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

transforms.Compose([
  transforms.Resize((128,128)),
  transforms.ToTensor(),                       # -> [0.0,1.0]
  transforms.Normalize((0.5,), (0.5,))         # -> roughly [−1.0, +1.0]
])


# 4. (Optional) Quick sanity check: visualize a batch
from multiprocessing import freeze_support
import torch
from torch.utils.data import DataLoader


# … your dataset and transform imports …

def main():
    # build datasets/transforms...
    train_loader = DataLoader(train_dataset, batch_size=32,
                              shuffle=True, num_workers=4, pin_memory=True)

    # now it’s safe to fetch a batch
    imgs, labels = next(iter(train_loader))
    print(imgs.shape, labels.shape)


if __name__ == "__main__":
    freeze_support()  # on Windows, this enables spawn safely
    main()

