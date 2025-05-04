from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='data', image_size=128, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_ds = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    test_ds = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader
