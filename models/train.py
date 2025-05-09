import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import time
from tqdm import tqdm

# Import the model
from face_detection import FaceDetectionCNN

# Configuration
batch_size = 32
epochs = 10
learning_rate = 0.001
image_size = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the base directory (project root)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# Create a custom dataset
class FaceDetectionDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):

        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Collect person images (label 1)
        person_dir = self.data_dir / 'person'
        if person_dir.exists():
            person_images = list(person_dir.glob('*.jpg'))
            self.image_paths.extend(person_images)
            self.labels.extend([1] * len(person_images))

        # Collect object images (label 0)
        object_dir = self.data_dir / 'object'
        if object_dir.exists():
            object_images = list(object_dir.glob('*.jpg'))
            self.image_paths.extend(object_images)
            self.labels.extend([0] * len(object_images))

        # Shuffle the dataset
        data = list(zip(self.image_paths, self.labels))
        random.shuffle(data)
        self.image_paths, self.labels = zip(*data)

        print(f"Loaded {split} dataset: {len(self.image_paths)} images "
              f"({self.labels.count(1)} person, {self.labels.count(0)} object)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and convert image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32).unsqueeze(0)


# Training function
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler=None):
    # For tracking training progress
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0

    # Start time
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        for inputs, labels in train_pbar:
            # Move tensors to the configured device
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

        # Calculate average training loss and accuracy
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                val_pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

        # Calculate average validation loss and accuracy
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        # Update learning rate if scheduler is provided
        if scheduler:
            scheduler.step(epoch_val_loss)

        # Print epoch statistics
        print(f'Epoch {epoch + 1}/{num_epochs} - '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')

        # Save the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
            }, MODELS_DIR / 'best_model.pth')
            print(f'Saved best model with validation accuracy: {epoch_val_acc:.4f}')

    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s')

    # Return training history
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies,
        'best_val_acc': best_val_acc
    }


# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'training_history.png')
    plt.show()


def main():

    # Print configuration
    print(f'Device: {DEVICE}')
    print(f'Epochs: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {learning_rate}')

    # Data transformations
    # For training: data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # For validation/testing: only normalization
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = FaceDetectionDataset(DATA_DIR, 'train', transform=train_transform)
    val_dataset = FaceDetectionDataset(DATA_DIR, 'val', transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = FaceDetectionCNN().to(DEVICE)
    print(model)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    # Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Plot training history
    plot_training_history(history)

    print(f'Best validation accuracy: {history["best_val_acc"]:.4f}')
    print(f'Model saved to: {MODELS_DIR / "best_model.pth"}')


if __name__ == "__main__":
    main()