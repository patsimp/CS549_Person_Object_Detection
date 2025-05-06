import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceDetectionCNN(nn.Module):
    """
    CNN for face detection (binary classification: person vs object)
    Input: 64x64x3 RGB images
    Output: Binary classification (0: object, 1: person)
    """

    def __init__(self):
        super(FaceDetectionCNN, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64x64x3 -> 32x32x32

        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32x32x32 -> 16x16x64

        # Third Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 16x16x64 -> 8x8x128

        # Fourth Convolutional Block
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 8x8x128 -> 4x4x256

        # Fully Connected Layers
        self.fc1 = nn.Linear(4 * 4 * 256, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)  # Binary classification

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Apply sigmoid for binary classification
        return torch.sigmoid(x)  # Output in range (0,1)


# Test the model with a random input
if __name__ == "__main__":
    # Create random input tensor (batch_size, channels, height, width)
    x = torch.randn(1, 3, 64, 64)

    # Initialize the model
    model = FaceDetectionCNN()

    # Print model summary
    print(model)

    # Forward pass
    output = model(x)

    # Print output shape
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.item():.4f}")  # Probability of being a person

    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")