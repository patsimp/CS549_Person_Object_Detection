import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceDetectionCNN(nn.Module):
    """
    A 3-layer CNN for face detection with group normalization instead of batch normalization
    to support single-sample inference.
    """

    def __init__(self):
        super(FaceDetectionCNN, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=32)  # Group norm instead of batch norm
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64x64x3 -> 32x32x32

        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=64)  # Group norm
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32x32x32 -> 16x16x64

        # Third Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(num_groups=32, num_channels=128)  # Group norm
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 16x16x64 -> 8x8x128

        # Fully Connected Layers
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.gn_fc1 = nn.GroupNorm(num_groups=32, num_channels=256)  # Group norm for FC layer
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)  # Binary classification

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))

        # Block 2
        x = self.pool2(F.relu(self.gn2(self.conv2(x))))

        # Block 3
        x = self.pool3(F.relu(self.gn3(self.conv3(x))))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers with regularization
        x = self.fc1(x)
        # Reshape for group norm which expects 2D input [N, C]
        batch_size = x.size(0)
        x = x.view(batch_size, 256, 1, 1)
        x = self.gn_fc1(x)
        x = x.view(batch_size, 256)
        x = F.relu(x)
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