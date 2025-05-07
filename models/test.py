import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np

from face_detection import FaceDetectionCNN
from train import FaceDetectionDataset

# Paths
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
OUTPUT_PATH = BASE_DIR / "models" / "test_outputs.npz"

# Transforms
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load test data
test_dataset = FaceDetectionDataset(DATA_DIR, 'test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load model
model = FaceDetectionCNN().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds = (outputs >= 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Save predictions and labels to file
np.savez(OUTPUT_PATH, preds=np.array(all_preds).flatten(), labels=np.array(all_labels).flatten())
print(f"✅ Saved predictions and labels to {OUTPUT_PATH}")
