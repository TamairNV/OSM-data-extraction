import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image


class HazardResNet(nn.Module):
    def __init__(self):
        super(HazardResNet, self).__init__()
        # Use ResNet18 (lightweight, fast)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features

        # Remove standard classification head
        self.backbone.fc = nn.Identity()

        # Build a SINGLE binary classification head
        self.hazard_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            # REMOVED nn.Sigmoid() !
        )

    def forward(self, x):
        # 1. Pass the image through ResNet to get the 512 features
        features = self.backbone(x)

        # 2. Pass the FEATURES into the linear head, NOT 'x'!
        hazard_score = self.hazard_head(features)

        return hazard_score


# --- 2. THE DATASET LOADER ---
class DroneSpotDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Grab images formatted like: spot-12345-1.png or spot-12345-0.png
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') and len(f.split('-')) >= 2]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # Safely parse the 1 or 0 from the end of the filename
        parts = filename.replace(".png", "").split("-")
        hazard_label = float(parts[-1])  # Will be 1.0 or 0.0

        if self.transform:
            image = self.transform(image)

        # Return a single target tensor
        return image, torch.tensor([hazard_label], dtype=torch.float32)


# --- 3. TRAINING PIPELINE ---
def train_model():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Training on device: {device}")

    # Data Augmentation (Crucial for small datasets)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = DroneSpotDataset(image_dir="dataset/training-yes-no", transform=train_transforms)

    # Split: 85% Train, 15% Validation
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    model = HazardResNet().to(device)
    # pos_weight = (Number of Zeros) / (Number of Ones)
    # 15% Zeros / 85% Ones = 0.176
    pos_weight = torch.tensor([0.176], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)


    # Training Loop
    epochs = 20
    print("Beginning training passes...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            true_hazard = targets.to(device)

            optimizer.zero_grad()
            pred_hazard = model(images)

            # Clean, native PyTorch loss calculation
            loss = criterion(pred_hazard, true_hazard)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_set)
        print(f"Epoch {epoch + 1}/{epochs} | Hazard Loss: {epoch_loss:.4f}")

    # Save the trained brain
    torch.save(model.state_dict(), "hazard_model.pth")
    print("Model saved successfully as hazard_model.pth!")


import io
import torch
from torchvision import transforms
from PIL import Image

# 1. INFERENCE TRANSFORMS (No random flips or rotations here!)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def check_hazard(image_source, model, device, threshold=0.5):
    # 2. HANDLE THE INPUT TYPE
    try:
        if isinstance(image_source, str):
            # It's a file path
            image = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, bytes):
            # It's raw API response.content
            image = Image.open(io.BytesIO(image_source)).convert("RGB")
        elif isinstance(image_source, Image.Image):
            # It's already a PIL Image
            image = image_source.convert("RGB")
        else:
            raise ValueError("Unsupported input type. Send a path, bytes, or PIL Image.")
    except Exception as e:
        print(f"Error opening image: {e}")
        return True, 1.0  # Default to hazard if the image is corrupted

    img_tensor = inference_transforms(image).unsqueeze(0).to(device)

    # 4. RUN INFERENCE
    model.eval()
    with torch.no_grad():
        raw_logits = model(img_tensor)
        probability = torch.sigmoid(raw_logits).item()

    is_hazard = probability > threshold

    return is_hazard, probability




if __name__ == "__main__":
    train_model()