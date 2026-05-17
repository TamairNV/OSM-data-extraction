import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image


# --- 1. THE ARCHITECTURE ---
class MultiTaskResNet(nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        # Use ResNet18 (lightweight, fast, perfect for smaller datasets)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features

        # Remove standard classification head
        self.backbone.fc = nn.Identity()

        # Build two separate regression heads
        self.scale_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Forces output between 0.0 and 1.0
        )

        self.flow_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Forces output between 0.0 and 1.0
        )

    def forward(self, x):
        features = self.backbone(x)
        scale = self.scale_head(features)
        flow = self.flow_head(features)
        return scale, flow


# --- 2. THE DATASET LOADER ---
class DroneSpotDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Only grab images that have been labeled with underscores
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') and len(f.split('-')) >= 3]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # Safely parse scores from the end of the filename (e.g., spot_12345_0.8_0.3.png)
        parts = filename.replace(".png", "").split("-")
        scale_score = float(parts[-2])
        flow_score = float(parts[-1])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([scale_score, flow_score], dtype=torch.float32)


# --- 3. TRAINING PIPELINE ---
def train_model():
    # Setup GPU acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Data Augmentation (Our data multiplier trick)
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
    dataset = DroneSpotDataset(image_dir="dataset/Training_Images", transform=train_transforms)

    # Split: 85% Train, 15% Validation
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # Initialize model, loss, and optimizer
    model = MultiTaskResNet().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error is perfect for continuous 0-1 values
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Training Loop
    epochs = 20
    print("Beginning training passes...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            true_scale = targets[:, 0].unsqueeze(1).to(device)
            true_flow = targets[:, 1].unsqueeze(1).to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_scale, pred_flow = model(images)

            # Calculate combined loss
            loss_scale = criterion(pred_scale, true_scale)
            loss_flow = criterion(pred_flow, true_flow)
            total_loss = loss_scale + loss_flow

            # Backward pass
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_set)
        print(f"Epoch {epoch + 1}/{epochs} | Combined Loss: {epoch_loss:.4f}")

    # Save the trained brain
    torch.save(model.state_dict(), "drone_model.pth")
    print("Model saved successfully as drone_model.pth!")


if __name__ == "__main__":
    train_model()