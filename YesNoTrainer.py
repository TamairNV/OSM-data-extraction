import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from torchgeo.models import ResNet18_Weights
import timm

import timm
import torch
import torch.nn as nn
from torchgeo.models import ResNet18_Weights


class HazardSatelliteResNet(nn.Module):

    def __init__(self):
        super(HazardSatelliteResNet, self).__init__()

        # 1. Fetch the Sentinel satellite weights object
        weights = ResNet18_Weights.SENTINEL2_ALL_MOCO

        # 2. Force the model to expect 3 channels (RGB) because our images are PNGs
        in_chans = 3

        # 3. Create a blank timm resnet18
        self.backbone = timm.create_model(
            "resnet18", pretrained=False, in_chans=in_chans
        )

        # 4. Download the satellite weights dictionary
        state_dict = weights.get_state_dict(progress=True)

        # --- THE FIX: EXTRACT ONLY RGB WEIGHTS ---
        # Sentinel-2 has 13 bands. In torchgeo, the order is:
        # Index 0: B1 (Coastal Aerosol)
        # Index 1: B2 (Blue)
        # Index 2: B3 (Green)
        # Index 3: B4 (Red)
        # Our dataset loads as RGB. So we extract the Red (3), Green (2), and Blue (1) channels.
        if 'conv1.weight' in state_dict and state_dict['conv1.weight'].shape[1] == 13:
            conv1_w = state_dict['conv1.weight']
            # Reorder and slice the 13 channels down to just the 3 RGB channels
            state_dict['conv1.weight'] = conv1_w[:, [3, 2, 1], :, :]

        # Strip out the final 1000-class layer so PyTorch doesn't throw a shape mismatch
        if 'fc.weight' in state_dict:
            del state_dict['fc.weight']
        if 'fc.bias' in state_dict:
            del state_dict['fc.bias']
        # ------------------------------------------

        # Load the newly sliced 3-channel dictionary into the backbone
        self.backbone.load_state_dict(state_dict, strict=False)

        # 5. Extract features and remove standard head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 6. Your binary classification head stays exactly the same
        self.hazard_head = nn.Sequential(
            nn.Linear(num_features, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.hazard_head(features)

# --- 2. THE DATASET LOADER ---
class DroneHarzardDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') and len(f.split('-')) >= 1]
        self.filenames.extend([f for f in os.listdir(image_dir) if f.endswith('.jpeg') and len(f.split('-')) >= 1])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # Safely remove extension
        name_without_ext = os.path.splitext(filename)[0]

        parts = name_without_ext.split("-")
        hazard_label = float(parts[-1])

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
    dataset = DroneHarzardDataset(image_dir="dataset/training-yes-no", transform=train_transforms)

    # --- DYNAMICALLY CALCULATE RATIO ---
    labels = [float(os.path.splitext(f)[0].split("-")[-1]) for f in dataset.filenames]
    num_zeros = labels.count(0.0)
    num_ones = labels.count(1.0)

    pos_weight_val = num_zeros / num_ones if num_ones > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)

    print(f"Dataset Distribution -> Zeros (Safe): {num_zeros} | Ones (Hazard): {num_ones}")
    print(f"Calculated pos_weight: {pos_weight_val:.4f}")
    # ------------------------------------

    # Split: 85% Train, 15% Validation
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    model = HazardSatelliteResNet().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Training Loop
    epochs = 20
    best_val_loss = float('inf')  # Start infinitely high
    print("Beginning training passes...")

    for epoch in range(epochs):

        # --- 1. TRAINING PHASE ---
        model.train()  # Tell the model it's study time
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            true_hazard = targets.to(device)

            optimizer.zero_grad()
            pred_hazard = model(images)
            loss = criterion(pred_hazard, true_hazard)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_epoch_loss = running_loss / len(train_set)

        # --- 2. VALIDATION PHASE (Using the val_loader!) ---
        model.eval()  # Tell the model to put its pencil down (test mode)
        val_running_loss = 0.0

        with torch.no_grad():  # CRITICAL: Do not learn from the validation data!
            for val_images, val_targets in val_loader:
                val_images = val_images.to(device)
                val_true_hazard = val_targets.to(device)

                # Make a prediction
                val_pred_hazard = model(val_images)

                # Check the score
                val_loss = criterion(val_pred_hazard, val_true_hazard)
                val_running_loss += val_loss.item() * val_images.size(0)

        val_epoch_loss = val_running_loss / len(val_set)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

        # --- SAVE THE BEST MODEL ---
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), "best_hazard_resnet.pth")
            print(f"  -> Best model saved! (Val Loss: {best_val_loss:.4f})")

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
