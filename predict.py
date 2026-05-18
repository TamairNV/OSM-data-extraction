import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys


class MultiTaskResNet(nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        self.backbone = models.resnet18()
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.scale_head = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.flow_head = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        features = self.backbone(x)
        return self.scale_head(features), self.flow_head(features)


def predict_spot(model, image_path=None, raw_image=None):
    start = time.time()

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
    else:
        image = raw_image

    # LOOK HERE: Automatically detects if the model is on CPU, CUDA, or MPS
    device = next(model.parameters()).device
    input_tensor = eval_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scale, pred_flow = model(input_tensor)

    #print(f"Inference time: {time.time() - start:.4f}s")
    return {'scale': pred_scale.item(), 'flow': pred_flow.item()}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        model = MultiTaskResNet()
        model.load_state_dict(torch.load("drone_model.pth", map_location=device))
        model.to(device)
        model.eval()

        print(f"\n--- AI Drone Spot Analysis ---")
        out = predict_spot(model, image_path=sys.argv[1])
        print(f"Predicted Scale (Whoop vs 5-inch): {out['scale']:.2f}")
        print(f"Predicted Flow  (Obstacle Density): {out['flow']:.2f}")
    else:
        print("Please provide an image path. Example: python predict.py test_image.png")