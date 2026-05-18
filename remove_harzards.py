import os

import pandas as pd
import torch

from predict import MultiTaskResNet

"""
file_path = "../master_candidates_reduced.csv"

df = pd.read_csv(file_path).sample(frac=1).reset_index(drop=True)
for index, row in df.iterrows():
    spot_id = row['id']
    row_type = row['type']
    lat = row['lat']
    lon = row['lon']

"""

device = torch.device("mps"if torch.backends.mps.is_available()else "cuda"if torch.cuda.is_available()else "cpu")
print(f"Using device: {device}")

from YesNoTrainer import check_hazard, HazardResNet


class harzardRemover:
    def __init__(self):
        self.model = HazardResNet()
        self.model.load_state_dict(
            torch.load("hazard_model.pth", map_location=device)
        )
        self.model.to(device)
        self.model.eval()

    def check_hazard_path(self, path):
        return check_hazard(path, self.model,device,0.35)


h = harzardRemover()


filenames = [f for f in os.listdir("Utils/images") if f.endswith('.png')]
rejected_spots = 0
i = 0
print(len(filenames))
import os
import shutil

# 1. Define your paths and make sure the rejected folder actually exists
base_dir = "Utils/images"
rejected_dir = os.path.join(base_dir, "rejected")
os.makedirs(rejected_dir, exist_ok=True)

for filename in filenames:
    file_path = os.path.join(base_dir, filename)
    out = h.check_hazard_path(file_path)

    # If it's a hazard (assuming out[0] == True means hazard, adjust if needed)
    if not out[0]:
        rejected_spots += 1

        # 2. Move the file into the rejected folder
        destination_path = os.path.join(rejected_dir, filename)
        shutil.move(file_path, destination_path)

    i += 1

    if i % 100 == 0:
        print(f"Processed: {i} | Rejected so far: {rejected_spots}")

print(f"Total rejected spots: {rejected_spots}")
print(f"Total Spots {len(filenames)}")
print(f"Proportion of rejected spots: {rejected_spots / len(filenames)}")

