import pandas as pd
import os
import requests
import time
import dotenv
dotenv.load_dotenv()
df = pd.read_csv("master_candidates.csv")
APPLE_MAPS_TOKEN = os.getenv("APPLE_MAPS_TOKEN")

# Create a clean folder just for the booster images
os.makedirs("dataset/booster", exist_ok=True)

# Filter for exactly what you need
# Swap out the old 'bandos' line for this:
bandos = df[df['type'].isin(['Historic Ruins', 'Military Bunker'])].head(50)

# Combine them into a single target list of 80 images
booster_targets = pd.concat([bandos])

print(f"Downloading {len(booster_targets)} booster images...")

for index, row in booster_targets.iterrows():
    spot_id = row['id']
    url = "https://snapshot.apple-mapkit.com/api/v1/snapshot"

    # Set zoom: 18 for bandos, 16 for wide terrain
    zoom = 16 if "Terrain" in row['type'] or "Mountain" in row['type'] else 18

    params = {
        "center": f"{row['lat']},{row['lon']}",
        "z": zoom,
        "size": "600x600",
        "scale": 2,
        "t": "satellite",
        "token": APPLE_MAPS_TOKEN
    }

    try:
        res = requests.get(url, params=params)
        if res.status_code == 200:
            with open(f"dataset/booster/spot_{spot_id}.png", "wb") as f:
                f.write(res.content)
            print(f"Downloaded booster: {row['type']} (ID: {spot_id})")
        time.sleep(0.5)
    except Exception as e:
        print(f"Error: {e}")