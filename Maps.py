import pandas as pd
import time
import os
import requests
import dotenv
dotenv.load_dotenv()
# Load the new master dataset
df = pd.read_csv("master_candidates.csv")

# CHOPPER TRICK: Shuffle the entire dataframe randomly.
# This ensures you get a beautiful geographic mix from all over the UK instantly,
# so you don't need to manually calculate skips anymore.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define exactly how many of each spot type you want in your final training batch.
# You can tweak these limits to get the exact balance you want!
target_limits = {
    "Walkable Bridge": 400,
    "Abandoned Industrial": 400,
    "Historic Ruins": 150,
    "Terrain": 150,  # Catches cliffs, ridges, rock formations
    "Mountain Peak": 100,
    "Skatepark / BMX Track": 50,
    "Water Pier": 50,
    "Military Bunker": 50,
    "Tower": 50  # Catches chimneys, water towers, silos
}

# Custom zoom levels for each type (Lower number = wider view)
category_zooms = {
    "Walkable Bridge": 18,         # Tight close-up
    "Abandoned Industrial": 18,    # Tight close-up
    "Historic Ruins": 17,          # Medium view to see the layout
    "Terrain": 16,                 # Zoomed out to see cliff faces and shadows
    "Mountain Peak": 15,           # Wide view to see valleys and ridges
    "Skatepark / BMX Track": 18,
    "Water Pier": 17,
    "Military Bunker": 18,
    "Tower": 18
}

# Keep track of how many we have successfully downloaded for each category
download_counts = {k: 0 for k in target_limits.keys()}

APPLE_MAPS_TOKEN = os.getenv("APPLE_MAPS_TOKEN")
os.makedirs("dataset/images", exist_ok=True)

print("Starting balanced download pipeline...")

for index, row in df.iterrows():
    spot_id = row['id']
    row_type = row['type']
    lat = row['lat']
    lon = row['lon']

    # Figure out which target bucket this row fits into using partial string matching
    # (This handles dynamic names like 'Mountain Peak (Ben Nevis)')
    matched_category = None
    for category in target_limits.keys():
        if row_type.startswith(category):
            matched_category = category
            break

    # If the spot doesn't match any of our defined categories, skip it
    if not matched_category:
        continue

    # If we have already hit our target limit for this specific category, skip it
    if download_counts[matched_category] >= target_limits[matched_category]:
        continue

    # --- API Request Setup ---
    url = "https://snapshot.apple-mapkit.com/api/v1/snapshot"
    chosen_zoom = category_zooms.get(matched_category, 18)

    params = {
        "center": f"{lat},{lon}",
        "z": chosen_zoom,  # Uses the dynamic zoom level
        "size": "600x600",  # Generates a much bigger, clearer image for your eyes
        "scale": 2,  # Keeps it high-res for Retina displays
        "t": "satellite",
        "token": APPLE_MAPS_TOKEN
    }

    try:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            # We save it with the raw ID first. You'll add the _SCALE_FLOW part during labeling!
            image_path = f"dataset/images/spot_{spot_id}.png"
            with open(image_path, "wb") as f:
                f.write(response.content)

            # Increment the counter for this specific type
            download_counts[matched_category] += 1
            print(
                f"Downloaded {matched_category} ({download_counts[matched_category]}/{target_limits[matched_category]})")
        else:
            print(f"API Error on spot {spot_id}: Status {response.status_code}")

        time.sleep(0.4)

    except Exception as e:
        print(f"Network Error on spot {spot_id}: {e}")

print("\n--- Pipeline Complete! ---")
print("Final Dataset Distribution:")
for category, count in download_counts.items():
    print(f" - {category}: {count} images downloaded")