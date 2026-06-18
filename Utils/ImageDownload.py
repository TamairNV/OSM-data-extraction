import io
import os
import time
import pandas as pd
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from Maps import category_zooms, target_limits, APPLE_MAPS_TOKEN

file_path = "balenced5000.csv"
TRACKER_FILE = "processed_ids.txt"
DAILY_LIMIT = 24900
MAX_WORKERS = 20

counter_lock = threading.Lock()
download_counts = 0


def download_image(row):
    global download_counts
    lat, lon, row_type, spot_id = row['lat'], row['lon'], row['type'], row['id']

    matched_cat = next((cat for cat in target_limits.keys() if row_type.startswith(cat)), None)


    url = "https://snapshot.apple-mapkit.com/api/v1/snapshot"
    params = {
        "center": f"{lat},{lon}",
        "z": category_zooms.get(matched_cat, 15),
        "size": "448x448",
        "scale": 2,
        "t": "satellite",
        "token": APPLE_MAPS_TOKEN
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            # Ensure directory exists
            os.makedirs("images", exist_ok=True)
            image_path = f"new_images_2/spot_{spot_id}.jpeg"
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image.save(image_path, "JPEG", quality=85)


            with counter_lock:
                download_counts += 1
                # Save the ID to the tracker immediately
                with open(TRACKER_FILE, "a") as f:
                    f.write(f"{spot_id}\n")
                print(f"Downloaded {matched_cat} (Today's Total: {download_counts})")
        else:
            print(f"API Error on spot {spot_id}: Status {response.status_code}")


    except Exception as e:
        print(f"Network Error on spot {spot_id}: {e}")


if __name__ == "__main__":
    print("\n--- Starting Daily Batch ---")

    # Load list
    df = pd.read_csv(file_path).sample(frac=1, random_state=42).reset_index(drop=True)

    # Check already downloaded
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            processed_ids = set(int(line.strip()) for line in f if line.strip())
    else:
        processed_ids = set()

    df_remaining = df[~df['id'].isin(processed_ids)]
    print(f"Total left to process: {len(df_remaining)}")

    if len(df_remaining) == 0:
        print("All spots downloaded! We are done here.")
        exit()


    df_today = df_remaining.head(DAILY_LIMIT)
    print(f"Queuing {len(df_today)} spots for this run.")
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            list(executor.map(download_image, [row for _, row in df_today.iterrows()]))

        print("\nBatch complete! The script will now exit safely.")

    except KeyboardInterrupt:
        print("\nEmergency Stop Triggered! Shutting down threads safely...")
        print(f"Progress saved in {TRACKER_FILE}. You can resume anytime.")