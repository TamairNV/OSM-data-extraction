import pandas as pd
import requests
import threading
from concurrent.futures import ThreadPoolExecutor

from Maps import category_zooms, target_limits, APPLE_MAPS_TOKEN

file_path = "../master_candidates_reduced.csv"
df = pd.read_csv(file_path).sample(frac=1).reset_index(drop=True)

# Thread-safe counter
download_counts = 0
counter_lock = threading.Lock()


def download_image(row):
    global download_counts
    lat, lon, row_type, spot_id = row['lat'], row['lon'], row['type'], row['id']

    matched_cat = next((cat for cat in target_limits.keys() if row_type.startswith(cat)), None)
    if not matched_cat:
        return
    url = "https://snapshot.apple-mapkit.com/api/v1/snapshot"
    params = {
        "center": f"{lat},{lon}",
        "z": category_zooms.get(matched_cat, 18),
        "size": "600x600",
        "scale": 2,
        "t": "satellite",
        "token": APPLE_MAPS_TOKEN
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            image_path = f"images/spot_{spot_id}.png"
            with open(image_path, "wb") as f:
                f.write(response.content)

            # Lock the counter so threads don't trip over each other
            with counter_lock:
                download_counts += 1
                print(f"Downloaded {matched_cat} (Total: {download_counts})")
        else:
            print(f"API Error on spot {spot_id}: Status {response.status_code}")

    except Exception as e:
        print(f"Network Error on spot {spot_id}: {e}")


# Choose how many jobs to run at once here
MAX_WORKERS = 17

# Fire up the threads
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(download_image, [row for _, row in df.iterrows()])