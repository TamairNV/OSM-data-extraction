import csv
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import torch
from PIL import Image

import predict
from Maps import *
from predict import MultiTaskResNet

device = torch.device("mps"if torch.backends.mps.is_available()else "cuda"if torch.cuda.is_available()else "cpu")
print(f"Using device: {device}")


class spotFinder:

    def __init__(self, threshold):
        self.spots = []
        self.threshold = threshold
        self.discard_count = 0
        self.processed_count = 0

        # Threading lock to keep operations safe
        self.lock = threading.Lock()
        self.csv_file = "master_spots.csv"

        # Initialize the CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["lat", "lon", "scale", "flow", "score"]
                )
                writer.writeheader()

        self.model = MultiTaskResNet()
        self.model.load_state_dict(
            torch.load("drone_model.pth", map_location=device)
        )
        self.model.to(device)
        self.model.eval()

    def save_spot_incremental(self, lat, lon, scale, flow):
        spot = {
            "lat": lat,
            "lon": lon,
            "scale": scale,
            "flow": flow,
            "score": scale + flow,
        }

        # Lock ensures only one thread writes to the shared list and CSV at a time
        with self.lock:
            self.spots.append(spot)
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["lat", "lon", "scale", "flow", "score"]
                )
                writer.writerow(spot)

    def get_image(self, spot_id, row_type, lat, lon, total_spots, start_time):
        matched_category = None
        mul = 1
        safe_type = str(row_type).lower()
        if "abandoned" in safe_type or "bando" in safe_type:
            mul = 1.5
        else:
            for category in target_limits.keys():
                if row_type.startswith(category):
                    matched_category = category
                    break

        if not matched_category:
            with self.lock:
                self.processed_count += 1
            return

        url = "https://snapshot.apple-mapkit.com/api/v1/snapshot"
        chosen_zoom = category_zooms.get(matched_category, 18)

        params = {
            "center": f"{lat},{lon}",
            "z": chosen_zoom,
            "size": "600x600",
            "scale": 2,
            "t": "satellite",
            "token": APPLE_MAPS_TOKEN,
        }

        try:
            response = requests.get(url, params=params)

            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content)).convert("RGB")

                # Inference is run inside the thread
                with torch.no_grad():
                    out = predict.predict_spot(self.model, raw_image=image)

                with self.lock:
                    self.processed_count += 1
                    current_processed = self.processed_count

                if out is not None:
                    score = out["scale"] + out["flow"]
                    if score*mul > self.threshold:
                        self.save_spot_incremental(
                            lat, lon, out["scale"], out["flow"]
                        )
                        log_msg = f"spot lat: {lat} , lon: {lon} added. Total Saved: {len(self.spots)}"
                    else:
                        with self.lock:
                            self.discard_count += 1
                            current_discard = self.discard_count
                        log_msg = f"spot lat: {lat} , lon: {lon} discarded. Total Discarded: {current_discard} {score}"
                else:
                    log_msg = f"Model returned None for spot {spot_id}"


                elapsed_time = time.time() - start_time
                avg_time_per_spot = elapsed_time / current_processed
                remaining_spots = total_spots - current_processed
                eta_seconds = remaining_spots * avg_time_per_spot

                hrs = int(eta_seconds // 3600)
                mins = int((eta_seconds % 3600) // 60)
                secs = int(eta_seconds % 60)
                eta_str = (
                    f"{hrs}h {mins}m {secs}s"
                    if hrs > 0
                    else f"{mins}m {secs}s"
                    if mins > 0
                    else f"{secs}s"
                )

                # Thread-safe console logs
                with self.lock:
                    print(log_msg)
                    print(
                        f"Progress: {current_processed}/{total_spots} | Time Elapsed: {int(elapsed_time)}s | ETA Remaining: {eta_str}"
                    )
                    print("-" * 50)
            else:
                with self.lock:
                    self.processed_count += 1
                print(
                    f"API Error on spot {spot_id}: Status {response.status_code}"
                )

        except Exception as e:
            with self.lock:
                self.processed_count += 1
            print(f"Network Error on spot {spot_id}: {e}")

    @staticmethod
    def get_all_spots():
        df = pd.read_csv("master_candidates_reduced.csv")
        df = df.sample(frac=1).reset_index(drop=True)
        finder = spotFinder(threshold=1.3)

        limit = 2500
        total_spots = min(len(df), limit)

        print(f"Starting multi-threaded scan for {total_spots} total spots...")
        start_time = time.time()

        # Execute using exactly 2 worker threads
        with ThreadPoolExecutor(max_workers=25) as executor:
            for index, row in df.iterrows():
                if index >= total_spots:
                    break

                executor.submit(
                    finder.get_image,
                    row["id"],
                    row["type"],
                    row["lat"],
                    row["lon"],
                    total_spots,
                    start_time,
                )

        print(
            f"Finished! Total execution time: {int(time.time() - start_time)} seconds. Check 'master_spots.csv' for results."
        )


if __name__ == "__main__":
    print("Starting spot finder...")
    spotFinder.get_all_spots()