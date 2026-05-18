import requests
import json
import dotenv
dotenv.load_dotenv()
import os
OPENAIP_API_KEY = os.getenv("OPENAIP_KEY")

import requests
import json


def fetch_drone_safe_airspaces(country_code="GB"):
    url = "https://api.core.openaip.net/api/airspaces"
    headers = {"x-openaip-api-key": OPENAIP_API_KEY, "Accept": "application/json"}
    params = {"country": country_code, "limit": 100, "page": 1}

    drone_hazards = []
    print(f"Connecting to OpenAIP for country: {country_code}...")

    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        airspaces = data.get("items", [])

        if not airspaces:
            break  # No more pages

        for space in airspaces:
            # IGNORING: CTRs (4), TMZs (5), RMZs (6), and Gliders (28)
            # These are the massive city-wide blocks that don't apply to FPV drones
            if space.get("type") in [4, 5, 6, 28]:
                continue

            lower_limit = space.get("lowerLimit", {})

            # Keep it only if the restriction touches the ground (0)
            if lower_limit.get("value") == 0:
                drone_hazards.append({
                    "name": space.get("name", "Unknown Airspace"),
                    "type": space.get("type"),
                    "geometry": space.get("geometry", {})
                })

        print(f"Processed page {params['page']} | Found {len(drone_hazards)} ground hazards so far...")
        params["page"] += 1

    print(f"\nSuccessfully compiled {len(drone_hazards)} total surface-level hazards.")
    return drone_hazards
# --- Quick Test Run ---
#ground_restrictions = fetch_drone_safe_airspaces("GB")