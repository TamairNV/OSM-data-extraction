import csv
import math

import pandas as pd
import time
import os
import requests
import dotenv

import geopandas as gpd
from shapely.geometry import Point

from getAirSpaceData import fetch_drone_safe_airspaces


def get_1km_bucket(lat, lon):
    # Divide by the degree-to-kilometer ratio
    lat_bucket = int(lat / 0.009)
    lon_bucket = int(lon / 0.015)

    # Returns a unique string key like "5922_231"
    return f"{lat_bucket}_{lon_bucket}"


def get_200m_bucket(lat, lon):
    # Use math.floor to prevent the Prime Meridian overlapping bug!
    lat_bucket = math.floor(float(lat) / 0.0018)
    lon_bucket = math.floor(float(lon) / 0.003)
    return f"{lat_bucket}_{lon_bucket}"

    return f"{lat_bucket}_{lon_bucket}"

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape


class Reducers:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        raw_hazards = fetch_drone_safe_airspaces("GB")

        for hazard in raw_hazards:
            hazard["geometry"] = shape(hazard["geometry"])
        self.air_space_hazards = gpd.GeoDataFrame(raw_hazards, geometry="geometry")

        self.new_spots = []
        self.new_spot_map = {}

        self.air_space_reduced = 0
        self.proximity_reduced = 0

    def check_airspace(self, lat, lon):

        if self.air_space_hazards.empty:
            return True

        point = Point(lon, lat)
        intersecting_zones = self.air_space_hazards[self.air_space_hazards.geometry.contains(point)]

        if not intersecting_zones.empty:
            self.air_space_reduced += 1
            return False

        return True

    def save_spot(self, osm_id, spot_type, lat, lon):
        self.spots.append({
            'id': osm_id,
            'type': spot_type,
            'lat': lat,
            'lon': lon
        })

    import math



    def proximity_check(self, spot):
        premium_spots = [
            "Abandoned Urban/Industrial", "Urban Action Park",
            "Historic Ruins",
            "Urban Brownfield", "Urban Graffiti Spot"
        ]
        if spot['type'] in premium_spots:
            return True

        spot_key = get_1km_bucket(spot['lat'], spot['lon'])

        if spot_key not in self.new_spot_map:
            self.new_spot_map[spot_key] = 1
            return True
        else:
            self.new_spot_map[spot_key] += 1

            if self.new_spot_map[spot_key] > 2:
                self.proximity_reduced += 1
                return False
            else:
                return True


    def reducer_spots(self,keep_tags=False):
        for index, row in self.df.iterrows():
            lat = row['lat']
            lon = row['lon']

            if self.proximity_check(row):
                if self.check_airspace(lat, lon):

                    if keep_tags:
                        self.new_spots.append({
                            'id': row['id'],
                            'type': row['type'],
                            'lat': lat,
                            'lon': lon,
                        "data" : row['data']})
                    else:
                        self.new_spots.append({
                            'id': row['id'],
                            'type': row['type'],
                            'lat': lat,
                            'lon': lon})

        with open('master_candidates_reduced.csv', 'w', newline='', encoding='utf-8') as f:

            if keep_tags:
                writer = csv.DictWriter(f, fieldnames=['id', 'type', 'lat', 'lon','data'])
            else:
                writer = csv.DictWriter(f, fieldnames=['id', 'type', 'lat', 'lon'])
            writer.writeheader()
            writer.writerows(self.new_spots)



if __name__ == "__main__":
    reducer = Reducers("master_candidates.csv")

    reducer.reducer_spots(keep_tags=False)
    print(f"Spots Reduced from air space {reducer.air_space_reduced}")
    print(f"Spots Reduced from proximity {reducer.proximity_reduced}")





