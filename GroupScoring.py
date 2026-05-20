import csv
import json
from pyproj import Transformer
import pandas as pd
from reduce_candidates import get_2km_bucket
location_grid = {}

df = pd.read_csv("master_candidates_reduced.csv")
df = df.sample(frac=1).reset_index(drop=True)


def get_lat_lon_from_2k_bucket(lat_bucket, lon_bucket):
    # Adding 0.5 gets you the exact center of the bucket
    lat = (float(lat_bucket) + 0.5) * 0.018
    lon = (float(lon_bucket) + 0.5) * 0.030

    return {"lat" : lat, "lon" : lon}




def create_spot(id,type,lat,lon,data):
    return {'id' : id, 'type' : type, 'lat' : lat, 'lon' : lon, 'data' : data}

for index, row in df.iterrows():
    spot_id = row['id']
    row_type = row['type']
    lat = row['lat']
    lon = row['lon']
    data = row['data']

    bucket = get_2km_bucket(lat, lon)
    tags = json.loads(data)
    if location_grid.get(bucket) is None:
        location_grid[bucket] = [create_spot(spot_id, row_type, lat, lon, tags)]
    else:
        location_grid[bucket].append(create_spot(spot_id, row_type, lat, lon, tags))

location_location_tags = {}
for location in location_grid:
    location_location_tags[location] = []

    for i in location_grid[location]:
        if i['data'] not in location_location_tags[location]:
            location_location_tags[location].append(i['data'])

best = max(location_location_tags.keys(), key=lambda x: len(location_location_tags[x]))
print(location_location_tags[best])
print(len(location_location_tags[best]))




cords = [
    get_lat_lon_from_2k_bucket(i.split('_')[0], i.split('_')[1])
    for i in location_location_tags
    if len(location_location_tags[i][0]) > 15
]
with open("test.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['lat', 'lon'])
    writer.writeheader()
    writer.writerows(cords)


