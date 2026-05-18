import json

import osmium
import csv

# ==========================================
# SPEED CONFIGURATION
# ==========================================
OSM_FILE = "great-britain-260515.osm.pbf"
OUTPUT_CSV = "master_candidates.csv"

# OPTIMIZATION: Converted lists to Sets {} for O(1) lightning-fast lookups
BLACKLIST_BUILDING = {"house", "detached", "semidetached_house", "terrace", "apartments", "office", "retail", "hotel",
                      "garages"}
BLACKLIST_AEROWAY = {"runway", "taxiway", "helipad"}
WALKABLE_HIGHWAYS = {"footway", "path", "cycleway", "pedestrian"}
BRIDGE_TYPES = {"yes", "viaduct", "aqueduct", "boardwalk"}
ABANDONED_TAGS = {"abandoned", "ruins", "collapsed"}
NATURE_TAGS = {"cliff", "ridge", "bare_rock"}
HISTORIC_TAGS = {"ruins", "castle", "fort"}
PIER_TAGS = {"pier", "jetty"}
TOWER_TAGS = {"water_tower", "chimney", "silo"}


class DroneSpotMasterHandler(osmium.SimpleHandler):
    def __init__(self):
        super(DroneSpotMasterHandler, self).__init__()
        self.spots = []

    def save_spot(self, osm_id, spot_type, lat, lon,data):
        self.spots.append({'id': osm_id, 'type': spot_type, 'lat': lat, 'lon': lon,'data': json.dumps(dict(data))})

    def node(self, n):
        natural = n.tags.get('natural')
        man_made = n.tags.get('man_made')

        if natural == 'peak':
            self.save_spot(n.id, "Mountain Peak", n.location.lat, n.location.lon,n.tags)
        elif man_made in TOWER_TAGS:
            self.save_spot(n.id, "Tower", n.location.lat, n.location.lon,n.tags)
        elif n.tags.get('tourism') == 'artwork' and n.tags.get('artwork_type') in ['graffiti', 'street_art']:
            self.save_spot(n.id, "Urban Graffiti Spot", n.location.lat, n.location.lon,n.tags)

    def way(self, w):
        try:
            # Grab coordinates first. If it fails, bail immediately.
            lat, lon = w.nodes[0].lat, w.nodes[0].lon
        except osmium.InvalidLocationError:
            return
        building = w.tags.get('building')
        bridge = w.tags.get('bridge')
        highway = w.tags.get('highway')
        landuse = w.tags.get('landuse')
        railway = w.tags.get('railway')

        # --- FAST ABANDONED CHECK ---
        is_abandoned = False
        if building in ABANDONED_TAGS or w.tags.get('abandoned') == 'yes' or w.tags.get('disused') == 'yes':
            is_abandoned = True

        # --- FAST BLACKLIST CHECK ---
        if not is_abandoned:
            if building in BLACKLIST_BUILDING or w.tags.get('aeroway') in BLACKLIST_AEROWAY:
                return

        # 1. Walkable Bridges
        if bridge in BRIDGE_TYPES and highway in WALKABLE_HIGHWAYS:
            self.save_spot(w.id, "Walkable Bridge", lat, lon,w.tags)
            return

        # 2. Bandos / Abandoned Sites
        if is_abandoned:
            self.save_spot(w.id, "Abandoned Urban/Industrial", lat, lon,w.tags)
            return

        # 3. Disused Railways
        if railway in ['abandoned', 'disused']:
            #self.save_spot(w.id, "Abandoned Railway", lat, lon,w.tags)
            return

        # 4. Urban Brownfields
        if landuse == 'brownfield':
            self.save_spot(w.id, "Urban Brownfield", lat, lon,w.tags)
            return

        # 5. Cliffs & Terrain
        if w.tags.get('natural') in NATURE_TAGS:
            self.save_spot(w.id, "Terrain", lat, lon,w.tags)
            return

        # 6. Historic Ruins
        if w.tags.get('historic') in HISTORIC_TAGS:
            self.save_spot(w.id, "Historic Ruins", lat, lon,w.tags)
            return

        # 7. Skateparks & Parkour
        leisure = w.tags.get('leisure')
        if leisure in ['skatepark', 'parkour'] or w.tags.get('sport') == 'bmx':
            self.save_spot(w.id, "Urban Action Park", lat, lon,w.tags)
            return

        # 8. Piers & Bunkers
        if w.tags.get('man_made') in PIER_TAGS:
            self.save_spot(w.id, "Water Pier", lat, lon,w.tags)
            return
        if w.tags.get('military') == 'bunker':
            self.save_spot(w.id, "Military Bunker", lat, lon,w.tags)
            return


import subprocess
import os

if __name__ == "__main__":

    TEMP_FILE = "temp_fpv_spots.osm.pbf"

    print("Executing multithreaded C++ pre-filter (this will be fast)...")

    filter_command = [
        "osmium", "tags-filter", OSM_FILE,
        "nwr/bridge=yes,viaduct,aqueduct,boardwalk",
        "nwr/building=abandoned,ruins,collapsed",
        "nwr/abandoned=yes",
        "nwr/disused=yes",
        "nwr/railway=abandoned,disused",
        "nwr/landuse=brownfield",
        "nwr/tourism=artwork",
        "nwr/leisure=skatepark,parkour",
        "nwr/sport=bmx",
        "nwr/natural=peak,cliff,ridge,bare_rock",
        "nwr/historic=ruins,castle,fort",
        "nwr/man_made=water_tower,chimney,silo,pier,jetty",
        "nwr/military=bunker",
        "-o", TEMP_FILE,
        "--overwrite"
    ]

    # Run the C++ tool directly from Python
    subprocess.run(filter_command, check=True)
    print(f"Pre-filter complete! Created lightweight map: {TEMP_FILE}")

    # 2. THE PYTHON PARSER
    # Now we run your exact Python logic, but it only has to look at a few thousand objects
    # instead of 100 million.
    handler = DroneSpotMasterHandler()
    print("Running Python logic on the filtered data...")
    handler.apply_file(TEMP_FILE, locations=True, idx='flex_mem')

    # 3. SAVE AND CLEANUP
    print(f"Writing {len(handler.spots)} premium spots to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'type', 'lat', 'lon','data'])
        writer.writeheader()
        writer.writerows(handler.spots)

    # Delete the temporary file to keep your folder clean
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    print("Done! Execution finished in record time.")