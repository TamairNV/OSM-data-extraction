import osmium
import csv

class DroneSpotMasterHandler(osmium.SimpleHandler):
    def __init__(self):
        super(DroneSpotMasterHandler, self).__init__()
        self.spots = []

    def save_spot(self, osm_id, spot_type, lat, lon):
        self.spots.append({
            'id': osm_id,
            'type': spot_type,
            'lat': lat,
            'lon': lon
        })

    # HANDLES NODES (Single points on the map)
    def node(self, n):
        # 1. Mountain Peaks
        if 'natural' in n.tags and n.tags['natural'] == 'peak':
            name = n.tags.get('name', 'Unnamed Peak')
            self.save_spot(n.id, f"Mountain Peak ({name})", n.location.lat, n.location.lon)
            return

        # 2. Industrial Towers, Chimneys, and Silos
        if 'man_made' in n.tags and n.tags['man_made'] in ['water_tower', 'chimney', 'silo']:
            self.save_spot(n.id, f"Tower ({n.tags['man_made']})", n.location.lat, n.location.lon)
            return

    # HANDLES WAYS (Lines, boundaries, and structures)
    def way(self, w):
        # Geolocation safety check for lines/polygons
        try:
            first_node = w.nodes[0]
            lat, lon = first_node.lat, first_node.lon
        except osmium.InvalidLocationError:
            return

        # 1. Walkable Bridges
        if 'bridge' in w.tags and w.tags['bridge'] == 'yes':
            if 'highway' in w.tags and w.tags['highway'] in ['footway', 'path', 'cycleway']:
                self.save_spot(w.id, "Walkable Bridge", lat, lon)
                return

        # 2. Bandos / Abandoned Industrial Sites
        is_bando = False
        if 'building' in w.tags and w.tags['building'] in ['abandoned', 'ruins', 'collapsed']:
            is_bando = True
        for tag in w.tags:
            if tag.k.startswith('abandoned:') or tag.k.startswith('disused:'):
                is_bando = True
                break
        if 'building' in w.tags and w.tags['building'] in ['industrial', 'warehouse', 'manufactory']:
            if 'abandoned' in w.tags and w.tags['abandoned'] == 'yes':
                is_bando = True
            elif 'disused' in w.tags and w.tags['disused'] == 'yes':
                is_bando = True
        if is_bando:
            self.save_spot(w.id, "Abandoned Industrial", lat, lon)
            return

        # 3. Cliffs, Ridges, and Massive Rock Formations
        if 'natural' in w.tags and w.tags['natural'] in ['cliff', 'ridge', 'bare_rock']:
            feature_name = w.tags.get('name', w.tags['natural'].capitalize())
            self.save_spot(w.id, f"Terrain ({feature_name})", lat, lon)
            return

        # 4. Historic Ruins, Castles, and Forts
        if 'historic' in w.tags and w.tags['historic'] in ['ruins', 'castle', 'fort']:
            self.save_spot(w.id, "Historic Ruins", lat, lon)
            return

        # 5. Skateparks & BMX Tracks (Micro playgrounds)
        if ('leisure' in w.tags and w.tags['leisure'] == 'skatepark') or ('sport' in w.tags and w.tags['sport'] == 'bmx'):
            self.save_spot(w.id, "Skatepark / BMX Track", lat, lon)
            return

        # 6. Concrete Piers and Jetties
        if 'man_made' in w.tags and w.tags['man_made'] in ['pier', 'jetty']:
            self.save_spot(w.id, "Water Pier", lat, lon)
            return

        # 7. WWII Bunkers & Pillboxes
        if 'military' in w.tags and w.tags['military'] == 'bunker':
            self.save_spot(w.id, "Military Bunker", lat, lon)
            return

# --- EXECUTION PIPELINE ---
handler = DroneSpotMasterHandler()

print("Scanning OSM file... This may take a minute for large regions.")
# 'locations=True' is absolutely mandatory to decode coordinates for ways
handler.apply_file("great-britain-260515.osm.pbf", locations=True)

# Write results directly to your dataset CSV
with open('master_candidates.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'type', 'lat', 'lon'])
    writer.writeheader()
    writer.writerows(handler.spots)

print(f"Success! Found {len(handler.spots)} diverse drone spots and saved to master_candidates.csv")