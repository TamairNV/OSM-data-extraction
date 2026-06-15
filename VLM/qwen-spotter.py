import csv
import math
import os
import time
from PIL import Image
import ollama
import json
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class QwenSpotter:
    def __init__(self, image_dir,prompt,spots_csv_dir):
        self.image_dir = image_dir
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpeg')]
        self.prompt = prompt
        self.results = {}
        self.tags_map = {}
        self.progress = 0

        with open(spots_csv_dir, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.tags_map[row['id']] = dict(row)



    def run_image(self,filepath):
        id = filepath.split('.jpeg')[0].split('/')[-1].split('_')[-1]
        tags = self.tags_map.get(id)
        print(tags['data'])
        if tags is None:
            return None
        response = ollama.chat(
            model='qwen2.5vl:7b',
            messages=[{
                'role': 'user',
                'content': self.prompt +str(tags),
                'images': [f'{self.image_dir}/{filepath}'],
            }]
        )
        self.progress +=1
        return [response.message.content,id]



    def run_qwen(self, start=0, end=99999999):
        targets = self.filenames[start:end]

        with ThreadPoolExecutor(max_workers=5) as executor:
            generator = executor.map(self.run_image, targets)

            self.results = list(
                tqdm(generator, total=len(targets), desc="Processing images")
            )

        return self.pass_results()

    def pass_results(self):
        new_results = []
        for result in self.results:

            try:
                data = json.loads(result[0])

                data['id'] = result[1]
                new_results.append(data)
                print("Parsed Dictionary:", data)
            except Exception as e:
                print(f"Error parsing JSON: {e}")
        return new_results




prompt = """
You are a critical, highly selective FPV drone pilot vetting satellite imagery for elite freestyle and cinematic flight locations. 
Most locations are boring and should receive very low scores. Reserve high scores (0.8+) ONLY for exceptional, highly clear structural features or scenic landmarks.

Analyze the provided top-down satellite image alongside its OpenStreetMap data.

### SCORING CALIBRATION RUBRIC (Apply strictly):

1. `freestyle_rating` (0.0 to 1.0):
   - 0.0 to 0.2: Open grass fields, standard parks, flat terrain, or tiny footbridges. (A basic green field with a minor bridge is a 0.1).
   - 0.3 to 0.6: Moderate tree canopies, active industrial parks with flat rooftops, standard multi-story buildings.
   - 0.7 to 1.0: Abandoned structures (bandos) with exposed rafters, complex multi-level concrete ruins, massive multi-tier bridge structures, tight architectural gaps.

2. `cinematic_rating` (0.0 to 1.0):
   - 0.0 to 0.2: Generic suburban roofs, flat fields, standard motorways. 
   - 0.3 to 0.6: Rolling hills, uniform forests, standard rivers/canals.
   - 0.7 to 1.0: Striking geographic features, lone historical structures, dramatic bridges over water, epic valley views.

3. `obstacle_density` (0.0 to 1.0):
   - 0.0: Perfectly flat, empty grass field.
   - 0.5: Scattered trees, light suburban housing, single-lane roads.
   - 1.0: Dense structural steel, crane yards, thick forest canopy, massive scaffolding matrix.

4. `busyness` (0.0 to 1.0):
   - 0.0: Total abandonment, overgrown concrete, zero parked cars, remote wilderness.
   - 0.5: Active warehouse docks (some trucks), quiet residential areas, public parks with walking paths.
   - 1.0: Active highways, packed parking lots, major construction zones, heavy pedestrian density.

### OUTPUT FORMAT:
You must return ONLY a raw, valid JSON object. Do not wrap it in markdown code blocks. Do not include any pre-text or post-text. 

Use this exact schema:
{{
  "freestyle_rating": float,
  "cinematic_rating": float,
  "obstacle_density": float,
  "busyness": float,
}}

### OPENSTREETMAP CONTEXT DATA:
The OpenStreetMap tags associated with this exact coordinate location are: 
"""

spotter = QwenSpotter('../Utils/images',prompt,'../master_candidates_reduced.csv')
t = time.time()
spotter.run_qwen(start = 100 ,end=110)

print(time.time()-t)

