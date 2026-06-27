import pandas as pd
import json
import random

df = pd.read_csv('labeled_images.csv')

IMAGE_Path = 'new_images_2'

PROMPT = """
You are a critical, highly selective FPV drone pilot vetting satellite imagery for elite freestyle and cinematic flight locations. 
Most locations are boring and should receive very low scores. Reserve high scores (0.8+) ONLY for exceptional, highly clear structural features or scenic landmarks.

Analyze the provided top-down satellite image alongside its OpenStreetMap data. 
CRITICAL DEPTH CUES: Top-down imagery hides elevation. You MUST actively look for high-contrast shadow lines, deep carved riverbeds, gorges, and jagged rock textures. These indicate steep, diveable cliffs.
CRITICAL HUMAN CUES: Tiny colored dots on beaches or in parks are people. Rectangles near buildings or roads are cars. 

### SCORING CALIBRATION RUBRIC (Apply strictly):

1. `freestyle_rating` (0.0 to 1.0):
   - 0.0 to 0.3: Open grass fields, standard parks, flat terrain, basic green fields.
   - 0.4 to 0.7: Moderate tree canopies, active industrial parks with flat rooftops, standard multi-story buildings.
   - 0.7 to 1.0: Abandoned structures (bandos), multi-level concrete ruins, tight architectural gaps, deep carved riverbeds/gorges, or isolated bridges crossing natural gaps (look for linear structures spanning dark shadowed areas).

2. `cinematic_rating` (0.0 to 1.0):
   - 0.0 to 0.2: Generic suburban roofs, flat fields, standard motorways. 
   - 0.3 to 0.6: Rolling hills (smooth textures), uniform forests, standard rivers.
   - 0.7 to 1.0: Striking geographic features, jagged mountain ridges/cliffs (heavy rock textures and long shadows), lone historical structures, epic valley views.

3. `obstacle_density` (0.0 to 1.0):
   - 0.0: Perfectly flat, empty grass/sand.
   - 0.5: Scattered trees, light suburban housing, single-lane roads.
   - 1.0: Dense structural steel, crane yards, thick forest canopy, complex ruins, dense jagged rock formations in gorges.

4. `busyness` (0.0 to 1.0):
   - 0.0: Total abandonment, remote wilderness, zero signs of human life.
   - 0.3: Very remote, maybe a lone dirt road, absolutely zero cars or people.
   - 0.6: Sparse houses, quiet rural roads, empty parks.
   - 0.8 to 1.0: ANY visible people (dots on a beach/park), parked or moving cars, dense residential neighborhoods, or active commercial buildings. If humans or their cars are there, score it high.

### OUTPUT FORMAT:
Return ONLY a raw, valid JSON object. No markdown code blocks. No pre-text or post-text.

{
  "freestyle_rating": float,
  "cinematic_rating": float,
  "obstacle_density": float,
  "busyness": float
}

### OPENSTREETMAP CONTEXT DATA:
The OpenStreetMap tags associated with this exact coordinate location are:
"""

dataList = []

for index, row in df.iterrows():

    data = {
        'image': f"{IMAGE_Path}/spot_{str(int(row['id']))}.jpeg",
        'conversations': [
            {
                'role': 'user',
                'content': PROMPT
            },
            {
                'role': 'assistant',
                'content': json.dumps({
                    'freestyle_rating' : row['freestyle_rating'],
                    'cinematic_rating': row['cinematic_rating'],
                    'obstacle_density': row['obstacle_density'],
                    'busyness': row['busyness']
                })
            }
        ]
    }

    dataList.append(data)

random.shuffle(dataList)

split = int(len(dataList) * 0.8)

trainingData = dataList[:split]
validationData = dataList[split:]

with open('training_dataSet.json', 'w') as output:
    json.dump(trainingData, output)
print(len(trainingData))

with open('validation_dataSet.json', 'w') as output:
    json.dump(validationData, output)
print(len(validationData))

