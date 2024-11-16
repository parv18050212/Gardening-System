import numpy as np
import pandas as pd
from dataset import create_plant_care_dataset
from ml1 import PlantCarePredictor


data = create_plant_care_dataset()

df = pd.DataFrame(data, columns=['plant_type', 'temperature', 'humidity'])
df = df.astype({'temperature': float, 'humidity': float})

df.to_csv('plant_care_dataset.csv', index=False)


predictor = PlantCarePredictor()

predictions = predictor.predict(df.values)


df['water_requirement'] = predictions[:, 0]
df['nutrient_frequency'] = predictions[:, 1]

# Save final dataset with predictions
df.to_csv('plant_care_dataset_with_predictions.csv', index=False)

print("Dataset generated and saved to 'plant_care_dataset.csv'")
print("Predictions added and saved to 'plant_care_dataset_with_predictions.csv'")