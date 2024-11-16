import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
file_path = 'plant_care_dataset_with_predictions.csv'
plant_data = pd.read_csv(file_path)

# Remove the 'plant_type' column
plant_data = plant_data.drop(columns=['plant_type'])

# Split data into features (X) and target (y)
X = plant_data[['temperature', 'humidity']]
y = plant_data[['water_requirement', 'nutrient_frequency']]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(random_state=42, n_estimators=100)

# Create and train the model
pipeline = Pipeline(steps=[('model', model)])
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (Watering, Nutrients):", mae)
print("R² Score:", r2)

# Save the trained model
joblib.dump(pipeline, 'plant_care_model_no_plant_type.joblib')
