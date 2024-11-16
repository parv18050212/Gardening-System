from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model and dataset
MODEL_PATH = 'model.joblib'  # Path to the saved model
DATASET_PATH = 'plant_care_dataset.csv'  # Path to the dataset
predictor = joblib.load(MODEL_PATH)
df = pd.read_csv(DATASET_PATH)

@app.route('/')
def index():
    # Display the first few rows of the dataset
    data_preview = df.head(10).to_html(classes='table table-striped', index=False)
    return render_template('index.html', table=data_preview)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs (only temperature and humidity)
        temperature = request.form.get('temperature')
        humidity = request.form.get('humidity')

        # Validate inputs
        if not temperature or not humidity:
            raise ValueError("Both temperature and humidity are required.")

        # Convert to float
        temperature = float(temperature)
        humidity = float(humidity)

        # Create a DataFrame with the exact columns the model expects (without plant_type)
        input_df = pd.DataFrame([{
            'temperature': temperature,
            'humidity': humidity
        }])

        # Make predictions
        predictions = predictor.predict(input_df)
        water_requirement = round(predictions[0, 0], 2)  # Water requirement
        nutrient_frequency = round(predictions[0, 1], 2)  # Nutrient frequency

        # Render results
        result = f"Water Requirement: {water_requirement} liters/week, Nutrient Frequency: {nutrient_frequency} days"
        return render_template('index.html', result=result, table=df.head(10).to_html(classes='table table-striped', index=False))

    except ValueError as e:
        # Handle invalid inputs
        return render_template(
            'index.html',
            result=f"Invalid input: {e}. Please enter valid values.",
            table=df.head(10).to_html(classes='table table-striped', index=False)
        )

    except Exception as e:
        # Handle any unexpected errors
        return render_template(
            'index.html',
            result=f"An error occurred: {e}. Please try again.",
            table=df.head(10).to_html(classes='table table-striped', index=False)
        )

if __name__ == '__main__':
    app.run(debug=True)
