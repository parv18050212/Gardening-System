# Plant Care Predictor

An intelligent system that provides personalized plant care recommendations based on environmental conditions.

## Features

- Predicts water requirements based on:
  - Plant type
  - Temperature
  - Humidity levels
- Recommends nutrient application frequency
- Adjusts recommendations based on environmental conditions
- Supports multiple plant types (tomatoes, lettuce, herbs)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python app.py
```

3. Open `index.html` in a web browser to use the application

## Supported Plant Types

- Tomato
- Lettuce
- Herbs
- Other (uses default parameters)

## Technical Details

- Frontend: HTML, CSS, JavaScript
- Backend: Flask, Python
- ML Model: Custom plant care predictor using scikit-learn base classes
- Data Processing: NumPy