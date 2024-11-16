import numpy as np
from sklearn.model_selection import train_test_split

# Dataset for ml.py - Simple linear regression
def create_linear_regression_dataset():
    """
    Creates a synthetic dataset for linear regression with one feature
    Returns X (features) and y (target) arrays
    """
    np.random.seed(42)  # for reproducibility
    X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
    y = 4 + 3 * X + np.random.randn(100, 1)  # true relationship: y = 4 + 3x + noise
    return X, y

# Dataset for ml1.py - Plant care prediction
def create_plant_care_dataset():
    """
    Creates a realistic dataset for plant care prediction
    Returns X array with [plant_type, temperature, humidity] for each sample
    """
    np.random.seed(42)
    
    # Define possible plant types
    plant_types = ['tomato', 'lettuce', 'herbs']
    
    # Create 1000 samples
    n_samples = 1000
    data = []
    
    for _ in range(n_samples):
        # Randomly select plant type
        plant_type = np.random.choice(plant_types)
        
        # Generate realistic temperature (15-35°C)
        temperature = np.random.uniform(15, 35)
        
        # Generate realistic humidity (30-90%)
        humidity = np.random.uniform(30, 90)
        
        data.append([plant_type, temperature, humidity])
    
    return np.array(data)

# Example usage:
if __name__ == "__main__":
    # Linear regression dataset
    X_linear, y_linear = create_linear_regression_dataset()
    print("Linear Regression Dataset Shape:", X_linear.shape, y_linear.shape)
    
    # Plant care dataset
    X_plant = create_plant_care_dataset()
    print("Plant Care Dataset Shape:", X_plant.shape)
    print("Sample data point:", X_plant[0])