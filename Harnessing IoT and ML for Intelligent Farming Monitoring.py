import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset

data = pd.read_csv("/content/Crop_recommendation.csv.xls")

# Display the first few rows of the dataset
print(data.head())
# Check for missing values
print(data.isnull().sum())

# Split the dataset into features and target variable
X = data.drop('label', axis=1)
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
# Function to predict the suitable crop based on sensor readings
def predict_crop(sensor_readings):
    sensor_readings = np.array(sensor_readings).reshape(1, -1)
    sensor_readings = scaler.transform(sensor_readings)
    prediction = model.predict(sensor_readings)
    return prediction[0]

# Example sensor readings: [N, P, K, temperature, humidity, ph, rainfall]
new_sensor_readings = [90, 42, 43, 20.87, 82.02, 6.5, 202.93]
predicted_crop = predict_crop(new_sensor_readings)
print(f"Predicted Crop: {predicted_crop}")
