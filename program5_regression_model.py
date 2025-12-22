# program5_regression_model.py
# ----------------------------------------
# Train a Linear Regression model on the MPG dataset
# Save the trained model in 'model/mpg_model.pkl'
# ----------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# -------------------------------
# Step 1: Load the cleaned dataset
# -------------------------------
data_path = "data/mpg_clean.csv"
data = pd.read_csv(data_path)

print("✅ Data loaded successfully!")
print(f"Dataset shape: {data.shape}\n")

# -------------------------------
# Step 2: Select features and target
# -------------------------------
# You can choose any features you like; here we use 4 that correlate well with MPG
X = data[['cylinders', 'displacement', 'horsepower', 'weight']]
y = data['mpg']

# -------------------------------
# Step 3: Split data into training and test sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 4: Train the model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Evaluate the model
# -------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("✅ Model Trained Successfully!")
print(f"R² Score: {r2}")
print(f"MSE: {mse}")

# -------------------------------
# Step 6: Save the model
# -------------------------------
os.makedirs("model", exist_ok=True)
model_path = "model/mpg_model.pkl"
joblib.dump(model, model_path)

print(f"\n✅ Model saved successfully at '{model_path}'")

# -------------------------------
# Step 7: (Optional) Quick test
# -------------------------------
print("\nEnter the following details to test prediction:")
cyl = float(input("Number of cylinders: "))
disp = float(input("Displacement: "))
hp = float(input("Horsepower: "))
wt = float(input("Weight: "))

# Prepare features for prediction (must match training feature order)
features = [[cyl, disp, hp, wt]]
prediction = model.predict(features)[0]

print(f"\n🔹 Predicted Miles Per Gallon (MPG): {prediction:.2f}")
