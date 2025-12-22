# program6_prediction_user_input.py
# ----------------------------------------
# Load the saved model and make MPG predictions using user input
# ----------------------------------------

import joblib
import os

model_path = "model/mpg_model.pkl"

# Check if model exists
if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found! Run program5_regression_model.py first.")

# Load the trained model
model = joblib.load(model_path)

print("🚗 MPG Prediction System 🚗")
print("Enter the following details:")

cylinders = float(input("Number of cylinders: "))
displacement = float(input("Displacement: "))
horsepower = float(input("Horsepower: "))
weight = float(input("Weight: "))

# Prepare the input in correct order
features = [[cylinders, displacement, horsepower, weight]]

# Predict the MPG
prediction = model.predict(features)[0]

print(f"\n🔹 Predicted Miles Per Gallon (MPG): {prediction:.2f}")
