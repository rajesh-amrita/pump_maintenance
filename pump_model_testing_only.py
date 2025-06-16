
import numpy as np
import joblib

# Define the order of features expected by the model
input_features = [
    'Phase1_Voltage', 'Phase2_Voltage', 'Phase3_Voltage',
    'Phase1_Current', 'Phase2_Current', 'Phase3_Current',
    'Active_Power', 'Reactive_Power', 'Power_Factor',
    'Frequency', 'Stator_Temperature', 'Bearing_Temperature',
    'Ambient_Temperature', 'Inlet_Pressure', 'Outlet_Pressure',
    'Flow_Rate', 'Vibration_X', 'Vibration_Y', 'Vibration_Z',
    'Motor_Speed', 'Torque', 'Insulation_Resistance', 'Oil_Level',
    'Current_Unbalance', 'THD_Current', 'Bearing_FFT_Harmonics'
]

# Load saved scaler and models
scaler = joblib.load("scaler.save")
models = {
    'RandomForest': joblib.load("RandomForest_model.pkl"),
    'XGBoost': joblib.load("XGBoost_model.pkl"),
    'KNN': joblib.load("KNN_model.pkl"),
    'LogisticRegression': joblib.load("LogisticRegression_model.pkl")
}

# Get input from user with validation
user_input = []
print("Enter sensor values for real-time prediction:")
for feature in input_features:
    while True:
        val = input(f"{feature}: ")
        try:
            user_input.append(float(val))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Scale input
user_input_scaled = scaler.transform([user_input])

# Predict with all models
print("\nPredictions from all models:")
for name, model in models.items():
    prediction = model.predict(user_input_scaled)[0]
    print(f"{name} => Predicted Failure Code: {prediction}")
