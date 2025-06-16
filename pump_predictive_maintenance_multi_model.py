
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ------------------- Step 1: Load & Preprocess Dataset -------------------
df = pd.read_csv("MBG12_1sec_data.csv")  # Ensure uncompressed

X = df.drop(columns=['Timestamp', 'Failure_Code'])
y = df['Failure_Code']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Save the scaler
joblib.dump(scaler, "scaler.save")

# ------------------- Step 2: Train and Save ML Models -------------------
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LogisticRegression': LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    joblib.dump(model, f"{name}_model.pkl")

# ------------------- Step 3: Test with User Input -------------------
print("\n\n==============================")
print("REAL-TIME TEST WITH NEW INPUT")
print("==============================\n")

# Load saved models
scaler = joblib.load("scaler.save")
loaded_models = {name: joblib.load(f"{name}_model.pkl") for name in models.keys()}

# Manual input
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

# Predict with all models
print("\nPredictions from all models:")
for name, model in loaded_models.items():
    pred = model.predict(user_input_scaled)[0]
    print(f"{name} => Predicted Failure Code: {pred}")
