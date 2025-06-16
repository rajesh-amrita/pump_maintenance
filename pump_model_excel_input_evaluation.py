
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

# ------------------ Load models and scaler ------------------

scaler = joblib.load("scaler.save")
models = {
    'RandomForest': joblib.load("RandomForest_model.pkl"),
    'XGBoost': joblib.load("XGBoost_model.pkl"),
    'KNN': joblib.load("KNN_model.pkl"),
    'LogisticRegression': joblib.load("LogisticRegression_model.pkl")
}

# ------------------ Input features ------------------

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

# ------------------ Read Excel Input ------------------

file_path = "test1.xlsx"
if not os.path.exists(file_path):
    print("❌ Excel file 'test_input.xlsx' not found. Please make sure it's in the same directory.")
    exit()

df_input = pd.read_excel(file_path)
missing_cols = [col for col in input_features if col not in df_input.columns]
if missing_cols:
    print(f"❌ Missing required columns: {missing_cols}")
    exit()

# Reorder and scale
df_input_ordered = df_input[input_features]
X_input_scaled = scaler.transform(df_input_ordered)

# ------------------ Predict on each row ------------------

predictions = []
for i, row in enumerate(X_input_scaled):
    result = {'Sample': i+1}
    for name, model in models.items():
        pred = model.predict(row.reshape(1, -1))[0]
        result[name] = pred
    predictions.append(result)

df_predictions = pd.DataFrame(predictions)
df_predictions.to_excel("predicted_results.xlsx", index=False)
print("✅ Predictions saved to predicted_results.xlsx")

# ------------------ Evaluation ------------------

df = pd.read_csv("MBG12_1sec_data.csv")
X = df.drop(columns=["Timestamp", "Failure_Code"])
y = df["Failure_Code"]
X_scaled = scaler.transform(X)

evaluation = []
for name, model in models.items():
    y_pred = model.predict(X_scaled)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    avg_f1 = report['weighted avg']['f1-score']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    evaluation.append({
        'Model': name,
        'F1 Score': avg_f1,
        'Precision': precision,
        'Recall': recall
    })

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

# ------------------ Feature Importance ------------------

def plot_feature_importance(model, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title(f"{model_name} - Feature Importances")
        sns.barplot(x=importances[indices], y=np.array(input_features)[indices])
        plt.tight_layout()
        plt.savefig(f"{model_name}_feature_importance.png")
        plt.close()

plot_feature_importance(models['RandomForest'], 'RandomForest')
plot_feature_importance(models['XGBoost'], 'XGBoost')

# ------------------ Summary Comparison ------------------

eval_df = pd.DataFrame(evaluation)

plt.figure(figsize=(10, 6))
sns.barplot(data=eval_df.melt(id_vars="Model"), x="Model", y="value", hue="variable")
plt.title("Model Comparison: F1 Score, Precision, Recall")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("model_comparison_metrics.png")
plt.close()

eval_df.to_csv("model_comparison_summary.csv", index=False)
print("📊 Metrics saved as model_comparison_summary.csv")
