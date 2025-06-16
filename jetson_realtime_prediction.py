import socket
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Define the custom ensemble class (must match training time)
class HybridWeightedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.base_models = {
            "xgb": XGBClassifier(n_estimators=30, use_label_encoder=False, eval_metric='mlogloss'),
            "rf": RandomForestClassifier(n_estimators=30),
            "knn": KNeighborsClassifier(n_neighbors=3)
        }
        self.meta_model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        self.base_models_ = {name: clone(model) for name, model in self.base_models.items()}
        self.meta_model_ = clone(self.meta_model)
        meta_X = []

        for name, model in self.base_models_.items():
            model.fit(X, y)
            preds = cross_val_predict(model, X, y, cv=3, method="predict_proba")
            meta_X.append(preds)

        self.meta_model_.fit(np.hstack(meta_X), y)
        return self

    def predict(self, X):
        meta_input = np.hstack([model.predict_proba(X) for model in self.base_models_.values()])
        return self.meta_model_.predict(meta_input)

    def predict_proba(self, X):
        meta_input = np.hstack([model.predict_proba(X) for model in self.base_models_.values()])
        return self.meta_model_.predict_proba(meta_input)

# Load trained model and scaler
scaler = joblib.load("hwe_pm_scaler.save")
pca = PCA(n_components=10)
model = joblib.load("HWE_PM_model.pkl")

# Fault class labels
fault_labels = {
    0: "Normal operation",
    1: "Bearing wear",
    2: "Winding insulation degradation",
    3: "Impeller damage",
    4: "Cavitation"
}

# UDP settings
UDP_IP = "0.0.0.0"
UDP_PORT = 8080
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"✅ Jetson Nano Fault Predictor is running on UDP port {UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(2048)
    try:
        values = list(map(float, data.decode().strip().split(",")))
        if len(values) != 5:
            raise ValueError("Expected 5 sensor values")

        # Transform and reduce
        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_reduced = pca.fit_transform(X_scaled)

        # Predict fault
        prediction = model.predict(X_reduced)[0]
        result = fault_labels[prediction]

        # Display and send back
        print(f"📡 From {addr} | Input: {values} --> 🚨 Fault: {result}")
        sock.sendto(result.encode(), addr)

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        sock.sendto(error_msg.encode(), addr)
