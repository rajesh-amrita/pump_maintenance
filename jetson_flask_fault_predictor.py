from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Define the custom ensemble class
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

# Load trained model and preprocessing tools
scaler = joblib.load("hwe_pm_scaler.save")
pca = joblib.load("hwe_pm_pca.save")
model = joblib.load("HWE_PM_model.pkl")

# Fault labels
fault_labels = {
    0: "Normal operation",
    1: "Bearing wear",
    2: "Winding insulation degradation",
    3: "Impeller damage",
    4: "Cavitation"
}

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_fault():
    try:
        data = request.json.get("sensor_data", [])
        if len(data) != 26:
            return jsonify({"error": "Expected 26 sensor values"}), 400

        # Preprocess input
        X = np.array(data).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_reduced = pca.transform(X_scaled)
        prediction = model.predict(X_reduced)[0]
        label = fault_labels.get(prediction, "Unknown")

        return jsonify({"fault_code": int(prediction), "fault_label": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
