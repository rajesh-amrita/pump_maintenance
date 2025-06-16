import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import joblib

# ------------------ Load Partial Data ------------------
df = pd.read_csv("MBG12_1sec_data.csv")
df_sample = df.sample(n=1500, random_state=42)  # ✅ Reduce dataset for quick training
X = df_sample.drop(columns=["Timestamp", "Failure_Code"])
y = df_sample["Failure_Code"]

# ------------------ Preprocessing ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "hwe_pm_scaler.save")

# Optional: Dimensionality Reduction
USE_PCA = True
if USE_PCA:
    pca = PCA(n_components=15)
    X_scaled = pca.fit_transform(X_scaled)
    joblib.dump(pca, "hwe_pm_pca.save")

# ------------------ Split ------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# ------------------ Base Models ------------------
models = {
    "xgb": XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='mlogloss'),
    "rf": RandomForestClassifier(n_estimators=50),
    "knn": KNeighborsClassifier(n_neighbors=3),
    "svm": SVC(probability=True, kernel='rbf', C=1.0)  # Reduced complexity
}

# Train and Save base models
base_predictions = []
for name, model in models.items():
    print(f"Training base model: {name}")
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}_model.pkl")
    preds = cross_val_predict(model, X_train, y_train, cv=3, method="predict_proba")
    base_predictions.append(preds)

# ------------------ Meta-Learner (Logistic Regression) ------------------
meta_X_train = np.hstack(base_predictions)
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(meta_X_train, y_train)
joblib.dump(meta_model, "meta_model.pkl")

# Save Final HWE-PM Wrapper
class HybridWeightedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def predict(self, X):
        base_preds = [model.predict_proba(X) for model in self.base_models.values()]
        meta_input = np.hstack(base_preds)
        return self.meta_model.predict(meta_input)

    def predict_proba(self, X):
        base_preds = [model.predict_proba(X) for model in self.base_models.values()]
        meta_input = np.hstack(base_preds)
        return self.meta_model.predict_proba(meta_input)

# Reload models and wrap
loaded_models = {name: joblib.load(f"{name}_model.pkl") for name in models}
loaded_meta = joblib.load("meta_model.pkl")
hwe_pm = HybridWeightedEnsemble(loaded_models, loaded_meta)
joblib.dump(hwe_pm, "HWE_PM_model.pkl")

# ------------------ Evaluate and Save ------------------
y_pred = hwe_pm.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("HWE_PM_classification_report.csv")

print("\n✅ HWE-PM training complete. All models and report saved.")
