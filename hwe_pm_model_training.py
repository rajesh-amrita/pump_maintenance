
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict
import joblib

# ------------------ Load Data ------------------
df = pd.read_csv("MBG12_1sec_data.csv")
X = df.drop(columns=["Timestamp", "Failure_Code"])
y = df["Failure_Code"]

# ------------------ Preprocessing ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "hwe_pm_scaler.save")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# ------------------ Define HWE-PM Model ------------------
class HybridWeightedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.base_models = {
            "xgb": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss'),
            "rf": RandomForestClassifier(n_estimators=100),
            "svm": SVC(probability=True),
            "knn": KNeighborsClassifier(n_neighbors=5)
        }
        self.meta_model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        self.base_models_ = {name: clone(model) for name, model in self.base_models.items()}
        self.meta_model_ = clone(self.meta_model)

        base_predictions = []
        for name, model in self.base_models_.items():
            print(f"Training base model: {name}")
            model.fit(X, y)
            preds = cross_val_predict(model, X, y, cv=3, method="predict_proba")
            base_predictions.append(preds)

        # Concatenate predictions for meta model
        meta_X = np.hstack(base_predictions)
        self.meta_model_.fit(meta_X, y)
        return self

    def predict(self, X):
        base_preds = [model.predict_proba(X) for model in self.base_models_.values()]
        meta_input = np.hstack(base_preds)
        return self.meta_model_.predict(meta_input)

    def predict_proba(self, X):
        base_preds = [model.predict_proba(X) for model in self.base_models_.values()]
        meta_input = np.hstack(base_preds)
        return self.meta_model_.predict_proba(meta_input)

# ------------------ Train & Evaluate ------------------
hwe_pm = HybridWeightedEnsemble()
hwe_pm.fit(X_train, y_train)
y_pred = hwe_pm.predict(X_test)

print("\nClassification Report - HWE-PM:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(hwe_pm, "HWE_PM_model.pkl")

# Save classification report to CSV
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("HWE_PM_classification_report.csv")

print("✅ HWE-PM training complete. Model and report saved.")
