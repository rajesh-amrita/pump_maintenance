import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# ------------------ Load & Sample Data ------------------
df = pd.read_csv("MBG12_1sec_data.csv")
df = df.sample(n=2000, random_state=42)  # Use 2000 rows for faster execution

X = df.drop(columns=["Timestamp", "Failure_Code"])
y = df["Failure_Code"]

# ------------------ Scale and Reduce Dimensions ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, stratify=y, test_size=0.2, random_state=42)

# ------------------ Define HWE-PM Ensemble ------------------
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

# ------------------ Define and Train All Models ------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=30),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "XGBoost": XGBClassifier(n_estimators=30, use_label_encoder=False, eval_metric='mlogloss'),
    "HWE_PM": HybridWeightedEnsemble()
}

for name, model in models.items():
    print(f"\nTraining model: {name}")
    model.fit(X_train, y_train)

# ------------------ Evaluate and Compare ------------------
print("\n✅ Model Comparison Reports:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred))
