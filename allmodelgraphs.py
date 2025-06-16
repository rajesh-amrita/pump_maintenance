import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Load and sample data (CHANGE PATH IF NEEDED)
df = pd.read_csv("D:/Amritha_pdf_preparation/manuscript/MBG12_1sec_data.csv")
df_sample = df.sample(n=2000, random_state=42)

X = df_sample.drop(columns=["Timestamp", "Failure_Code"])
y = df_sample["Failure_Code"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "hwe_pm_scaler.save")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, objective='multi:softprob',
                          num_class=len(np.unique(y)), eval_metric='mlogloss', base_score=0.5).fit(X_train, y_train)
knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
svm_model = SVC(probability=True).fit(X_train, y_train)

# Save models
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(knn_model, "knn_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")

# Hybrid Weighted Ensemble
class HybridWeightedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.base_models = {
            "xgb": clone(xgb_model),
            "rf": clone(rf_model),
            "svm": clone(svm_model),
            "knn": clone(knn_model)
        }
        self.meta_model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        base_predictions = []
        for model in self.base_models.values():
            model.fit(X, y)
            preds = cross_val_predict(model, X, y, cv=3, method="predict_proba")
            base_predictions.append(preds)
        meta_X = np.hstack(base_predictions)
        self.meta_model.fit(meta_X, y)
        return self

    def predict(self, X):
        base_preds = [model.predict_proba(X) for model in self.base_models.values()]
        meta_input = np.hstack(base_preds)
        return self.meta_model.predict(meta_input)

    def predict_proba(self, X):
        base_preds = [model.predict_proba(X) for model in self.base_models.values()]
        meta_input = np.hstack(base_preds)
        return self.meta_model.predict_proba(meta_input)

# Train and save HWE-PM
hwe_pm = HybridWeightedEnsemble().fit(X_train, y_train)
joblib.dump(hwe_pm, "HWE_PM_model.pkl")

# Evaluate all models
models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "KNN": knn_model,
    "SVM": svm_model,
    "HWE-PM": hwe_pm
}

metrics = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics[name] = {
        "Accuracy": report["accuracy"],
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1 Score": report["weighted avg"]["f1-score"]
    }

metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv("model_comparison_metrics.csv")
print("Model evaluation metrics saved.")

# Plot metrics
for metric in metrics_df.columns:
    plt.figure(figsize=(8, 4))
    sns.barplot(x=metrics_df.index, y=metrics_df[metric])
    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png")
    plt.close()

# Plot confusion matrix for HWE-PM
y_pred_hwe = hwe_pm.predict(X_test)
cm = confusion_matrix(y_test, y_pred_hwe)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - HWE-PM")
plt.savefig("confusion_matrix_hwe_pm.png")
plt.close()

# Feature Importance
features = X.columns
plt.figure(figsize=(10, 5))
sns.barplot(x=features, y=xgb_model.feature_importances_)
plt.title("XGBoost Feature Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("xgb_feature_importance.png")
plt.close()

plt.figure(figsize=(10, 5))
sns.barplot(x=features, y=rf_model.feature_importances_)
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.close()

print("✅ All models trained, saved, and graphs generated.")
