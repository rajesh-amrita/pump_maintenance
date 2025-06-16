import joblib
model = joblib.load("xgb_model.pkl")
print(type(model))
