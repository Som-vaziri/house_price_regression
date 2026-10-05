import joblib

from src.model import build_xgboost_model
from src.preprocessing import load_data


# Load the full training dataset
X, y = load_data("data/train.csv")


# Build the final XGBoost model
model = build_xgboost_model(X)


# Train on all available training data
model.fit(X, y)


# Save the complete preprocessing + model pipeline
joblib.dump(
    model,
    "models/house_price_xgboost.joblib"
)


print("Final XGBoost model trained successfully.")
print("Model saved to:")
print("models/house_price_xgboost.joblib")