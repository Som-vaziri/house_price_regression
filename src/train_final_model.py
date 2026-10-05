import joblib
import numpy as np

from src.preprocessing import load_data
from src.model import build_gradient_boosting_model


# Load the full dataset
X, y = load_data("data/train.csv")


# Build the final Gradient Boosting model
model = build_gradient_boosting_model(
    X,
    n_estimators=500,
    learning_rate=0.2,
    max_depth=4
)


# Train on all available data
model.fit(X, y)


# Save the complete pipeline
joblib.dump(
    model,
    "models/house_price_gradient_boosting.joblib"
)


print("Final model trained successfully.")
print("Model saved to:")
print("models/house_price_gradient_boosting.joblib")