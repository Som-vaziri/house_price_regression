import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "models/house_price_xgboost.joblib"
TEST_DATA_PATH = "data/test.csv"
OUTPUT_PATH = "output/house_price_predictions.csv"


# Load the trained preprocessing + model pipeline
model = joblib.load(MODEL_PATH)

# Load unseen house data
X_test = pd.read_csv(TEST_DATA_PATH)

# Generate predictions in log scale
predicted_log_price = model.predict(X_test)

# Convert predictions back to dollar prices
predicted_price = np.expm1(predicted_log_price)

# Create prediction output
predictions = pd.DataFrame(
    {
        "Id": X_test["Id"],
        "SalePrice": predicted_price,
    }
)

# Save predictions
predictions.to_csv(OUTPUT_PATH, index=False)

print("Predictions generated successfully.")
print(f"Saved to: {OUTPUT_PATH}")
print()
print(predictions.head())