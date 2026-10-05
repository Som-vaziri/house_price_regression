# House Price Regression

A focused machine learning regression project using the **Ames Housing dataset** to predict house sale prices.

## Approach

* Applied `log1p` transformation to the skewed `SalePrice` target.
* Built a Scikit-learn preprocessing pipeline:

  * Median imputation for numerical features
  * Missing-value handling for categorical features
  * One-hot encoding for categorical variables
* Compared three regression models:

  * Ridge Regression
  * Random Forest
  * Gradient Boosting
* Tuned key hyperparameters using a fixed validation set.
* Selected the final model based primarily on **MAE**.

## Model Optimization

| Model                 | Best Configuration              |         MAE |
| --------------------- | ------------------------------- | ----------: |
| Ridge                 | α = 100                         |     $17,848 |
| Random Forest         | 200 trees, `min_samples_leaf=2` |     $17,297 |
| **Gradient Boosting** | **500 trees, lr=0.2, depth=4**  | **$15,493** |

### Final Model

```text
GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.2,
    max_depth=4
)
```

**Validation performance:**

* MAE: **$15,493**
* RMSE: **$26,695**

The target was trained in log scale, while the reported MAE and RMSE above are calculated after converting predictions back to the original dollar scale.

## Inference

The trained preprocessing + model pipeline is used directly on unseen test data.

```bash
python -m src.predict
```

Predictions are written to:

```text
output/house_price_predictions.csv
```

The trained model is saved locally as:

```text
models/house_price_gradient_boosting.joblib
```

Generated models and prediction files are excluded from Git.

## Data

This project uses the **House Prices: Advanced Regression Techniques** dataset from Kaggle.

The dataset files are intentionally kept out of the Git repository. Place the following files inside `data/` before running the project:

```text
data/
├── train.csv
├── test.csv
└── data_description.txt
```

## Project Structure

```text
house_price_regression/
├── data/
├── models/
├── output/
├── src/
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train_final_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies

Python · Pandas · NumPy · Scikit-learn · Joblib
