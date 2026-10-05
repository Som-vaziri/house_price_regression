# House Price Regression

A machine learning regression project using the **Ames Housing dataset** to predict house sale prices.

## Approach

* Applied `log1p` transformation to the skewed `SalePrice` target.
* Built a Scikit-learn preprocessing pipeline:

  * Median imputation for numerical features
  * Missing-value handling and one-hot encoding for categorical features
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

The final pipeline was retrained on the full training dataset and saved as:

```text
models/house_price_gradient_boosting.joblib
```

## Project Structure

```text
house_price_regression/
├── data/
├── models/
├── output/
├── src/
└── README.md
```

## Technologies

Python · Pandas · NumPy · Scikit-learn · Joblib
