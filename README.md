# House Price Regression

A focused machine learning regression project using the **Ames Housing dataset** to predict house sale prices.

## Approach

- Applied `log1p` transformation to the skewed `SalePrice` target.
- Built a Scikit-learn preprocessing pipeline:
  - Median imputation for numerical features
  - Missing-value handling for categorical features
  - One-hot encoding for categorical variables
- Compared several regression model families.
- Tuned the strongest models using cross-validation.
- Performed feature, target-transformation, loss-function, and outlier experiments.
- Selected the final model based primarily on **5-fold cross-validated MAE in dollar scale**.

## Model Comparison

Initial models were compared using a fixed validation set:

| Model | Best MAE |
|---|---:|
| Ridge | $17,848 |
| Random Forest | $17,297 |
| **Gradient Boosting** | **$15,493** |

The Gradient Boosting model was initially the strongest model.

## Model Investigation

The initial Gradient Boosting model was then evaluated using 5-fold cross-validation:

- Mean MAE: **$16,413**
- MAE standard deviation: **$1,478**
- Mean RMSE: **$28,919**

Several focused experiments were performed:

**Feature engineering:** Total floor area, bathroom/porch totals, age features, and quality interactions did not produce a consistent MAE improvement.

**Target transformation:**

| Target | MAE |
|---|---:|
| `log1p(SalePrice)` | **$15,493** |
| Raw `SalePrice` | $16,543 |

The log-transformed target was retained.

**Gradient Boosting loss:**

| Loss | MAE |
|---|---:|
| **squared_error** | **$15,493** |
| huber | $16,702 |
| absolute_error | $16,662 |

The standard squared-error loss performed best.

**Outlier handling:** Removing extreme high-area, low-price training examples slightly improved RMSE but did not improve MAE, so the original data was retained.

## Final Model Selection

Additional tree-based models were compared using 5-fold cross-validation:

| Model | Mean CV MAE |
|---|---:|
| Extra Trees | $17,575 |
| HistGradientBoosting | $16,699 |
| Gradient Boosting | $16,413 |
| XGBoost | $15,290 |

XGBoost was then tuned.

The final configuration was:

```text
XGBRegressor(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.9,
    min_child_weight=1
)