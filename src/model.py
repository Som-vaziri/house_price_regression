from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor


def build_xgboost_model(X):
    """Build the final XGBoost regression pipeline."""

    preprocessor = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=700,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    min_child_weight=1,
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model