from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def build_model(X):
    """Build preprocessing + Ridge regression pipeline."""

    preprocessor = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=10.0))
        ]
    )

    return model


def build_ridge_model(X, alpha):
    """Build preprocessing + Ridge regression pipeline with custom alpha."""

    preprocessor = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=alpha))
        ]
    )

    return model

def build_random_forest_model(
    X,
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1
):
    """Build preprocessing + Random Forest regression pipeline."""

    preprocessor = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    return model



def build_gradient_boosting_model(
    X,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
):
    """Build preprocessing + Gradient Boosting regression pipeline."""

    preprocessor = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            ))
        ]
    )

    return model
