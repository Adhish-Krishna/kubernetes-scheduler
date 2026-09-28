import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from scipy.optimize import lsq_linear
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
    random_seed: int = 42,
    min_weight_floor: float = 0.005,
) -> Tuple[Any, XGBRegressor, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Fits:
      1. Bounded Least Squares (NNLS with positive floor) to guarantee NO weight is 0.000000000.
      2. XGBoost Regressor to capture non-linear multi-criteria feature importances.
      3. Blended Ensemble weights combining linear regression and tree importance.

    Returns:
      linear_model: Fitted bounded linear model object (or Ridge)
      xgb_model: Fitted XGBoost model
      blended_weights: 9-decimal normalized weights (sum == 1.000000000)
      linear_weights: 9-decimal normalized linear weights
      xgb_weights: 9-decimal normalized XGBoost importance weights
      raw_coefs: Dict of feature -> raw linear coefficient
    """
    print(f"[ml.train] Fitting Bounded Non-Negative Least Squares (min floor: {min_weight_floor})...")

    # Lower bound ensures every factor has a positive contribution and never zeroes out
    lower_bounds = np.full(X_train.shape[1], min_weight_floor)
    res = lsq_linear(X_train.values, y_train.values, bounds=(lower_bounds, np.inf), lsmr_tol="auto")
    raw_coefs = dict(zip(X_train.columns, res.x))

    # Normalize linear coefficients
    sum_linear = sum(raw_coefs.values())
    raw_linear_norm = {k: v / sum_linear for k, v in raw_coefs.items()}

    # Train Ridge for standard Scikit-learn estimator interface (predict/score)
    ridge_model = Ridge(alpha=alpha, positive=True, fit_intercept=False, random_state=random_seed)
    ridge_model.fit(X_train, y_train)
    # Inject bounded non-zero coefficients into ridge estimator
    ridge_model.coef_ = res.x

    print("[ml.train] Fitting XGBoost Regressor baseline...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_seed,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    print("[ml.train] XGBoost baseline trained successfully.")

    # Extract XGBoost normalized feature importances
    xgb_imp = xgb_model.feature_importances_
    sum_xgb = np.sum(xgb_imp)
    raw_xgb_norm = {col: float(imp / sum_xgb) for col, imp in zip(X_train.columns, xgb_imp)}

    # Blended weights: 50% Linear Bounded Regression + 50% XGBoost Non-Linear Importance
    # This provides the theoretical grounding of linear equations with the empirical robustness of gradient boosted trees!
    raw_blended = {
        col: 0.5 * raw_linear_norm[col] + 0.5 * raw_xgb_norm[col]
        for col in X_train.columns
    }

    def format_to_9_decimals(weights_dict: Dict[str, float]) -> Dict[str, float]:
        """Rounds to 9 decimal places and adjusts delta on largest key so sum is strictly 1.000000000."""
        rounded = {k: round(v, 9) for k, v in weights_dict.items()}
        diff = round(1.0 - sum(rounded.values()), 9)
        largest_key = max(rounded, key=rounded.get)
        rounded[largest_key] = round(rounded[largest_key] + diff, 9)
        return rounded

    linear_weights = format_to_9_decimals(raw_linear_norm)
    xgb_weights = format_to_9_decimals(raw_xgb_norm)
    blended_weights = format_to_9_decimals(raw_blended)

    print("\n" + "=" * 65)
    print("        DERIVED WEIGHTS COMPARISON (9 DECIMALS)")
    print("=" * 65)
    print(f"{'Feature':18s} | {'Linear Bounded':15s} | {'XGBoost Gain':15s} | {'Final Blended':15s}")
    print("-" * 65)
    for col in X_train.columns:
        print(f"{col:18s} | {linear_weights[col]:.9f}   | {xgb_weights[col]:.9f}   | {blended_weights[col]:.9f}")
    print("-" * 65)
    print(f"{'SUM':18s} | {sum(linear_weights.values()):.9f}   | {sum(xgb_weights.values()):.9f}   | {sum(blended_weights.values()):.9f}")
    print("=" * 65 + "\n")

    return ridge_model, xgb_model, blended_weights, linear_weights, xgb_weights, raw_coefs
