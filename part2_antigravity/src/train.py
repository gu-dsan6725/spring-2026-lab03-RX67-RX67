import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import polars as pl
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
PROCESSED_DATA_DIR: Path = Path("output/processed")
MODELS_DIR: Path = Path("output/models")
RANDOM_STATE: int = 42
CV_FOLDS: int = 5


def run_training() -> None:
    """Train XGBoost model on the Wine dataset."""
    logging.info("Starting Training process")

    # Ensure output directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load processed data
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    if not train_path.exists():
        logging.error(f"Train data not found at {train_path}")
        raise FileNotFoundError(f"Train data not found at {train_path}")

    logging.info(f"Loading train data from {train_path}")
    df_train = pl.read_parquet(train_path)

    # Separate features and target
    target_col = "target"
    feature_cols = [col for col in df_train.columns if col != target_col]

    X_train = df_train.select(feature_cols).to_pandas()
    y_train = df_train.select(target_col).to_pandas().values.ravel()

    # Setup XGBoost
    logging.info("Setting up XGBoost Classifier")
    xgb = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

    # Hyperparameter tuning
    param_grid: Dict[str, Any] = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    logging.info(f"Starting GridSearchCV with {CV_FOLDS} folds")
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logging.info(f"Best parameters: {json.dumps(best_params, indent=2)}")
    logging.info(f"Best Cross-Validation Score: {best_score:.4f}")

    # Build final model
    best_model = grid_search.best_estimator_

    # Save model and artifacts
    model_path = MODELS_DIR / "xgb_model.joblib"
    joblib.dump(best_model, model_path)
    logging.info(f"Saved model to {model_path}")

    # Save best params
    params_path = MODELS_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logging.info(f"Saved best params to {params_path}")

    logging.info("Training completed successfully")


if __name__ == "__main__":
    run_training()
