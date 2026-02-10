"""Train and evaluate an XGBoost classification model on the Wine dataset.

Loads the prepared train/test splits, trains an XGBoost classifier,
evaluates performance, and saves the model and evaluation artifacts.
Supports cross-validation and hyperparameter tuning via CLI flags.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from xgboost import XGBClassifier

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output"
MODEL_FILENAME: str = "xgboost_model.joblib"
FIGURE_DPI: int = 150
N_ESTIMATORS: int = 200
MAX_DEPTH: int = 6
LEARNING_RATE: float = 0.1
RANDOM_STATE: int = 42
NUM_CLASSES: int = 3

# Cross-validation constants
CV_FOLDS: int = 5
CV_SCORING: str = "accuracy"
N_ITER_SEARCH: int = 20

# Hyperparameter search space
PARAM_DISTRIBUTIONS: dict = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3],
}


def _load_splits(
    output_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/test splits from parquet files."""
    path = Path(output_dir)

    x_train = pl.read_parquet(path / "x_train.parquet").to_numpy()
    x_test = pl.read_parquet(path / "x_test.parquet").to_numpy()
    y_train = pl.read_parquet(path / "y_train.parquet").to_numpy().ravel()
    y_test = pl.read_parquet(path / "y_test.parquet").to_numpy().ravel()

    logger.info(f"Loaded splits: train={x_train.shape}, test={x_test.shape}")
    return x_train, x_test, y_train, y_test


def _train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> XGBClassifier:
    """Train an XGBoost classifier with default hyperparameters."""
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

    model.fit(x_train, y_train)
    logger.info(
        f"Trained XGBoost classifier with n_estimators={N_ESTIMATORS}, "
        f"max_depth={MAX_DEPTH}, learning_rate={LEARNING_RATE}"
    )
    return model


def _run_cross_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model: XGBClassifier,
) -> dict:
    """Run k-fold stratified cross-validation and return score statistics.

    Args:
        x_train: Training feature matrix.
        y_train: Training target array.
        model: Fitted or unfitted XGBClassifier to evaluate.

    Returns:
        Dictionary with cv_mean_accuracy, cv_std_accuracy, and cv_scores.
    """
    logger.info(f"Running {CV_FOLDS}-fold stratified cross-validation...")

    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=skf,
        scoring=CV_SCORING,
        n_jobs=-1,
    )

    cv_results = {
        "cv_mean_accuracy": round(float(np.mean(scores)), 4),
        "cv_std_accuracy": round(float(np.std(scores)), 4),
        "cv_scores": [round(float(s), 4) for s in scores],
    }

    logger.info(
        f"Cross-validation accuracy: {cv_results['cv_mean_accuracy']} "
        f"(+/- {cv_results['cv_std_accuracy']})"
    )
    logger.info(f"Per-fold accuracy scores: {cv_results['cv_scores']}")

    return cv_results


def _run_hyperparameter_tuning(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[XGBClassifier, RandomizedSearchCV]:
    """Run randomized search for hyperparameter tuning.

    Args:
        x_train: Training feature matrix.
        y_train: Training target array.

    Returns:
        Tuple of (best estimator, full RandomizedSearchCV object).
    """
    logger.info(
        f"Starting hyperparameter tuning with {N_ITER_SEARCH} iterations "
        f"and {CV_FOLDS}-fold stratified CV. This may take a while."
    )

    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        cv=skf,
        scoring=CV_SCORING,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(x_train, y_train)

    best_accuracy = search.best_score_
    logger.info(f"Best CV accuracy: {best_accuracy:.4f}")
    logger.info(f"Best parameters:\n{json.dumps(search.best_params_, indent=2, default=str)}")

    return search.best_estimator_, search


def _save_tuning_results(
    search: RandomizedSearchCV,
    output_path: Path,
) -> None:
    """Save hyperparameter tuning results to a JSON file.

    Args:
        search: Fitted RandomizedSearchCV object.
        output_path: Directory to save the results file.
    """
    cv_results = search.cv_results_

    candidates = []
    for i in range(len(cv_results["params"])):
        candidates.append(
            {
                "rank": int(cv_results["rank_test_score"][i]),
                "mean_accuracy": round(float(cv_results["mean_test_score"][i]), 4),
                "std_accuracy": round(float(cv_results["std_test_score"][i]), 4),
                "params": {
                    k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                    for k, v in cv_results["params"][i].items()
                },
            }
        )

    candidates.sort(key=lambda x: x["rank"])

    results = {
        "best_params": {
            k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in search.best_params_.items()
        },
        "best_cv_accuracy": round(float(search.best_score_), 4),
        "n_iterations": N_ITER_SEARCH,
        "cv_folds": CV_FOLDS,
        "top_10_candidates": candidates[:10],
    }

    filepath = output_path / "tuning_results.json"
    filepath.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Tuning results saved to {filepath}")


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute classification evaluation metrics."""
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted"))
    recall = float(recall_score(y_true, y_pred, average="weighted"))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    logger.info(f"Evaluation metrics:\n{json.dumps(metrics, indent=2, default=str)}")
    return metrics


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Generate a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    class_labels = [f"Class {i}" for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    filepath = output_path / "confusion_matrix.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Confusion matrix saved to {filepath}")


def _plot_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Generate a feature importance bar chart."""
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        range(len(importances)),
        importances[sorted_indices],
        align="center",
        alpha=0.8,
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in sorted_indices],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    ax.set_title("XGBoost Feature Importance")

    plt.tight_layout()
    filepath = output_path / "feature_importance.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Feature importance plot saved to {filepath}")


def _save_model(
    model: XGBClassifier,
    output_path: Path,
) -> None:
    """Save the trained model to disk."""
    filepath = output_path / MODEL_FILENAME
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def _write_evaluation_report(
    metrics: dict,
    output_path: Path,
    cv_results: Optional[dict] = None,
    best_params: Optional[dict] = None,
) -> None:
    """Write an evaluation report to a markdown file.

    Args:
        metrics: Dictionary of test set evaluation metrics.
        output_path: Directory to write the report.
        cv_results: Optional cross-validation results dictionary.
        best_params: Optional best hyperparameters from tuning.
    """
    report = "# Model Evaluation Report\n\n"
    report += "## Metrics Summary\n\n"
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    report += f"| Accuracy | {metrics['accuracy']} |\n"
    report += f"| Precision (weighted) | {metrics['precision']} |\n"
    report += f"| Recall (weighted) | {metrics['recall']} |\n"
    report += f"| F1-score (weighted) | {metrics['f1']} |\n\n"

    if cv_results is not None:
        report += "## Cross-Validation Results\n\n"
        report += f"- **Folds**: {CV_FOLDS}\n"
        report += f"- **Mean Accuracy**: {cv_results['cv_mean_accuracy']}\n"
        report += f"- **Std Accuracy**: {cv_results['cv_std_accuracy']}\n"
        report += f"- **Per-fold Accuracy**: {cv_results['cv_scores']}\n\n"

    if best_params is not None:
        report += "## Best Hyperparameters (from tuning)\n\n"
        report += "| Parameter | Value |\n"
        report += "|-----------|-------|\n"
        for param, value in sorted(best_params.items()):
            report += f"| {param} | {value} |\n"
        report += "\n"

    report += "## Artifacts\n\n"
    report += "- `confusion_matrix.png`: Confusion matrix heatmap\n"
    report += "- `feature_importance.png`: XGBoost feature importance ranking\n"
    report += "- `xgboost_model.joblib`: Trained model file\n"

    if best_params is not None:
        report += "- `tuning_results.json`: Hyperparameter tuning results\n"

    filepath = output_path / "evaluation_report.md"
    filepath.write_text(report)
    logger.info(f"Evaluation report saved to {filepath}")


def run_training_and_evaluation(
    tune: bool = False,
    cv_only: bool = False,
) -> None:
    """Run the full model training and evaluation pipeline.

    Args:
        tune: If True, run hyperparameter tuning before training.
        cv_only: If True, run cross-validation on the default model.
    """
    start_time = time.time()
    logger.info("Starting model training and evaluation...")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = _load_splits(OUTPUT_DIR)

    cv_results = None
    best_params = None

    if tune:
        model, search = _run_hyperparameter_tuning(x_train, y_train)
        best_params = search.best_params_
        _save_tuning_results(search, output_path)
        cv_results = _run_cross_validation(x_train, y_train, model)

    elif cv_only:
        model = _train_model(x_train, y_train)
        cv_results = _run_cross_validation(x_train, y_train, model)

    else:
        model = _train_model(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = _compute_metrics(y_test, y_pred)

    _plot_confusion_matrix(y_test, y_pred, output_path)
    feature_names = pl.read_parquet(output_path / "x_train.parquet").columns
    _plot_feature_importance(model, feature_names, output_path)

    _save_model(model, output_path)
    _write_evaluation_report(
        metrics,
        output_path,
        cv_results=cv_results,
        best_params=best_params,
    )

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    if minutes > 0:
        logger.info(
            f"Training and evaluation completed in {minutes} minutes and {seconds:.1f} seconds"
        )
    else:
        logger.info(f"Training and evaluation completed in {seconds:.1f} seconds")


def main() -> None:
    """Parse CLI arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate an XGBoost classifier on the Wine dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Default training with fixed hyperparameters
    uv run python -m part1_claude_code.src.03_xgboost_model

    # Run with cross-validation on default model
    uv run python -m part1_claude_code.src.03_xgboost_model --cv-only

    # Run with hyperparameter tuning (20 iterations, 5-fold stratified CV)
    uv run python -m part1_claude_code.src.03_xgboost_model --tune
""",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run randomized hyperparameter search (20 iters, 5-fold stratified CV)",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        default=False,
        help="Run cross-validation on the default model (no tuning)",
    )

    args = parser.parse_args()

    if args.tune and args.cv_only:
        parser.error("--tune and --cv-only are mutually exclusive")

    run_training_and_evaluation(
        tune=args.tune,
        cv_only=args.cv_only,
    )


if __name__ == "__main__":
    main()
