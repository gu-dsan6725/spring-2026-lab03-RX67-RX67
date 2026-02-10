"""Generate a comprehensive model evaluation report from trained model artifacts.

Loads a trained model, test data, tuning results, and existing evaluation
metrics, then fills in a report template and saves the completed report.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
TOP_N_FEATURES: int = 5
DEFAULT_OUTPUT_DIR: str = "output"
REPORT_FILENAME: str = "full_report.md"
TUNING_FILENAME: str = "tuning_results.json"
TARGET_COLUMN: str = "target"
KEY_PARAM_NAMES: list[str] = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "objective",
    "random_state",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "eval_metric",
]


def _find_model_file(
    output_dir: Path,
) -> Path:
    """Find the first .joblib or .pkl model file in the output directory."""
    for pattern in ["*.joblib", "*.pkl"]:
        files = list(output_dir.glob(pattern))
        if files:
            logger.info(f"Found model file: {files[0]}")
            return files[0]

    raise FileNotFoundError(f"No .joblib or .pkl model file found in {output_dir}")


def _extract_model_info(
    model_path: Path,
) -> dict:
    """Load model and extract type and hyperparameters."""
    model = joblib.load(model_path)
    model_type = type(model).__name__

    params = model.get_params()
    key_params = {k: v for k, v in params.items() if v is not None and k in KEY_PARAM_NAMES}

    logger.info(f"Model type: {model_type}")
    logger.info(
        f"Hyperparameters:\n{json.dumps({k: str(v) for k, v in key_params.items()}, indent=2)}"
    )

    return {
        "model": model,
        "model_type": model_type,
        "params": key_params,
    }


def _load_tuning_results(
    output_dir: Path,
) -> Optional[dict]:
    """Load tuning results JSON if it exists."""
    tuning_path = output_dir / TUNING_FILENAME
    if not tuning_path.exists():
        logger.info("No tuning_results.json found.")
        return None

    tuning = json.loads(tuning_path.read_text())
    logger.info(f"Loaded tuning results: best_cv_accuracy={tuning['best_cv_accuracy']}")
    return tuning


def _load_test_data(
    output_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load test features and labels from parquet files."""
    x_test = pl.read_parquet(output_dir / "x_test.parquet")
    y_test = pl.read_parquet(output_dir / "y_test.parquet")

    logger.info(f"Loaded test data: x={x_test.shape}, y={y_test.shape}")
    return x_test, y_test


def _compute_metrics(
    model: object,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
) -> dict:
    """Compute classification metrics and prediction distribution."""
    x_np = x_test.to_numpy()
    y_arr = y_test.to_numpy().flatten().astype(int)
    preds = model.predict(x_np)
    proba = model.predict_proba(x_np)

    accuracy = float(accuracy_score(y_arr, preds))
    precision = float(precision_score(y_arr, preds, average="weighted"))
    recall = float(recall_score(y_arr, preds, average="weighted"))
    f1 = float(f1_score(y_arr, preds, average="weighted"))
    mean_max_proba = float(np.mean(np.max(proba, axis=1)))

    metrics = {
        "Accuracy": round(accuracy, 4),
        "Precision (weighted)": round(precision, 4),
        "Recall (weighted)": round(recall, 4),
        "F1-score (weighted)": round(f1, 4),
        "Mean max probability": round(mean_max_proba, 4),
    }

    logger.info(f"Computed metrics:\n{json.dumps(metrics, indent=2, default=str)}")

    report_str = classification_report(
        y_arr,
        preds,
        target_names=[f"class_{i}" for i in range(len(np.unique(y_arr)))],
    )
    logger.info(f"Classification report:\n{report_str}")

    # Prediction distribution
    classes = sorted(np.unique(y_arr))
    pred_dist = {
        f"class_{c}": {"predicted": int(np.sum(preds == c)), "actual": int(np.sum(y_arr == c))}
        for c in classes
    }
    logger.info(f"Prediction distribution:\n{json.dumps(pred_dist, indent=2, default=str)}")

    return {
        "metrics": metrics,
        "pred_dist": pred_dist,
    }


def _get_feature_importance(
    model: object,
    feature_names: list[str],
    top_n: int = TOP_N_FEATURES,
) -> list:
    """Extract top N features ranked by importance."""
    importance = model.feature_importances_
    ranked = sorted(
        zip(feature_names, importance),
        key=lambda x: -x[1],
    )

    top_features = [
        {"rank": i + 1, "name": name, "score": round(float(score), 4)}
        for i, (name, score) in enumerate(ranked[:top_n])
    ]

    logger.info(f"Top {top_n} features:\n{json.dumps(top_features, indent=2, default=str)}")
    return top_features


def _get_dataset_info(
    output_dir: Path,
    x_test: pl.DataFrame,
) -> dict:
    """Gather dataset size information."""
    x_train = pl.read_parquet(output_dir / "x_train.parquet")

    total = x_train.shape[0] + x_test.shape[0]
    info = {
        "total": total,
        "train": x_train.shape[0],
        "test": x_test.shape[0],
        "n_features": x_test.shape[1],
        "feature_names": x_test.columns,
    }

    logger.info(
        f"Dataset: {info['total']} total, "
        f"{info['train']} train, "
        f"{info['test']} test, "
        f"{info['n_features']} features"
    )
    return info


def _build_report(
    dataset_info: dict,
    model_info: dict,
    eval_data: dict,
    top_features: list,
    tuning: Optional[dict],
) -> str:
    """Build the comprehensive markdown report from template sections."""
    metrics = eval_data["metrics"]
    pred_dist = eval_data["pred_dist"]
    lines: list[str] = []

    lines.append("# Model Evaluation Report")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    accuracy_val = metrics.get("Accuracy", "N/A")
    f1_val = metrics.get("F1-score (weighted)", "N/A")
    n_features = dataset_info["n_features"]
    n_test = dataset_info["test"]
    summary = (
        f"An {model_info['model_type']} model was trained on the UCI Wine "
        f"dataset to classify wines into 3 cultivar classes using {n_features} "
        f"features (13 original chemical measurements plus 3 engineered ratios). "
        f"The model achieves {accuracy_val} accuracy and {f1_val} weighted "
        f"F1-score on the held-out test set ({n_test} samples)."
    )
    if tuning:
        summary += (
            f" Hyperparameter tuning via RandomizedSearchCV "
            f"({tuning['n_iterations']} iterations, "
            f"{tuning['cv_folds']}-fold stratified CV) selected the best "
            f"configuration with a cross-validation accuracy of "
            f"{tuning['best_cv_accuracy']}."
        )
    lines.append(summary)
    lines.append("")

    # Dataset Overview
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Total samples | {dataset_info['total']:,} |")
    lines.append(f"| Training samples | {dataset_info['train']:,} |")
    lines.append(f"| Test samples | {dataset_info['test']:,} |")
    lines.append(f"| Number of features | {n_features} |")
    lines.append("| Target variable | Wine cultivar class (0, 1, 2) |")
    lines.append("| Train/test split | 80/20 stratified (random_state=42) |")
    lines.append("")

    # Model Configuration
    lines.append("## Model Configuration")
    lines.append("")
    lines.append("| Hyperparameter | Value |")
    lines.append("|----------------|-------|")
    lines.append(f"| Model type | {model_info['model_type']} |")
    for pname, pval in sorted(model_info["params"].items()):
        lines.append(f"| {pname} | {pval} |")
    if tuning:
        lines.append("| Tuning method | RandomizedSearchCV |")
        lines.append(f"| Tuning iterations | {tuning['n_iterations']} |")
        lines.append(f"| CV folds | {tuning['cv_folds']} |")
        lines.append(f"| Best CV accuracy | {tuning['best_cv_accuracy']} |")
    lines.append("")

    # Performance Metrics
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for mname, mval in metrics.items():
        lines.append(f"| {mname} | {mval} |")
    if tuning:
        cv_std = tuning["top_10_candidates"][0]["std_accuracy"]
        lines.append(f"| CV Accuracy (mean +/- std) | {tuning['best_cv_accuracy']} +/- {cv_std} |")
    lines.append("")

    # Prediction Distribution
    lines.append("### Prediction Distribution")
    lines.append("")
    lines.append("| Class | Predicted Count | Actual Count |")
    lines.append("|-------|-----------------|--------------|")
    for cls_name, counts in pred_dist.items():
        lines.append(f"| {cls_name} | {counts['predicted']} | {counts['actual']} |")
    lines.append("")

    # Feature Importance
    lines.append(f"## Feature Importance (Top {len(top_features)})")
    lines.append("")
    lines.append("| Rank | Feature | Importance Score |")
    lines.append("|------|---------|-----------------:|")
    for feat in top_features:
        lines.append(f"| {feat['rank']} | {feat['name']} | {feat['score']} |")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations for Improvement")
    lines.append("")
    lines.append(
        "1. **Increase tuning budget**: Expand to 50-100 iterations or "
        "switch to Bayesian optimization (e.g., Optuna) to explore the "
        "hyperparameter space more thoroughly."
    )
    lines.append(
        "2. **Nested cross-validation**: Use an outer CV loop around "
        "tuning to get an unbiased generalization estimate, since the "
        "current CV accuracy was used for model selection."
    )
    lines.append(
        "3. **Feature selection**: Prune low-importance features to "
        "reduce overfitting risk and improve interpretability on this "
        "small dataset (178 samples, 16 features)."
    )
    lines.append(
        "4. **Ensemble stacking**: Combine XGBoost with complementary "
        "classifiers (SVM, Random Forest) via stacking to improve "
        "robustness."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Generate a model evaluation report from artifacts in the output dir."""
    parser = argparse.ArgumentParser(
        description="Generate a model evaluation report from trained artifacts",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing model artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    start_time = time.time()
    output_dir = Path(args.output_dir)

    logger.info(f"Generating report from artifacts in: {output_dir}")

    # Load model
    model_path = _find_model_file(output_dir)
    model_info = _extract_model_info(model_path)

    # Load tuning results (optional)
    tuning = _load_tuning_results(output_dir)

    # Load test data and compute metrics
    x_test, y_test = _load_test_data(output_dir)
    eval_data = _compute_metrics(
        model_info["model"],
        x_test,
        y_test,
    )

    # Feature importance
    top_features = _get_feature_importance(
        model_info["model"],
        x_test.columns,
    )

    # Dataset info
    dataset_info = _get_dataset_info(output_dir, x_test)

    # Build and save report
    report_content = _build_report(
        dataset_info,
        model_info,
        eval_data,
        top_features,
        tuning,
    )

    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(report_content)
    logger.info(f"Report saved to {report_path}")

    elapsed = time.time() - start_time
    logger.info(f"Report generation completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
