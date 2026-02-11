import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
PROCESSED_DATA_DIR: Path = Path("output/processed")
MODELS_DIR: Path = Path("output/models")
EVALUATION_DIR: Path = Path("output/evaluation")


def _save_plot(filename: str) -> None:
    """Save the current plot to the evaluation directory."""
    filepath = EVALUATION_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    logging.info(f"Saved plot to {filepath}")


def run_evaluation() -> None:
    """Evaluate the trained XGBoost model on the test set."""
    logging.info("Starting Evaluation process")

    # Ensure output directory exists
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_path = PROCESSED_DATA_DIR / "test.parquet"
    if not test_path.exists():
        logging.error(f"Test data not found at {test_path}")
        raise FileNotFoundError(f"Test data not found at {test_path}")

    logging.info(f"Loading test data from {test_path}")
    df_test = pl.read_parquet(test_path)

    target_col = "target"
    feature_cols = [col for col in df_test.columns if col != target_col]

    X_test = df_test.select(feature_cols).to_pandas()
    y_test = df_test.select(target_col).to_pandas().values.ravel()

    # Load model
    model_path = MODELS_DIR / "xgb_model.joblib"
    if not model_path.exists():
        logging.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")

    logging.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    # Predict
    logging.info("Making predictions on test set")
    y_pred = model.predict(X_test)

    # Classification Report
    logging.info("Generating Classification Report")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)

    logging.info("\n" + report_str)

    # Save report
    report_path = EVALUATION_DIR / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    logging.info(f"Saved classification report to {report_path}")

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {acc:.4f}")

    # Confusion Matrix
    logging.info("Generating Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    _save_plot("confusion_matrix.png")

    logging.info("Evaluation completed successfully")


if __name__ == "__main__":
    run_evaluation()
