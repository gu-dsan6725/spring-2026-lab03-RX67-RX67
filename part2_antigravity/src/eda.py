import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR: Path = Path("output/eda")
PROCESSED_DATA_DIR: Path = Path("output/processed")


def _save_plot(filename: str) -> None:
    """Save the current plot to the output directory."""
    filepath = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    logging.info(f"Saved plot to {filepath}")


def run_eda() -> None:
    """Run exploratory data analysis on the Wine dataset."""
    logging.info("Starting EDA process")

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logging.info("Loading Wine dataset")
    wine_data = load_wine(as_frame=True)
    # Convert to polars, passing the pandas DataFrame directly
    df = pl.from_pandas(wine_data.frame)

    # Basic info
    logging.info(f"Dataset shape: {df.shape}")
    logging.info("Columns: %s", df.columns)

    # Summary statistics
    stats = df.describe()
    logging.info("Summary statistics:\n%s", stats)

    # Check for missing values
    null_counts = df.null_count()
    logging.info("Missing values:\n%s", null_counts)

    # Class distribution
    logging.info("Plotting class distribution")
    plt.figure(figsize=(8, 6))
    # Count targets
    class_counts = df.group_by("target").len().sort("target")
    sns.barplot(x=class_counts["target"], y=class_counts["len"])
    plt.title("Class Distribution")
    plt.xlabel("Target Class")
    plt.ylabel("Count")
    _save_plot("class_distribution.png")

    # Correlation matrix
    logging.info("Plotting correlation matrix")
    plt.figure(figsize=(12, 10))
    # Convert to pandas for seaborn correlation calculation
    corr_matrix = df.to_pandas().corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    _save_plot("correlation_matrix.png")

    # Pairplot for selected features
    selected_features = ["alcohol", "flavanoids", "color_intensity", "hue", "target"]
    logging.info(f"Plotting pairplot for selected features: {selected_features}")
    # Use pandas for pairplot
    sns.pairplot(df.select(selected_features).to_pandas(), hue="target", palette="viridis")
    _save_plot("pairplot_selected.png")

    # Save raw data as parquet for next steps
    raw_data_path = PROCESSED_DATA_DIR / "wine_raw.parquet"
    df.write_parquet(raw_data_path)
    logging.info(f"Saved raw data to {raw_data_path}")

    logging.info("EDA completed successfully")


if __name__ == "__main__":
    run_eda()
