"""Exploratory Data Analysis on the UCI Wine dataset.

Loads the dataset, computes summary statistics, generates distribution
plots, creates a correlation heatmap, plots class distribution, and
identifies outliers.
"""

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output"
FIGURE_DPI: int = 150
IQR_MULTIPLIER: float = 1.5
TARGET_COLUMN: str = "target"


def _ensure_output_dir(
    output_dir: str,
) -> Path:
    """Create the output directory if it does not exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_dataset() -> tuple[pl.DataFrame, list[str]]:
    """Load the Wine dataset and return as a polars DataFrame with class names."""
    wine = load_wine()
    feature_names: list[str] = list(wine.feature_names)

    df = pl.DataFrame({name: wine.data[:, i] for i, name in enumerate(feature_names)})
    df = df.with_columns(pl.Series(TARGET_COLUMN, wine.target))

    class_names: list[str] = list(wine.target_names)

    logger.info(f"Loaded dataset with shape: {df.shape}")
    logger.info(f"Class names: {class_names}")
    return df, class_names


def _compute_summary_statistics(
    df: pl.DataFrame,
) -> dict:
    """Compute summary statistics for all columns."""
    stats = {}
    for col in df.columns:
        col_data = df[col]
        stats[col] = {
            "mean": round(float(col_data.mean()), 4),
            "median": round(float(col_data.median()), 4),
            "std": round(float(col_data.std()), 4),
            "min": round(float(col_data.min()), 4),
            "max": round(float(col_data.max()), 4),
        }

    logger.info(f"Summary statistics:\n{json.dumps(stats, indent=2, default=str)}")
    return stats


def _check_missing_values(
    df: pl.DataFrame,
) -> dict:
    """Check for missing values in each column."""
    missing = {}
    for col in df.columns:
        null_count = df[col].null_count()
        missing[col] = null_count

    total_missing = sum(missing.values())
    logger.info(f"Total missing values: {total_missing}")
    if total_missing > 0:
        logger.warning(f"Missing values found:\n{json.dumps(missing, indent=2, default=str)}")
    else:
        logger.info("No missing values found in the dataset.")

    return missing


def _plot_distributions(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Generate histogram distribution plots for each feature."""
    columns = [c for c in df.columns if c != TARGET_COLUMN]
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        values = df[col].to_list()
        axes[i].hist(values, bins=30, edgecolor="black", alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    # Hide unused subplots
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    filepath = output_path / "distributions.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Distribution plots saved to {filepath}")


def _plot_correlation_matrix(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Generate a correlation matrix heatmap."""
    columns = [c for c in df.columns if c != TARGET_COLUMN]
    corr_data = {}
    for col in columns:
        correlations = []
        for other_col in columns:
            corr_value = df.select(pl.corr(col, other_col)).item()
            correlations.append(round(float(corr_value), 3))
        corr_data[col] = correlations

    corr_df = pl.DataFrame(corr_data)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_df.to_numpy(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=columns,
        yticklabels=columns,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    filepath = output_path / "correlation_matrix.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Correlation matrix saved to {filepath}")


def _plot_class_distribution(
    df: pl.DataFrame,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Generate a bar chart showing the distribution of target classes."""
    counts = df.group_by(TARGET_COLUMN).len().sort(TARGET_COLUMN)
    target_values = counts[TARGET_COLUMN].to_list()
    count_values = counts["len"].to_list()

    labels = [class_names[int(v)] for v in target_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, count_values, edgecolor="black", alpha=0.8)

    for bar, count in zip(bars, count_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_xlabel("Wine Class")
    ax.set_ylabel("Count")
    ax.set_title("Wine Class Distribution")
    plt.tight_layout()
    filepath = output_path / "class_distribution.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Class distribution plot saved to {filepath}")


def _identify_outliers(
    df: pl.DataFrame,
) -> dict:
    """Identify outliers using the IQR method."""
    outlier_counts = {}
    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr

        outlier_count = df.filter((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)).height

        outlier_counts[col] = outlier_count

    logger.info(
        "Outlier counts (IQR method):\n%s",
        json.dumps(outlier_counts, indent=2, default=str),
    )
    return outlier_counts


def run_eda() -> None:
    """Run the full exploratory data analysis pipeline."""
    start_time = time.time()
    logger.info("Starting exploratory data analysis on the Wine dataset...")

    output_path = _ensure_output_dir(OUTPUT_DIR)
    df, class_names = _load_dataset()
    _compute_summary_statistics(df)
    _check_missing_values(df)
    _plot_distributions(df, output_path)
    _plot_correlation_matrix(df, output_path)
    _plot_class_distribution(df, class_names, output_path)
    _identify_outliers(df)

    elapsed = time.time() - start_time
    logger.info(f"EDA completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    run_eda()
