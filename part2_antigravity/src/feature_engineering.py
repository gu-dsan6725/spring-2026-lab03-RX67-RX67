import logging
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
PROCESSED_DATA_DIR: Path = Path("output/processed")
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2


def run_feature_engineering() -> None:
    """Run feature engineering on the Wine dataset."""
    logging.info("Starting Feature Engineering process")

    # Load raw data
    raw_data_path = PROCESSED_DATA_DIR / "wine_raw.parquet"
    if not raw_data_path.exists():
        logging.error(f"Raw data not found at {raw_data_path}")
        raise FileNotFoundError(f"Raw data not found at {raw_data_path}")

    logging.info(f"Loading raw data from {raw_data_path}")
    df = pl.read_parquet(raw_data_path)

    # Separate features and target
    target_col = "target"
    feature_cols = [col for col in df.columns if col != target_col]

    X = df.select(feature_cols).to_pandas()
    y = df.select(target_col).to_pandas()

    # Split data (Stratified)
    logging.info(f"Splitting data with test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Standard Scaling
    logging.info("Applying Standard Scaling")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to Polars for saving
    # We need to recombine features and target

    # Create Train DataFrame
    df_train = pl.from_numpy(X_train_scaled, schema=feature_cols)
    df_train = df_train.with_columns(pl.Series(target_col, y_train[target_col].values))

    # Create Test DataFrame
    df_test = pl.from_numpy(X_test_scaled, schema=feature_cols)
    df_test = df_test.with_columns(pl.Series(target_col, y_test[target_col].values))

    # Save processed data
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    test_path = PROCESSED_DATA_DIR / "test.parquet"

    df_train.write_parquet(train_path)
    df_test.write_parquet(test_path)

    logging.info(f"Saved train data to {train_path}")
    logging.info(f"Saved test data to {test_path}")

    logging.info("Feature Engineering completed successfully")


if __name__ == "__main__":
    run_feature_engineering()
