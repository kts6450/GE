from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TICKER = "GE"
START_DATE = "2014-01-01"
INTERVAL = "1d"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = RAW_DATA_DIR / "ge_raw.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "ge_processed.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "ge_train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "ge_test.csv"
LATEST_PREDICTION_PATH = PREDICTIONS_DIR / "latest_prediction.json"
PREDICTION_HISTORY_PATH = PREDICTIONS_DIR / "prediction_history.csv"

DENSE_MODEL_PATH = MODELS_DIR / "ge_dense_model.keras"
LSTM_MODEL_PATH = MODELS_DIR / "ge_lstm_model.keras"
FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = MODELS_DIR / "target_scaler.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TEST_SIZE = 0.2
RANDOM_SEED = 42
LSTM_LOOKBACK_DAYS = 30
EPOCHS = 80
BATCH_SIZE = 32


def ensure_directories() -> None:
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
