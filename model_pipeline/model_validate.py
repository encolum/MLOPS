import pandas as pd
import mlflow
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Thiết lập tracking URI trỏ đến thư mục mlruns trong MLOPS
MODEL_PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_PIPELINE_DIR.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
LATEST_RUNS_PATH = PROJECT_ROOT / "latest_runs.json"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
load_dotenv(MODEL_PIPELINE_DIR / ".env")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri()))

def validate_model(model_uri, test_data_path):
    test_df = pd.read_csv(test_data_path)

    # Kiểm tra cột đầu vào
    if 'text' not in test_df.columns or 'sentiment_num' not in test_df.columns:
        raise ValueError("CSV phải có cột 'text' và 'sentiment_num'")

    X_test = test_df[['text']]
    y_test = test_df['sentiment_num']

    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
    rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    print(f"   Evaluation for {model_uri}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print("")

if __name__ == "__main__":
    # Đọc latest_runs.json từ thư mục gốc (MLOPS)
    if not LATEST_RUNS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {LATEST_RUNS_PATH}")

    with LATEST_RUNS_PATH.open("r") as f:
        latest_runs = json.load(f)

    for model_name, run_id in latest_runs.items():
        print(f"Validating model: {model_name}")
        validate_model(f"runs:/{run_id}/model", TEST_DATA_PATH)
