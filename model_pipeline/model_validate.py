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
VALIDATED_RUNS_PATH = PROJECT_ROOT / "validated_runs.json"
TEST_DATA_PATH = MODEL_PIPELINE_DIR / "test_data.csv"
load_dotenv(MODEL_PIPELINE_DIR / ".env")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri()))

def validate_model(model_uri, test_data_path):
    test_df = pd.read_csv(test_data_path)

    # Kiểm tra cột đầu vào
    if 'text' not in test_df.columns or 'sentiment_num' not in test_df.columns:
        raise ValueError("CSV must contain 'text' and 'sentiment_num' columns")

    X_test = test_df[['text']]
    y_test = test_df['sentiment_num']

    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
    rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_test, predictions, average='macro', zero_division=0)

    print(f"   Evaluation for {model_uri}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   Weighted F1: {f1:.4f}")
    print(f"   Macro F1   : {macro_f1:.4f}")
    print("")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "weighted_f1": f1,
        "macro_f1": macro_f1,
    }

if __name__ == "__main__":
    # Đọc latest_runs.json từ thư mục gốc (MLOPS)
    if not LATEST_RUNS_PATH.exists():
        raise FileNotFoundError(f"File not found: {LATEST_RUNS_PATH}")

    with LATEST_RUNS_PATH.open("r") as f:
        latest_runs = json.load(f)

    min_weighted_f1 = float(os.getenv("MODEL_MIN_WEIGHTED_F1", "0.55"))
    validated_runs = {}
    client = mlflow.tracking.MlflowClient()

    for model_name, run_id in latest_runs.items():
        print(f"Validating model: {model_name}")
        metrics = validate_model(f"runs:/{run_id}/model", TEST_DATA_PATH)
        for metric_name, metric_value in metrics.items():
            client.log_metric(run_id, f"external_test_{metric_name}", metric_value)

        if metrics["weighted_f1"] >= min_weighted_f1:
            validated_runs[model_name] = run_id
            print(
                f"Validation passed: weighted F1 {metrics['weighted_f1']:.4f} "
                f">= {min_weighted_f1:.4f}"
            )
        else:
            print(
                f"Validation failed: weighted F1 {metrics['weighted_f1']:.4f} "
                f"< {min_weighted_f1:.4f}"
            )

    with VALIDATED_RUNS_PATH.open("w") as f:
        json.dump(validated_runs, f, indent=4)

    if not validated_runs:
        raise RuntimeError(
            f"No model passed the external test threshold of {min_weighted_f1:.4f}"
        )

    print(f"Validated {len(validated_runs)} model(s). Results saved to {VALIDATED_RUNS_PATH}")
