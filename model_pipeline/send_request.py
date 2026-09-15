import mlflow
import requests
from mlflow.tracking import MlflowClient
import os
from pathlib import Path
from dotenv import load_dotenv

MODEL_PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_PIPELINE_DIR.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
load_dotenv(MODEL_PIPELINE_DIR / ".env")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri()))
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:5001").rstrip("/")

def get_champion_model_info(prefix="sentiment_"):
    client = MlflowClient()
    for rm in client.search_registered_models():
        if rm.name.startswith(prefix):
            for v in client.search_model_versions(f"name='{rm.name}'"):
                if v.current_stage == "Production" or v.tags.get("champion") == "True":
                    return rm.name, v.version, v.current_stage
    return None, None, None

def send_request():
    # === Lấy thông tin mô hình champion ===
    model_name, model_version, model_stage = get_champion_model_info()
    if not model_name:
        raise RuntimeError("No champion model found")

    # === Dữ liệu test mẫu ===
    input_text = "Donald Trump is the 45th president of the United States."
    payload = {
        "instances": [
            {"text": input_text}
        ]
    }

    url = f"{FASTAPI_URL}/predict"

    # === Gửi request đến endpoint đã serve ===
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        prediction = response.json()["predictions"]
        print(f"Prediction: {prediction}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request error: {e}") from e

    # === Ghi log vào MLflow ===
    mlflow.set_experiment("sentiment-analysis")
    with mlflow.start_run(run_name="serve_request"):
        mlflow.log_param("input_text", input_text)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("model_stage", model_stage)
        if prediction is not None:
            mlflow.log_param("prediction", prediction[0] if isinstance(prediction, list) else str(prediction))

if __name__ == "__main__":
    send_request()
