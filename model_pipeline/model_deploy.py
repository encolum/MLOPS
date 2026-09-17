import mlflow
import json
import os
from pathlib import Path
from dotenv import load_dotenv

MODEL_PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_PIPELINE_DIR.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
VALIDATED_RUNS_PATH = PROJECT_ROOT / "validated_runs.json"
load_dotenv(MODEL_PIPELINE_DIR / ".env")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri()))

def register_models(model_name_prefix="sentiment"):
    if not VALIDATED_RUNS_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {VALIDATED_RUNS_PATH}. Run model_validate.py first."
        )

    with VALIDATED_RUNS_PATH.open("r") as f:
        validated_runs = json.load(f)

    if not validated_runs:
        raise RuntimeError("No validated models are available for deployment")

    client = mlflow.tracking.MlflowClient()

    for model_name, run_id in validated_runs.items():
        full_model_name = f"{model_name_prefix}_{model_name}"
        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(model_uri, full_model_name)
        print(f"Registered model {full_model_name}, version {result.version}")

if __name__ == "__main__":
    register_models()
