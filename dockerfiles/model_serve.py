import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

DEFAULT_MLRUNS_DIR = Path(__file__).resolve().parent / "mlruns"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLRUNS_DIR.as_uri()))


# mlflow.set_experiment("sentiment-analysis")
# === Bước 1: Lấy danh sách mô hình theo prefix ===
def get_registered_models(prefix="sentiment_"):
    client = MlflowClient()
    try:
        return [
            rm.name
            for rm in client.search_registered_models()
            if rm.name.startswith(prefix)
        ]
    except mlflow.exceptions.MlflowException as e:
        print(f"Failed to fetch model registry: {e}")
        return []


# === Bước 2: Tìm champion và challenger ===
def find_best_model(registered_models):
    client = MlflowClient()
    best_f1 = -1
    challenger_f1 = -1
    best_model, challenger_model = (None,) * 2
    best_run_id, challenger_run_id = (None,) * 2

    for model in registered_models:
        versions = client.search_model_versions(f"name='{model}'")
        for v in versions:
            run = client.get_run(v.run_id)
            f1 = run.data.metrics.get(
                "external_test_weighted_f1",
                run.data.metrics.get("f1_score", -1),
            )
            if f1 > best_f1:
                challenger_f1, challenger_model, challenger_run_id = (
                    best_f1,
                    best_model,
                    best_run_id,
                )
                best_f1, best_model, best_run_id = f1, model, v.run_id
            elif f1 > challenger_f1:
                challenger_f1, challenger_model, challenger_run_id = f1, model, v.run_id

    print(f"Champion: {best_model} (F1={best_f1:.4f})")
    if challenger_model:
        print(f"Challenger: {challenger_model} (F1={challenger_f1:.4f})")
    return (
        best_model,
        best_run_id,
        best_f1,
        challenger_model,
        challenger_run_id,
        challenger_f1,
    )


# === Bước 3: Gắn tag champion/challenger ===
def update_tags(best_model, best_run_id, _, challenger_model, challenger_run_id, __):
    client = MlflowClient()

    def set_tag(model, run_id, tag):
        versions = client.search_model_versions(f"name='{model}'")
        for v in versions:
            if v.run_id == run_id:
                client.set_model_version_tag(model, v.version, tag, "True")
                print(f" Set {tag} tag for {model} v{v.version}")
                return

    if best_model and best_run_id:
        set_tag(best_model, best_run_id, "champion")
    if challenger_model and challenger_run_id:
        set_tag(challenger_model, challenger_run_id, "challenger")


# === Bước 4: Lấy URI của mô hình champion và chuyển sang Production ===
def get_model_uri(best_model):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{best_model}'")
    for v in versions:
        if v.tags.get("champion") == "True":
            client.transition_model_version_stage(
                name=best_model,
                version=v.version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"Promoted {best_model} v{v.version} to Production.")
            return f"models:/{best_model}/production"
    print("No champion found.")
    return None


# === Bước 5: Khởi tạo FastAPI và load mô hình ===
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for serving MLflow sentiment models",
)


# Định nghĩa schema cho input
class PredictionInput(BaseModel):
    text: str


class PredictionRequest(BaseModel):
    instances: list[PredictionInput]


# Biến toàn cục để lưu mô hình
model = None


# Hàm khởi tạo và load mô hình khi server khởi động
@app.on_event("startup")
async def startup_event():
    global model
    print("Starting model loading process...")
    models = get_registered_models()
    print(f"Registered models: {models}")
    if not models:
        print("No registered models found.")
        return

    (
        best_model,
        best_run_id,
        best_f1,
        challenger_model,
        challenger_run_id,
        challenger_f1,
    ) = find_best_model(models)
    print(f"Best model: {best_model}, Best F1: {best_f1}")
    update_tags(
        best_model,
        best_run_id,
        best_f1,
        challenger_model,
        challenger_run_id,
        challenger_f1,
    )

    uri = get_model_uri(best_model)
    print(f"Model URI: {uri}")
    if uri:
        print(f"Loading model from URI: {uri}")
        try:
            model = mlflow.pyfunc.load_model(uri)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            model = None
    else:
        print("Failed to load model: model URI not found.")


# Endpoint kiểm tra sức khỏe
@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


# Endpoint dự đoán
@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Chuyển đổi input thành định dạng MLflow
        input_data = pd.DataFrame({"text": [item.text for item in request.instances]})
        # Dự đoán
        predictions = model.predict(input_data)
        prediction_list = (
            predictions.tolist()
            if hasattr(predictions, "tolist")
            else list(predictions)
        )
        return {"predictions": prediction_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e!s}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
