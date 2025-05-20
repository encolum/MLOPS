import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# === Thiết lập tracking URI ===
# tracking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mlruns"))
# mlflow.set_tracking_uri(f"file://{tracking_path}")
tracking_path = "/app/mlruns"  # Đường dẫn trong container
mlflow.set_tracking_uri(f"file://{tracking_path}")
# mlflow.set_experiment("sentiment-analysis")
# === Bước 1: Lấy danh sách mô hình theo prefix ===
def get_registered_models(prefix="sentiment_"):
    client = MlflowClient()
    try:
        return [
            rm.name for rm in client.search_registered_models()
            if rm.name.startswith(prefix)
        ]
    except mlflow.exceptions.MlflowException as e:
        print(f"Lỗi khi lấy model registry: {e}")
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
            f1 = run.data.metrics.get("f1_score", -1)
            if f1 > best_f1:
                challenger_f1, challenger_model, challenger_run_id = best_f1, best_model, best_run_id
                best_f1, best_model, best_run_id = f1, model, v.run_id
            elif f1 > challenger_f1:
                challenger_f1, challenger_model, challenger_run_id = f1, model, v.run_id

    print(f"Champion: {best_model} (F1={best_f1:.4f})")
    if challenger_model:
        print(f"⚔️  Challenger: {challenger_model} (F1={challenger_f1:.4f})")
    return best_model, best_run_id, best_f1, challenger_model, challenger_run_id, challenger_f1

# === Bước 3: Gắn tag champion/challenger ===
def update_tags(best_model, best_run_id, _, challenger_model, challenger_run_id, __):
    client = MlflowClient()

    def set_tag(model, run_id, tag):
        versions = client.search_model_versions(f"name='{model}'")
        for v in versions:
            if v.run_id == run_id:
                client.set_model_version_tag(model, int(v.version), tag, "True")
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
                name=best_model, version=v.version, stage="Production", archive_existing_versions=True
            )
            print(f"Promoted {best_model} v{v.version} to Production.")
            return f"models:/{best_model}/production"
    print("Không tìm thấy champion.")
    return None

# === Bước 5: Khởi tạo FastAPI và load mô hình ===
app = FastAPI(title="Sentiment Analysis API", description="API for serving MLflow sentiment models")

# Định nghĩa schema cho input
class PredictionInput(BaseModel):
    text: str

class PredictionRequest(BaseModel):
    instances: List[PredictionInput]

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
        print("Không tìm thấy mô hình nào trong registry.")
        return

    best_model, best_run_id, best_f1, challenger_model, challenger_run_id, challenger_f1 = find_best_model(models)
    print(f"Best model: {best_model}, Best F1: {best_f1}")
    update_tags(best_model, best_run_id, best_f1, challenger_model, challenger_run_id, challenger_f1)

    uri = get_model_uri(best_model)
    print(f"Model URI: {uri}")
    if uri:
        print(f"Loading model from URI: {uri}")
        try:
            model = mlflow.pyfunc.load_model(uri)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Không thể load mô hình: {e}")
            model = None
    else:
        print("Không thể load mô hình vì không tìm thấy URI.")

# Endpoint kiểm tra sức khỏe
@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {"status": "healthy"}

# Endpoint dự đoán
@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Chuyển đổi input thành định dạng MLflow
        input_data = [{"text": item.text} for item in request.instances]
        # Dự đoán
        predictions = model.predict(input_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
# @app.post("/validate")
# async def validate_model_endpoint():
#     latest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../latest_runs.json"))
#     try:
#         with open(latest_path, "r") as f:
#             latest_runs = json.load(f)
#     except FileNotFoundError:
#         raise HTTPException(status_code=500, detail=f"File not found: {latest_path}")
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail="Invalid JSON format in latest_runs.json")

#     results = {}
#     test_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../test_data.csv"))
#     for model_name, run_id in latest_runs.items():
#         try:
#             model_uri = f"runs:/{run_id}/model"
#             test_df = pd.read_csv(test_data_path)
#             if 'text' not in test_df.columns or 'sentiment_num' not in test_df.columns:
#                 raise ValueError("CSV phải có cột 'text' và 'sentiment_num'")
#             if test_df['text'].isnull().any():
#                 raise ValueError("Cột 'text' chứa giá trị NaN")
#             if not test_df['text'].apply(lambda x: isinstance(x, str)).all():
#                 raise ValueError("Cột 'text' phải chứa chuỗi (string)")

#             X_test = test_df[['text']]
#             y_test = test_df['sentiment_num']
#             model = mlflow.pyfunc.load_model(model_uri)
#             predictions = model.predict(X_test)

#             acc = accuracy_score(y_test, predictions)
#             prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
#             rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
#             f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

#             results[model_name] = {
#                 "accuracy": acc,
#                 "precision": prec,
#                 "recall": rec,
#                 "f1_score": f1
#             }
#         except Exception as e:
#             logger.error(f"Validation error for {model_name}: {str(e)}")
#             results[model_name] = {"error": str(e)}
#             continue  # Tiếp tục với model tiếp theo thay vì dừng

#     # Kiểm tra nếu tất cả model đều lỗi
#     if all("error" in result for result in results.values()):
#         raise HTTPException(status_code=500, detail="Validation failed for all models")
#     return results
# === Chạy server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)