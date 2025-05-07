import mlflow
import json
import os

# 🔧 Thiết lập đúng tracking URI trỏ tới thư mục mlruns ở cấp trên
tracking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mlruns"))
mlflow.set_tracking_uri(f"file://{tracking_path}")

def register_models(model_name_prefix="sentiment"):
    # Đường dẫn tuyệt đối đến latest_runs.json tại thư mục MLOPS
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_path = os.path.join(base_dir, "latest_runs.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with open(file_path, "r") as f:
        latest_runs = json.load(f)

    client = mlflow.tracking.MlflowClient()

    for model_name, run_id in latest_runs.items():
        full_model_name = f"{model_name_prefix}_{model_name}"
        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(model_uri, full_model_name)
        print(f"Registered model {full_model_name}, version {result.version}")

if __name__ == "__main__":
    register_models()
