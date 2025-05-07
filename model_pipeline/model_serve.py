import mlflow
from mlflow.tracking import MlflowClient
import subprocess
import os

# === Thiết lập tracking URI ===
tracking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mlruns"))
mlflow.set_tracking_uri(f"file://{tracking_path}")

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

# === Bước 5: Serve mô hình ===

def serve_model_automatically():
    models = get_registered_models()
    if not models:
        print("Không tìm thấy mô hình nào.")
        return

    best_model, best_run_id, best_f1, challenger_model, challenger_run_id, challenger_f1 = find_best_model(models)
    update_tags(best_model, best_run_id, best_f1, challenger_model, challenger_run_id, challenger_f1)

    uri = get_model_uri(best_model)
    if uri:
        print(f"Serving model from URI: {uri}")
        log_path = os.path.join(os.path.dirname(__file__), "mlflow_serve.log")
        working_dir = os.path.abspath(os.path.dirname(__file__)) 

        with open(log_path, "w") as log_file:
            subprocess.Popen(
                [
                    "mlflow", "models", "serve",
                    "--model-uri", uri,
                    "--host", "0.0.0.0",
                    "--port", "5001",
                    "--no-conda"
                ],
                stdout=log_file,
                stderr=log_file,
                cwd=working_dir  
            )
        print(f"Serve started in background. Logs: {log_path}")
    else:
        print("Không thể serve vì không tìm thấy URI.")

if __name__ == "__main__":
    serve_model_automatically()
