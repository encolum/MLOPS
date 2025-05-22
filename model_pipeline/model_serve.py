# import mlflow
# from mlflow.tracking import MlflowClient
# import subprocess
# import os

# # === Thiết lập tracking URI ===
# tracking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mlruns"))
# mlflow.set_tracking_uri(f"file://{tracking_path}")

# # === Bước 1: Lấy danh sách mô hình theo prefix ===
# def get_registered_models(prefix="sentiment_"):
#     client = MlflowClient()
#     try:
#         return [
#             rm.name for rm in client.search_registered_models()
#             if rm.name.startswith(prefix)
#         ]
#     except mlflow.exceptions.MlflowException as e:
#         print(f"Lỗi khi lấy model registry: {e}")
#         return []

# # === Bước 2: Tìm champion và challenger ===
# def find_best_model(registered_models):
#     client = MlflowClient()
#     best_f1 = -1
#     challenger_f1 = -1
#     best_model, challenger_model = (None,) * 2
#     best_run_id, challenger_run_id = (None,) * 2

#     for model in registered_models:
#         versions = client.search_model_versions(f"name='{model}'")
#         for v in versions:
#             run = client.get_run(v.run_id)
#             f1 = run.data.metrics.get("f1_score", -1)
#             if f1 > best_f1:
#                 challenger_f1, challenger_model, challenger_run_id = best_f1, best_model, best_run_id
#                 best_f1, best_model, best_run_id = f1, model, v.run_id
#             elif f1 > challenger_f1:
#                 challenger_f1, challenger_model, challenger_run_id = f1, model, v.run_id

#     print(f"Champion: {best_model} (F1={best_f1:.4f})")
#     if challenger_model:
#         print(f"⚔️  Challenger: {challenger_model} (F1={challenger_f1:.4f})")
#     return best_model, best_run_id, best_f1, challenger_model, challenger_run_id, challenger_f1

# # === Bước 3: Gắn tag champion/challenger ===
# def update_tags(best_model, best_run_id, _, challenger_model, challenger_run_id, __):
#     client = MlflowClient()

#     def set_tag(model, run_id, tag):
#         versions = client.search_model_versions(f"name='{model}'")
#         for v in versions:
#             if v.run_id == run_id:
#                 client.set_model_version_tag(model, int(v.version), tag, "True")
#                 print(f" Set {tag} tag for {model} v{v.version}")
#                 return

#     if best_model and best_run_id:
#         set_tag(best_model, best_run_id, "champion")
#     if challenger_model and challenger_run_id:
#         set_tag(challenger_model, challenger_run_id, "challenger")

# # === Bước 4: Lấy URI của mô hình champion và chuyển sang Production ===
# def get_model_uri(best_model):
#     client = MlflowClient()
#     versions = client.search_model_versions(f"name='{best_model}'")
#     for v in versions:
#         if v.tags.get("champion") == "True":
#             client.transition_model_version_stage(
#                 name=best_model, version=v.version, stage="Production", archive_existing_versions=True
#             )
#             print(f"Promoted {best_model} v{v.version} to Production.")
#             return f"models:/{best_model}/production"
#     print("Không tìm thấy champion.")
#     return None

# # === Bước 5: Serve mô hình ===

# def serve_model_automatically():
#     models = get_registered_models()
#     if not models:
#         print("Không tìm thấy mô hình nào.")
#         return

#     best_model, best_run_id, best_f1, challenger_model, challenger_run_id, challenger_f1 = find_best_model(models)
#     update_tags(best_model, best_run_id, best_f1, challenger_model, challenger_run_id, challenger_f1)

#     uri = get_model_uri(best_model)
#     if uri:
#         print(f"Serving model from URI: {uri}")
#         log_path = os.path.join(os.path.dirname(__file__), "mlflow_serve.log")
#         working_dir = os.path.abspath(os.path.dirname(__file__)) 

#         with open(log_path, "w") as log_file:
#             subprocess.Popen(
#                 [
#                     "mlflow", "models", "serve",
#                     "--model-uri", uri,
#                     "--host", "0.0.0.0",
#                     "--port", "5001",
#                     "--no-conda"
#                 ],
#                 stdout=log_file,
#                 stderr=log_file,
#                 cwd=working_dir  
#             )
#         print(f"Serve started in background. Logs: {log_path}")
#     else:
#         print("Không thể serve vì không tìm thấy URI.")

# if __name__ == "__main__":
#     serve_model_automatically()
import mlflow
from mlflow.tracking import MlflowClient
import subprocess
import os
import json

# === Setup ===
tracking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mlruns"))
mlflow.set_tracking_uri(f"file://{tracking_path}")
client = MlflowClient()

def get_current_champion():
    # Find any model in Production stage
    for rm in client.search_registered_models():
        for v in client.search_model_versions(f"name='{rm.name}'"):
            if v.current_stage == "Production":
                run = client.get_run(v.run_id)
                f1 = run.data.metrics.get("f1_score", -1)
                return rm.name, v.version, v.run_id, f1
    return None, None, None, -1

def get_latest_runs():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../latest_runs.json"))
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def find_best_challenger(latest_runs):
    best_model = None
    best_f1 = -1
    best_run_id = None
    for model_name, run_id in latest_runs.items():
        try:
            run = client.get_run(run_id)
            f1 = run.data.metrics.get("f1_score", -1)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name
                best_run_id = run_id
        except:
            continue
    return best_model, best_run_id, best_f1

def promote_if_better():
    latest_runs = get_latest_runs()
    if not latest_runs:
        print("⚠️ No latest runs found.")
        return None

    # Challenger = best in latest_runs
    challenger_model, challenger_run_id, challenger_f1 = find_best_challenger(latest_runs)
    if not challenger_model:
        print("⚠️ No challenger found.")
        return None

    # Champion = any model in Production
    champion_model, champion_version, champion_run_id, champion_f1 = get_current_champion()

    print(f"Champion: {champion_model} (F1={champion_f1:.4f}), Challenger: {challenger_model} (F1={challenger_f1:.4f})")

    if challenger_f1 > champion_f1:
        versions = client.search_model_versions(f"name='{challenger_model}'")
        for v in versions:
            if v.run_id == challenger_run_id:
                version_int = int(v.version)
                client.transition_model_version_stage(
                    name=challenger_model,
                    version=version_int,
                    stage="Production",
                    archive_existing_versions=True
                )
                client.set_model_version_tag(challenger_model, version_int, "champion", "True")
                print(f"Promoted v{version_int} of {challenger_model} to Production.")
                # Save champion info for prediction step
                champion_info = {"name": challenger_model, "version": version_int}
                with open("/mnt/d/python/MLOps/clone/MLOPS/current_champion.json", "w") as f:
                    json.dump(champion_info, f)
                return challenger_model
    else:
        print("Challenger is not better. Keep current champion.")
        # Save current champion info for prediction step
        if champion_model and champion_version:
            champion_info = {"name": champion_model, "version": champion_version}
            with open("/mnt/d/python/MLOps/clone/MLOPS/current_champion.json", "w") as f:
                json.dump(champion_info, f)
        return champion_model  # Serve the current champion


def serve_model(model_name: str, port=5001):
    uri = f"models:/{model_name}/Production"
    log_path = os.path.join(os.path.dirname(__file__), "mlflow_serve.log")

    with open(log_path, "w") as log_file:
        subprocess.Popen(
            [
                "mlflow", "models", "serve",
                "--model-uri", uri,
                "--host", "0.0.0.0",
                "--port", str(port),
                "--no-conda"
            ],
            stdout=log_file,
            stderr=log_file,
            cwd=os.path.dirname(__file__)
        )
    print(f"Serving model {model_name} on {uri} (port {port})")

if __name__ == "__main__":
    champion_model = promote_if_better()
    if champion_model:
        serve_model(champion_model)
