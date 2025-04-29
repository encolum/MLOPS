import mlflow
from mlflow.tracking import MlflowClient
import subprocess

# Lấy tất cả các mô hình đã đăng ký với prefix nhất định
def get_registered_models(prefix="sentiment_"):
    client = MlflowClient()
    try:
        registered_models = [
            rm.name for rm in client.search_registered_models()
            if rm.name.startswith(prefix)
        ]
        if not registered_models:
            print(f"No registered models found with prefix: {prefix}")
        return registered_models
    except mlflow.exceptions.MlflowException as e:
        print(f"Error fetching registered models: {str(e)}")
        return []

# Tìm mô hình tốt nhất dựa trên f1_score
def find_best_model(registered_models):
    client = MlflowClient()
    best_f1 = -1
    best_model_name = None
    best_run_id = None
    challenger_f1 = -1
    challenger_model_name = None
    challenger_run_id = None

    for model_name in registered_models:
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            for v in versions:
                run_id = v.run_id
                run = client.get_run(run_id)
                f1_score = run.data.metrics.get("f1_score", -1)
                if f1_score > best_f1:
                    # Nếu f1_score cao nhất, thì là champion
                    best_f1 = f1_score
                    best_model_name = model_name
                    best_run_id = run_id
                elif f1_score > challenger_f1:
                    # Nếu f1_score cao hơn challenger nhưng thấp hơn champion
                    challenger_f1 = f1_score
                    challenger_model_name = model_name
                    challenger_run_id = run_id
        except mlflow.exceptions.MlflowException as e:
            print(f"Error fetching model versions for {model_name}: {str(e)}")

    if best_model_name:
        print(f"Best model found: {best_model_name} (F1={best_f1:.4f})")
    if challenger_model_name:
        print(f"Challenger model found: {challenger_model_name} (F1={challenger_f1:.4f})")
    return best_model_name, best_run_id, best_f1, challenger_model_name, challenger_run_id, challenger_f1

# Cập nhật tag cho mô hình tốt nhất và challenger (sử dụng tags thay vì alias)
def update_tags(best_model_name, best_run_id, best_f1, challenger_model_name, challenger_run_id, challenger_f1):
    client = MlflowClient()

    if not best_model_name or not challenger_model_name:
        print("No best model or challenger available for tag update.")
        return

    try:
        # Cập nhật tag cho champion
        versions = client.search_model_versions(f"name='{best_model_name}'")
        best_version = None
        for v in versions:
            if v.run_id == best_run_id:
                best_version = v.version
                break

        if best_version is None:
            raise Exception(f"Version for run_id {best_run_id} not found for model {best_model_name}.")

        best_version_int = int(best_version)
        client.set_model_version_tag(best_model_name, best_version_int, "champion", "True")
        print(f"Set version {best_version_int} of model {best_model_name} as Champion.")

        # Cập nhật tag cho challenger
        versions = client.search_model_versions(f"name='{challenger_model_name}'")
        challenger_version = None
        for v in versions:
            if v.run_id == challenger_run_id:
                challenger_version = v.version
                break

        if challenger_version is None:
            raise Exception(f"Version for run_id {challenger_run_id} not found for model {challenger_model_name}.")

        challenger_version_int = int(challenger_version)
        client.set_model_version_tag(challenger_model_name, challenger_version_int, "challenger", "True")
        print(f"Set version {challenger_version_int} of model {challenger_model_name} as Challenger.")

    except mlflow.exceptions.MlflowException as e:
        print(f"Error while updating tags for {best_model_name}: {str(e)}")

# Lấy URI của mô hình champion và đảm bảo mô hình được chuyển sang production
def get_model_uri(best_model_name):
    client = MlflowClient()

    try:
        # Tìm mô hình có tag "champion" trong stage "production"
        versions = client.search_model_versions(f"name='{best_model_name}'")
        for v in versions:
            version = client.get_model_version(name=best_model_name, version=v.version)
            if version.tags.get('champion') == 'True':  # Kiểm tra tag champion
                model_uri = f"models:/{best_model_name}/production"
                print(f"Model URI for champion: {model_uri}")
                
                # Chuyển mô hình vào stage "production" nếu chưa
                client.transition_model_version_stage(
                    name=best_model_name,
                    version=v.version,
                    stage="Production"
                )
                print(f"Model {best_model_name} version {v.version} moved to 'production' stage.")
                
                return model_uri
        raise Exception(f"No champion model found for {best_model_name}.")
    
    except mlflow.exceptions.MlflowException as e:
        print(f"Error while fetching champion model URI: {str(e)}")
        return None

# Hàm để phục vụ mô hình champion tự động
def serve_model_automatically():
    # Step 1: Lấy tất cả các mô hình đã đăng ký
    registered_models = get_registered_models(prefix="sentiment_")
    if not registered_models:
        print("No registered models found.")
        return

    # Step 2: Tìm mô hình tốt nhất
    best_model_name, best_run_id, best_f1, challenger_model_name, challenger_run_id, challenger_f1 = find_best_model(registered_models)

    # Step 3: Cập nhật tag cho mô hình champion và challenger
    update_tags(best_model_name, best_run_id, best_f1, challenger_model_name, challenger_run_id, challenger_f1)

    # Step 4: Lấy model_uri cho mô hình champion và chuyển stage thành "production"
    model_uri = get_model_uri(best_model_name)

    if model_uri:
        # Step 5: Serve mô hình champion tự động
        print(f"Serving the model: {best_model_name}")
        subprocess.run([
        "mlflow", "models", "serve",
        "--model-uri", model_uri,
        "--host", "0.0.0.0",
        "--port", "5001",
        "--no-conda"
        ])

if __name__ == "__main__":
    # Tự động chọn mô hình champion và phục vụ
    serve_model_automatically()
