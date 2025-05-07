import os
import glob
import time
import pandas as pd
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from datetime import datetime
import mlflow
import requests

MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def wait_for_mlflow(timeout=30):
    """Chờ MLflow server sẵn sàng trước khi gọi API"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(MLFLOW_TRACKING_URI)
            if r.status_code == 200:
                print("MLflow server is ready.")
                return
        except requests.exceptions.ConnectionError:
            pass
        print("Waiting for MLflow...")
        time.sleep(3)
    raise RuntimeError("MLflow server not ready after waiting.")

def get_champion_model_uri(prefix="sentiment_"):
    """
    Tìm model đang ở stage 'Production' trên MLflow Registry
    """
    client = MlflowClient()
    for rm in client.search_registered_models():
        if not rm.name.startswith(prefix):
            continue
        for mv in client.search_model_versions(f"name='{rm.name}'"):
            if mv.current_stage == "Production":
                print(f"Found production model: {rm.name}, version: {mv.version}")
                return f"models:/{rm.name}/Production"
    raise RuntimeError("Không tìm thấy model nào ở stage Production.")

def find_latest_processed_file(processed_dir="/mnt/d/MLOps2/data/processed"):
    """
    Lấy file .csv mới nhất trong thư mục processed/
    """
    pattern = os.path.join(processed_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file processed nào trong {processed_dir}")
    return max(files, key=os.path.getmtime)

def main():
    # 1. Đợi MLflow server sẵn sàng
    wait_for_mlflow()

    # 2. Lấy model URI
    model_uri = get_champion_model_uri(prefix="sentiment_")
    print(f"Loading champion model from '{model_uri}'")
    model = mlflow.pyfunc.load_model(model_uri)

    # 3. Đọc file processed mới nhất
    input_file = find_latest_processed_file("/mnt/d/MLOps2/data/processed")
    print(f"Reading processed data from '{input_file}'")
    df = pd.read_csv(input_file)

    # 4. Chuẩn bị input cho model
    model_input = df[['cleaned_text']].rename(columns={'cleaned_text': 'text'})

    # 5. Chạy inference
    print("Running inference...")
    preds = model.predict(model_input)
    df['sentiment'] = preds

    # 6. Ghi kết quả ra thư mục labeled/
    out_dir = "/mnt/d/MLOps2/data/labeled"
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_file = os.path.join(out_dir, f"predicted_twitter_{date_str}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved predictions to '{out_file}'")

if __name__ == "__main__":
    main()
