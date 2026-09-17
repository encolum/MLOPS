import os
import glob
import time
import pandas as pd
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from datetime import datetime
import mlflow
import requests
from pathlib import Path

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:5001").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data_pipeline" / "processed"
LABELED_DIR = PROJECT_ROOT / "data_pipeline" / "labeled"

def wait_for_fastapi(timeout=30):
    """Chờ FastAPI server sẵn sàng trước khi gọi API"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{FASTAPI_URL}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                print("FastAPI server is ready.")
                return
        except (requests.exceptions.RequestException, ValueError):
            pass
        print("Waiting for FastAPI...")
        time.sleep(3)
    raise RuntimeError("FastAPI server not ready after waiting.")

# def get_champion_model_uri(prefix="sentiment_"):
#     """
#     Tìm model đang ở stage 'Production' trên MLflow Registry
#     """
#     client = MlflowClient()
#     for rm in client.search_registered_models():
#         if not rm.name.startswith(prefix):
#             continue
#         for mv in client.search_model_versions(f"name='{rm.name}'"):
#             if mv.current_stage == "Production":
#                 print(f"Found production model: {rm.name}, version: {mv.version}")
#                 return f"models:/{rm.name}/Production"
#     raise RuntimeError("Không tìm thấy model nào ở stage Production.")

def find_latest_processed_file(processed_dir=PROCESSED_DIR):
    """Lấy file .csv mới nhất trong thư mục processed/"""
    pattern = os.path.join(processed_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No processed files found in {processed_dir}")
    return max(files, key=os.path.getmtime)

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def predict_batch(batch_data):
    """Gửi batch dữ liệu đến endpoint /predict và lấy kết quả"""
    payload = {"instances": [{"text": str(text)} for text in batch_data]}
    response = requests.post(
        f"{FASTAPI_URL}/predict",
        json=payload,
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Prediction failed: {response.text}")
    return response.json()["predictions"]

def predict_in_batches(df, batch_size=20):
    """Chia dữ liệu thành batch và dự đoán"""
    predictions = []
    for i in range(0, len(df), batch_size):
        batch = df['cleaned_text'][i:i + batch_size].tolist()
        print(f"Predicting batch {i // batch_size + 1} ({len(batch)} samples)...")
        batch_preds = predict_batch(batch)
        predictions.extend(batch_preds)
    return predictions

def main():
    # 1. Đợi MLflow server sẵn sàng
    wait_for_fastapi()

    # # 2. Lấy model URI
    # model_uri = get_champion_model_uri(prefix="sentiment_")
    # print(f"Loading champion model from '{model_uri}'")
    # model = mlflow.pyfunc.load_model(model_uri)

    # 3. Đọc file processed mới nhất
    input_file = find_latest_processed_file()
    print(f"Reading processed data from '{input_file}'")
    df = pd.read_csv(input_file)

    if 'cleaned_text' not in df.columns:
        raise ValueError("Processed data must contain a 'cleaned_text' column")

    # 4. Chạy inference
    print("Running inference...")
    preds = predict_in_batches(df, batch_size=20)
    df['sentiment'] = preds

    # 6. Ghi kết quả ra thư mục labeled/
    out_dir = LABELED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"predicted_twitter_{date_str}.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved predictions to '{out_file}'")

if __name__ == "__main__":
    main()
