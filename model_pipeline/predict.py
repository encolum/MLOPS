
import os
import glob
import pandas as pd
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from datetime import datetime
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

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
                return f"models:/{rm.name}/Production"
    raise RuntimeError("Không tìm thấy champion model ở stage Production")

def find_latest_processed_file(processed_dir="/mnt/d/MLOps2/data/processed"):
    """
    Lấy file mới nhất trong thư mục processed/ có dạng .csv
    """
    pattern = os.path.join(processed_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No processed CSV files found in {processed_dir}")
    return max(files, key=os.path.getmtime)

def main():
    # 1. Lấy model URI
    model_uri = get_champion_model_uri(prefix="sentiment_")
    print(f"Loading champion model from '{model_uri}'")
    model = mlflow.pyfunc.load_model(model_uri)

    # 2. Đọc file processed mới nhất
    input_file = find_latest_processed_file("/mnt/d/MLOps2/data/processed")
    print(f"Reading processed data from '{input_file}'")
    df = pd.read_csv(input_file)

    # 3. Chuẩn bị input cho model (giả sử model nhận DataFrame với cột 'cleaned_text')
    model_input = df[['cleaned_text']].rename(columns={'cleaned_text':'text'})

    # 4. Chạy inference
    print("Running inference...")
    preds = model.predict(model_input)
    df['Sentiment'] = preds

    # 5. Ghi ra thư mục predicted/
    out_dir = "/mnt/d/MLOps2/data/labeled"
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_file = os.path.join(out_dir, f"predicted_twitter_{date_str}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved predictions to '{out_file}'")

if __name__ == "__main__":
    main()
