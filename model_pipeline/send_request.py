# import mlflow
# import requests
# import json

# def send_request():
#     # Gửi yêu cầu dự đoán đến MLflow Serving API
#     url = 'http://localhost:5001/invocations'

#     # Định dạng đúng cho MLflow DataFrame input
#     data = {
#         "dataframe_records": [
#             {"text": "Donald Trump is stupid."}
#         ]
#     }

#     headers = {'Content-Type': 'application/json'}
#     response = requests.post(url, data=json.dumps(data), headers=headers)

#     # Kiểm tra và in kết quả dự đoán
#     if response.status_code == 200:
#         prediction = response.json()
#         print(f"Prediction result: {prediction}")
#     else:
#         print(f"Failed to get prediction, status code: {response.status_code}, response: {response.text}")
#         prediction = None

#     # Log kết quả vào MLflow experiment 'sentiment-analysis'
#     mlflow.set_experiment("sentiment-analysis")
#     with mlflow.start_run():
#         mlflow.log_param("input_text", data["dataframe_records"][0]["text"])
#         if prediction is not None:
#             mlflow.log_param("prediction", prediction)

# if __name__ == "__main__":
#     send_request()
import mlflow
import requests
import json
from mlflow.tracking import MlflowClient

def get_champion_model_info(prefix="sentiment_"):
    client = MlflowClient()
    for rm in client.search_registered_models():
        if rm.name.startswith(prefix):
            for v in client.search_model_versions(f"name='{rm.name}'"):
                # Check for champion tag or Production stage
                if v.current_stage == "Production" or v.tags.get("champion") == "True":
                    return rm.name, v.version, v.current_stage
    return None, None, None

def send_request():
    # Get champion model info automatically
    model_name, model_version, model_stage = get_champion_model_info()
    if not model_name:
        print("No champion model found.")
        return

    # Gửi yêu cầu dự đoán đến MLflow Serving API
    url = 'http://localhost:5001/invocations'

    data = {
        "dataframe_records": [
            {"text": "Donald Trump is the 45th president of the United States."}
        ]
    }

    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)

    # Kiểm tra và in kết quả dự đoán
    if response.status_code == 200:
        prediction = response.json()
        print(f"Prediction result: {prediction}")
    else:
        print(f"Failed to get prediction, status code: {response.status_code}, response: {response.text}")
        prediction = None

    # Log kết quả vào MLflow experiment 'sentiment-analysis'
    mlflow.set_experiment("sentiment-analysis")
    with mlflow.start_run():
        mlflow.log_param("input_text", data["dataframe_records"][0]["text"])
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("model_stage", model_stage)
        if prediction is not None:
            mlflow.log_param("prediction", prediction)

if __name__ == "__main__":
    send_request()