import mlflow
import json

def register_models(model_name_prefix="sentiment"):
    with open("latest_runs.json", "r") as f:
        latest_runs = json.load(f)

    client = mlflow.tracking.MlflowClient()

    for model_name, run_id in latest_runs.items():
        full_model_name = f"{model_name_prefix}_{model_name}"
        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(model_uri, full_model_name)
        print(f"Registered model {full_model_name}, version {result.version}")

if __name__ == "__main__":
    register_models()


# if __name__ == "__main__":
#     # đọc run_id vừa train từ latest_run.txt
#     with open("latest_run.txt", "r") as f:
#         run_id = f.read().strip()

#     model_names = [
#         "LogisticRegression_TFIDF",
#         "BERT_Transformer",
#         "RoBERTa_Transformer",
#         "DistilBERT_Transformer",
#         "VADER_Sentiment",
#     ]

#     for name in model_names:
#         model_uri = f"runs:/{run_id}/{name}_model"
#         register_model(
#             model_uri=model_uri,
#             model_name=name,
#             tags={"author": "mlops", "task": "sentiment_analysis"}
#         )
