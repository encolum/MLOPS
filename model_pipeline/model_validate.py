import pandas as pd
import mlflow
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def validate_model(model_uri, test_data_path):
    test_df = pd.read_csv(test_data_path)
    X_test = test_df[['text']]  # Nếu cần đổi cột phù hợp
    y_test = test_df['sentiment_num']

    model = mlflow.pyfunc.load_model(model_uri)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted')
    rec = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}")

if __name__ == "__main__":
    with open("latest_runs.json", "r") as f:
        latest_runs = json.load(f)

    for model_name, run_id in latest_runs.items():
        print(f" Validating model: {model_name}")
        validate_model(f"runs:/{run_id}/model", "./test_data.csv")
