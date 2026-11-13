# MLOps: Tweet Sentiment Pipeline

An end-to-end MLOps pipeline that collects tweets, labels their sentiment with an LLM, trains a transformer classifier, and serves the best model behind a REST API. Every step is orchestrated by Airflow, tracked in MLflow, and shipped through a CI/CD workflow.

## Architecture

```
crawl (twikit) ─► preprocess ─► label (Gemini) ─► validate ─► ingest (PostgreSQL)
                                                                   │
           ┌───────────────────────────────────────────────────────┘
           ▼
  train (DistilBERT, MLflow) ─► validate (weighted F1 gate) ─► register ─► serve (FastAPI)
                                                                              │
                                                       predictions ─► Grafana dashboard
```

| Stage | Code | What it does |
|---|---|---|
| Data | `data_pipeline/` | Crawls tweets, cleans text, labels sentiment with `gemini-2.0-flash-lite`, validates and loads the data into PostgreSQL |
| Training | `model_pipeline/model_training.py` | Fine-tunes `distilbert-base-uncased`, logs runs and artifacts to MLflow |
| Validation | `model_pipeline/model_validate.py` | Scores candidates on held-out data and only promotes models above `MODEL_MIN_WEIGHTED_F1` |
| Registry | `model_pipeline/model_deploy.py` | Registers validated runs in the MLflow Model Registry |
| Serving | `dockerfiles/model_serve.py` | FastAPI service (`GET /health`, `POST /predict`) that loads the best registered model |
| Orchestration | `airflow/dags/full_dag.py` | One DAG: crawl → preprocess → validate → ingest → train (if needed) → validate → deploy → serve → predict |
| Monitoring | `data_pipeline/connect_grafana.py` | Connects Grafana to PostgreSQL and provisions the dashboard |
| CI/CD | `.github/workflows/ci-cd.yml` | Lint, build the image, smoke-test the real API container, push to Docker Hub |

## Tech stack

Python · PyTorch · Hugging Face Transformers · MLflow · Apache Airflow · FastAPI · PostgreSQL · Grafana · Docker · GitHub Actions

## Getting started

```bash
cp .env.example .env          # fill in your own values
pip install -r dockerfiles/requirements.txt

# serve the model API
docker compose -f dockerfiles/docker-compose.yml up --build
curl -X POST localhost:5001/predict -H "Content-Type: application/json" -d '{"instances": [{"text": "great news today"}]}'
```

To run the full pipeline, start Airflow with `airflow/` as `AIRFLOW_HOME` and trigger the `full_dag` DAG.

The CI/CD workflow needs two repository secrets: `DOCKER_USERNAME` and `DOCKERHUB_TOKEN`.

## Team

- Mai Thanh Loc ([@encolum](https://github.com/encolum))
- [@pphanhh](https://github.com/pphanhh)
