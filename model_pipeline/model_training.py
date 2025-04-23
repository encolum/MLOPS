# model_training.py: trains new models using MLflow and logs the model.
# This script includes loading data from feature_store in Postgre database
# and training a model using different algorithms.
# Input: Postgre database file containing features.
# Output: MLflow model logged to the tracking server.