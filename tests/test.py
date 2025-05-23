# tests/test_model_serve.py
import requests
import pytest

# URL của API (chạy local trên port 5001)
BASE_URL = "http://localhost:5001"

def test_health_endpoint():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "unhealthy"]

def test_predict_endpoint_valid_input():
    # Test với input hợp lệ
    payload = {"instances": [{"text": "I love Trump!"}]}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 1
    assert data["predictions"][0] in [0,1,2]

def test_predict_endpoint_invalid_input():
    # Test với input không hợp lệ (thiếu "instances")
    payload = {"text": "I love Trump!"}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 422  # FastAPI tự động trả 422 cho input sai định dạng
    data = response.json()
    assert "detail" in data
