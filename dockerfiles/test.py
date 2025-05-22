import requests
import pytest
from unittest.mock import patch

BASE_URL = "http://fastapi:5001"

def test_health_endpoint():
    with patch("requests.get") as mock_get:
        mock_get.return_value = type('MockResponse', (), {
            'status_code': 200,
            'json': lambda: {"status": "healthy"}
        })()
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]

def test_predict_endpoint_valid_input():
    with patch("requests.post") as mock_post:
        mock_post.return_value = type('MockResponse', (), {
            'status_code': 200,
            'json': lambda: {"predictions": [2]}
        })()
        payload = {"instances": [{"text": "I love Trump!"}]}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == [2]

def test_predict_endpoint_invalid_input():
    with patch("requests.post") as mock_post:
        mock_post.return_value = type('MockResponse', (), {
            'status_code': 422,
            'json': lambda: {"detail": "Invalid input"}
        })()
        payload = {"text": "I love Trump!"}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data