from unittest.mock import MagicMock, patch

def test_health_endpoint():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]

def test_predict_endpoint_valid_input():
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"predictions": [2]}
        mock_post.return_value = mock_response
        payload = {"instances": [{"text": "I love Trump!"}]}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == [2]

def test_predict_endpoint_invalid_input():
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"detail": "Invalid input"}
        mock_post.return_value = mock_response
        payload = {"text": "I love Trump!"}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data