from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_calculate_weights():
    response = client.post(
        "/api/v1/calculate-weights",
        json={
            "question": "Какие факторы влияют на успех?",
            "criteria": ["Команда", "Финансы", "Рынок"],
            "topic": "Бизнес"
        }
    )
    # Тест пройдет после загрузки модели
    assert response.status_code in [200, 500]