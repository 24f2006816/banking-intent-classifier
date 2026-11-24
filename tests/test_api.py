from fastapi.testclient import TestClient

from intents.serve import create_app


class FakeClassifier:
    def predict(self, text, top_k=3):
        return [{"intent": "card_arrival", "probability": 0.9}][:top_k]


def test_predict_endpoint():
    client = TestClient(create_app(FakeClassifier()))
    response = client.post("/predict", json={"text": "Where is my card?", "top_k": 1})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["intent"] == "card_arrival"


def test_rejects_empty_text():
    client = TestClient(create_app(FakeClassifier()))
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_health_and_missing_model(monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
    assert client.post("/predict", json={"text": "hi"}).status_code == 503
