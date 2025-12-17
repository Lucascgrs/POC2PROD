from fastapi.testclient import TestClient
import IA.POC2PROD.app as app

# TestClient permet de faire des requêtes HTTP sur ton API sans lancer le serveur
client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    # On envoie un vrai payload
    payload = {"title": "How to use pandas in python?", "top_k": 1}

    response = client.post("/predict", json=payload)

    # On vérifie que ça répond 200
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) > 0
    # Vérifie que la structure de réponse est correcte
    assert "tag_id" in data["predictions"][0]