from __future__ import annotations

from fastapi.testclient import TestClient

from spectracnn.api import create_app
from spectracnn.config import Settings


def test_root_and_health(client):
    root = client.get("/")
    assert root.status_code == 200

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_loaded": True, "version": "0.1.0"}


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "SpectraCNN"
    assert payload["length"] == 256
    assert payload["classes"] == ["BPSK", "QPSK", "16QAM", "64QAM"]
    assert payload["metrics"]["accuracy"] == 0.9


def test_predict(client, valid_iq):
    response = client.post("/predict", json={"iq": valid_iq})
    assert response.status_code == 200
    payload = response.json()
    assert payload["modulation"] in {"BPSK", "QPSK", "16QAM", "64QAM"}
    assert 0 <= payload["confidence"] <= 1
    assert abs(sum(payload["probabilities"].values()) - 1) < 1e-5
    assert payload["latency_ms"] >= 0


def test_predict_rejects_wrong_length(client, valid_iq):
    response = client.post("/predict", json={"iq": valid_iq[:100]})
    assert response.status_code == 422
    assert "expected exactly 256" in response.json()["detail"]


def test_predict_rejects_zero_power(client):
    response = client.post("/predict", json={"iq": [[0.0, 0.0] for _ in range(256)]})
    assert response.status_code == 422
    assert "zero power" in response.json()["detail"]


def test_demo(client):
    response = client.get("/demo", params={"modulation": "16QAM", "snr_db": 8})
    assert response.status_code == 200
    payload = response.json()
    assert payload["true_modulation"] == "16QAM"
    assert len(payload["constellation"]) == 256
    assert payload["snr_db"] == 8.0
    assert payload["modulation"] in {"BPSK", "QPSK", "16QAM", "64QAM"}


def test_demo_rejects_unknown_modulation(client):
    response = client.get("/demo", params={"modulation": "8PSK"})
    assert response.status_code == 422


def test_api_key_is_enforced(secured_client, valid_iq):
    assert secured_client.post("/predict", json={"iq": valid_iq}).status_code == 401
    response = secured_client.post(
        "/predict",
        json={"iq": valid_iq},
        headers={"X-API-Key": "secret-key"},
    )
    assert response.status_code == 200
    assert secured_client.get("/demo").status_code == 200


def test_missing_model_degrades_cleanly(tmp_path, valid_iq):
    settings = Settings(
        model_path=tmp_path / "missing.pt",
        meta_path=tmp_path / "missing.json",
        web_dir=tmp_path / "web",
        api_key=None,
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"iq": valid_iq}).status_code == 503
