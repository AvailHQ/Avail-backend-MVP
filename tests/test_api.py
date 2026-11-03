from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_and_athletes():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "data_source": "synthetic", "validated": False}
    response = client.get("/athletes")
    assert response.status_code == 200
    assert response.json()["count"] == 20


def test_latest_response_contract():
    response = client.get("/athletes/ATH-001/latest")
    assert response.status_code == 200
    payload = response.json()
    required = {"athlete_id", "date", "pre_training_readiness", "completed_training_load", "data_quality", "data_source", "validated"}
    assert required <= payload.keys()
    assert set(payload["pre_training_readiness"]) >= {"score", "components", "validated"}
    assert set(payload["completed_training_load"]) >= {"score", "internal_load", "external_load", "validated"}
    assert payload["data_source"] == "synthetic"
    assert payload["validated"] is False


def test_history_and_unknown_athlete():
    response = client.get("/athletes/ATH-001/history?limit=4")
    assert response.status_code == 200
    assert response.json()["count"] == 4
    assert len(response.json()["records"]) == 4
    assert client.get("/athletes/NOT-REAL/latest").status_code == 404


def test_separate_score_endpoints_and_decision_support():
    readiness = client.get("/athletes/ATH-001/readiness/latest")
    load = client.get("/athletes/ATH-001/training-load/latest")
    assert readiness.status_code == 200
    assert load.status_code == 200
    assert "completed_training_load" not in readiness.json()
    assert "pre_training_readiness" not in load.json()

    response = client.post(
        "/athletes/ATH-001/decision-support",
        json={"planned_internal_load": 90, "planned_external_load": 90},
    )
    assert response.status_code == 200
    decision = response.json()["decision_support"]
    assert decision["planned_training_load_score"] == 90
    assert decision["validated"] is False
    assert "planned_demand_materially_above_recent_load" in decision["attention_points"]


def test_decision_support_validates_planned_components():
    response = client.post(
        "/athletes/ATH-001/decision-support",
        json={"planned_internal_load": 101, "planned_external_load": 50},
    )
    assert response.status_code == 422
