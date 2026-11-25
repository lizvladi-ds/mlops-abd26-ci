import pandas as pd
import pytest
from fastapi.testclient import TestClient
from wine_project.api import create_app

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_api_matches_pipeline(client, valid_row, fitted_pipeline):
    rows = [valid_row, {**valid_row, "alcohol": 12.0}]
    response = client.post("/predict", json={"rows": rows})
    assert response.status_code == 200
    expected = fitted_pipeline.predict(pd.DataFrame(rows)).tolist()
    assert response.json() == {"predictions": expected}

@pytest.mark.parametrize("field", ["alcohol", "malic_acid", "color_intensity", "proline"])
def test_missing_feature_returns_422(client, valid_row, field):
    del valid_row[field]
    assert client.post("/predict", json={"rows": [valid_row]}).status_code == 422

@pytest.mark.parametrize("change", [
    {"alcohol": -1}, {"alcohol": "bad"}, {"proline": None}, {"target": 0}
])
def test_bad_feature_values_return_422(client, valid_row, change):
    row = {**valid_row, **change}
    assert client.post("/predict", json={"rows": [row]}).status_code == 422

def test_empty_batch_returns_422(client):
    assert client.post("/predict", json={"rows": []}).status_code == 422


def test_missing_model_stops_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path / "missing.pkl")):
            pass
