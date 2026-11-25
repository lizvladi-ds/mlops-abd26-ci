import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from wine_project.api import create_app
from wine_project.data import FEATURES
from wine_project.model import make_pipeline, save_pipeline

@pytest.fixture
def sample_data():
    # 9 заранее заданных строк. В unit-тестах полный Wine не читается.
    x = pd.DataFrame([
        [14.0, 1.5, 5.0, 1100], [13.7, 1.7, 5.3, 1050],
        [13.9, 1.6, 5.1, 1080], [12.0, 1.2, 2.0, 450],
        [12.2, np.nan, 2.2, 470], [12.1, 1.1, 2.1, 460],
        [13.0, 3.0, 8.0, 700], [13.2, 3.3, 8.5, 730],
        [13.1, 3.1, 8.2, 710],
    ], columns=FEATURES)
    y = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2], name="target")
    return x, y

@pytest.fixture
def fitted_pipeline(sample_data):
    x, y = sample_data
    return make_pipeline().fit(x, y)

@pytest.fixture
def model_path(fitted_pipeline, tmp_path):
    path = tmp_path / "model.pkl"
    save_pipeline(fitted_pipeline, path)
    return path

@pytest.fixture
def client(model_path):
    # TestClient работает в том же процессе, отдельный uvicorn не нужен.
    with TestClient(create_app(model_path)) as test_client:
        yield test_client

@pytest.fixture
def valid_row():
    return {"alcohol": 14.0, "malic_acid": 1.5,
            "color_intensity": 5.0, "proline": 1100}
