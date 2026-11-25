"""FastAPI загружает сохранённый Pipeline. Обучения внутри API нет."""
import os
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from wine_project.data import prepare_features
from wine_project.model import load_pipeline

class WineRow(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    alcohol: float = Field(gt=0)
    malic_acid: float = Field(ge=0)
    color_intensity: float = Field(ge=0)
    proline: float = Field(ge=0)

class PredictRequest(BaseModel):
    rows: list[WineRow] = Field(min_length=1, max_length=1000)

def create_app(model_path=None):
    path = model_path or os.getenv("MODEL_PATH", "artifacts/model.pkl")

    @asynccontextmanager
    async def lifespan(app):
        # Один раз при старте. Ошибка загрузки останавливает запуск API.
        app.state.pipeline = load_pipeline(path)
        yield

    app = FastAPI(title="Wine API", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(request: PredictRequest):
        frame = pd.DataFrame([row.model_dump() for row in request.rows])
        prediction = app.state.pipeline.predict(prepare_features(frame))
        return {"predictions": prediction.tolist()}

    return app
