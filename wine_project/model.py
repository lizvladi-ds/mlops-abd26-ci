"""Создание Pipeline и сохранение через pickle."""
import pickle
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_pipeline(c=1.0):
    if c <= 0:
        raise ValueError("C must be positive")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=1000, random_state=42)),
    ])

def save_pipeline(pipeline, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(pipeline, file)

def load_pipeline(path):
    # Загружаем только собственный, доверенный файл.
    with Path(path).open("rb") as file:
        return pickle.load(file)
