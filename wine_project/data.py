"""Небольшой датасет Wine и проверка входных признаков."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

FEATURES = ["alcohol", "malic_acid", "color_intensity", "proline"]

def load_dataset():
    # Wine входит в sklearn. Скачивание и S3 не нужны.
    wine = load_wine(as_frame=True)
    return wine.data[FEATURES].copy(), wine.target.copy()

def prepare_features(frame):
    """Выбираем признаки в фиксированном порядке, не меняя исходную таблицу."""
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    result = frame.loc[:, FEATURES].copy()
    for name in FEATURES:
        result[name] = pd.to_numeric(result[name], errors="raise")
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError("Infinite values are not allowed")
    # NaN допустим: пропуски обработает SimpleImputer внутри Pipeline.
    return result
