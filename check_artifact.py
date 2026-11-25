"""Проверка результатов обучения в новой job, без повторного fit."""
import argparse
import json
from pathlib import Path
import pandas as pd
from wine_project.data import prepare_features
from wine_project.model import load_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="artifacts")
    args = parser.parse_args()
    folder = Path(args.directory)
    pipeline = load_pipeline(folder / "model.pkl")
    x = prepare_features(pd.read_csv(folder / "test.csv"))
    prediction = pipeline.predict(x)
    metrics = json.loads((folder / "metrics.json").read_text())
    assert len(prediction) == metrics["test_rows"]
    assert set(prediction).issubset({0, 1, 2})
    assert 0 <= metrics["accuracy"] <= 1
    print(f"Loaded model; predicted {len(prediction)} rows without fit")

if __name__ == "__main__":
    main()
