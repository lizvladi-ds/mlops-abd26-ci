"""Читаем CSV, загружаем готовый Pipeline, записываем предсказания."""
import argparse
from pathlib import Path
import pandas as pd
from wine_project.data import prepare_features
from wine_project.model import load_pipeline

def predict_file(model_path, input_path, output_path):
    frame = prepare_features(pd.read_csv(input_path))
    pipeline = load_pipeline(model_path)
    result = frame.copy()
    result["prediction"] = pipeline.predict(frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result

def main():
    parser = argparse.ArgumentParser(description="Предсказания Wine из CSV")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predict_file(args.model, args.input, args.output)

if __name__ == "__main__":
    main()
