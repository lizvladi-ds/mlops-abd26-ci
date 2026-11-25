"""python -m wine_project.train --output-dir artifacts --c 1.0"""
import argparse
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from wine_project.data import load_dataset, prepare_features
from wine_project.model import make_pipeline, save_pipeline

def train(output_dir, c=1.0):
    x, y = load_dataset()
    x = prepare_features(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )
    pipeline = make_pipeline(c)
    pipeline.fit(x_train, y_train)
    prediction = pipeline.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "f1_macro": float(f1_score(y_test, prediction, average="macro")),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "C": c,
    }
    output_dir = Path(output_dir)
    save_pipeline(pipeline, output_dir / "model.pkl")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Отдельный вход для последующего batch, без целевой переменной.
    x_test.to_csv(output_dir / "test.csv", index=False)
    print(json.dumps(metrics, indent=2))
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Обучить Pipeline на Wine")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()
    train(args.output_dir, c=args.c)

if __name__ == "__main__":
    main()
