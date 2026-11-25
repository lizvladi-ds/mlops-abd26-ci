"""Дождаться реального API и проверить один HTTP-запрос."""
import argparse
import json
import time
from http.client import HTTPException
from urllib.request import Request, urlopen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    for _ in range(45):
        try:
            with urlopen(args.url + "/health", timeout=2) as response:
                if json.load(response) == {"status": "ok"}:
                    break
        except (OSError, HTTPException):
            pass
        time.sleep(1)
    else:
        raise RuntimeError("API did not become ready")
    payload = {"rows": [{"alcohol": 14.0, "malic_acid": 1.5,
                         "color_intensity": 5.0, "proline": 1100}]}
    request = Request(args.url + "/predict", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        body = json.load(response)
    predictions = body["predictions"]
    assert len(predictions) == 1 and predictions[0] in [0, 1, 2]
    print("HTTP smoke check passed:", body)

if __name__ == "__main__":
    main()
