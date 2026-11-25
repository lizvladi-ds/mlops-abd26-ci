"""Чтение приватного CSV из S3 по настройкам из окружения."""
import io
import os
import pandas as pd


def main():
    required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET", "S3_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Missing settings: " + ", ".join(missing))

    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
        region_name=os.getenv("AWS_DEFAULT_REGION", "ru-central1"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    response = s3.get_object(Bucket=os.environ["S3_BUCKET"], Key=os.environ["S3_KEY"])
    with response["Body"] as body:
        payload = body.read()
    frame = pd.read_csv(io.BytesIO(payload))
    if frame.empty:
        raise ValueError("CSV contains no rows")
    print(f"Private CSV read successfully: {len(frame)} rows, {len(frame.columns)} columns")
    # Не печатаем ключи, весь os.environ или строки приватной таблицы.


if __name__ == "__main__":
    main()
