"""Upload a single submission file to your team's MinIO/S3 bucket.

Credentials and endpoint come from environment variables (or a local .env
file, gitignored, never committed) so nothing secret ever lands in the repo:

    PRC_S3_ENDPOINT     e.g. s3.opensky-network.org (host[:port], no scheme)
    PRC_S3_ACCESS_KEY
    PRC_S3_SECRET_KEY
    PRC_S3_BUCKET       your team's submission bucket name

See .env.example for the template.

Usage:
    python -m src.post.upload data/submissions/smart-jigsaw_v1.parquet
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio

load_dotenv()


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m src.post.upload <local_parquet_path>")
        sys.exit(1)

    local_path = Path(sys.argv[1])
    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    endpoint = os.environ["PRC_S3_ENDPOINT"]
    access_key = os.environ["PRC_S3_ACCESS_KEY"]
    secret_key = os.environ["PRC_S3_SECRET_KEY"]
    bucket = os.environ["PRC_S3_BUCKET"]

    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)

    object_name = local_path.name
    client.fput_object(bucket, object_name, str(local_path))

    size = client.stat_object(bucket, object_name).size
    print(f"uploaded {local_path} -> {bucket}/{object_name}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
