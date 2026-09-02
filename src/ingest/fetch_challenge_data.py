"""Download the PRC 2026 challenge parquet files from the OpenSky S3 bucket.

Idempotent: skips files already present locally. Routes files by name into
data/raw/ (monthly training files) vs data/ranking/ (ranking + submitting
templates) so the rest of the pipeline can treat them differently (ranking
schema has DEP taxi/off-block times blanked; see CLAUDE.md's hard invariant).

Requires pyopensky credentials configured first — see README.md.
"""

from __future__ import annotations

from pathlib import Path

from pyopensky.s3 import S3Client

BUCKET = "prc-2026-datasets"
RAW_DIR = Path("data/raw")
RANKING_DIR = Path("data/ranking")


def _target_dir(object_name: str) -> Path:
    name = Path(object_name).name
    if name.startswith("training_"):
        return RAW_DIR
    if name in ("ranking.parquet", "submitting.parquet"):
        return RANKING_DIR
    return RAW_DIR


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RANKING_DIR.mkdir(parents=True, exist_ok=True)

    s3 = S3Client()
    objects = list(s3.s3client.list_objects(BUCKET, recursive=True))
    print(f"Found {len(objects)} objects in '{BUCKET}'")

    for obj in objects:
        name = Path(obj.object_name).name
        dest = _target_dir(obj.object_name) / name
        if dest.exists():
            print(f"skip  {dest} (already present)")
            continue
        print(f"fetch {obj.object_name} -> {dest}")
        s3.download_object(obj, filename=dest)

    print("Done.")


if __name__ == "__main__":
    main()
