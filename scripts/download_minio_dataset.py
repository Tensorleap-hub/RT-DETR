import argparse
import json
import os
import sys
from pathlib import Path

import boto3
import yaml
from botocore.client import Config
from dotenv import load_dotenv
from tqdm import tqdm


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _minio_client(auth_secret: str, minio_cfg: dict) -> boto3.client:
    creds = json.loads(auth_secret)
    return boto3.client(
        "s3",
        endpoint_url=minio_cfg.get("endpoint_url") or None,
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        region_name=minio_cfg.get("region", "us-east-1"),
        verify=minio_cfg.get("verify_tls", True),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": minio_cfg.get("addressing_style", "path")},
        ),
    )


def _list_objects(client, bucket: str, prefix: str) -> dict:
    objects = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                objects[key] = obj["Size"]
    return objects


def _scan_local(dest: Path, prefix: str) -> dict:
    local = {}
    if not dest.exists():
        return local
    for dirpath, _, filenames in os.walk(dest):
        for fname in filenames:
            full = Path(dirpath) / fname
            rel = full.relative_to(dest)
            key = f"{prefix}/{rel}".replace("\\", "/")
            local[key] = full.stat().st_size
    return local


def download(config_path: str, env_file: str, dest_override: str = None):
    load_dotenv(env_file)
    auth_secret = os.environ.get("AUTH_SECRET")
    if not auth_secret:
        sys.exit("AUTH_SECRET not set")

    config = _load_config(config_path)
    minio_cfg = config.get("minio", {})
    bucket = minio_cfg["bucket_name"]
    prefix = minio_cfg["prefix"].rstrip("/")

    if dest_override:
        dest = Path(dest_override)
    else:
        dataset_path = config.get("dataset_path")
        if isinstance(dataset_path, list):
            dataset_path = dataset_path[0]
        dest = Path(dataset_path)

    client = _minio_client(auth_secret, minio_cfg)

    tqdm.write("Listing MinIO objects...")
    remote_objects = _list_objects(client, bucket, prefix)
    tqdm.write(f"Found {len(remote_objects)} objects in minio://{bucket}/{prefix}")

    tqdm.write(f"Scanning local files in {dest}...")
    local_index = _scan_local(dest, prefix)

    to_download = {
        key: size
        for key, size in remote_objects.items()
        if key not in local_index or local_index[key] != size
    }

    if not to_download:
        tqdm.write("All files up to date.")
        return

    total_files = len(to_download)
    total_bytes = sum(to_download.values())
    tqdm.write(f"Downloading {total_files} files ({total_bytes / 1e6:.1f} MB) → {dest}")

    files_done = 0
    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Progress") as bar:
        for key, size in to_download.items():
            rel = key[len(prefix) + 1:]
            local_path = dest / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)

            files_done += 1
            bar.set_description(f"[{files_done}/{total_files}]", refresh=False)

            def _callback(n, _bar=bar):
                _bar.update(n)

            client.download_file(bucket, key, str(local_path), Callback=_callback)

    tqdm.write("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="leap_config.yaml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dest", default=None, help="Override dataset_path from config")
    args = parser.parse_args()
    download(args.config, args.env_file, args.dest)


if __name__ == "__main__":
    main()
