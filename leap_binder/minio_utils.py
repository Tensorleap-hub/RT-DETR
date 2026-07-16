import json
import os

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from .common import CONFIG

_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}

_client = None


def _minio_client() -> boto3.client:
    global _client
    if _client is None:
        _client = _build_minio_client()
    return _client


def _build_minio_client() -> boto3.client:
    auth_secret_string = os.getenv("AUTH_SECRET")
    if auth_secret_string is None:
        raise ValueError("AUTH_SECRET environment variable not set")

    creds = json.loads(auth_secret_string)
    access_key = creds.get("access_key")
    secret_key = creds.get("secret_key")

    if not access_key or not secret_key:
        raise ValueError("MinIO credentials missing in AUTH_SECRET")

    minio_config = CONFIG.get("minio", {})
    endpoint_url = minio_config.get("endpoint_url") or None
    region = minio_config.get("region", "us-east-1")
    verify_tls = minio_config.get("verify_tls", True)
    addressing_style = minio_config.get("addressing_style", "path")

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        verify=verify_tls,
        config=Config(signature_version="s3v4", s3={"addressing_style": addressing_style}),
    )


def object_exists(bucket: str, key: str) -> bool:
    client = _minio_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in _NOT_FOUND_CODES:
            return False
        raise


def download_file_if_missing(bucket: str, key: str, local_path: str) -> str:
    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client = _minio_client()
    client.download_file(bucket, key, local_path)
    return local_path


def download_annotations(bucket: str, prefix: str, annotation_files: dict, dataset_root: str) -> None:
    client = None
    for relative_path in annotation_files.values():
        local_path = os.path.join(dataset_root, relative_path)
        if os.path.exists(local_path):
            continue
        key = f"{prefix}/{relative_path}".lstrip("/")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if client is None:
            client = _minio_client()
        try:
            client.download_file(bucket, key, local_path)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in _NOT_FOUND_CODES:
                print(f"Annotation file not found in storage, skipping: {key}")
                continue
            raise
