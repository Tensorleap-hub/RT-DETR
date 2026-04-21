import json
import os
from typing import Optional

import boto3


def _connect_to_s3() -> boto3.client:
    auth_secret_string = os.getenv("AUTH_SECRET")
    if auth_secret_string is None:
        raise ValueError("AUTH_SECRET environment variable not set")

    auth_secret_dict = json.loads(auth_secret_string)
    aws_access_key_id = auth_secret_dict.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = auth_secret_dict.get("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS credentials missing in AUTH_SECRET")

    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def download_file_if_missing(bucket: str, s3_key: str, local_path: str) -> str:
    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client = _connect_to_s3()
    s3_client.download_file(bucket, s3_key, local_path)
    return local_path
