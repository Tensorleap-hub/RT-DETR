import json
import os
import sys

import boto3
import yaml
from botocore.client import Config
from dotenv import load_dotenv

cfg = yaml.safe_load(open("leap_config.yaml"))["minio"]
load_dotenv(".env")
creds = json.loads(os.environ["AUTH_SECRET"])
endpoint, bucket = cfg["endpoint_url"], cfg["bucket_name"]
prefix = cfg["prefix"].rstrip("/")

if "--debug" in sys.argv:
    boto3.set_stream_logger("")


def client(style):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        region_name=cfg.get("region", "us-east-1"),
        verify=cfg.get("verify_tls", True),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": style},
            proxies={},
            connect_timeout=15,
            retries={"max_attempts": 1},
        ),
    )


def show(label, fn):
    try:
        print(f"[OK]   {label}: {fn()}")
    except Exception as e:
        r = getattr(e, "response", {}) or {}
        code = r.get("Error", {}).get("Code")
        status = r.get("ResponseMetadata", {}).get("HTTPStatusCode")
        print(f"[FAIL] {label}: status={status} code={code} :: {type(e).__name__}: {e}")


print(f"endpoint={endpoint}  bucket={bucket}  prefix={prefix or '(empty)'}\n")

key = f"{prefix}/train/annotations_train.json" if prefix else "train/annotations_train.json"
c = client("path")
show("head_bucket (path)", lambda: c.head_bucket(Bucket=bucket))
show(f"head_object {key}", lambda: c.head_object(Bucket=bucket, Key=key)["ContentLength"])
show("list_objects_v2 (path)", lambda: len(c.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1).get("Contents", [])))
show("list_objects v1 (path)", lambda: len(c.list_objects(Bucket=bucket, Prefix=prefix, MaxKeys=1).get("Contents", [])))
show("list_objects_v2 (virtual)", lambda: len(client("virtual").list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1).get("Contents", [])))
