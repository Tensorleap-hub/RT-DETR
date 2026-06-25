import argparse
import json
import os
import sys
from pathlib import Path

import boto3
import yaml
from botocore.client import Config
from botocore.exceptions import (
    ClientError,
    ConnectionError as BotoConnectionError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    ParamValidationError,
    SSLError,
)
from dotenv import load_dotenv
from tqdm import tqdm


def _fail(msg: str, hint: str = None, detail: str = None):
    sys.stdout.flush()
    print("\n" + "=" * 70, file=sys.stderr)
    print(f"ERROR: {msg}", file=sys.stderr)
    if hint:
        print(f"\nLikely cause / fix:\n  {hint}", file=sys.stderr)
    if detail:
        print(f"\nUnderlying error:\n  {detail}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.exit(1)


def _load_config(config_path: str) -> dict:
    if not Path(config_path).exists():
        _fail(f"Config file not found: {config_path}",
              "Pass the correct path with --config, or run from the repository root.")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_credentials(env_file: str) -> dict:
    load_dotenv(env_file)
    auth_secret = os.environ.get("AUTH_SECRET")
    if not auth_secret:
        _fail("AUTH_SECRET is not set",
              f'Create {env_file} with a line:\n'
              f'  AUTH_SECRET={{"access_key": "...", "secret_key": "..."}}')
    try:
        creds = json.loads(auth_secret)
    except json.JSONDecodeError as e:
        _fail("AUTH_SECRET is not valid JSON",
              'It must be a JSON object, e.g. {"access_key": "...", "secret_key": "..."}',
              str(e))
    missing = [k for k in ("access_key", "secret_key") if not creds.get(k)]
    if missing:
        _fail(f"AUTH_SECRET is missing required key(s): {', '.join(missing)}",
              'Expected format: {"access_key": "...", "secret_key": "..."}')
    return creds


def _resolve_minio_config(config: dict) -> dict:
    minio_cfg = config.get("minio")
    if not minio_cfg:
        _fail("No 'minio' section found in the config file",
              "Add a 'minio:' block with endpoint_url, bucket_name and prefix.")
    missing = [k for k in ("endpoint_url", "bucket_name", "prefix") if not minio_cfg.get(k)]
    if missing:
        _fail(f"minio config is missing required value(s): {', '.join(missing)}",
              "Fill these in under the 'minio:' block in the config file.")
    return minio_cfg


def _make_client(creds: dict, minio_cfg: dict) -> boto3.client:
    return boto3.client(
        "s3",
        endpoint_url=minio_cfg["endpoint_url"],
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        region_name=minio_cfg.get("region", "us-east-1"),
        verify=minio_cfg.get("verify_tls", True),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": minio_cfg.get("addressing_style", "path")},
            connect_timeout=15,
            retries={"max_attempts": 2},
        ),
    )


def _handle_boto_error(e: Exception, minio_cfg: dict):
    endpoint = minio_cfg["endpoint_url"]
    bucket = minio_cfg["bucket_name"]
    if isinstance(e, SSLError):
        _fail(f"TLS/SSL error connecting to {endpoint}",
              "If the MinIO server uses a self-signed certificate, set 'verify_tls: false' "
              "in the config, or set verify_tls to the path of the CA bundle. "
              "If the server is plain HTTP, the endpoint_url must start with 'http://'.",
              str(e))
    if isinstance(e, (EndpointConnectionError, ConnectTimeoutError, BotoConnectionError)):
        _fail(f"Could not connect to MinIO at {endpoint}",
              "Check that endpoint_url (host, port, http/https) is correct and reachable "
              "from this machine (firewall / VPN / DNS).",
              str(e))
    if isinstance(e, NoCredentialsError):
        _fail("No credentials were provided to the client", detail=str(e))
    if isinstance(e, ClientError):
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("InvalidAccessKeyId", "SignatureDoesNotMatch", "403") or status == 403:
            _fail("Authentication/authorization failed (HTTP 403)",
                  "The access_key/secret_key are wrong, or the account lacks permission "
                  f"to read bucket '{bucket}'.", str(e))
        if code in ("NoSuchBucket", "404") or status == 404:
            _fail(f"Bucket '{bucket}' does not exist on the server",
                  "Check 'bucket_name' in the config matches the bucket on the MinIO server.",
                  str(e))
        _fail(f"Server returned an error (code={code or status})", detail=str(e))
    raise e


def _list_objects(client, bucket: str, prefix: str, minio_cfg: dict) -> dict:
    objects = {}
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/"):
                    objects[key] = obj["Size"]
    except (ClientError, EndpointConnectionError, ConnectTimeoutError,
            BotoConnectionError, SSLError, NoCredentialsError, ParamValidationError) as e:
        _handle_boto_error(e, minio_cfg)
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


def _resolve_dest(config: dict, dest_override: str) -> Path:
    if dest_override:
        return Path(dest_override)
    dataset_path = config.get("dataset_path")
    if not dataset_path:
        _fail("No download destination set",
              "Set 'dataset_path' in the config file, or pass --dest /path/to/destination.")
    if isinstance(dataset_path, list):
        dataset_path = dataset_path[0]
    return Path(dataset_path)


def download(config_path: str, env_file: str, dest_override: str = None):
    config = _load_config(config_path)
    creds = _load_credentials(env_file)
    minio_cfg = _resolve_minio_config(config)
    bucket = minio_cfg["bucket_name"]
    prefix = minio_cfg["prefix"].rstrip("/")
    dest = _resolve_dest(config, dest_override)

    print("MinIO download configuration:")
    print(f"  endpoint_url     : {minio_cfg['endpoint_url']}")
    print(f"  bucket_name      : {bucket}")
    print(f"  prefix           : {prefix}")
    print(f"  region           : {minio_cfg.get('region', 'us-east-1')}")
    print(f"  verify_tls       : {minio_cfg.get('verify_tls', True)}")
    print(f"  addressing_style : {minio_cfg.get('addressing_style', 'path')}")
    print(f"  destination      : {dest}")
    print(f"  credentials      : access_key={creds['access_key'][:4]}*** (from {env_file})")
    print()

    client = _make_client(creds, minio_cfg)

    print("Connecting and listing objects...")
    remote_objects = _list_objects(client, bucket, prefix, minio_cfg)
    print(f"Found {len(remote_objects)} objects in minio://{bucket}/{prefix}")

    if not remote_objects:
        _fail(f"No objects found under minio://{bucket}/{prefix}",
              "The connection and credentials worked, but the prefix is empty. "
              "Check that 'prefix' matches the path of the dataset inside the bucket.")

    print(f"Scanning local files in {dest}...")
    local_index = _scan_local(dest, prefix)

    to_download = {
        key: size
        for key, size in remote_objects.items()
        if key not in local_index or local_index[key] != size
    }

    if not to_download:
        print("All files up to date.")
        return

    total_files = len(to_download)
    total_bytes = sum(to_download.values())
    print(f"Downloading {total_files} files ({total_bytes / 1e6:.1f} MB) -> {dest}")

    failures = []
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

            try:
                client.download_file(bucket, key, str(local_path), Callback=_callback)
            except Exception as e:
                failures.append((key, str(e)))
                tqdm.write(f"  FAILED: {key} -> {e}")

    if failures:
        print(f"\n{len(failures)} of {total_files} files failed to download:", file=sys.stderr)
        for key, err in failures:
            print(f"  - {key}: {err}", file=sys.stderr)
        sys.exit(1)

    print("Done. All files downloaded successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="leap_config.yaml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dest", default=None, help="Override dataset_path from config")
    args = parser.parse_args()
    download(args.config, args.env_file, args.dest)


if __name__ == "__main__":
    main()
