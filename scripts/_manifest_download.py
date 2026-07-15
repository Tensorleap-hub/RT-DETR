import json
import os
import sys
from pathlib import Path, PurePosixPath

import boto3
import yaml
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm


def fail(msg, hint=None, detail=None):
    sys.stdout.flush()
    print("\n" + "=" * 70, file=sys.stderr)
    print(f"ERROR: {msg}", file=sys.stderr)
    if hint:
        print(f"\nLikely cause / fix:\n  {hint}", file=sys.stderr)
    if detail:
        print(f"\nUnderlying error:\n  {detail}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.exit(1)


def load_config(config_path):
    if not Path(config_path).exists():
        fail(f"Config file not found: {config_path}", "Run from the repository root or pass --config.")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_credentials(env_file, secret_var):
    load_dotenv(env_file)
    raw = os.environ.get(secret_var)
    if not raw:
        fail(f"{secret_var} is not set",
             f'Add to {env_file}:\n  {secret_var}={{"access_key": "...", "secret_key": "..."}}')
    try:
        creds = json.loads(raw)
    except json.JSONDecodeError as e:
        fail(f"{secret_var} is not valid JSON",
             'Expected: {"access_key": "...", "secret_key": "..."}', str(e))
    missing = [k for k in ("access_key", "secret_key") if not creds.get(k)]
    if missing:
        fail(f"{secret_var} is missing key(s): {', '.join(missing)}")
    return creds


def make_client(creds, section):
    return boto3.client(
        "s3",
        endpoint_url=section["endpoint_url"],
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        region_name=section.get("region", "us-east-1"),
        verify=section.get("verify_tls", True),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": section.get("addressing_style", "path")},
            proxies={},
            connect_timeout=15,
            retries={"max_attempts": 2},
        ),
    )


def _normalize_annotation_files(annotation_files):
    if isinstance(annotation_files, str):
        return {"val": annotation_files}
    return annotation_files or {}


def _download(client, bucket, key, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(local_path))


def manifest_download(client, bucket, key_prefix, annotation_files, dest, limit=None):
    dest = Path(dest)
    annotation_files = _normalize_annotation_files(annotation_files)
    if not annotation_files:
        fail("No annotation_file entries in config", "Set annotation_file with the split->json mapping.")

    prefix = key_prefix.strip("/")
    image_tasks = []

    for split, rel in annotation_files.items():
        rel_posix = PurePosixPath(rel.replace("\\", "/"))
        ann_key = f"{prefix}/{rel_posix}" if prefix else str(rel_posix)
        ann_local = dest / rel_posix
        print(f"[{split}] annotation -> {ann_key}")
        try:
            _download(client, bucket, ann_key, ann_local)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            fail(f"Failed to download annotation {ann_key} (status={status} code={code})",
                 "Confirm the object exists at this key and the credentials allow GetObject.", str(e))

        with open(ann_local) as f:
            coco = json.load(f)
        split_dir = rel_posix.parent
        for img in coco.get("images", []):
            file_name = img["file_name"].replace("\\", "/")
            image_rel = split_dir / "images" / file_name
            image_key = f"{prefix}/{image_rel}" if prefix else str(image_rel)
            image_tasks.append((image_key, dest / image_rel))

    print(f"\nManifest lists {len(image_tasks)} images.")
    if limit is not None:
        image_tasks = image_tasks[:limit]
        print(f"--limit set: downloading first {len(image_tasks)}.")

    failures = []
    for image_key, local_path in tqdm(image_tasks, desc="Images", unit="file"):
        if local_path.exists():
            continue
        try:
            _download(client, bucket, image_key, local_path)
        except ClientError as e:
            failures.append((image_key, str(e)))
            tqdm.write(f"  FAILED: {image_key} -> {e}")

    if failures:
        print(f"\n{len(failures)} of {len(image_tasks)} images failed:", file=sys.stderr)
        for key, err in failures[:20]:
            print(f"  - {key}: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Dataset available at: {dest}")
