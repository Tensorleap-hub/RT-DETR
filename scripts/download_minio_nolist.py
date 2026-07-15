import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _manifest_download import fail, load_config, load_credentials, make_client, manifest_download


def _resolve_dest(config, section, dest_override):
    if dest_override:
        return Path(dest_override)
    dataset_path = section.get("dataset_path") or config.get("dataset_path")
    if not dataset_path:
        fail("No download destination set", "Set dataset_path in config or pass --dest.")
    if isinstance(dataset_path, list):
        dataset_path = dataset_path[0]
    return Path(dataset_path)


def main():
    parser = argparse.ArgumentParser(description="List-free MinIO dataset download (GetObject only).")
    parser.add_argument("--config", default="leap_config.yaml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dest", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N images (quick test).")
    args = parser.parse_args()

    config = load_config(args.config)
    section = config.get("minio")
    if not section:
        fail("No 'minio' section in config")
    for key in ("endpoint_url", "bucket_name", "prefix"):
        if not section.get(key):
            fail(f"minio config missing '{key}'")

    creds = load_credentials(args.env_file, "AUTH_SECRET")
    dest = _resolve_dest(config, section, args.dest)

    print("MinIO (no-list) download configuration:")
    print(f"  endpoint_url : {section['endpoint_url']}")
    print(f"  bucket_name  : {section['bucket_name']}")
    print(f"  prefix       : {section['prefix']}")
    print(f"  destination  : {dest}")
    print(f"  credentials  : access_key={creds['access_key'][:4]}*** (AUTH_SECRET from {args.env_file})")
    print()

    client = make_client(creds, section)
    manifest_download(
        client=client,
        bucket=section["bucket_name"],
        key_prefix=section["prefix"],
        annotation_files=config.get("annotation_file", {}),
        dest=dest,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
