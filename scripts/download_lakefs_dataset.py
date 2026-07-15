import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _manifest_download import fail, load_config, load_credentials, make_client, manifest_download


def _resolve_dest(config, section, dest_override):
    if dest_override:
        return Path(dest_override)
    if section.get("dataset_path"):
        return Path(section["dataset_path"])
    base = config.get("dataset_path")
    if isinstance(base, list):
        base = base[0]
    if not base:
        fail("No download destination set", "Set lakefs.dataset_path or pass --dest.")
    return Path(str(base) + "-lakefs")


def main():
    parser = argparse.ArgumentParser(description="List-free lakeFS dataset download via the S3 gateway (GetObject only).")
    parser.add_argument("--config", default="leap_config.yaml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dest", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N images (quick test).")
    args = parser.parse_args()

    config = load_config(args.config)
    section = config.get("lakefs")
    if not section or not section.get("enabled"):
        fail("lakeFS is not configured yet",
             "Fill in the 'lakefs:' section in the config (endpoint_url, repository, branch, prefix) "
             "and set enabled: true, then add LAKEFS_SECRET to the .env file.")
    for key in ("endpoint_url", "repository", "branch"):
        if not section.get(key):
            fail(f"lakefs config missing '{key}'")

    creds = load_credentials(args.env_file, "LAKEFS_SECRET")
    dest = _resolve_dest(config, section, args.dest)

    # lakeFS S3 gateway: bucket = repository, keys are prefixed with the branch/ref.
    prefix = section.get("prefix", "").strip("/")
    key_prefix = f"{section['branch']}/{prefix}" if prefix else section["branch"]

    print("lakeFS (no-list) download configuration:")
    print(f"  endpoint_url : {section['endpoint_url']}")
    print(f"  repository   : {section['repository']}")
    print(f"  branch       : {section['branch']}")
    print(f"  prefix       : {prefix or '(none)'}")
    print(f"  key_prefix   : {key_prefix}")
    print(f"  destination  : {dest}")
    print(f"  credentials  : access_key={creds['access_key'][:4]}*** (LAKEFS_SECRET from {args.env_file})")
    print()

    client = make_client(creds, section)
    manifest_download(
        client=client,
        bucket=section["repository"],
        key_prefix=key_prefix,
        annotation_files=config.get("annotation_file", {}),
        dest=dest,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
