import ssl
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener

import yaml


def _host_from_config():
    try:
        url = yaml.safe_load(open("leap_config.yaml"))["minio"]["endpoint_url"]
        p = urlparse(url if "://" in url else "//" + url)
        return p.hostname, p.port
    except Exception:
        return None, None


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else None
    cfg_host, cfg_port = _host_from_config()
    host = host or cfg_host
    if not host:
        host = input("MinIO host (no scheme, no port): ").strip()

    extra_ports = [int(p) for p in sys.argv[2:] if p.isdigit()]
    ports = []
    for p in ([cfg_port] if cfg_port else []) + [9000, 443, 80, 9001, 9443, 9090] + extra_ports:
        if p and p not in ports:
            ports.append(p)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=ctx))

    print(f"Probing host: {host}\n")
    winners = []
    for scheme in ("https", "http"):
        for port in ports:
            url = f"{scheme}://{host}:{port}"
            try:
                resp = opener.open(Request(url + "/", method="GET"), timeout=6)
                status, headers, body = resp.status, resp.headers, resp.read(200)
            except HTTPError as e:
                status, headers, body = e.code, e.headers, e.read(200)
            except (URLError, OSError, ValueError) as e:
                print(f"  {url:32s} -> unreachable ({type(e).__name__})")
                continue

            server = headers.get("Server", "")
            amz = headers.get("x-amz-request-id") or headers.get("X-Amz-Request-Id")
            is_s3 = bool(amz) or "MinIO" in server or b"<Error" in body or b"ListAllMyBucketsResult" in body
            mark = "  <-- S3 API" if is_s3 else ""
            print(f"  {url:32s} -> HTTP {status}  Server={server or '-'}  x-amz-request-id={'yes' if amz else 'no'}{mark}")
            if is_s3:
                winners.append(url)

    print()
    if winners:
        print("=" * 60)
        print("Correct endpoint_url for leap_config.yaml:")
        for w in winners:
            print(f"    {w}")
        print("=" * 60)
    else:
        print("No S3 API endpoint found on the probed ports.")
        print("Ask whoever runs MinIO for the S3 API endpoint, or pass extra ports:")
        print("    python scripts/find_minio_endpoint.py <host> 9000 9001 <other-ports>")


if __name__ == "__main__":
    main()
