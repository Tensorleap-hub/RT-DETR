# MinIO connectivity diagnostics

Run these steps in order from the **repository root** (where `leap_config.yaml` and
`.env` live), in **Windows PowerShell**. Each step isolates one possible cause; stop as
soon as one gives a clear answer and send back the output.

All commands bypass any corporate HTTP proxy so we test the direct connection to MinIO.
If your MinIO uses a self-signed certificate (`verify_tls: false` in the config), add
`-k` to every `curl.exe` command.

---

## Step 0 — Set variables once (copy-paste, then reuse everywhere)

This reads the values straight from `leap_config.yaml`, so nothing is typed by hand and
the format matches the real script exactly.

```powershell
$ENDPOINT = (python -c "import yaml;print(yaml.safe_load(open('leap_config.yaml'))['minio']['endpoint_url'])").Trim()
$BUCKET   = (python -c "import yaml;print(yaml.safe_load(open('leap_config.yaml'))['minio']['bucket_name'])").Trim()
$U        = [System.Uri]$ENDPOINT
$MHOST    = $U.Host
$MPORT    = if ($U.Port -ge 0) { $U.Port } elseif ($U.Scheme -eq "https") { 443 } else { 80 }
echo "ENDPOINT=$ENDPOINT  HOST=$MHOST  PORT=$MPORT  BUCKET=$BUCKET"
```

Also record whether a proxy is configured (context for everything below):

```powershell
echo "HTTP_PROXY=$env:HTTP_PROXY  HTTPS_PROXY=$env:HTTPS_PROXY  NO_PROXY=$env:NO_PROXY"
```

---

## Step 1 — TCP reachability

```powershell
Test-NetConnection $MHOST -Port $MPORT
```

**Tells us:** whether the port is open at all. `TcpTestSucceeded : False` → firewall / DNS
/ wrong port, and nothing below will work.

---

## Step 2 — What is answering at the endpoint?

```powershell
curl.exe --noproxy "*" -sS -i "$ENDPOINT/"
```

**Tells us:** the `Server:` header. Genuine MinIO returns `Server: MinIO`. If it's
`nginx` / `traefik` / `envoy` or an HTML page, something is fronting MinIO — the leading
suspect.

---

## Step 3 — Raw body of the failing list call (decisive)

```powershell
curl.exe --noproxy "*" -sS -i "$ENDPOINT/$BUCKET/?list-type=2"
```

**Tells us:** this is unsigned, so real MinIO answers with **XML**
(`<Error><Code>AccessDenied</Code>...` or `NoSuchBucket`). If instead you get an **HTML
page or a bare 404 with no XML**, a proxy/ingress is intercepting the S3 path rather than
MinIO answering.

---

## Step 4 — Signed probe battery

Runs the real code path with the client's exact config and credentials, and bisects the
problem: list-vs-get, ListObjectsV2-vs-v1, path-vs-virtual addressing.

```powershell
python scripts/probe_minio.py
```

**Tells us**, from the pattern of `[OK]` / `[FAIL]`:

- `head_object` **OK** but every `list` **FAIL** → auth, creds, endpoint and bucket are
  all fine; the fault is the **LIST operation specifically** (ingress dropping
  `?list-type=2`, or a `ListBucket` policy gap vs `GetObject`).
- `head_bucket` **FAIL 404** as well → the bucket path itself isn't reachable as S3.
- `list_objects v1` **OK** but `list_objects_v2` **FAIL** → the backend doesn't implement
  ListObjectsV2.
- `list (virtual)` **OK** but `(path)` **FAIL** → addressing-style mismatch; fix by
  setting `addressing_style: "virtual"` under `minio:` in the config.
- Any `403 SignatureDoesNotMatch` / `InvalidAccessKeyId` → credentials/region issue.

The probe uses `MaxKeys=1`, so it also doubles as the "limit the list request" test — if
that still fails, an expensive/slow list is not the cause.

---

## Step 5 — Wire-level capture (only if Steps 1–4 are ambiguous)

```powershell
python scripts/probe_minio.py --debug
```

**Tells us:** the exact signed URL/path boto3 sends and the raw HTTP response headers and
body — the ground truth when the higher-level results don't converge.
