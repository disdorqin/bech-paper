import json, os, sys, urllib.request, urllib.error, time

OUT = os.path.dirname(os.path.abspath(__file__))
REC = "https://zenodo.org/api/records/4624804"

def http_get(url, binary=False, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data

# 1) metadata
print("[1] fetching Zenodo record metadata ...")
meta = json.loads(http_get(REC).decode("utf-8"))
files = meta.get("files", [])
print(f"    found {len(files)} files")
urls = {}
for f in files:
    key = f.get("key")
    link = (f.get("links", {}).get("self") or f.get("links", {}).get("download"))
    urls[key] = link
    print(f"    {key:10s} -> {link}")

# 2) download each
for key, link in urls.items():
    path = os.path.join(OUT, key)
    if os.path.exists(path) and os.path.getsize(path) > 100000:
        print(f"[skip] {key} already present ({os.path.getsize(path)} bytes)")
        continue
    ok = False
    for attempt in range(3):
        try:
            print(f"[dl] {key} (attempt {attempt+1}) ...")
            data = http_get(link, timeout=180)
            with open(path, "wb") as fh:
                fh.write(data)
            print(f"    saved {key} ({len(data)} bytes)")
            ok = True
            break
        except Exception as e:
            print(f"    ERROR: {e}")
            time.sleep(2)
    if not ok:
        print(f"[FAIL] {key}")
        sys.exit(1)

print("[done] all benchmark CSVs downloaded to:", OUT)
