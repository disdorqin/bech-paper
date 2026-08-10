# -*- coding: utf-8 -*-
"""v8-E1: 重建 31a_sources.csv + 31a_validation.json（真实 CSV writer，完整 SHA256）"""
import csv
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = r"D:\AI_Memory\papers\raw"
REFS = os.path.join(HERE, "refs")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# (arxiv_id, local_path, stable_url, coord, quote)
papers = [
    ("2512.22428", os.path.join(RAW, "2512.22428.pdf"), "https://arxiv.org/abs/2512.22428",
     "p4 Eq(9)", "The unconstrained correction is \u0394_i = \u0394_ridge"),
    ("2601.20280", os.path.join(RAW, "2601.20280.pdf"), "https://arxiv.org/abs/2601.20280",
     "p3 Eq1.3", "Output-side correction: \u02dcY=F(X)+\u03b4 A_out"),
    ("2605.21088", os.path.join(RAW, "2605.21088.pdf"), "https://arxiv.org/abs/2605.21088",
     "p4 decomposition", "we decompose the backbone prediction into trend and seasonal"),
    ("2605.08935", os.path.join(RAW, "2605.08935.pdf"), "https://arxiv.org/abs/2605.08935",
     "p10 L422", "outputs a corrected state \u02dcX_{t+1}"),
    ("2505.15354", os.path.join(RAW, "2505.15354.pdf"), "https://arxiv.org/abs/2505.15354",
     "p4 L3", "applying an affine correction g_{a,b}(y)=ay+b"),
    ("2505.23583", os.path.join(REFS, "2505.23583.pdf"), "https://arxiv.org/abs/2505.23583",
     "p6 Eq(4)", "y_global = WeightedSum(p, Y_re)"),
    ("2101.02703", os.path.join(REFS, "2101.02703.pdf"), "https://arxiv.org/abs/2101.02703",
     "p5 Theorem 1", "Theorem 1 (Validity of UCB calibration)"),
    ("2110.01052", os.path.join(REFS, "2110.01052.pdf"), "https://arxiv.org/abs/2110.01052",
     "p3 L17", "control the risk, abbreviated as R(\u03bb)=R(T_\u03bb)"),
]

rows = []
fileinfo = {}
for aid, path, url, coord, quote in papers:
    exists = os.path.exists(path)
    h = sha256(path) if exists else ""
    fileinfo[aid] = {"path": path, "exists": exists, "sha256": h}
    rows.append({
        "arxiv_id": aid,
        "local_pdf": path,
        "url": url,
        "sha256": h,
        "coordinate": coord,
        "verbatim_quote": quote,
        "counted": "yes",
    })

csv_path = os.path.join(HERE, "31a_sources.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# validation: parse back
with open(csv_path, encoding="utf-8") as f:
    back = list(csv.DictReader(f))

val = {
    "row_count": len(back),
    "counted_count": sum(1 for r in back if r["counted"] == "yes"),
    "per_file": fileinfo,
    "csv_roundtrip": len(back) == 8 and sum(1 for r in back if r["counted"] == "yes") == 8,
    "all_hashes_full_64": all(len(r["sha256"]) == 64 for r in back),
}
with open(os.path.join(HERE, "31a_validation.json"), "w", encoding="utf-8") as f:
    json.dump(val, f, ensure_ascii=False, indent=2)

print("CSV rows:", val["row_count"])
print("counted:", val["counted_count"])
print("roundtrip:", val["csv_roundtrip"])
print("all_hashes_full_64:", val["all_hashes_full_64"])
for aid, fi in fileinfo.items():
    print(aid, "exists=", fi["exists"], "sha256=", fi["sha256"][:16] + "..." if fi["sha256"] else "MISSING")
