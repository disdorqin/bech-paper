"""WP-2 (P0-B/C) — provenance + naming freeze.

Writes 00_RUN_CONFIG.json and 01_CODE_PROVENANCE.json (protocol §4 / §18).
P0-C: PatchTST host is labelled "PatchTST-style" in every artifact; never
claimed as official/fidelity-equivalent PatchTST.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "09-paper-gate" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SRC = ROOT / "src"
EX = ROOT / "experiments" / "08-hch-v2"
PEERS = ROOT / "experiments" / "07-route-e" / "peers"
DATA = ROOT / "data"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> tuple[str, str]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        return sha, branch
    except Exception:
        return "unknown", "unknown"


def hashes_of(files: list[Path]) -> dict[str, str]:
    out = {}
    for f in sorted(files):
        if f.is_file():
            out[str(f.relative_to(ROOT))] = sha256_file(f)[:16]
    return out


def main():
    sha, branch = git_head()

    host_code = [SRC / "backbones.py"]
    math_core = [SRC / "hch_v2_pipeline.py", SRC / "iah_candidate.py",
                 SRC / "iah_crps_loss.py", SRC / "universal_trainer.py",
                 SRC / "hch_v2_context.py", SRC / "hch_v2_bundle.py"]
    split_code = [SRC / "common.py", SRC / "eval_manifest.py"]
    runner_code = [EX / f for f in ("r1a_run.py", "r1a9_action_calibration.py",
                                    "r1a11_prequential_calibration_router.py",
                                    "r1b_generalization_screen.py",
                                    "r1b_stage2a_panel.py",
                                    "r1b_stage2d_action_chain.py",
                                    "r1b_stage2e_extension.py",
                                    "r1b_stage2f_localcore.py",
                                    "host_cache.py", "baselines_v2.py",
                                    "_final_point.py")]
    peer_adapters = list(PEERS.glob("*.py")) + list(PEERS.glob("*/**/*.py")) \
        if PEERS.exists() else []

    dataset_files = sorted(DATA.rglob("*"))
    dataset_files = [p for p in dataset_files if p.is_file() and p.suffix.lower() in
                     (".csv", ".xlsx", ".parquet")]

    config = {
        "protocol": "hch_v2_paper_benchmark_gate_comparative_experiment_protocol_v0.1_2026-08-14",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": sha,
        "branch": branch,
        "architecture_version": "HCH-v2",
        "math_core_version": "v2-host-relative-asinh-IAH-3atom",
        "split_version": "H0/S1R/S2T/S2V/S3M/S3C/S4 (four-segment rolling origin)",
        "host_matrix": {"H1": "Linear", "H2": "MLP", "H3": "LSTM",
                        "H4": "PatchTST-style"},
        "host_naming_policy": ("P0-C: code key 'PatchTST' is rendered as "
                               "'PatchTST-style' in all artifacts; never "
                               "claimed as official/fidelity-equivalent PatchTST."),
        "peer_naming_policy": ("P0-C: PIR/δ-Adapter used under fidelity-smoke "
                               "status only; accepted/patch status in "
                               "05_PEER_FIDELITY_REPORT.md."),
        "comparison_matrix": ["B0_HostIdentity", "B1_ResidualL1",
                              "B2_QuantileResidual", "B3_deltaAdapter",
                              "B4_PIR", "B5_HCHUniversal", "B6_HCHLocal"],
        "primary_metrics": ["MAE", "sMAPE_nofloor"],
        "secondary_metrics": ["RMSE", "rMAE", "CRPS_candidate", "neg_price_MAE",
                              "high_tail_MAE", "negative_sign_miss_rate"],
        "redlines": ["P0-A pass before any headline comparison",
                     "Shandong/private never mixed into public-universal training",
                     "sealed test not tuned",
                     "30% never called a majority",
                     "PatchTST always 'PatchTST-style'",
                     "R1A/R1B S4 not called pristine manuscript test"],
    }

    provenance = {
        "git": {"sha": sha, "branch": branch},
        "date": config["date"],
        "architecture_version": config["architecture_version"],
        "math_core_version": config["math_core_version"],
        "split_version": config["split_version"],
        "host_code_hashes": hashes_of(host_code),
        "math_core_hashes": hashes_of(math_core),
        "split_code_hashes": hashes_of(split_code),
        "runner_code_hashes": hashes_of(runner_code),
        "peer_adapter_hashes": hashes_of(peer_adapters),
        "dataset_hashes": hashes_of(dataset_files),
        "n_dataset_files_hashed": len(dataset_files),
        "note": ("'unknown' git_sha is FORBIDDEN in formal artifacts (P0-B). "
                 "Regenerate this file if HEAD moves."),
    }

    with open(OUT / "00_RUN_CONFIG.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(OUT / "01_CODE_PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)

    print(f"[provenance] git_sha={sha[:10]} branch={branch}")
    print(f"[provenance] wrote 00_RUN_CONFIG.json / 01_CODE_PROVENANCE.json -> {OUT}")
    print(f"[provenance] peer_adapters_hashed={len(peer_adapters)} "
          f"dataset_files_hashed={len(dataset_files)}")
    assert sha != "unknown", "P0-B: git_sha must not be unknown"


if __name__ == "__main__":
    main()
