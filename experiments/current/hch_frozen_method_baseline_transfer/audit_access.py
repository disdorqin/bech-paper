"""Standalone partition-access and provenance audit for the transfer stage.

Reads the frozen panel metadata only; executes no baseline and computes no metric.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.current.hch_frozen_method_baseline_transfer import panel  # noqa: E402

CURRENT = ROOT / "experiments/current/hch_frozen_method_baseline_transfer"
PROTOCOL = ROOT / "docs/current/HCH_FROZEN_METHOD_BASELINE_TRANSFER_ADJUDICATION_20260911.md"
OFFICIAL_PIR = ROOT / "experiments/foundation/reference_deps/official_repos/PIR"
OFFICIAL_DELTA = ROOT / "experiments/foundation/reference_deps/official_repos/delta-Adapter/Adapter-X+Y"

PIR_COMMIT = "fc372bb02090da887d4a20b614a6cfecbfd813d0"
DELTA_COMMIT = "0add06ea7b4d2e0a84c364a8be72eef2676a92f2"


def stringify_keys(x):
    """JSON needs string keys; the panel's access maps are keyed by tuples."""
    if isinstance(x, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): stringify_keys(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [stringify_keys(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def main():
    cells, host_paths, host_hash, access = panel.build_cells()
    audit = {
        **access,
        "host_cache_sha256": host_hash,
        "host_cache_paths": {k: str(Path(v).relative_to(ROOT)) for k, v in host_paths.items()},
        "seeds_run": list(map(int, panel.SEEDS)),
        "stamp": time.strftime("%Y%m%d_%H%M%S"),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": panel.sha(PROTOCOL),
        "cell_gap_breaks": {f"{m}/{h}": int(c["gap_breaks"]) for (m, h), c in cells.items()},
        "cell_split": {f"{m}/{h}": c["split_manifest"] for (m, h), c in cells.items()},
        "official_lineage": {
            "PIR": {"commit": PIR_COMMIT,
                    "root": str(OFFICIAL_PIR.relative_to(ROOT)),
                    "sha256": {f: panel.sha(OFFICIAL_PIR / f) for f in (
                        "models/PIR.py", "exp/exp_long_term_forecasting_pir.py",
                        "data_provider/data_factory.py", "data_provider/data_loader.py",
                        "utils/tools.py", "utils/timefeatures.py", "run.py")}},
            "delta_Adapter": {"commit": DELTA_COMMIT,
                              "root": str(OFFICIAL_DELTA.relative_to(ROOT)),
                              "sha256": {"experiments/exp_post_y_add.py":
                                         panel.sha(OFFICIAL_DELTA / "experiments/exp_post_y_add.py")}},
        },
        "stage_sources": {p.name: panel.sha(p) for p in sorted(CURRENT.glob("*.py"))},
    }
    out = CURRENT / "_access_audit.json"
    out.write_text(json.dumps(stringify_keys(audit), indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("S3_reads", "S4_reads", "protected_reads",
                                            "final_reads", "other_China_markets_read",
                                            "roles_read", "cell_gap_breaks")},
                     indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
