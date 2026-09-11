"""Read-only probe: validate the frozen panel loader before any baseline runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.current.hch_frozen_method_baseline_transfer import panel  # noqa: E402


def main() -> int:
    cells, host_paths, host_hash, access = panel.build_cells()
    ref = panel.frozen_s1_reference()
    out = {"cells": {}, "access": access}
    for (m, h), c in sorted(cells.items()):
        sp = c["span"]
        out["cells"][f"{m}/{h}"] = {
            "cache_sha256": host_hash[f"{m}/{h}"],
            "n_fit": c["n_fit"], "n_eval": c["n_eval"],
            "n_runs": sp["n_runs"], "n_gap_breaks": sp["n_gap_breaks"],
            "runs": sp["runs"],
            "role_hours": panel.role_hours(c["runs"], c["n_fit"], max(1, int(0.10 * c["n_fit"]))),
            "split_manifest": c["split_manifest"],
            "fit_avail": int(c["fi"].repair_available.sum()),
            "eval_avail": int(c["ei"].repair_available.sum()),
            "tail_frac": float(c["tail"].mean()),
            "q05": c["q05"], "q95": c["q95"],
            "host_overall_mae": float(panel.mae(c["y"], c["hp"])),
            "host_normal_mae": float(panel.mae(c["y"], c["hp"], c["normal"])),
            "host_tail_mae": float(panel.mae(c["y"], c["hp"], c["tail"])),
            "frozen_s1_overall_mae": float(ref.loc[(m, h), "Overall_MAE"]),
            "frozen_s1_gain": float(ref.loc[(m, h), "relative_gain_pct"]),
            "host_meta_keys": sorted(k for k in c["host_metadata"] if "param" in k or k.startswith("T_")),
        }
    out_path = ROOT / "experiments/current/hch_frozen_method_baseline_transfer/_probe_panel.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["cells"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
