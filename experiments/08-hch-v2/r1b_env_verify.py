"""R1B env-verify: GPU + imports + repo smoke. Run via venv python on server."""
from __future__ import annotations

import platform
import sys
from pathlib import Path

OUT = []


def log(msg):
    print(msg, flush=True)
    OUT.append(msg)


def main():
    log(f"[env] python {platform.python_version()} {platform.platform()}")

    import numpy, pandas, scipy, sklearn, lightgbm
    log(f"[env] numpy {numpy.__version__} | pandas {pandas.__version__} | "
        f"scipy {scipy.__version__} | sklearn {sklearn.__version__} | "
        f"lightgbm {lightgbm.__version__}")

    import torch
    log(f"[env] torch {torch.__version__} | cuda_available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"[env] gpu {torch.cuda.get_device_name(0)} | "
            f"cap {torch.cuda.get_device_capability(0)} | "
            f"mem_GB {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}")

    import matplotlib
    log(f"[env] matplotlib {matplotlib.__version__}")
    import yaml
    log(f"[env] pyyaml {yaml.__version__}")

    # repo smoke: imports of src modules used by R1B
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(HERE))

    from common import load_dataset, DATASETS, DATA
    for mk in ["LAGO_DE", "LAGO_PJM", "NEM_SA1", "NORD_DK1"]:
        spec = DATASETS[mk]
        p = Path(DATA) / spec["path"]
        ok = p.exists()
        log(f"[env] dataset {mk}: exists={ok} "
            f"({p.stat().st_size / 1e6:.1f}MB)" if ok else f"[env] dataset {mk}: MISSING {p}")
    # smoke: actually load one source + one unseen market
    for mk in ["LAGO_DE", "NORD_DK1"]:
        d = load_dataset(mk)
        log(f"[env] load {mk}: n={len(d['price'])} ts_range="
            f"{d['ts'].iloc[0]}..{d['ts'].iloc[-1]} cols={list(d['exog_fc'])}"
            f"{['+' + x for x in list(d['exog_act'])]}")

    import eval_manifest, hch_v2_bundle, hch_v2_context, hch_v2_pipeline, iah_candidate
    log("[env] imports ok: eval_manifest hch_v2_bundle hch_v2_context hch_v2_pipeline iah_candidate")

    import universal_trainer
    log(f"[env] universal_trainer ok ({universal_trainer.__file__})")

    import host_cache
    log(f"[env] host_cache ok ({host_cache.__file__})")

    log("[env] ENV_VERIFY_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
