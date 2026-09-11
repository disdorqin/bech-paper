"""Read-only smoke test of the PIR paper-protocol transfer plumbing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.current.hch_frozen_method_baseline_transfer import panel  # noqa: E402
from experiments.current.hch_frozen_method_baseline_transfer import pir_transfer as pt  # noqa: E402

out = {"pir_module_file": pt.__file__,
       "official_repo": str(pt.OFFICIAL),
       "official_exists": pt.OFFICIAL.exists(),
       "utils_path": list(pt._utils.__path__),
       "imported_exp": pt.Exp_Long_Term_Forecast_PIR.__module__,
       "model_module_file": sys.modules[pt.Exp_Long_Term_Forecast_PIR.__module__].__file__}

cells, _, _, _ = panel.build_cells()
cell = cells[("GANSU_DA", "PatchTST")]
args = pt.build_args(ROOT / "_smoke", "PatchTST", panel.SEQ, panel.H)
out["args_seq_len"] = args.seq_len
out["args_pred_len"] = args.pred_len
out["args_batch_size"] = args.batch_size

bundles, meta, scaled, stamps = pt.build_bundles(cell, args, ROOT / "_smoke")
out["window_counts"] = meta["window_counts"]
out["scaler"] = {"mean": meta["scaler_mean"], "scale": meta["scaler_scale"]}
out["n_eval_expected"] = cell["n_eval"]

# shapes of the first training window
x, y, xm, ym, idx = bundles["train"][0][0]
out["sample_shapes"] = {"x": list(x.shape), "y": list(y.shape),
                        "xm": list(xm.shape), "ym": list(ym.shape), "index": int(idx)}

eval_windows = pt.day_origin_windows(cell["runs"], cell["n_fit"])
out["n_eval_windows"] = len(eval_windows)
out["eval_window_head"] = eval_windows[:3]
out["eval_window_tail"] = eval_windows[-3:]
# the eval windows must reproduce the frozen y_true exactly
ev_y = cell["ev"].y_true[:, :, 0]
got = []
for rid, t in eval_windows:
    got.append(scaled[rid][t:t + panel.H] * meta["scaler_scale"] + meta["scaler_mean"])
got = __import__("numpy").array(got)
out["eval_target_max_abs_err"] = float(__import__("numpy").abs(got - ev_y).max())

# model construction
exp = pt.PanelPIRExp(args, bundles)
out["model_params_total"] = int(sum(p.numel() for p in exp.model.parameters()))
out["model_params_backbone"] = int(sum(p.numel() for p in exp.model.model.parameters()))
out["has_quality_estimator"] = hasattr(exp.model, "quality_estimator")
out["has_retrieval"] = hasattr(exp.model, "retrieval")
out["refiner_layers"] = len(exp.model.refiner.attn_layers)

Path(ROOT / "experiments/current/hch_frozen_method_baseline_transfer/_smoke_pir.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
