"""PIR paper-protocol transfer for the frozen six-cell DA panel.

Controlling protocol:
    docs/current/HCH_FROZEN_METHOD_BASELINE_TRANSFER_ADJUDICATION_20260911.md

Lineage (B0 evidence):

* The executed code is the *exact* accepted official checkout at commit
  ``fc372bb02090da887d4a20b614a6cfecbfd813d0`` --
  ``models/PIR.py`` (QualityEstimator, Transformer ``refiner``,
  ``retrieval`` construction, ``intermediate_results + alpha * refine_out +
  beta * retrieval_results``) and
  ``exp/exp_long_term_forecasting_pir.py`` (pretrain loop, retrieval-index
  construction, refine loop with the quality loss).
* The historical HCH ``PIR_PROXY_10epoch_K10`` and ``src/baselines/pir.py``
  ridge/refiner substitute are **not** imported, referenced or executed here.
* The only method overridden is ``_get_data``, which the protocol explicitly
  permits: cached-Host use would change official semantics (PIR is a standalone
  forecaster), so the verified official backbone + PIR pair is executed on the
  same dataset/task/Host family and labelled ``PIR-paper-protocol``.

Task-level adaptation (disclosed in the transfer-fidelity audit, not tuned):

* ``pred_len`` is the frozen day-ahead horizon (24) rather than the anchor's 96.
* ``seq_len`` is the frozen panel's registered context width (168) rather than
  the anchor's 96, so that PIR sees exactly the information set the frozen Host
  family sees.  The frozen Host caches themselves record the same adaptation
  (``official_config.seq_len = 168, pred_len = 24``).
* ``enc_in = dec_in = c_out = 1`` because the frozen DA task is the single
  price channel.
* ``load_pretrained_backbone = 0`` so the official code trains its own backbone
  on this data; there is no pre-existing checkpoint for this panel.
* ``fix_seed = 2021`` is preserved from the official ``run.py``.

Everything else (backbone configs P-A1/P-A2, ``refine_d_model/d_ff/layers``,
``retrieval_num``/``retrieval_stride``, Adam ``1e-4``, ``lradj='type1'``,
patience 3, the 0.7/0.1/0.2-style chronology and the official scaling) is the
paper-native configuration.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from experiments.current.hch_frozen_method_baseline_transfer import panel  # noqa: E402

ROOT = panel.ROOT
OFFICIAL = ROOT / "experiments/foundation/reference_deps/official_repos/PIR"
RUNTIME = ROOT / "experiments/current/hch_baseline_fidelity_reproduction/_runtime"
PIR_COMMIT = "fc372bb02090da887d4a20b614a6cfecbfd813d0"

ANCHOR_SEED = 2021     # official run.py fix_seed contract
NUM_THREADS = 4        # PIR has no bit-exact frozen reference; S1 is never replayed here

# The official checkout publishes a top-level ``utils`` package while this
# repository keeps its own helpers at ``src/utils``.  ``panel`` has already
# imported the repository helpers, so the ``utils`` package is re-pointed at the
# verified official checkout for every later import.
import utils as _utils  # noqa: E402

_utils.__path__ = [str(OFFICIAL / "utils")]
for _p in (str(OFFICIAL), str(RUNTIME)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_provider.data_loader import Dataset_Custom  # noqa: E402,F401  (lineage anchor)
from exp.exp_long_term_forecasting_pir import Exp_Long_Term_Forecast_PIR  # noqa: E402

# --------------------------------------------------------------------------
# Paper-native configuration (audited anchor P-A1 / P-A2)
# --------------------------------------------------------------------------

BACKBONE_CONFIG = {
    "PatchTST": {"e_layers": 3, "n_heads": 4, "d_model": 16, "d_ff": 128,
                 "dropout": 0.1, "batch_size": 32},
    "TimeMixer": {"e_layers": 2, "n_heads": 4, "d_model": 16, "d_ff": 32,
                  "dropout": 0.1, "batch_size": 32, "down_sampling_layers": 3,
                  "down_sampling_window": 2, "down_sampling_method": "avg"},
}

PIR_COMMON = {
    "refine_epochs": 10, "refine_d_model": 128, "refine_d_ff": 128, "refine_layers": 1,
    "refine_lr": 1e-4, "retrieval_num": 50, "retrieval_stride": 1,
    "including_time_features": 1, "train_epochs": 10, "patience": 3,
    "learning_rate": 1e-4, "lradj": "type1", "use_norm": 1, "percent": 100,
}

PIR_LABEL = "PAPER_FAITHFUL_EXACT_ACCEPTED"


def build_args(out_dir: Path, backbone: str, seq_len: int, pred_len: int) -> SimpleNamespace:
    """The official anchor argument namespace, adapted to the frozen panel."""
    cfg = BACKBONE_CONFIG[backbone]
    return SimpleNamespace(
        task_name="long_term_forecast", is_training=1,
        model_id=f"PIR_{seq_len}_{pred_len}",
        model="PIR", backbone=backbone,
        checkpoints=str(Path(out_dir) / "checkpoints"),
        bakcbone_checkpoints=str(Path(out_dir) / "checkpoints"),
        load_pretrained_backbone=0, output_index=1,
        data="custom", root_path=str(out_dir), data_path="panel.csv",
        features="M", target="OT", freq="h", inverse=False, seasonal_patterns="Monthly",
        seq_len=seq_len, label_len=0, pred_len=pred_len,
        enc_in=1, dec_in=1, c_out=1,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
        d_layers=1, d_ff=cfg["d_ff"], moving_avg=25, factor=1, distil=True,
        dropout=cfg["dropout"], embed="timeF", activation="gelu", output_attention=False,
        down_sampling_layers=cfg.get("down_sampling_layers", 3),
        down_sampling_window=cfg.get("down_sampling_window", 2),
        down_sampling_method=cfg.get("down_sampling_method", "avg"),
        channel_independence=1, decomp_method="moving_avg", period_len=1,
        use_future_temporal_feature=0, top_k=5, num_kernels=6,
        num_workers=0, itr=1, des="Exp", loss="MSE", use_amp=False,
        use_gpu=False, gpu=0, use_multi_gpu=False, devices="0",
        p_hidden_dims=[128, 128], p_hidden_layers=2,
        batch_size=int(cfg["batch_size"]),
        **PIR_COMMON,
    )


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def time_marks(times: pd.DatetimeIndex) -> np.ndarray:
    """The official ``timeenc=0`` calendar encoding (PIR's data_factory forces 0)."""
    return np.stack([
        times.month / 12 - 0.5,
        times.day / 31 - 0.5,
        times.weekday / 6 - 0.5,
        times.hour / 23 - 0.5,
        times.minute / 59 - 0.5,
    ], axis=-1).astype(np.float64)


def scale_runs(runs, n_fit, n_val):
    """Fit the official StandardScaler on the frozen train span and apply it.

    Only hours belonging to frozen ``train`` role day blocks are used, so the
    scaler statistics never touch DIAG_EVAL or the validation tail.
    """
    spans = panel.role_hours(runs, n_fit, n_val)
    train_hours = np.concatenate([runs[rid]["series"][h0:h1]
                                  for rid, role, h0, h1 in spans if role == "train"])
    mean = float(train_hours.mean())
    scale = float(train_hours.std())
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("degenerate train-span scale for the PIR transfer")
    series = [np.asarray(r["series"], dtype=np.float64) for r in runs]
    stamps = [time_marks(r["series_time"]) for r in runs]
    scaled = [(s - mean) / scale for s in series]
    return scaled, stamps, mean, scale


def windows_for(runs, n_fit, n_val, role, seq_len, pred_len, stride=1):
    """Chronological ``(run_id, target_start)`` windows for one role."""
    out = []
    for rid, r, h0, h1 in panel.role_hours(runs, n_fit, n_val):
        if r != role:
            continue
        for t in range(h0, h1 - pred_len + 1, stride):
            if t - seq_len >= 0:
                out.append((rid, t))
    return out


def day_origin_windows(runs, n_fit):
    """Exactly the frozen DIAG_EVAL day origins, in frozen row order."""
    out = []
    for r in runs:
        for j, day in enumerate(r["days"]):
            if int(day) < n_fit:
                continue
            t = int(r["day_base"][j])
            if t - panel.SEQ < 0:
                raise RuntimeError("a frozen DIAG_EVAL day has no full context window")
            out.append((int(r["run_id"]), t))
    return out


class PanelWindowDataset(Dataset):
    """``Dataset_Custom`` semantics over the reconstructed frozen runs.

    ``index`` is the window ordinal inside this dataset, which is exactly the
    quantity the official retrieval self-exclusion mask uses (official:
    ``s_begin = index * stride``; here the enumeration is chronological with
    stride 1, so ordinal distance equals hour distance inside a run).  Across a
    run boundary the ordinal distance is *shorter* than the true hour distance,
    so the mask is only ever more conservative -- it can drop candidate keys but
    can never expose a temporally overlapping one.
    """

    def __init__(self, series, stamps, windows, seq_len, pred_len, label_len=0):
        self.series = series
        self.stamps = stamps
        self.windows = list(windows)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.label_len = int(label_len)
        # The official `Dataset_Custom.data_x` is the whole scaled series; it is
        # only ever consumed by `get_person_similarity`, which the official exp
        # assigns and never reads.
        self.data_x = np.concatenate(list(series), axis=0)[:, None]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        rid, t = self.windows[i]
        s, st = self.series[rid], self.stamps[rid]
        return (s[t - self.seq_len:t, None], s[t:t + self.pred_len, None],
                st[t - self.seq_len:t], st[t:t + self.pred_len], i)

    def get_person_similarity(self):
        """Official ``(enc_in, enc_in)`` contract.

        ``np.corrcoef`` collapses to a 0-d array for a single channel, which the
        official ``torch.FloatTensor(...)`` cannot consume.  The frozen DA task
        is univariate, so the value is restored to its declared 1x1 shape; the
        official exp assigns this tensor and never reads it, so no computation
        depends on it.
        """
        return np.atleast_2d(np.corrcoef(self.data_x.T))


def build_bundles(cell, args, out_dir):
    """Official flag -> (dataset, loader) mapping for one cell."""
    runs = cell["runs"]
    n_fit, n_val = cell["n_fit"], max(1, int(panel.VAL_FRACTION * cell["n_fit"]))
    scaled, stamps, mean, scale = scale_runs(runs, n_fit, n_val)
    w = {role: windows_for(runs, n_fit, n_val, role, args.seq_len, args.pred_len)
         for role in ("train", "val", "test")}
    if not w["train"] or not w["val"] or not w["test"]:
        raise RuntimeError(f"empty PIR window role: train={len(w['train'])} "
                           f"val={len(w['val'])} test={len(w['test'])}")

    def ds(role):
        return PanelWindowDataset(scaled, stamps, w[role], args.seq_len, args.pred_len, args.label_len)

    bundles = {}
    for flag, role in (("train", "train"), ("retrieval", "train"),
                       ("val", "val"), ("test", "test")):
        d = ds(role)
        bundles[flag] = (d, DataLoader(d, batch_size=args.batch_size, shuffle=(role == "train"),
                                       drop_last=False, num_workers=0))
    meta = {"scaler_mean": mean, "scaler_scale": scale,
            "window_counts": {k: len(v) for k, v in w.items()},
            "n_val_days": int(n_val),
            "train_span_hours": [list(map(int, s[2:])) for s in panel.role_hours(runs, n_fit, n_val)]}
    return bundles, meta, scaled, stamps


class PanelPIRExp(Exp_Long_Term_Forecast_PIR):
    """The official PIR experiment with only ``_get_data`` bound to the panel."""

    def __init__(self, args, bundles):
        self.bundles = bundles
        super().__init__(args)

    def _get_data(self, flag):
        return self.bundles[flag]


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_day_origins(exp, dataset, args, mean, scale):
    """Predict exactly the frozen DIAG_EVAL day origins, in original units."""
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=0)
    exp.model.eval()
    chunks = []
    t0 = time.perf_counter()
    for x, y, xm, ym, _ in loader:
        x, y, xm, ym = x.float(), y.float(), xm.float(), ym.float()
        dec_inp = torch.zeros_like(y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([y[:, :args.label_len, :], dec_inp], dim=1).float()
        out = exp.model(x, xm, dec_inp, ym, mode="refine")[0]
        chunks.append(out[:, -args.pred_len:, :].numpy())
    infer_seconds = time.perf_counter() - t0
    pred = np.concatenate(chunks, axis=0)[:, :, 0] * scale + mean
    return pred, infer_seconds


def run_pir(cell, out_dir, seed=ANCHOR_SEED, log=None):
    """Train and evaluate the official PIR pair on one frozen cell."""
    torch.set_num_threads(NUM_THREADS)
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    args = build_args(out_dir, cell["host"], panel.SEQ, panel.H)
    bundles, meta, scaled, stamps = build_bundles(cell, args, out_dir)

    _seed_everything(seed)
    started = time.perf_counter()
    exp = PanelPIRExp(args, bundles)
    backbone_parameters = int(sum(p.numel() for p in exp.model.model.parameters()))
    setting = f"panel_{cell['market']}_{cell['host']}_{seed}"
    exp.train(setting)
    train_seconds = time.perf_counter() - started

    eval_windows = day_origin_windows(cell["runs"], cell["n_fit"])
    eval_ds = PanelWindowDataset(scaled, stamps, eval_windows, args.seq_len, args.pred_len, args.label_len)
    pred, infer_seconds = predict_day_origins(exp, eval_ds, args, meta["scaler_mean"], meta["scaler_scale"])
    if pred.shape != (cell["n_eval"], panel.H):
        raise RuntimeError(f"PIR produced {pred.shape}, expected {(cell['n_eval'], panel.H)}")

    total_params = int(sum(p.numel() for p in exp.model.parameters()))
    return {
        "pred": pred,
        "training_seconds": float(train_seconds),
        "inference_seconds": float(infer_seconds),
        "parameter_count": total_params,
        "backbone_parameter_count": backbone_parameters,
        "fidelity": PIR_LABEL,
        "seed": int(seed),
        "source": f"official PIR @ {PIR_COMMIT} :: models/PIR.py + exp/exp_long_term_forecasting_pir.py",
        "eval_windows": [[int(r), int(t)] for r, t in eval_windows],
        "meta": meta,
    }


def load_pir_result(path):
    z = np.load(path, allow_pickle=False)
    return z["pred"]


def main(argv=None):  # pragma: no cover - thin CLI for the stage driver
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("market", choices=list(panel.MARKETS))
    ap.add_argument("host", choices=list(panel.HOSTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=ANCHOR_SEED)
    ns = ap.parse_args(argv)

    cells, _, _, _ = panel.build_cells()
    cell = cells[(ns.market, ns.host)]
    out_dir = Path(ns.out) / f"{ns.market}__{ns.host}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_pir(cell, out_dir, seed=ns.seed)
    np.savez_compressed(out_dir / "pred.npz", pred=res.pop("pred"))
    (out_dir / "result.json").write_text(
        json.dumps({**res, "market": ns.market, "host": ns.host}, indent=2, default=str),
        encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
