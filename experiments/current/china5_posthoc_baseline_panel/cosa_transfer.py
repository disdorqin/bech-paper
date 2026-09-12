"""COSA online/TTA transfer onto the frozen China-5 Hosts (supplementary track).

Executes the **official** `SimpleOutputAdapter` class, compiled unchanged from the
pinned checkout, inside a driver whose ordering is copied line-by-line from the
official `tta/cosa.py::SimpleAdapter.adapt_simple` (commit
`43a8c8da4de74d5745a8713f6130c523b7df2694`):

    adapt_simple, per batch, in this exact order
      1. base forecast                     -> original_pred
      2. if adapters_enabled:              pred = output_adapter(original_pred, ctx_pre)
      3. metrics on pred
      4. update_memory_buffer(gt, original_pred)      # appends gt.mean(); enables adapters
      5. if adapters_enabled:              ctx_post = context(); STEPS grad steps on
                                           output_adapter(original_pred, ctx_post)

The predict-then-update chronology is therefore strict: the adapter that produces
day D's prediction has seen only days < D, and the gradient step that consumes day
D happens after D has been predicted.

Disclosed adaptation (task-level only, never tuned):

  * the official batch source (ETTh1 dataloader with an FFT-derived
    ``period + 1`` batch size) is replaced by one cache day per step, so
    ``batch_size = 1`` and ``targets.mean()`` is that day's 24h target mean.
    The official batch/period machinery does not exist in a 24h day-origin
    DA panel and is not reproducible there.
  * ``pred_len = 24``, ``n_vars = 1`` (the frozen DA task).
  * ``sample_history`` is seeded with the observed POST_TRAIN daily target means,
    which are strictly prior to every DEV_EVAL day.
  * all knobs (BASE_LR 0.001, WEIGHT_DECAY 1e-4, Adam, STEPS 3,
    BUFFER_CONTEXT_SIZE 10, FAST_ADAPTATION, ADAPTIVE_LR, PER_BATCH_LR_RESET,
    MAX_LR 0.005, MIN_LR 1e-4, ADAPTER_LAYERS 1, hidden 64) are the audited
    paper-era configuration from
    `experiments/current/hch_baseline_fidelity_reproduction/run_cosa_paper_era_anchor.py`.
  * the FFT period estimator is not used (PERIOD_N/AAS path is inactive here).

This is a **transferred implementation**: fidelity is
`TRANSFERRED_IMPLEMENTATION / PAPER_FIDELITY_UNRESOLVED`. It does not modify and
does not upgrade the repo's existing COSA adjudication
(`ONLINE_PARTIAL_FIDELITY / PAPER_NATIVE_METHOD_UNRESOLVED`).
"""
from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
_OFFICIAL = ROOT / "experiments/foundation/reference_deps/official_repos/COSA_paper_era_43a8c8d"
_SOURCE = _OFFICIAL / "tta/cosa.py"
COSA_COMMIT = "43a8c8da4de74d5745a8713f6130c523b7df2694"
COSA_LABEL = "TRANSFERRED_IMPLEMENTATION / PAPER_FIDELITY_UNRESOLVED"

BASE_LR, WEIGHT_DECAY = 0.001, 0.0001
STEPS, BUFFER_CONTEXT_SIZE = 3, 10
FAST_ADAPTATION, ADAPTIVE_LR, PER_BATCH_LR_RESET = True, True, True
MAX_LR, MIN_LR, ADAPTER_LAYERS, HIDDEN = 0.005, 0.0001, 1, 64
L2_COEF = 1e-4


def _official_output_adapter_cls():
    """Compile only `SimpleOutputAdapter`, avoiding the official package imports."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"), filename=str(_SOURCE))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "SimpleOutputAdapter")
    ns: dict = {"torch": torch, "nn": nn, "F": torch.nn.functional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_SOURCE), "exec"), ns)
    return ns["SimpleOutputAdapter"]


class CosaOnline:
    """Official output adapter driven with the official online ordering."""

    def __init__(self, pred_len: int, seed: int = 2021):
        torch.manual_seed(seed)
        cls = _official_output_adapter_cls()
        self.adapter = cls(pred_len=pred_len, buffer_context_size=BUFFER_CONTEXT_SIZE,
                           n_vars=1, var_wise_gating=False, num_layers=ADAPTER_LAYERS,
                           hidden_dim=HIDDEN)
        self.opt = torch.optim.Adam(self.adapter.parameters(), lr=BASE_LR,
                                    weight_decay=WEIGHT_DECAY)
        self.params = list(self.adapter.parameters())
        self.sample_history: list[float] = []
        self.adapters_enabled = False
        self.current_lr = BASE_LR
        self.loss_history: list[float] = []

    # -- official _get_individual_context_for_batch -------------------------
    def _context(self, batch_size: int) -> torch.Tensor:
        if len(self.sample_history) == 0:
            return torch.zeros(batch_size, BUFFER_CONTEXT_SIZE)
        n = min(BUFFER_CONTEXT_SIZE, len(self.sample_history))
        vals = [self.sample_history[-(i + 1)] for i in range(n)]
        if len(vals) < BUFFER_CONTEXT_SIZE:
            last = vals[-1] if vals else 0.0
            vals.extend([last] * (BUFFER_CONTEXT_SIZE - len(vals)))
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)

    def _update_buffer(self, gt: torch.Tensor) -> None:
        self.sample_history.append(float(gt.mean().item()))
        self.adapters_enabled = True

    # -- official _adaptive_learning_rate ----------------------------------
    def _adaptive_lr(self, loss: float, step: int) -> float:
        if step == 0 and PER_BATCH_LR_RESET:
            self.current_lr = BASE_LR
            self.loss_history.append(loss)
            return self.current_lr
        self.loss_history.append(loss)
        recent = list(self.loss_history)[-3:]
        if len(recent) < 2:
            return self.current_lr
        trend = recent[-1] - recent[0]
        var = float(torch.tensor(recent).var().item())
        if trend > 0 and var < 1e-6:
            self.current_lr = min(self.current_lr * 1.2, MAX_LR)
        elif trend < -0.01:
            self.current_lr = min(self.current_lr * 1.05, MAX_LR)
        elif abs(trend) < 1e-6:
            self.current_lr = max(self.current_lr * 0.8, MIN_LR)
        if step >= 1:
            cos = 0.5 * (1 + float(torch.cos(torch.tensor(step * 3.14159 / STEPS))))
            self.current_lr = MIN_LR + (self.current_lr - MIN_LR) * cos
        return self.current_lr


def run_cosa(host_pred: np.ndarray, y_true: np.ndarray,
             seed_means: np.ndarray, seed: int = 2021) -> dict:
    """Strict online/TTA over an ordered day sequence.

    host_pred, y_true : (n_days, 24) float64
    seed_means        : observed daily target means from POST_TRAIN (strictly prior)
    """
    n, h = host_pred.shape
    eng = CosaOnline(h, seed=seed)
    eng.sample_history = [float(v) for v in np.asarray(seed_means).reshape(-1)]
    out = np.empty((n, h), dtype=np.float64)
    n_adapted, corrections = 0, []
    t0 = time.perf_counter()
    for i in range(n):
        yp = torch.as_tensor(host_pred[i:i + 1, :, None], dtype=torch.float32)
        gt = torch.as_tensor(y_true[i:i + 1, :, None], dtype=torch.float32)

        # 1-2. predict with the pre-update context
        ctx_pre = eng._context(1)
        if eng.adapters_enabled:
            eng.adapter.eval()
            with torch.no_grad():
                pred = eng.adapter(yp, ctx_pre)
            n_adapted += 1
        else:
            pred = yp
        out[i] = pred[0, :, 0].numpy()
        corrections.append(float(np.mean(np.abs(pred[0, :, 0].numpy() - host_pred[i]))))

        # 4. buffer update happens BEFORE the gradient step, as in the official loop
        eng._update_buffer(gt)

        # 5. adapt on the post-update context, using the *original* host prediction
        ctx_post = eng._context(1)
        n_steps = min(STEPS, 5) if FAST_ADAPTATION else STEPS
        for step in range(n_steps):
            eng.adapter.train()
            adapted = eng.adapter(yp, ctx_post)
            loss = nn.functional.mse_loss(adapted, gt)
            l2 = sum(p.pow(2).sum() for p in eng.params if p.requires_grad)
            loss = loss + L2_COEF * l2
            if ADAPTIVE_LR and FAST_ADAPTATION:
                lr = eng._adaptive_lr(float(loss.item()), step)
                for g in eng.opt.param_groups:
                    g["lr"] = lr
            eng.opt.zero_grad()
            loss.backward()
            max_norm = max(0.05, min(0.5, float(loss.item()))) if FAST_ADAPTATION else 0.1
            torch.nn.utils.clip_grad_norm_(eng.adapter.parameters(), max_norm=max_norm)
            eng.opt.step()
        eng.adapter.eval()
    return {
        "pred": out.astype(np.float32),
        "fidelity": COSA_LABEL,
        "setting": "strict_online_TTA_predict_then_update",
        "source": f"official COSA @ {COSA_COMMIT} :: tta/cosa.py::SimpleOutputAdapter",
        "source_file_sha256": __import__("hashlib").sha256(_SOURCE.read_bytes()).hexdigest().upper(),
        "n_days": int(n),
        "n_days_adapted": int(n_adapted),
        "first_day_is_unadapted_host": bool(n_adapted == max(n - 1, 0)),
        "mean_abs_correction": float(np.mean(corrections)),
        "context_seed_days": int(len(seed_means)),
        "adaptation_seconds": time.perf_counter() - t0,
        "knobs": {"BASE_LR": BASE_LR, "WEIGHT_DECAY": WEIGHT_DECAY, "STEPS": STEPS,
                  "BUFFER_CONTEXT_SIZE": BUFFER_CONTEXT_SIZE, "MAX_LR": MAX_LR,
                  "MIN_LR": MIN_LR, "ADAPTER_LAYERS": ADAPTER_LAYERS, "HIDDEN": HIDDEN,
                  "OPTIMIZER": "adam", "batch_size": 1},
        "disclosed_adaptation": [
            "official FFT period+1 batch construction replaced by one cache day per step",
            "pred_len=24, n_vars=1 for the frozen DA task",
            "sample_history seeded with observed POST_TRAIN daily target means",
        ],
    }
