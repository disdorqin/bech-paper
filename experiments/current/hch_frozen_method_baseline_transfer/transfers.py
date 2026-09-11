"""Audited post-hoc baseline transfers for the frozen six-cell panel.

Controlling protocol:
    docs/current/HCH_FROZEN_METHOD_BASELINE_TRANSFER_ADJUDICATION_20260911.md

Two of the three admitted offline baselines correct the *frozen Host* and are
therefore defined directly on the frozen day rows:

* ``MatchedDirectResidual`` -- the frozen internal same-information control.
  The implementation is replayed verbatim from
  ``hch_unified_da_shape_upgrade.run_canary.fit_direct`` (``nn.Linear(624,24)``,
  AdamW ``lr=1e-3`` ``weight_decay=1e-4``, 1000 L1 steps, best-loss state kept).
  Its capacity and target are not modified.

* ``delta-Adapter Ada-Y`` -- descends from the independently audited official
  checkout at commit ``0add06ea7b4d2e0a84c364a8be72eef2676a92f2``.  The official
  ``PostProcessingNet`` class is AST-extracted from that checkout; the audited
  transfer semantics are preserved exactly:

    - ``pred = base + delta * adapter(base)`` with the source ``delta = 0.1``;
    - Adam at the source ``post_train_lr = 1e-4``;
    - ``clip_grad_norm_(adapter.parameters(), 0.01)``;
    - ``num_episodes = 10`` with patience-3 early stopping on the *source*
      validation split;
    - the source normalization choice -- the bounded ``tanh`` correction is only
      meaningful in the standardized units the official datasets are served in,
      so the adapter is trained and applied in train-span standardized units and
      the result is de-standardized before the frozen metric is computed.

Nothing here is tuned on DIAG_EVAL.  The only quantities read from DIAG_EVAL are
the frozen features and the frozen Host prediction that the metric needs.
"""
from __future__ import annotations

import ast
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from experiments.current.hch_frozen_method_baseline_transfer import panel
from experiments.current.hch_unified_da_shape_upgrade.run_canary import fit_direct
from src.mvp.hch_minimal_repair.training import set_deterministic
from utils.timing import parameter_count

ROOT = panel.ROOT
OFFICIAL_DELTA = ROOT / "experiments/foundation/reference_deps/official_repos/delta-Adapter/Adapter-X+Y"
DELTA_COMMIT = "0add06ea7b4d2e0a84c364a8be72eef2676a92f2"

DELTA = 0.1          # source delta convention
POST_TRAIN_LR = 1e-4 # source post-training learning rate
NUM_EPISODES = 10    # source num_episodes
PATIENCE = 3         # source early-stopping patience
CLIP = 0.01          # source grad-norm clip on the adapter
BATCH_SIZE = 16      # source batch size for the Ada-Y adapter stage

DIRECT_LABEL = "FROZEN_INTERNAL_CONTROL"
DELTA_LABEL = "ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY"


def load_official_postprocessingnet():
    """Compile only the official class, avoiding unrelated optional model imports."""
    source_path = OFFICIAL_DELTA / "experiments/exp_post_y_add.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PostProcessingNet")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["PostProcessingNet"]


PostProcessingNet = load_official_postprocessingnet()


def delta_source_sha256() -> str:
    """Hash the exact official file the adapter class is extracted from."""
    return panel.sha(OFFICIAL_DELTA / "experiments/exp_post_y_add.py")


def adapter_hidden_dim(cell) -> int:
    """The source's ``args.d_model``: the width of the backbone being adapted.

    The official code builds ``PostProcessingNet(dim, args.d_model, dim)``, so the
    adapter hidden width is the adapted backbone's ``d_model``.  Here that is the
    frozen Host's own backbone width, read from the frozen cache metadata rather
    than chosen.
    """
    cfg = cell["host_metadata"]["official_config"]
    return int(cfg["d_model"])


# --------------------------------------------------------------------------
# MatchedDirectResidual
# --------------------------------------------------------------------------

def run_direct(cell, seed):
    """Replay the frozen same-information direct residual control."""
    model, train_seconds = fit_direct(
        cell["fi"].x, cell["ef"], cell["fi"].repair_available, seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        dn = model(torch.as_tensor(cell["ei"].x, dtype=torch.float32)).numpy()
    infer_seconds = time.perf_counter() - t0
    pred = cell["hp"] + dn * cell["scale"] * cell["avail"][:, None]
    return {
        "pred": pred,
        "training_seconds": float(train_seconds),
        "inference_seconds": float(infer_seconds),
        "parameter_count": int(parameter_count(model)),
        "fidelity": DIRECT_LABEL,
        "source": "experiments/current/hch_unified_da_shape_upgrade/run_canary.py::fit_direct",
    }


# --------------------------------------------------------------------------
# delta-Adapter Ada-Y
# --------------------------------------------------------------------------

def _standardizer(y_train_rows):
    """StandardScaler over the training rows -- the source normalization choice."""
    flat = np.asarray(y_train_rows, dtype=np.float64).reshape(-1)
    return float(flat.mean()), float(flat.std())


def _adapter_mse(base_std, y_std, adapter):
    adapter.eval()
    with torch.no_grad():
        b = torch.as_tensor(base_std, dtype=torch.float32)
        t = torch.as_tensor(y_std, dtype=torch.float32)
        corr = DELTA * adapter(b.reshape(b.shape[0], -1)).view_as(b)
        return float(torch.mean((b + corr - t) ** 2))


def run_delta(cell, seed, record_history=False):
    """Transfer the audited Ada-Y adapter onto the frozen Host of one cell."""
    set_deterministic(seed)
    rng = np.random.default_rng(seed)

    y = np.asarray(cell["fit"].y_true[:, :, 0], dtype=np.float64)
    hp = np.asarray(cell["fit"].host_pred[:, :, 0], dtype=np.float64)
    n_fit = len(y)
    n_val = max(1, int(panel.VAL_FRACTION * n_fit))
    tr = np.arange(0, n_fit - n_val)
    va = np.arange(n_fit - n_val, n_fit)
    if len(tr) < 2 or len(va) < 1:
        raise RuntimeError("cell has too few DIAG_FIT rows for the Ada-Y transfer")

    mu, sd = _standardizer(y[tr])
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError("degenerate train-span scale for the Ada-Y transfer")
    y_std, base_std = (y - mu) / sd, (hp - mu) / sd

    dim = int(y.shape[1] * 1)  # pred_len * enc_in, the source's flattened output dim
    hidden = adapter_hidden_dim(cell)
    torch.manual_seed(seed)
    adapter = PostProcessingNet(dim, hidden, dim)
    opt = torch.optim.Adam(adapter.parameters(), lr=POST_TRAIN_LR)
    criterion = nn.MSELoss()

    best, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.perf_counter()
    for epoch in range(NUM_EPISODES):
        adapter.train()
        order = rng.permutation(tr)
        losses = []
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            if len(idx) < 2:  # BatchNorm1d cannot take a singleton batch in train mode
                continue
            b = torch.as_tensor(base_std[idx], dtype=torch.float32)
            t = torch.as_tensor(y_std[idx], dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            corr = DELTA * adapter(b.reshape(b.shape[0], -1)).view_as(b)
            loss = criterion(b + corr, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), CLIP)
            opt.step()
            losses.append(float(loss.detach()))
        val = _adapter_mse(base_std[va], y_std[va], adapter)
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "val_mse": val})
        if val < best:
            best, best_state, bad = val, deepcopy(adapter.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    training_seconds = time.perf_counter() - t0
    if best_state is None:
        raise RuntimeError("Ada-Y transfer never produced a validation state")
    adapter.load_state_dict(best_state)

    ev_base = (np.asarray(cell["ev"].host_pred[:, :, 0], dtype=np.float64) - mu) / sd
    t0 = time.perf_counter()
    adapter.eval()
    with torch.no_grad():
        b = torch.as_tensor(ev_base, dtype=torch.float32)
        corr = DELTA * adapter(b.reshape(b.shape[0], -1)).view_as(b)
    infer_seconds = time.perf_counter() - t0
    pred = (ev_base + corr.numpy().astype(np.float64)) * sd + mu

    out = {
        "pred": pred,
        "training_seconds": float(training_seconds),
        "inference_seconds": float(infer_seconds),
        "parameter_count": int(parameter_count(adapter)),
        "fidelity": DELTA_LABEL,
        "hidden_dim": hidden,
        "source_dim": dim,
        "epochs_run": len(history),
        "best_val_mse": float(best),
        "mu": mu, "sd": sd,
        "source": f"official delta-Adapter Adapter-X+Y @ {DELTA_COMMIT} :: PostProcessingNet",
        "source_file_sha256": delta_source_sha256(),
    }
    if record_history:
        out["history"] = history
    return out
