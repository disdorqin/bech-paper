"""Fixed execution contract for the RCL v4.1 ladder.

Preparation only: this module defines *what* the run must hold fixed.  It reads
no data and fits nothing.  The runner, once authorized, must import these
constants rather than re-declaring them, so the structural tests can verify the
contract against the same object the execution will use.
"""
from __future__ import annotations

from typing import Sequence

__all__ = [
    "CELLS", "SEEDS", "TEST_LABEL_READ_COUNT", "WARMUP_FRACTION",
    "OOF_BLOCKS", "LOCAL_TRAIN_BLOCKS", "VAL_CAL_FRACTION",
    "oof_partition", "local_train_days", "assert_stage1_frozen",
    "stage2_optimizer", "PrequentialQHistory",
]

#: The fixed 8-cell development panel (protocol section 3).  Frozen before any
#: fit; nothing may add, drop or reorder a cell.
CELLS: tuple[tuple[str, str], ...] = (
    ("GANSU_DA", "LSTM"),
    ("GANSU_DA", "TimeMixer"),
    ("GANSU_DA", "PatchTST"),
    ("SHANDONG_DA", "TimeMixer"),
    ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"),
    ("SHAANXI_DA", "PatchTST"),
    ("NINGXIA_DA", "LSTM"),
)

SEEDS: tuple[int, ...] = (7, 17, 37)

#: V2 TEST is sealed.  The only legal value, and the one the audit must report.
TEST_LABEL_READ_COUNT = 0

#: TRAIN chronology: first 40% is warm-up, the rest splits into four OOF blocks.
WARMUP_FRACTION = 0.40
OOF_BLOCKS: tuple[str, str, str, str] = ("B1", "B2", "B3", "B4")

#: LOCAL_TRAIN is B2 u B3 u B4: the largest support on which a causal OOF Level
#: prediction and a strictly earlier-block target are both defined.
LOCAL_TRAIN_BLOCKS: tuple[str, str, str] = ("B2", "B3", "B4")

#: VAL splits chronologically; the first half is CAL, the second is EVAL.
VAL_CAL_FRACTION = 0.50


def oof_partition(n_days: int) -> dict[str, range]:
    """Deterministic chronological warm-up / B1..B4 partition of TRAIN days.

    Blocks are contiguous, strictly increasing and disjoint.  ``B1`` is present so
    the runner can use it as a Local-history warm start only; it is never a
    Local training target day.
    """
    if n_days < 5:
        raise ValueError("need at least five TRAIN days for a 40/4 partition")
    warm_end = int(n_days * WARMUP_FRACTION)
    remaining = n_days - warm_end
    if remaining < 4:
        raise ValueError("not enough post-warm-up days for four OOF blocks")
    base, extra = divmod(remaining, 4)
    blocks: dict[str, range] = {}
    start = warm_end
    for index, name in enumerate(OOF_BLOCKS):
        size = base + (1 if index < extra else 0)
        blocks[name] = range(start, start + size)
        start += size
    return {"WARMUP": range(0, warm_end), **blocks}


def local_train_days(n_days: int) -> list[int]:
    """Sorted LOCAL_TRAIN day indices: B2 u B3 u B4, never warm-up or B1."""
    partition = oof_partition(n_days)
    out: set[int] = set()
    for name in LOCAL_TRAIN_BLOCKS:
        out.update(partition[name])
    excluded = set(partition["WARMUP"]) | set(partition["B1"])
    assert not (out & excluded), "LOCAL_TRAIN must exclude warm-up and B1"
    return sorted(out)


def assert_stage1_frozen(level_model, local_model) -> None:
    """Fail unless Stage 1 is fully frozen and Stage 2 holds none of it.

    Three separate breaches are checked, in this order:

    1. Stage 2 nests a Stage-1 *module* — the boundary must be structural;
    2. Stage 2 owns one of Stage-1's *parameter objects*;
    3. a Stage-1 parameter still has ``requires_grad=True`` — freezing is not
       merely "Stage 2 does not list it", it is "Stage 1 cannot receive a
       gradient at all".

    The third check is the one that was missing: sharing checks pass trivially on a
    Stage 1 that was simply never frozen, so the contract could hold while the
    runner forgot to freeze.  Call this before building the Stage-2 optimizer.

    Checks 1 and 2 run before 3 so a smuggled shared parameter is still reported as
    a smuggling failure rather than being masked by an unrelated freeze failure.
    """
    import torch

    stage1_ids = {id(p) for p in level_model.parameters()}
    for name, module in local_model.named_modules():
        if type(module).__name__ == type(level_model).__name__:
            raise AssertionError(
                f"Stage-2 model nests a Stage-1 module at {name!r}: the freeze "
                "boundary must be structural, not conventional")
    for name, param in local_model.named_parameters():
        assert id(param) not in stage1_ids, (
            f"Stage-2 model owns a Stage-1 parameter: {name}")
    assert isinstance(local_model, torch.nn.Module)
    for name, param in level_model.named_parameters():
        if param.requires_grad:
            raise AssertionError(
                f"Stage-1 parameter {name!r} still requires grad: Stage 1 must be "
                "fully frozen before the Stage-2 optimizer is created")


def stage2_optimizer(local_model, level_model, *, lr: float = 1e-3,
                     weight_decay: float = 1e-4):
    """Build the Stage-2 optimizer and prove it carries no Stage-1 parameter.

    Called only after Stage 1 is frozen.  If a Stage-1 parameter ever reaches
    this optimizer the freeze boundary has been breached, which is exactly what
    the structural test checks.
    """
    import torch

    assert_stage1_frozen(level_model, local_model)
    level_ids = {id(p) for p in level_model.parameters()}
    parameters = [p for p in local_model.parameters() if id(p) not in level_ids]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    for group in optimizer.param_groups:
        for param in group["params"]:
            assert id(param) not in level_ids, "Stage-1 parameter in Stage-2 optimizer"
    return optimizer


class PrequentialQHistory:
    """VAL q-history that only ever receives already-revealed post-Level q.

    The discipline the protocol requires is: predict Level -> predict Local ->
    persist -> reveal target -> compute q -> append.  This class enforces it by
    refusing to expose a day before its q has been appended, refusing to append
    twice, and refusing to append a day that is not strictly later than the last
    one already appended.

    The chronology rule is not bookkeeping.  ``q`` is built against a frozen
    Stage-1 Level fitted on strictly earlier days, so appending day 3 after day 5
    would splice a later-vintage history in front of an earlier one and hand the
    Shape GRU a future it should not have.  Replayed or reordered delivery is a
    protocol breach, and it fails closed rather than being silently sorted.
    """

    def __init__(self, seed_windows: Sequence):
        self._windows = list(seed_windows)
        self._appended: set[int] = set()
        self._last_day: int | None = None

    def windows(self):
        return list(self._windows)

    def reveal_and_append(self, day_index: int, q) -> None:
        day = int(day_index)
        if day in self._appended:
            raise ValueError(f"day {day} already appended; no replay")
        if self._last_day is not None and day <= self._last_day:
            raise ValueError(
                f"day {day} is not strictly after the last appended day "
                f"{self._last_day}; prequential q-history must be strictly "
                "increasing in delivery order")
        self._windows.append(q)
        self._appended.add(day)
        self._last_day = day

    def appended_days(self) -> list[int]:
        return sorted(self._appended)
