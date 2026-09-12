"""Common-benchmark data contract — `COMMON_BENCHMARK_701020_V1`.

This stage does not define a new *eligibility* rule.  It re-partitions the target-day
sequence each market already admits under its frozen contract, into the literature
chronological 70/10/20 benchmark split:

```text
n_train = int(0.70 * M)          M = the currently OPEN eligible target-day sequence
n_test  = int(0.20 * M)
n_val   = M - n_train - n_test
```

where the fractions are evaluated as exact integer arithmetic (`(70*M)//100`,
`(20*M)//100`) so that no floating-point representation can move a boundary.  The
denominator `M` is the **open** block — `HOST_TRAIN+HOST_VAL+POST_TRAIN+DEV_EVAL` for
the China-5 family and `S1+DIAG_FIT+DIAG_EVAL` for GANSU.  `PROTECTED_FINAL` /
`S3+S4` is not part of the new split, is never repartitioned, and is never read: an
old sealed-final role is not silently repurposed as TEST.

Why `M` and not the registry's raw eligible count: the two families were frozen under
the old five-role allocation, and the only day sequence either contract actually
admits to a training/evaluation consumer is its open block.  Applying the new
fractions to `M` reproduces all five pre-registered expected counts exactly
(GANSU 203/30/58, SHANDONG 809/116/231, SHAANXI 198/30/56, NINGXIA 70/11/20,
QINGHAI 65/11/18); applying them to the registry total does not reproduce any of them.

Two read disciplines are enforced here, not merely documented:

* **Timestamp-only split construction.**  The split is built from the timestamp column
  alone, before any price value is read.  `split_plan()` never opens the target column.
* **TEST targets stay closed.**  `load_pretest_windows()` performs a *projected* read of
  the TRAIN and VAL day rows (plus their strictly-prior 168h context) and skips every
  other row at parse time, so a TEST target value is never materialised during PRETEST.
  `load_windows(..., roles=(...))` is the general form; asking it for `TEST` during
  PRETEST is refused by `assert_pretest_safe()`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
IMPL = Path(__file__).resolve().parent
BREADTH_IMPL = ROOT / "experiments/current/hch_china_host_breadth_expansion/implementation"
PARENT_STAGE = ROOT / "experiments/current/china5_posthoc_baseline_panel"
GANSU_STAGE = ROOT / "experiments/current/hch_unified_da_shape_upgrade"

for _p in (str(IMPL), str(BREADTH_IMPL), str(PARENT_STAGE), str(GANSU_STAGE),
           str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as BC                                                      # noqa: E402
import panel_contract as pc                                                 # noqa: E402
import run_canary as gs                                                     # noqa: E402
from utils.benchmark_cache import stable_hash                               # noqa: E402
from utils.benchmark_contracts import ForecastWindows                       # noqa: E402

PROTOCOL_ID = "COMMON_BENCHMARK_701020_V1"
MARKETS = ("GANSU_DA", "SHANDONG_DA", "SHAANXI_DA", "NINGXIA_DA", "QINGHAI_DA")
HOSTS = ("PatchTST", "TimeMixer", "iTransformer", "LSTM")
ROLES = ("TRAIN", "VAL", "TEST")

TRAIN_PCT, TEST_PCT = 70, 20

# Pre-registered by the protocol document.  A mismatch is a stop condition, never a
# silent re-derivation.
EXPECTED_COUNTS = {
    "GANSU_DA": (203, 30, 58),
    "SHANDONG_DA": (809, 116, 231),
    "SHAANXI_DA": (198, 30, 56),
    "NINGXIA_DA": (70, 11, 20),
    "QINGHAI_DA": (65, 11, 18),
}

SEQ, H = 168, 24

EV = ROOT / "experiments/evidence/hch_china5_common_benchmark_701020_20260912"

DAY_CONVENTION = {"GANSU_DA": "origin_00:00", "SHANDONG_DA": "hour_ending",
                  "SHAANXI_DA": "hour_ending", "NINGXIA_DA": "hour_ending",
                  "QINGHAI_DA": "hour_ending"}

FAMILY = {"GANSU_DA": "GANSU", "SHANDONG_DA": "CHINA5", "SHAANXI_DA": "CHINA5",
          "NINGXIA_DA": "CHINA5", "QINGHAI_DA": "CHINA5"}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest().upper()


# --------------------------------------------------------------------- contract
@dataclass
class OpenBlock:
    """The currently-open eligible target-day sequence of one market, timestamps only.

    No price value has been read when this object exists.
    """
    market: str
    dataset_id: str
    family: str
    day_ids: tuple[str, ...]              # chronological, the full open block
    positions: dict[str, np.ndarray]      # day id -> source row positions
    timestamps: pd.DatetimeIndex          # the source timestamp column
    source_path: str
    source_sha256: str
    target_column: str
    eligibility_contract: str
    exclusion_contract: str
    native_roles: tuple[str, ...]

    @property
    def M(self) -> int:
        return len(self.day_ids)


def open_block(market: str) -> OpenBlock:
    """Timestamp-only view of one market's open eligible target-day sequence."""
    if market not in MARKETS:
        raise KeyError(f"market not in the common-benchmark coordinate system: {market}")
    family = FAMILY[market]

    if family == "GANSU":
        contract = gs.timestamp_contract_gansu()
        keep = {"S1", "DIAG_FIT", "DIAG_EVAL"}
        days = tuple(d for d, s in contract.segments.items() if s in keep)
        positions = {d: np.asarray(contract.positions[d], int) for d in days}
        native = ("S1", "DIAG_FIT", "DIAG_EVAL")
        elig = ("run_canary.timestamp_contract_gansu: a natural day is eligible iff its 24 "
                "hourly rows are contiguous, it starts at or after source row SEQ, and its "
                "168 strictly-prior hours are contiguous; day id = calendar date of the day "
                "origin (00:00 convention)")
        excl = ("forbidden target-day inputs = 日前电价 / 实时电价; S3+S4 sealed, never read")
        return OpenBlock(market, "GANSU_DA", family, days, positions,
                         contract.timestamps, "data/CHINA/GANSU/甘肃24h电价数据集.xlsx",
                         sha256_file(ROOT / "data/CHINA/GANSU/甘肃24h电价数据集.xlsx"),
                         "日前电价", elig, excl, native)

    phys = BC.CHINA5[market]
    contract = pc.build_day_contract(phys)
    days = tuple(contract.open_days)
    positions = {d: np.asarray(contract.positions[d], int) for d in days}
    return OpenBlock(market, pc.MARKETS[phys]["dataset_id"], family, days, positions,
                     contract.timestamps, pc.MARKETS[phys]["path"],
                     pc.sha256_file(pc.source_path(phys)), pc.MARKETS[phys]["target"],
                     ("panel_contract.build_day_contract: a natural day is eligible iff it has "
                      "exactly 24 hour-ending rows one hour apart, it starts at or after source "
                      "row SEQ, and its 168 strictly-prior hours are contiguous; day id = "
                      "(timestamp - 1h).date (hour-ending convention)"),
                     ("竞价空间预测值 excluded project-wide (pre-DA publication time "
                      "unconfirmed); realized columns never inputs; PROTECTED_FINAL sealed, "
                      "never read"),
                     tuple(pc.OPEN_ROLES))


# ------------------------------------------------------------------------ split
@dataclass
class SplitPlan:
    market: str
    dataset_id: str
    family: str
    day_ids: tuple[str, ...]
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    M: int

    @property
    def counts(self) -> tuple[int, int, int]:
        return len(self.train_ids), len(self.val_ids), len(self.test_ids)

    def ids(self, role: str) -> tuple[str, ...]:
        return {"TRAIN": self.train_ids, "VAL": self.val_ids, "TEST": self.test_ids}[role]


def split_plan(market: str) -> SplitPlan:
    """The 70/10/20 chronological partition of the open block. Timestamps only."""
    ob = open_block(market)
    m = ob.M
    n_train = (TRAIN_PCT * m) // 100
    n_test = (TEST_PCT * m) // 100
    n_val = m - n_train - n_test
    if min(n_train, n_val, n_test) < 1:
        raise RuntimeError(f"{market}: degenerate 70/10/20 partition of M={m}")
    days = ob.day_ids
    return SplitPlan(market, ob.dataset_id, ob.family, days,
                     days[:n_train], days[n_train:n_train + n_val], days[n_train + n_val:], m)


def assert_expected_counts(plan: SplitPlan) -> None:
    exp = EXPECTED_COUNTS[plan.market]
    if plan.counts != exp:
        raise RuntimeError(
            f"{plan.market}: split counts {plan.counts} differ from the protocol's "
            f"pre-registered {exp} (M={plan.M}); source/eligibility identity must be "
            f"diagnosed before any further PRETEST step")


# ------------------------------------------------------------------ manifest
def split_manifest(market: str) -> dict:
    """The frozen PRETEST split record.  Carries day IDs and hashes; no TEST values."""
    plan = split_plan(market)
    ob = open_block(market)
    assert_expected_counts(plan)
    return {
        "schema": "china5_common_benchmark_split.v1",
        "protocol_id": PROTOCOL_ID,
        "market": plan.market,
        "dataset_id": plan.dataset_id,
        "family": plan.family,
        "split_rule": "n_train=int(0.70*M); n_test=int(0.20*M); n_val=M-n_train-n_test",
        "split_rule_integer_form": "n_train=(70*M)//100; n_test=(20*M)//100",
        "n_open_days": int(plan.M),
        "counts": {"TRAIN": len(plan.train_ids), "VAL": len(plan.val_ids),
                   "TEST": len(plan.test_ids)},
        "expected_counts": {"TRAIN": EXPECTED_COUNTS[market][0],
                            "VAL": EXPECTED_COUNTS[market][1],
                            "TEST": EXPECTED_COUNTS[market][2]},
        "counts_match_preregistration": True,
        # -- day identity -----------------------------------------------------
        "ordered_eligible_target_day_ids": list(plan.day_ids),
        "TRAIN_ids": list(plan.train_ids),
        "VAL_ids": list(plan.val_ids),
        "TEST_ids": list(plan.test_ids),
        "day_id_convention": DAY_CONVENTION[market],
        "first_train_id": plan.train_ids[0], "last_train_id": plan.train_ids[-1],
        "first_val_id": plan.val_ids[0], "last_val_id": plan.val_ids[-1],
        "first_test_id": plan.test_ids[0], "last_test_id": plan.test_ids[-1],
        "chronological": True,
        # -- hashes -----------------------------------------------------------
        "ordered_day_ids_sha256": sha256_text("\n".join(plan.day_ids)),
        "TRAIN_id_sha256": sha256_text("\n".join(plan.train_ids)),
        "VAL_id_sha256": sha256_text("\n".join(plan.val_ids)),
        "test_target_blind_id_sha256": sha256_text("\n".join(plan.test_ids)),
        "test_blind_hash_meaning": ("sha256 over the ordered TEST target-day IDs only; it "
                                    "commits to which days are TEST without revealing or "
                                    "depending on any TEST target value"),
        # -- provenance / contract identity ----------------------------------
        "source": ob.source_path,
        "source_sha256": ob.source_sha256,
        "target_column": ob.target_column,
        "eligibility_contract": ob.eligibility_contract,
        "exclusion_contract": ob.exclusion_contract,
        "eligibility_contract_sha256": sha256_text(ob.eligibility_contract + "||" + ob.exclusion_contract),
        "native_roles_of_open_block": list(ob.native_roles),
        "sealed_roles_excluded_from_split": ("PROTECTED_FINAL" if ob.family == "CHINA5"
                                             else "S3+S4"),
        "sealed_roles_partitioned": False,
        "denominator_basis": ("the currently-open eligible block only; the old sealed-final "
                              "role is not repartitioned and no old final role is used as TEST"),
        "thresholds_fit_partition": "TRAIN",
        "scaler_fit_partition": "TRAIN",
        "host_selection_partition": "VAL",
        "generated_before_any_test_label_read": True,
    }


def manifest_hash(manifest: dict) -> str:
    return stable_hash(manifest)


# ------------------------------------------------------------------- readers
_TARGET_CACHE: dict[tuple[str, tuple[str, ...]], tuple[dict, dict]] = {}


def _read_target_values(market: str, day_ids) -> tuple[dict, dict]:
    """Projected target read of exactly the requested days plus their prior context.

    Every other row of the source file is skipped at parse time, so an unrequested
    (e.g. TEST) target value is never materialised in memory.
    """
    key = (market, tuple(day_ids))
    if key in _TARGET_CACHE:
        return _TARGET_CACHE[key]
    ob = open_block(market)
    wanted = set(day_ids)
    unknown = wanted - set(ob.day_ids)
    if unknown:
        raise RuntimeError(f"{market}: requested days outside the open block: {sorted(unknown)[:5]}")

    target_pos: set[int] = set()
    for d in day_ids:
        target_pos.update(ob.positions[d].tolist())
    needed: set[int] = set(target_pos)
    for d in day_ids:
        start = int(ob.positions[d][0])
        if start < SEQ:
            raise RuntimeError(f"{market}: day {d} has no full {SEQ}h context")
        needed.update(range(start - SEQ, start))

    source = ROOT / ob.source_path
    if ob.family == "GANSU":
        skip = [i + 1 for i in range(len(ob.timestamps)) if i not in needed]
        df = gs._reader(source, usecols=[gs.TCOL["GANSU_DA"], gs.TARGET["GANSU_DA"]],
                        skiprows=skip)
        tcol, tgt = gs.TCOL["GANSU_DA"], gs.TARGET["GANSU_DA"]
        ts = pd.to_datetime(df[tcol], errors="raise")
        val = pd.to_numeric(df[tgt], errors="raise")
    else:
        phys = BC.CHINA5[market]
        skip = [i + 1 for i in range(len(ob.timestamps)) if i not in needed]
        df = pc.read_projected(phys, [ob.target_column], set(needed))
        tcol, tgt = pc.MARKETS[phys]["tcol"], ob.target_column
        ts = pd.to_datetime(df[tcol], errors="raise")
        val = pd.to_numeric(df[tgt], errors="raise")

    if len(ts) != len(needed):
        raise RuntimeError(f"{market}: projected read returned {len(ts)} rows, expected {len(needed)}")
    mp = {pd.Timestamp(t): float(v) for t, v in zip(ts, val)}
    if len(mp) != len(ts):
        raise RuntimeError(f"{market}: projected read collapsed duplicate timestamps")
    missing = [t for t in (pd.Timestamp(ob.timestamps[i]) for i in sorted(needed)) if t not in mp]
    if missing:
        raise RuntimeError(f"{market}: projected read incomplete ({len(missing)} rows)")

    audit = {
        "target_column": ob.target_column,
        "days_requested": len(day_ids),
        "target_day_values_read": len(target_pos),
        "causal_context_values_read": len(needed - target_pos),
        "rows_materialised": len(needed),
        "rows_in_source": int(len(ob.timestamps)),
        "rows_skipped_at_parse_time": int(len(ob.timestamps)) - len(needed),
        "test_target_values_read": 0,
        "sealed_final_target_values_read": 0,
        "read_discipline": ("projected parse-time skiprows; unrequested rows are never parsed, "
                            "so their target values never enter memory"),
    }
    _TARGET_CACHE[key] = (mp, audit)
    return mp, audit


def load_windows(market: str, roles=("TRAIN", "VAL")) -> tuple[ForecastWindows, dict]:
    """Chronological windows for the requested roles, projected read."""
    plan = split_plan(market)
    for r in roles:
        if r not in ROLES:
            raise KeyError(f"unknown role {r!r}")
    ob = open_block(market)
    day_ids = [d for r in ROLES if r in roles for d in plan.ids(r)]
    mp, audit = _read_target_values(market, day_ids)

    ctx, y, stamps, segs = [], [], [], []
    for role in ROLES:
        if role not in roles:
            continue
        for d in plan.ids(role):
            pos = ob.positions[d]
            origin = pd.Timestamp(ob.timestamps[int(pos[0])])
            hts = [origin - pd.Timedelta(hours=k) for k in range(SEQ, 0, -1)]
            tts = [origin + pd.Timedelta(hours=k) for k in range(H)]
            if any(t not in mp for t in hts + tts):
                raise RuntimeError(f"{market}: projected read incomplete for {role} day {d}")
            ctx.append(np.asarray([mp[t] for t in hts], np.float32)[:, None])
            y.append(np.asarray([mp[t] for t in tts], np.float32)[:, None])
            stamps.append(str(origin))
            segs.append(role)
    w = ForecastWindows(np.stack(ctx), np.stack(y), np.asarray(stamps), np.asarray(segs),
                        plan.dataset_id, ob.market if ob.family == "CHINA5" else "GANSU")
    return w, audit


def load_pretest_windows(market: str) -> tuple[ForecastWindows, dict]:
    """TRAIN + VAL only.  The PRETEST-safe loader — TEST is not materialised."""
    return load_windows(market, roles=("TRAIN", "VAL"))


def role_view(market: str, role: str) -> ForecastWindows:
    w, _ = load_windows(market, roles=(role,))
    return w


# -------------------------------------------------------------- test seal
class TestSeal:
    """Central gate: every attempt to obtain a TEST target value goes through here.

    The counter is the object of record for the PRETEST claim "TEST-label read
    count = 0".  It is deliberately not a convention: code paths that need TEST
    values must call `reveal()`, and any call before the seal is lifted raises and is
    recorded as a refusal.
    """

    def __init__(self, lifted: bool = False):
        self.lifted = bool(lifted)
        self.granted = 0
        self.refused = 0
        self.refusals: list[str] = []

    def reveal(self, why: str = "") -> None:
        if not self.lifted:
            self.refused += 1
            self.refusals.append(why)
            raise PermissionError(
                f"TEST targets are sealed during PRETEST (attempt: {why or 'unspecified'})")
        self.granted += 1

    @property
    def test_label_read_count(self) -> int:
        return int(self.granted)

    def as_dict(self) -> dict:
        return {"lifted": self.lifted, "test_label_read_count": self.test_label_read_count,
                "refused_attempts": self.refused, "refusal_reasons": list(self.refusals),
                "meaning": ("PRETEST requires test_label_read_count == 0; the only legal "
                            "reader is TestSeal.reveal(), and it raises while the seal holds")}


PROCESS_SEAL = TestSeal(lifted=False)


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
