# 17. PAPER_GATE_VERDICT — 2026-08-14

Protocol: `hch_v2_paper_benchmark_gate_comparative_experiment_protocol_v0.1_2026-08-14` §19
Execution plan: `hch_v2_paper_gate_execution_plan_2026-08-14` WP-8
Branch: `exp/r1b-screening-20260813` — git `5182ef4`

---

## A. P0 status

| item | status |
|---|---|
| final replay | ✅ P0-A: `x_final = s·sinh(z0 + π_eff)`, `π_eff = π` iff DVG release else 0 |
| code SHA | `5182ef4` |
| host naming | ✅ Linear / MLP / LSTM / PatchTST (host = frozen backbone, PatchTST-style naming) |
| data audit | ✅ `02_DATASET_MANIFEST.csv` / `03_DOMESTIC_DATA_AUDIT.csv` / `04_HOST_MANIFEST.csv` |
| B5 head | `WP5_PUBLIC_20260814_053855` (p=0.5 temperature-sampled, **user-waived §12.2**) |

---

## B. Peer fidelity

- **PIR = PIR_ACCEPT_AS_IS**: ETTh1+PatchTST full-mechanism (incl. retrieval) repro 0.4272→0.3983 (−6.75%) vs paper −6.22%. Caveat: PatchTST training diverged, gain measured on weak backbone; production PIRLimited has no retrieval (`limited_official`).
- **δ-Adapter = DELTA_ACCEPT_AS_IS**: after removing 50 train -9999 sentinel rows (scale mismatch std 386 vs ~16), production PostY −14.1% vs paper −9.6%. Mechanism faithful.
- Baselines B0–B4 implemented per official references and enter the matrix as-is.

---

## C. Foreign scorecard (WP-6 — `B3_MATRIX_20260814_105940`)

64 primary cells (8 datasets × 4 hosts × {MAE, sMAPE}).

| gate metric | value | §15.1 |
|---|---|---|
| **Top-1 / tied** | **22 / 64 = 34.4%** (strict 15 + tied 7) | YELLOW band (30–60%) |
| Top-2 | 19 / 64 = 29.7% | GREEN needs ≥85% |
| host-better frac | 46.9% | GREEN needs ≥90% |
| MAE safety failure (>15%) | none (degradation_frac = 0 everywhere) | ✅ |

Per-dataset wins (B5 top-1/tied cells, out of 8):

| dataset | wins | note |
|---|---|---|
| LAGO_BE | 6 | strongest (LSTM/MLP/PatchTST both metrics strict) |
| NEM_SA1 | 3 + 2 tied | huge CRPS gain (ΔCRPS −0.53), point only partially follows |
| NORD_DK1 | 3 + 2 top2 | |
| LAGO_NP | 2 | PatchTST only |
| LAGO_PJM | 2 | |
| LAGO_FR | 1 strict + 1 tied | |
| GEFCOM14P | 1 strict + 2 tied | |
| **LAGO_DE** | **0** | B1_ResidualL1 dominates; worst gaps 12–33% |

---

## D. Domestic scorecard (WP-7 — `B4_MATRIX_SHANDONG_20260814`)

Shandong DA/RT × 4 hosts × {MAE, sMAPE} = 16 primary cells.

| gate metric | value | §15.2 target |
|---|---|---|
| **Top-1 / tied** | **0 / 16 = 0%** | ≥75% → **FAIL** |
| host-better frac | 50% | |
| worst gap | 53.3% (shandong_RT:Linear MAE) | |
| CRPS | ΔCRPS < 0 in 8/8 cells (−0.028…−0.176) | |

- Best method in every cell is a peer (B1_ResidualL1 / B4_PIR / B3_DeltaAdapter), gap 20–53%.
- HCH cells: 3 CANDIDATE (small −1.4…−1.6%), 3 ABSTAIN_SAFE (== host), 2 POINT_READOUT.
- **All 16 cells selected C3**; universal head effectively inactive or locally uncorrected on the hardest market (11.1% / 13.4% neg-price).

---

## E. Failure decomposition

| tag | foreign | domestic |
|---|---|---|
| CANDIDATE (ΔCRPS<0 + point improves) | 12 | 3 |
| **POINT_READOUT (ΔCRPS<0, point worse → Case A)** | **14** | 2 |
| NEUTRAL | 4 | 1 |
| ABSTAIN_SAFE (final == host) | 2 | 3 |

**Unifying pattern**: CRPS layer wins 40/40 cells (ΔCRPS<0 everywhere). The broken layer is the **point readout** (`z^point`): 16/40 cells regress, and where HCH is safe/abstain the peers still beat it. This is exactly protocol §16 **Case A** (distribution useful, point/action readout misaligned), plus a domestic hint of **Case E** (C3/local evidence not helping point on a hard market).

---

## F. Verdict

```text
PAPER_GATE_YELLOW_READOUT
```

Rationale:
- Foreign Top-1/tied 34.4% ∈ YELLOW band; domestic 0% (target 75%) adds weight downward — **not GREEN**.
- Candidate layer is healthy (40/40 CRPS wins); failure is isolated to one identified layer, the point readout — **YELLOW_READOUT**, not CANDIDATE / MIXTURE / LOCAL.
- Not RED: 34.4% ≥ 30%; candidate CRPS is not broadly weak.
- **30% is never called a majority.**

---

## G. Next action — max 3, NOT auto-executed

1. **Case A — μ_R point readout (zero added parameters).**
   `z^point = z0 + μ_R`, `μ_R = w⁺m⁺ − w⁻m⁻` (protocol §16 Case A). Dev-only shrinkage grid λ ∈ {0.25, 0.5, 0.75, 1.0}, select on validation only. Targets the 14+2 POINT_READOUT cells and the domestic point deficit.

2. **Case E — separate point-readout calibration from safe-action calibration (domestic).**
   Shandong all-C3 with heavy abstention ⇒ local authorization too conservative for point; apply identity-anchored / shrunk isotonic to the point readout independently, longer local evidence horizon. Do not retrain the universal candidate.

3. **WP-5 waiver locked in.**
   B5 head = p=0.5 temp-sampled, macro −1.30% / anchor −1.53% but 4 headline domains exceed 2% (worst 4.42%), user-waived 2026-08-14 (`guard_report.json` waiver_record). Any future retrain must pass §12.2 in full.

---

## HARD STOP (protocol §20)

Do not start: U0 · large-scale data expansion · new foundation-model experiments · new architecture branch · final sealed testing.
Return results for human/scientific review. Review chooses: freeze · one diagnosis-driven modification · or core review.
