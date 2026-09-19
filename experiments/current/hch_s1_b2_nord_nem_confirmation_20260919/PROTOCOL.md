# PROTOCOL — bounded NORD_DK1 / NEM_SA1 untouched confirmation

AUTHORITY: `docs/current/HCH_S1_B2_NORD_NEM_CONFIRMATION_20260919.md`
LAUNCHER:  `experiments/current/hch_s1_b2_nord_nem_confirmation_20260919/AI_EXECUTION_PROMPT.md`
PREDECESSOR (closed, blocked): `experiments/evidence/hch_s1_b2_international_confirmation_20260919/P0_BLOCKED.json`

## 0. Scope

This stage exists because the four-market stage stopped correctly at P0: the frozen
Host identity of the GEFCOM_NP family is unrecoverable, so no protected-final row of
that matrix could be scored. It is a **bounded successor**, not a retry.

Fixed panel: `NORD_DK1` / `NEM_SA1` × `PatchTST` / `TimeMixer` / `iTransformer` /
`LSTM`, HCH seeds `7` / `17` / `37`, scoring role `PROTECTED_FINAL`. 8 cells,
24 HCH candidates.

`LAGO_NP` and `GEFCOM14P` are excluded and are structurally unreachable from this
stage: `nn_shared.MARKETS` is the two-market tuple, so no code path here can read,
re-derive or proxy them. The blocker is inherited unchanged.

## 1. Method

Frozen S1-B2 exactly as adjudicated in
`docs/current/HCH_S1_B2_PAPER_READY_ADJUDICATION_20260919.md`, with the freeze's flag
vector verbatim:

```
{"scalar": "normalized", "scale": "mean", "ray_norm": "l2",
 "objective": "mse", "checkpoint": "inner", "c": False, "b3": False}
```

No architecture, feature, loss, scale, alpha, history-window, optimizer, B2-rule or
seed change is permitted in this stage. A failure is reported, never redesigned.

## 2. P0 — the seal stays closed until it passes

P0 is zero-fit and reads open roles only. It certifies:

1. the frozen torch build (`2.13.0+cpu`) — the gate that fails closed *before* a Host
   is touched, because re-deriving under a different build is silently wrong;
2. source and split identity for both markets;
3. every frozen module the method or a comparator executes, by digest, including the
   reused contract layer (`intl_shared.py`, digest-pinned) and the extent of the
   five-global rebinding this stage applies to it;
4. all 8 frozen Host checkpoints re-derive **bit-exactly** (digest equality against
   `FREEZE_MANIFEST.json` *and* bit-exactness of the open-role prediction against the
   frozen `.npz`);
5. the stage-local reader reproduces the frozen open-role windows bit-for-bit;
6. `POST_TRAIN` support: the frozen inner split admits the fit block and the 168h
   original-origin history resolves;
7. the legal-state manifest, with realised-demand substitution pinned to zero;
8. `PROTECTED_FINAL` target values read = **0**.

P0 PASS is written to `EVID/P0_PASS.json` and is the only thing that unlocks the
protected reader. A failure writes no token and the seal stays closed.

## 3. Confirmation execution

Fit stays the frozen `POST_TRAIN`; only the scoring partition moves to
`PROTECTED_FINAL`. During the protected span: parameters frozen, no gradient, no
update, no scalar refit, no checkpoint selection, no COSA-style adaptation by HCH. A
protected day's 168h residual history may draw on strictly earlier revealed days
(including earlier protected days), which is observation, not fitting.

All 8 cells run in one pass regardless of early outcomes. No conditional stopping.

## 4. Comparison set

| method | setting | cells | in `best_static` |
|---|---|---|---|
| Host | `OFFLINE_HOST_REFERENCE` | 8 | yes |
| delta-Adapter | `OFFLINE_STATIC_POSTHOC` | 8 | yes |
| PIR | `OFFLINE_STATIC_POSTHOC` | 4 (`PatchTST`, `TimeMixer` only) | yes |
| MatchedDirectResidual | `INTERNAL_CONTROL_ONLY` | 8 | **no** |
| COSA | `ONLINE_TTA` | 8 | **no** |

`UEC-STD` and `OMPB` stay inherited blockers: no proxy row, no re-probe. `PIR` is not
extended to `iTransformer` / `LSTM`, whose frozen backbone configs do not exist.
`best_static` = minimum MAE among the legally available comparators of that cell.

Cell statistic: median MAE across seeds `7/17/37`. The Host-relative cell gain is the
**median of the per-seed gains**, and the best_static-relative gain is
`(best_static_MAE − cell median MAE) / best_static_MAE`. Both are the frozen domestic
paper-ready panel's own definitions (`hch_s1_b2_paper_ready_closure_20260919`), re-used
rather than re-chosen. The gain-of-the-median is carried as a disclosed diagnostic only.

## 5. Gate (authority section 7)

`CONFIRMED_FOR_PAPER` requires all six:

1. panel median HCH gain vs Host > 0;
2. ≥ 6/8 cells Host-nonworse within −1.0%;
3. panel median gain vs best_static ≥ −1.0%;
4. ≥ 6/8 cells within −2.0% of best_static;
5. neither market median vs Host below −3.0%;
6. no access / provenance / chronology failure.

`STRONG_CONFIRMED` additionally requires ≥ 6/8 beating Host, ≥ 5/8 beating
best_static, and panel median gain vs best_static > 0.

Otherwise `MIXED`. Access, provenance or chronology failure ⇒ `INVALID`, which is not
a weak result but an unusable one.

## 6. Continuity (not rerun)

The four `LAGO_DE` / `LAGO_PJM` `PatchTST`/`TimeMixer` cells are development
continuity, already verified in the predecessor stage. They are cited, hashed and
excluded from the primary aggregate. They are not untouched confirmation.

## 7. Prohibitions

No method change; no LAGO/GEFCOM blocker relaxation; no proxy Host; no tolerance
relaxation on the digest or bit-exactness checks; no result-conditioned rescue; no
added dataset; no post-hoc threshold movement; no edit to `src/`, `paper/` or any
frozen artifact of a closed stage.

Writes are confined to
`experiments/current/hch_s1_b2_nord_nem_confirmation_20260919/**` and
`experiments/evidence/hch_s1_b2_nord_nem_confirmation_20260919/**`.

## 8. Return

Independent verifier, then the authority's 11-item return format, ≤ 80 lines, with a
terminal token from
`HCH_S1_B2_NORD_NEM_{STRONG_CONFIRMED,CONFIRMED_FOR_PAPER,MIXED,INVALID}`.
