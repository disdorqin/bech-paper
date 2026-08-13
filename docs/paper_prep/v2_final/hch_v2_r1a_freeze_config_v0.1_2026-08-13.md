# HCH-v2 R1A Frozen Configuration (T0 Acceptance Record)

**Date:** 2026-08-13
**Depends on:** first-round training protocol v0.1 (audited baseline `769e421`)
**Status:** T0 accepted — P0 gates closed, non-empty evidence reproduced.

---

## 1. Protocol §24 checklist — implementation status

| # | Item | Status | Where |
|---|------|--------|-------|
| 1 | H0/S1R split added to manifest | ✅ | `eval_manifest.py` `SPLIT_7`/`FRAC_7`, `assert_7seg_disjoint` |
| 2 | S2T/S2V split + split hash | ✅ | same; `split_hash` in bundle `local_hashes` |
| 3 | domain descriptor as forward context | ✅ G0-1 | `DataSignature.forward(core, domain_det)`; `IAHCandidateHead.forward(..., domain_det)` |
| 4 | zero-init identity FiLM | ✅ G0-4 | `h'=(1+Δγ)h+β`, `mod_head` zero-init |
| 5 | remove/repair order-sensitive descriptors | ✅ G0-5 | 8-dim order-free `[q05,q25,q50,q75,q95,iqr,mean_abs,neg_mass]` |
| 6 | UniversalCoreTrainer implemented | ✅ P0-4 | `src/universal_trainer.py` |
| 7 | macro-domain validation | ✅ P0-3/G0-3 | `MacroCRPS_S2V` selection + `L_worst`; host baseline `L^host_g` |
| 8 | optional U2 params disabled | ✅ | `d_value=0` default; no optional branch |
| 9 | source datasets/hosts in universal manifest | ✅ P1-3 | `bundle.source_datasets/source_hosts` + `training_provenance` |
| 10 | regenerate non-empty v0.4 smoke | ✅ G0-6 | execute_rate=0.048 (21/437), roundtrip_hash_match=True |
| 11 | freeze first-round config | ✅ | this document |

All G0-1..G0-6 gates verified. 18/18 unit tests green
(`experiments/08-hch-v2/tests/test_p1.py`).

---

## 2. T0 acceptance evidence (LAGO_DE × Linear, seed=0)

| Metric | Value |
|--------|-------|
| S2 checkpoint | 0.1788 (S2V-selected, 88 validation days) |
| S3-M memory / k-val | 81 / 28 days, selected k=5 |
| S3-C calibration | n=109, q=0.0655 |
| S4 days | 437 |
| execute_rate | 0.048 (21/437) |
| roundtrip_hash_match | True |

Evidence: `experiments/08-hch-v2/results/v0.3/smoke_v4_lago_de_linear.json`

---

## 3. R1A frozen hyperparameters (protocol §10-§12)

| Parameter | Value |
|-----------|-------|
| model | d_model=64, d_sig=32, optional branch disabled (protocol §9) |
| T0 smoke scale | d_model=32 (fast validation; R1A uses 64) |
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight decay | 1e-4 |
| gradient clipping | 1.0 |
| seed | one HCH seed (0) initial; host cache fixed at host seed 0 |
| minibatch | 16–32 days/step, domain-homogeneous |
| domain sampling | uniform over G (`L_universal = (1/|G|) Σ_g E[d~g]`) |
| validation | macro S2V CRPS every epoch + `L_worst` + `L^host_g` + `Δ_g` |
| §13 health | mass (mean w⁻/w⁰/w⁺ + entropy), shift (frac m>tiny, med/p95), signature (\|Δγ\|,\|β\|), gradient (grad norm, NaN/Inf batches, scale-unidentified days) per validation checkpoint |
| early stopping | patience 3 on macro S2V CRPS |
| checkpoint restore | best S2V state |
| S3M/k/DVG | S3M→k selection→S3C DVG (protocol §7-§8), k from forward validation |
| S4 | batch through `predict_s4`, no hyperparameter search vs S4 |

---

## 4. Scope freeze (do not change without version bump)

- splits: H0/S1R/S2T/S2V/S3M/S3C/S4 (7-way)
- objective: IAH-CRPS (Eq 10) only
- domain sampling: uniform, domain-homogeneous minibatch
- freeze scope: universal core + local profile separation (P1-3)
- evaluation: macro S2V CRPS / L_worst / Δ_g; S4 held out

Per master plan §8: any change affecting splits, objective semantics, domain
sampling, freeze scope or final evaluation must increment the document version
and be recorded in the run manifest.

---

## 5. Reproduce

```bash
python experiments/08-hch-v2/smoke_v4.py
python experiments/08-hch-v2/tests/test_p1.py
```
