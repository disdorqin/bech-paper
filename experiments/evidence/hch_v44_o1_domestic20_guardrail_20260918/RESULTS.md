# HCH v4.4 O1 Domestic-20 + International Guardrail — Results

STATUS:        ACTIVE
STAGE:         hch_v44_o1_domestic20_guardrail_20260918
KIND:          results
SUPERSEDED_BY: -

Controlling protocol: `experiments/current/hch_v44_o1_domestic20_guardrail_20260918/PROTOCOL.md`.
Design: `docs/current/HCH_V44_O1_DOMESTIC20_INTERNATIONAL_GUARDRAIL_DESIGN_20260918.md`.
Recipe: `O1_AUX_WARMUP_400_THEN_MAE`, switch step 400 (non-tunable).

## Terminal token

`HCH_V44_O1_DOMESTIC20_NOT_READY`

## New fits

- Phase D: 36 registered new fits (12 cells x seeds 7/17/37),
  36 executed here, 0 already present; 24 reused runs never retrained.
- Phase I: 0 fits (not reached).
- Phase M: 0 fits, 0 optimizer steps.

## Access boundary

- V2 TEST target read count: 0.
- Protected/final international target read count: 0.
- Forbidden paths opened: 0.
- Source tree `src/core` digest: `6F18C0E4241A299E32C8D8A617061BAEAA7B8DC98B4EE069DC67199B87531577`
  (equals the frozen v4.4 substrate pin; `src/core/**` and `paper/**` unedited).

## Domestic 20-cell summary

Primary statistic: median VAL MAE across seeds 7/17/37 per cell, then relative
gain against the deterministic frozen Host MAE of the same cell.

| cell | O1 seed-median VAL MAE | Host MAE | gain vs Host | seeds beating Host |
|---|---|---|---|---|
| GANSU_DA/PatchTST | 75.7672 | 77.1040 | +1.7337% | 3/3 |
| GANSU_DA/TimeMixer | 76.2325 | 76.6070 | +0.4889% | 3/3 |
| GANSU_DA/iTransformer | 79.0445 | 79.1737 | +0.1632% | 2/3 |
| GANSU_DA/LSTM | 79.4816 | 80.6607 | +1.4618% | 3/3 |
| SHANDONG_DA/PatchTST | 73.7958 | 76.8667 | +3.9951% | 3/3 |
| SHANDONG_DA/TimeMixer | 78.4921 | 78.9922 | +0.6332% | 3/3 |
| SHANDONG_DA/iTransformer | 74.8497 | 75.0703 | +0.2939% | 3/3 |
| SHANDONG_DA/LSTM | 78.4936 | 84.7114 | +7.3400% | 3/3 |
| SHAANXI_DA/PatchTST | 106.8355 | 111.9654 | +4.5817% | 2/3 |
| SHAANXI_DA/TimeMixer | 103.4710 | 103.5922 | +0.1169% | 3/3 |
| SHAANXI_DA/iTransformer | 112.4108 | 112.5065 | +0.0851% | 3/3 |
| SHAANXI_DA/LSTM | 109.3026 | 109.4564 | +0.1405% | 3/3 |
| NINGXIA_DA/PatchTST | 62.0734 | 65.0741 | +4.6113% | 3/3 |
| NINGXIA_DA/TimeMixer | 72.5092 | 78.3469 | +7.4511% | 3/3 |
| NINGXIA_DA/iTransformer | 69.2431 | 71.3142 | +2.9042% | 3/3 |
| NINGXIA_DA/LSTM | 67.5466 | 70.2218 | +3.8096% | 3/3 |
| QINGHAI_DA/PatchTST | 128.4313 | 143.2564 | +10.3487% | 3/3 |
| QINGHAI_DA/TimeMixer | 151.1069 | 157.5714 | +4.1026% | 3/3 |
| QINGHAI_DA/iTransformer | 150.7839 | 161.4467 | +6.6045% | 3/3 |
| QINGHAI_DA/LSTM | 162.2515 | 171.2948 | +5.2794% | 3/3 |

## Domestic gate

- panel median gain: +3.3569% (requires >= +1.5%)
- strict Host-positive cells: 20/20 (requires >= 16)
- cells with gain >= +2.0%: 11/20 (requires >= 8)
- worst cell: +0.0851%; best cell: +10.3487%
- market medians: {"GANSU_DA": 0.9753623522835875, "SHANDONG_DA": 2.314122995707326, "SHAANXI_DA": 0.12868746961947752, "NINGXIA_DA": 4.210419080766486, "QINGHAI_DA": 5.941950839857343}
- blocks: {"D0": true, "D1": true, "D2": true, "D3": false, "D4": true}
- domestic PASS: False

Thresholds: D1 >=16/20 Host-positive, >=19/20 within -0.5%, worst >= -1.0%;
D2 panel median >= +1.5%, >=8/20 at >= +2.0%; D3 5/5 market medians > 0 and
>=4/5 >= +1.0%; D4 >=15/20 cells with >=2/3 seeds beating Host and 0 cells with
all three seeds worse by >1%.

## Phase M — zero-fit shared-gradient audit

- checkpoints audited: 60 (all available domestic seeds; superset of the 20-cell reading)
- optimizer steps: 0; optimizer ever constructed: False
- all checkpoints byte-unchanged: True
- negative-cosine fractions (overall): {"cos_rec_b": 0.2631578947368421, "cos_rec_B": 0.47368421052631576, "cos_rec_S": 0.3, "cos_rec_auxsum": 0.36666666666666664}
- cosines defined / total per pair (a cosine is undefined exactly when one of its
  two gradients is the zero vector; undefined cases are excluded from the
  fractions, so these are their denominators): cos_rec_b 57/60 defined; cos_rec_B 57/60 defined; cos_rec_S 60/60 defined; cos_rec_auxsum 60/60 defined
- runs with an undefined cosine: GANSU_DA__iTransformer__seed7, NINGXIA_DA__PatchTST__seed37, SHAANXI_DA__PatchTST__seed7
- by Host family: {"PatchTST": {"n_runs": 15, "cos_rec_b": 0.15384615384615385, "cos_rec_B": 0.38461538461538464, "cos_rec_S": 0.13333333333333333, "cos_rec_auxsum": 0.2, "norm_ratio_b_over_rec": 0.17767106573936756, "norm_ratio_B_over_rec": 10.82027339724129, "norm_ratio_S_over_rec": 3.293443802948863}, "TimeMixer": {"n_runs": 15, "cos_rec_b": 0.13333333333333333, "cos_rec_B": 0.5333333333333333, "cos_rec_S": 0.4, "cos_rec_auxsum": 0.3333333333333333, "norm_ratio_b_over_rec": 0.03541892628255168, "norm_ratio_B_over_rec": 3.0942535120452193, "norm_ratio_S_over_rec": 4.664823750451817}, "iTransformer": {"n_runs": 15, "cos_rec_b": 0.35714285714285715, "cos_rec_B": 0.5714285714285714, "cos_rec_S": 0.3333333333333333, "cos_rec_auxsum": 0.5333333333333333, "norm_ratio_b_over_rec": 0.11221125796155575, "norm_ratio_B_over_rec": 2.0696836284841393, "norm_ratio_S_over_rec": 4.777180049715965}, "LSTM": {"n_runs": 15, "cos_rec_b": 0.4, "cos_rec_B": 0.4, "cos_rec_S": 0.3333333333333333, "cos_rec_auxsum": 0.4, "norm_ratio_b_over_rec": 0.023690091916620828, "norm_ratio_B_over_rec": 3.8124205761603354, "norm_ratio_S_over_rec": 3.4969117335869573}}
- strongest Host-family deviation on cos(rec, aux_sum): {"pair": "cos_rec_auxsum", "host": "iTransformer", "fraction": 0.5333333333333333, "overall_fraction": 0.36666666666666664, "absolute_deviation": 0.16666666666666669}
- diagonal subset rule: np.unique(np.linspace(0, n-1, min(n, 256))) over registered TRAIN row order

This phase is descriptive only; no gate, recipe, checkpoint or decision reads it.

## Phase I — international guardrail

Not reached, and correctly so: the domestic gate failed on **D3**, and PROTOCOL.md sections 3.4 and 5 forbid Phase I after a domestic FAIL. No international fit, comparison table or international gate exists, and none may be created under this token.

**Out-of-order artifact, disclosed.** A zero-fit preflight for the four registered foreign cells is present at `INTERNATIONAL_PREFLIGHT.json`, written 2026-09-18T12:05:21.502912 — inside the Phase-D fit sweep (2026-09-18T12:02:58.000570 to 2026-09-18T12:17:49.263080) and before `DOMESTIC_GATE.json` existed (2026-09-18T12:17:51.385627). It therefore cannot have been conditioned on the domestic verdict, and it executed **0 fits**, authorized **0**, wrote no international run directory, no comparison table and no international gate, and read no TEST or protected target. It selected nothing; the blocker it records is a property of the information setting, re-verified here. Full record: `INTERNATIONAL_PREFLIGHT_DISCLOSURE.json`.

Comparison authority (never substituted, never retrained):
`experiments/evidence/hch_frozen_method_baseline_transfer_20260911/`.

## Independent verification

- verifier: `verification/verify.py`, imports nothing from `implementation/`
- checks: 73, failed: 0
- verdict: `VERIFIED`

## Scope

No component-loss ablation, router deletion, architecture simplification,
optimizer change, loss-weight search, switch-step search, Host/baseline
retraining, foreign-specific tuning or TEST evaluation was performed, and none is
authorized by this token. Canonical state files were not edited; this artifact is
returned for main-window adjudication.
