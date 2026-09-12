# Independent result audit

- schema: `signed_mass_result_audit.v1`
- evidence root: `D:\作业\science\solar_leak_price_model\experiments\evidence\hch_signed_mass_method_20260912`
- checks: 752 (0 failed)
- result: **PASS**
- derived verdict: `HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION`
- recorded verdict: `HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION`

## Coverage

- n_ok: `300`
- n_failed: `0`
- cells: `['GANSU_DA::LSTM', 'GANSU_DA::PatchTST', 'GANSU_DA::TimeMixer', 'GANSU_DA::iTransformer', 'NINGXIA_DA::LSTM', 'NINGXIA_DA::PatchTST', 'NINGXIA_DA::TimeMixer', 'NINGXIA_DA::iTransformer', 'QINGHAI_DA::LSTM', 'QINGHAI_DA::PatchTST', 'QINGHAI_DA::TimeMixer', 'QINGHAI_DA::iTransformer', 'SHAANXI_DA::LSTM', 'SHAANXI_DA::PatchTST', 'SHAANXI_DA::TimeMixer', 'SHAANXI_DA::iTransformer', 'SHANDONG_DA::LSTM', 'SHANDONG_DA::PatchTST', 'SHANDONG_DA::TimeMixer', 'SHANDONG_DA::iTransformer']`
- configs: `['FULL', 'NO_RARE_MASS', 'NO_SHAPE_CONTEXT', 'NO_TCN', 'TIED_AMPLITUDE']`
- seeds: `[7, 17, 37]`

## Checks

### algebra

1/1 pass


### artifacts

721/721 pass


### comparison

8/8 pass


### coverage

6/6 pass


### decisions

3/3 pass


### frozen

3/3 pass


### gates

4/4 pass


### host

1/1 pass


### metrics

1/1 pass


### protocol

1/1 pass


### sealed

3/3 pass


## Worst recomputation differences

| field | max abs difference | at |
| --- | --- | --- |
| alpha | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| alpha_consistency_max_abs_err | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| amplitude_scale | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| branch_reconstruction_max_abs_err | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| final_epochs | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| fit_seconds | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| high_mass_negative_l1 | 0.000e+00 | GANSU_DA::PatchTST::FULL::s7 |
| high_mass_negative_n_days | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| high_mass_positive_l1 | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| high_mass_positive_n_days | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| host_mae | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| inference_seconds | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| lower_tail_mae | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| mae_gain_abs | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| mae_gain_pct | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| mass_negative_l1 | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| mass_positive_l1 | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| n_days | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| n_days_any_valid | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| n_entries_evaluated | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| n_folds | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| n_parameters | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| negative_price_mae | 0.000e+00 | SHANDONG_DA::LSTM::FULL::s7 |
| negative_price_mae_host | 0.000e+00 | SHANDONG_DA::LSTM::FULL::s7 |
| normal_mae | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| normal_mae_host | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| normal_relative_harm_pct | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| overall_mae | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| shape_w1_negative | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| shape_w1_positive | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| tail_mae | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| tail_mae_host | 0.000e+00 | GANSU_DA::LSTM::FULL::s7 |
| upper_tail_mae | 0.000e+00 | QINGHAI_DA::LSTM::FULL::s7 |

## Instrument changes made after the first audit run

Disclosed because they were made *after* seeing this audit fail, which is the circumstance in which a change to a measuring instrument is most easily mistaken for a result.  None of them alters a metric, a component decision or a gate; the first two decide only what this report can see, and the third was a tolerance error in the report itself.

1. **The Shape-simplex tolerance was wrong, not the method.** The check required row sums within `1e-9`, a float64 figure, of rows produced by a float32 `softmax`. It therefore failed all 360 artifacts -- including the ones whose Shapes are exactly the simplex the model emitted -- and the observed residuals cluster on float32 epsilon (min 6.0e-8, median 1.6e-7, max 3.8e-7; 0 of 360 above `1e-6`). The tolerance is now derived as `H * eps32` from the stored horizon, and the check still refuses an unnormalised Shape, which is wrong by O(1) rather than by 1e-7 (see `test_the_simplex_check_accepts_a_softmax_and_rejects_a_broken_one`). No gate reads this flag.
2. **The frozen method's own fits were not being audited.** For a derived vector the frozen method is a sixth configuration with its own status table, and the reported comparison is built from *that* table. The audit previously recomputed the 300 screen fits only, so it adjudicated a configuration whose predictions it had never opened. It now audits both tables (300 + 60 = 360 artifacts), which strengthens the audit rather than relaxing it.
3. **The frozen label was reconstructed instead of read.** The verifier derived the configuration name from the decision table alone, which yields the placeholder `FROZEN` for a derived vector; the comparison then found no rows and returned an empty table -- an audit reporting zero cells compared against zero cells, which reads as 'nothing to report' rather than as the label mismatch it was. The label is now read from the frozen record and *checked* against the derived vector.

