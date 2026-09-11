# HCH Host-Relative State Interaction

Status: **AUTHORIZED FINAL STRUCTURAL DEVELOPMENT CLOSURE**

Controlling protocol:

`../../../docs/current/HCH_HOST_RELATIVE_STATE_INTERACTION_FINAL_CLOSURE_20260911.md`

Launcher:

`CODEX_GOAL_PROMPT_20260911.md`

Fixed panel:

`GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`, seeds `7/17/37`.

Scientific question:

> Can forecast-time semantic state improve frozen-Host residual Shape when its effect is interpreted relative to the Host's causal recent signed residual-bias direction, rather than mapped directly to an absolute residual direction?

Only incremental learner: one **global** `beta in R^5` shared across the entire six-cell panel, zero-initialized in

`u_HRSI = normalize(u_S1 + Z beta)`, `Z[h,k] = b_hat[h] * R[h,k]`.

For every OOF fold, beta is fit once on earlier rows pooled across all six cells with equal total weight per cell, then the same beta predicts the current fold in every market/Host.

No S1 retraining, new history window, excursion/MAD, conditional Amplitude, Verification/Gate, deeper network, role selection, market-specific hyperparameters, extra markets or protected/final access.

This stage is terminal:

- PASS -> method target reached, freeze HRSI and stop for human adjudication.
- FAIL -> freeze S1 + pooled fixed alpha and stop structural search.
