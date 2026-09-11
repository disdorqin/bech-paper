# Implementation and test audit

- Canonical root: `D:\作业\science\solar_leak_price_model`
- Controlling protocol: `docs/current/HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md`
- Frozen S1 metric replay maximum absolute difference: `1.421e-14` (tolerance `1e-10`)
- beta=0 exact S1 direction replay maximum absolute difference: `1.192e-07` (tolerance `1e-06`)
- Frozen artifact immutability: `True`
- S1 code tree immutability: `True`
- Amplitude code tree immutability: `True`
- Host cache immutability: `True`
- Stacked chronological OOF self-exclusion: `True`
- Primary-only role channels, forecast-time legal: `True`
- Exactly five trainable HSA parameters per market: `True`
- Forbidden/protected reads: `0`
- Targeted unit suite exit code: `0`
- Targeted unit suite result: `60 checks passed`
- Targeted unit suite output: `tests.txt`
- Independently re-verified by `verify_gates.py` (does not import this runner): recomputed H0-H4 and the token from the written evidence alone
