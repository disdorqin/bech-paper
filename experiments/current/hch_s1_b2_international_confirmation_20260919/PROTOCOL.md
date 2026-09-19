# HCH S1-B2 International Confirmation Protocol

STATUS: ACTIVE
STAGE: hch_s1_b2_international_confirmation_20260919
KIND: protocol
SUPERSEDED_BY: -

Authority:
docs/current/HCH_S1_B2_INTERNATIONAL_BENCHMARK_FREEZE_20260919.md

Execute exactly the frozen method and matrix.

Primary panel:
LAGO_NP / GEFCOM14P / NORD_DK1 / NEM_SA1
x PatchTST / TimeMixer / iTransformer / LSTM.

HCH seeds 7/17/37.

Primary role:
PROTECTED_FINAL only after P0 proves it has never been read.

Static comparisons:
Host, delta-Adapter, PIR where legal.
MDR internal control only.
COSA separate ONLINE_TTA.
UEC/OMPB and PIR-new-host blockers remain blockers.

No method or baseline tuning after protected-final access.
No proxy rows.

Writes only under:
experiments/current/hch_s1_b2_international_confirmation_20260919/**
experiments/evidence/hch_s1_b2_international_confirmation_20260919/**

Independent verifier must recompute hashes, chronology, cell medians, rankings, gates and protected
access counts without importing the executor.

## Return format

Follow the authority document exactly; hard cap 100 lines.
