# Compatibility adaptations

- This is a read-only V2 registry reconstruction; no baseline was retrained or re-inferred.
- COSA is represented as `ONLINE_TTA` in the registry while preserving the parent strict online chronology provenance.
- PIR is numeric only for PatchTST/TimeMixer; iTransformer/LSTM remain inherited `Q-INCOMPATIBLE`.
- UEC-STD retains the inherited `Q-FIDELITY` blocker; OMPB retains the inherited `Q-DATA` blocker.
- TEST metrics are preserved from the final metric-only V2 export where no TEST raw array is exposed; no substitute metric was manufactured.
