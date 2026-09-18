# hch_final_geometry_coupled_canary_20260918

STATUS:        ACTIVE
STAGE:         hch_final_geometry_coupled_canary_20260918
KIND:          index
SUPERSEDED_BY: -

Current domestic final-method stage.

Authority:
- docs/current/HCH_FINAL_GEOMETRY_COUPLED_METHOD_FREEZE_20260918.md
- docs/current/HCH_FINAL_GEOMETRY_COUPLED_IMPLEMENTATION_EXPERIMENT_PLAN_20260918.md

Direct children:
- README.md — stage index/status.
- PROTOCOL.md — executable scientific protocol.
- AI_EXECUTION_PROMPT.md — code + experiment launcher.
- implementation/ — the final method's source and phase driver (see implementation/README.md).
- verification/ — the correctness gate and the independent verifier (see verification/README.md).

Child indices:
- implementation/README.md — L1 table of the eight active-path modules and their boundary.
- verification/README.md — L1 table of the two check programs, their import boundary and order of use.

Execution:
source implementation -> correctness gate -> 8-cell GEOM_FLAT vs GEOM_COUPLED canary -> conditional DIRECT/NOHIST paper ablations -> conditional full-20 TRAIN+VAL completion.

V2 TEST remains quarantined. Foreign markets are out of scope.
