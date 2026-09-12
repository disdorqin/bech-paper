# Archive — `signed_mass_development_core_20260912`

Date: 2026-09-12
Status: **byte-preserving archive; never rewritten after creation**

## Why this archive exists

The user-authorized refactor

`docs/current/HCH_ALIGNMENT_SAFE_CORE_IMPLEMENTATION_PROMPT_20260912.md` (section 3)

requires the completed signed-mass development core to be archived **before** any file
under `src/core/**` is mutated. The same requirement is stated in
`docs/current/HCH_ALIGNMENT_SAFE_CORE_REFACTOR_PLAN_20260912.md` (section 2).

`src/core/` is **not tracked by git** in this repository (`git ls-files src/core` is
empty). There is therefore no version-control safety net for this tree: the hashes
below are the only fidelity evidence, and the archive is the only recovery path.

Archive placement is lifecycle metadata. It does **not** change the scientific status
of the completed experiment, and it does **not** reopen any closed component.

## What is archived

- source root: `src/core`
- archive root: `src/archive/signed_mass_development_core_20260912/legacy_core/`
- files: **20** (all `*.py`, `*.md` and `*.json` under `src/core`; `__pycache__/` excluded as compiled bytecode, not source)
- closed experiment this source served:
  `HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION`

## Tree digest

```
0c53faab6457ce489cf697543ef498a93e15c99e1098b9d98ffeb2b1e828bb99
```

Algorithm (also recorded in `ARCHIVE_HASHES.json`): sha256 over the concatenation of,
for each file in case-sensitive ASCII order of its relative POSIX path,

```
utf8(relpath) || 0x00 || ascii(sha256(file_bytes)) || 0x0a
```

## Per-file manifest

Machine-readable form with the same fields plus byte counts: `ARCHIVE_HASHES.json`.

| old path | sha256 | role | historical dependency |
|---|---|---|---|
| `src/core/__init__.py` | `5ae77a2396dee851c2f243443e8c824d1c6abb22295d34f81199c31649855059` | package export surface (40 names) | every downstream signed-mass fit imported through it |
| `src/core/amplitude.py` | `b5531b7c5286cc272bb5560a9c6c6925353627f760f25d1d38d0efa8e7db5031` | `AmplitudeBranch` trunk + tied/untied non-negative heads | **removed component** `untied_amplitude_heads` (9/20); tied semantics survive as one scalar |
| `src/core/calibration.py` | `6996f7302a165821288ff70a4c15b3dd71c0f85591547d769f7d9c76ea541b2b` | `weighted_median`, `fit_mae_scalar`, `apply_mae_scalar`, `mae_objective` | produced the per-cell OOF `alpha_0` of the closed fit |
| `src/core/contracts.py` | `539e05927bc0e56bb72f9b1088e4ae53eed1571758309c184bdc12db023cb230` | dataclass contracts incl. `ModelConfig` | defined the four-switch search surface consumed by every closed fit |
| `src/core/CORE_READINESS_AUDIT_20260912.md` | `3d4ec78c59ec9f3b09abcac5497124b839ab7a4177da171e95aed893cc61e52d` | predecessor core-readiness audit | lifecycle metadata of the archived core |
| `src/core/core_reorganization_inventory.json` | `51350bc567e840928bf87ea20cf9952c783246392a2deb842d527d7d5eacd5b4` | predecessor inventory, machine-readable | lifecycle metadata of the archived core |
| `src/core/CORE_REORGANIZATION_INVENTORY.md` | `31e3c08916910db66a91da75a63059af9cdf2a1284518b71fef458bf3714cafb` | predecessor reorganization inventory | lifecycle metadata of the archived core |
| `src/core/CORE_REORGANIZATION_PLAN_20260912.md` | `97273414d8f53c02461a3e1631140d990264f304f86ceec6fea39b9ae7e5fe13` | predecessor reorganization plan | lifecycle metadata of the archived core |
| `src/core/DESIGN_CONTRACT.md` | `2968c7fa19d3a550e6c4e3a6fdf679e929046b3d55401f7e556876d1168235b8` | signed-mass invariants I1–I12 + four removal switches | rewritten to the alignment-safe contract |
| `src/core/encoders.py` | `25df8e9411a349c39683e96a75fc0073722946fb06d08c1c787d21a6880916ff` | `LocalTCN` + `WindowRNN` + `BranchEncoder` | **removed components** `use_tcn` (2/20); LSTM/bidirectional were search switches never selected |
| `src/core/fusion.py` | `71e7c451d7a316b96b2816ef0d12ba29f55bcf3ce3e1fd185f42c2730a037810` | `safe_shape`, `decompose_residual`, `fuse`, `repair` | signed-mass identity; survive as pure helpers |
| `src/core/losses.py` | `f97a2cbfc719145d1a0f65bdb98222bd41051560348db923ed3a4a0d52457f54` | repair MAE, Shape W1/L1, mass-normalized Amplitude L1, combined loss | the tied-head Amplitude loss survives rewritten as `L_A^bal` |
| `src/core/model.py` | `cb05a014ffc961b070ff6ef8866ce243a3e78334becaf1566d54ede93ea2f285` | `SignedMassRepairModel` + `default_config` | instantiated all four switches; executed the 60 fits / 0 failed of the closed experiment |
| `src/core/preprocessing.py` | `dc2d2e5ff55f9c46de29f319b68cf63dc4546b4acdf71487bef2d5d76358a1c4` | deterministic coordinates, `TrainFrozenScaler`, twin history views | target-day Shape semantic-view construction was a registered ablation component |
| `src/core/README.md` | `ee78963d481d27a0f5f5b6722bb21b81a92d74357ecd684b71f3bd16a64cc0d4` | signed-mass core description | rewritten to describe the minimal generator + safety layer |
| `src/core/sampling.py` | `da8bc0fe62f0f0ce32d8a4ae837475ff655994c1df95cc8263335931347797bb` | `RareMassSampler`, `RareMassSamplerConfig` | **removed component** `rare_mass_sampling` (9/20) |
| `src/core/semantic_views.py` | `7d8a1fcee7dc4436da720f87f8dce40cb182baee12bebd59fe70a5c73d0b2393` | `SemanticRouter` task routing | **removed component** `shape_semantic_context` (1/20) |
| `src/core/shape.py` | `b6b7b794a8cca9191e3f2c6223857d08bcc5c2e690b5a836dddc2cca9befc03e` | `ShapeBranch` horizon softmax `S+`,`S-` | surviving object; context argument removed |
| `src/core/similarity.py` | `616f5abc7b39ce073ae0741806c3d259bba0f1b1e5ef47b37a88f53e7b6bc5cf` | `KNNShapeContext`, `ShapeMemoryBank`, `KNNShapeContextProvider` | disabled retrieval interface carried in pre-adjudication source; never active in the closed fit |
| `src/core/stem.py` | `667e0d1d10c2153f29b0a9d2625f6a52571949c5e230cfa86305593c742946ee` | `UnifiedFeatureStem` (one shared MLP) | unchanged in the refactor apart from default widths |

## Verification performed

1. every pre-refactor file under `src/core` was hashed (sha256, raw bytes);
2. the tree was copied with `shutil.copy2` (contents plus file metadata), preserving relative paths;
3. every archived file was re-hashed and compared to its pre-refactor digest — **20/20 exact matches, 0 mismatches**;
4. the archive was checked for extra files not present in the pre-refactor tree — **none**.

The archive is **never rewritten**. Any later change belongs in the active `src/core`
tree, not here.

## Provenance rules attached to this archive

- `src/AGENTS.md` forbids rewriting `src/archive/**` source merely to modernize paths.
  The archived modules therefore keep their original `from .contracts import ...`
  relative imports and their original symbols verbatim. They are a historical
  record, not an importable runtime.
- The compat layer for the *earlier* V2.5 archive (different archive,
  `core_pre_extreme_repair_20260912/`) is `src/legacy_core_compat.py`. It does not
  touch this archive.
- No historical evidence artifact, frozen baseline, or signed-mass metric was
  modified in the course of creating this archive. Creating the archive was a
  read-only operation on the rest of the repository.
- The four removed components recorded above were already deleted by adjudication at
  `HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION`. Archiving their
  source does **not** rescue them; it preserves the exact code those registered
  ablations were executed against.
