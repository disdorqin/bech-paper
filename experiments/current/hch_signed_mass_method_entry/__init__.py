"""Signed-mass China-5 method-experiment harness.

Experiment-side integration only.  The scientific object lives in ``src/core``
and is imported, never re-implemented.  This package builds the legal data path
(five markets x four frozen Hosts), the chronological ``POST_TRAIN`` OOF trainer,
the nonnegative pooled MAE calibration, metrics and provenance -- and stops there.

No scientific result is produced here, and ``PROTECTED_FINAL`` is structurally
unreachable: the canonical dataset contracts declare it a *closed* role, so it
never appears in any frozen artifact's segment vocabulary.

The module list below is the read order: registries, then the legal data path,
then the trainer, then the entry points.  ``core_bridge`` is the only module that
touches ``sys.path`` or ``src/core``; ``training`` / ``oof`` / ``calibration`` /
``metrics`` hold no fitting mathematics of their own and delegate every
scientific definition back to the frozen core.
"""

__all__ = [
    # registries and contracts
    "config",
    "contracts",
    # legal data path
    "host_prediction_loader",
    "china5_adapter",
    "core_bridge",
    # trainer
    "training",
    "oof",
    "calibration",
    "metrics",
    # entry points
    "synthetic",
    "runner",
    "verify_preexecution",
]
