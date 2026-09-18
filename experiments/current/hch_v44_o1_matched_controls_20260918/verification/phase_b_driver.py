"""Drive the registered E2 Phase B fits through the runner's own phase machinery.

Why this file exists
--------------------
``matched_runner.phase_b`` passes a **lambda** into ``ProcessPoolExecutor``::

    return _fit_phase(PHASE_B, direct_variant,
                      lambda a: fit_task_b((a[0], a[1], a[2], direct_variant)),
                      workers, "FIT_LOG_PHASE_B.json")

``_fit_phase`` sends that callable through ``pool.map`` under the ``spawn`` start
method, and a lambda is not picklable, so Phase B aborts with
``AttributeError: Can't pickle local object 'phase_b.<locals>.<lambda>'`` before
any fit runs.  ``phase_a`` does not have this problem because it passes the
module-level ``fit_task_a``.

``matched_runner.py`` cannot be edited: the 24 Phase A freeze records pinned
``matched_code_hash`` over the **entire stage directory** (``matched_common.
matched_tree_digest`` walks ``STAGE.rglob("*")``), so any byte written into the
stage tree now would leave those 24 records unreproducible.  The verifier asserts
exactly that reproducibility, so editing the runner would invalidate Phase A.

What this does instead
----------------------
It imports the runner and calls the **same** ``_fit_phase`` with the **same**
``fit_task_b``, the same phase name, the same variant and the same log name --
substituting only a module-level (therefore picklable) shim for the lambda.  The
scientific code path is untouched: every fit still goes through
``matched_train.train_matched(market, host, seed, out, variant=G3_SHARED,
device="cuda")``, which each freeze record pins by its own ``matched_train_sha256``.

This file deliberately lives **outside** the stage tree so that the fit-time
matched_code_hash stays byte-identical to Phase A's.  Its own sha256 is recorded
into EVID/PHASE_B_DRIVER_PROVENANCE.json and it is archived to
STAGE/verification/ afterwards, so the dispatcher is auditable and hash-pinned
even though it is not part of the digest that the runner computes.
"""
import ast
import datetime as dt
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(r"D:\作业\science\solar_leak_price_model")
STAGE = REPO / "experiments/current/hch_v44_o1_matched_controls_20260918"
IMPL = STAGE / "implementation"
EVID = REPO / "experiments/evidence/hch_v44_o1_matched_controls_20260918"

for p in (str(IMPL), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matched_common as MC          # noqa: E402
import matched_runner as MR          # noqa: E402


def fit_task_b_shared(args) -> dict:
    """The registered Phase B task, with the variant bound at module level.

    Delegates to ``matched_runner.fit_task_b`` -- the runner's own task body, so
    it is not duplicated here -- with the same 4-tuple call shape as the lambda
    it replaces, ``lambda a: fit_task_b((a[0], a[1], a[2], direct_variant))``,
    given ``direct_variant == MC.G3_SHARED``.  The only difference is *where* the
    variant argument comes from, which changes no fit.
    """
    market, host, seed = args
    return MR.fit_task_b((market, host, seed, MC.G3_SHARED))


def _resolve_task():
    """Return a picklable handle whose ``__module__`` a spawned child can import.

    Under ``spawn`` the child unpickles the task by module + qualname and must be
    able to import that module.  Importing this file under its own name (in
    addition to ``__main__``) makes ``__module__ == "phase_b_driver"``, and the
    parent's ``sys.path`` is carried to the child in spawn's preparation data.
    """
    try:
        if __name__ != "phase_b_driver":
            import phase_b_driver as mod
            return mod.fit_task_b_shared
    except Exception as exc:                                   # pragma: no cover
        print(f"  [driver] self-import unavailable ({exc!r}); "
              f"relying on spawn's __main__ fixup", flush=True)
    return fit_task_b_shared


def _fn_src(text: str, name: str) -> str:
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    return ""


def _lambda_of(text: str, fn_name: str) -> str:
    """Unparse the first lambda inside ``fn_name`` -- the shim being replaced."""
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Lambda):
                    return ast.unparse(sub)
    return ""


def _forwards_to_fit_task_b(src: str) -> bool:
    """True iff ``src`` calls ``fit_task_b`` with one 4-element tuple argument."""
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name != "fit_task_b" or len(node.args) != 1:
            continue
        a = node.args[0]
        if isinstance(a, ast.Tuple) and len(a.elts) == 4:
            return True
    return False


def provenance(workers: int) -> dict:
    self_src = Path(__file__).read_bytes()
    self_text = Path(__file__).read_text(encoding="utf-8")
    runner_src = (IMPL / "matched_runner.py").read_text(encoding="utf-8")
    task_src = _fn_src(self_text, "fit_task_b_shared")
    lam = _lambda_of(runner_src, "phase_b")
    return {
        "schema": "hch_v44_o1_matched_phase_b_driver_provenance.v1",
        "protocol_id": MC.PROTOCOL_ID,
        "driver_sha256": hashlib.sha256(self_src).hexdigest().upper(),
        "driver_bytes": len(self_src),
        "driver_path_when_run": str(Path(__file__).resolve()),
        "driver_inside_stage_tree_at_fit_time": str(
            Path(__file__).resolve()).lower().startswith(str(STAGE).lower()),
        "reason": (
            "matched_runner.phase_b binds the variant in a local lambda, which "
            "ProcessPoolExecutor cannot pickle under spawn "
            "(AttributeError: Can't pickle local object 'phase_b.<locals>.<lambda>', "
            "raised from _fit_phase's pool.map before any fit). matched_runner.py "
            "cannot be edited without leaving the 24 Phase A matched_code_hash "
            "records unreproducible, because matched_tree_digest covers the whole "
            "stage directory."),
        "equivalence": (
            "This driver calls matched_runner._fit_phase(MR.PHASE_B, MC.G3_SHARED, "
            "task, workers, 'FIT_LOG_PHASE_B.json') with a module-level task whose "
            "body is matched_runner.fit_task_b((market, host, seed, MC.G3_SHARED)). "
            "Same phase, same variant, same run dirs, same log name, same worker "
            "count, same trainer. The only substitution is a picklable shim for the "
            "lambda; no scientific code path differs."),
        "driver_task_source_verbatim": task_src,
        "replaced_lambda_source_verbatim": lam,
        "runner_fit_task_b_source_verbatim": _fn_src(runner_src, "fit_task_b"),
        "bound_variant_equals_phase_b_direct_variant": (
            MC.G3_SHARED == "G3_SHARED_DIRECT_CONTEXT_CTRL"),
        "replaced_lambda_forwards_to_fit_task_b": _forwards_to_fit_task_b(lam),
        "driver_task_forwards_to_fit_task_b": _forwards_to_fit_task_b(task_src),
        "both_forward_the_same_call_shape": (
            _forwards_to_fit_task_b(lam) and _forwards_to_fit_task_b(task_src)),
        "runner_phase_b_source_verbatim": _fn_src(runner_src, "phase_b"),
        "runner_fit_phase_source_verbatim": _fn_src(runner_src, "_fit_phase"),
        "runner_matched_runner_sha256": hashlib.sha256(
            runner_src.encode("utf-8")).hexdigest().upper(),
        "matched_train_sha256": hashlib.sha256(
            (IMPL / "matched_train.py").read_bytes()).hexdigest().upper(),
        "registered_direct_variant": MC.G3_SHARED,
        "workers": int(workers),
        "written": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def main(workers: int = 2) -> int:
    task = _resolve_task()
    print(f"  [driver] direct family: {MC.G3_SHARED}", flush=True)
    print(f"  [driver] task module: {task.__module__}.{task.__qualname__}", flush=True)
    print(f"  [driver] matched_code_hash will be recomputed by the runner from the "
          f"stage tree; driver is outside it: "
          f"{not str(HERE).lower().startswith(str(STAGE).lower())}", flush=True)

    prov = provenance(workers)
    MR.U.json_dump(EVID / "PHASE_B_DRIVER_PROVENANCE.json", prov)

    # _fit_phase builds its own pool, with its own initializer and its own log
    # write, so the driver passes the task in and mirrors nothing.
    MR._fit_phase(MR.PHASE_B, MC.G3_SHARED, task, max(1, int(workers)),
                  "FIT_LOG_PHASE_B.json")
    print("  [driver] Phase B complete", flush=True)
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    raise SystemExit(main(n))
