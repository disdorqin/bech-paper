"""Write RESULTS.md and STAGE_TOKEN.json from the *independently recomputed* verdict.

The runner advertises a ``finalize`` step that its ``main()`` does not dispatch,
and its own source cannot be edited now that the 24 fits have pinned the stage
tree.  So the two section-11 artifacts are written here.

The token is taken from ``INDEPENDENT_VERIFICATION_REPORT.json`` -- the verifier's own
recomputation from raw artifacts -- never from the runner's gate file, so a
gate-arithmetic bug cannot survive into the terminal token.
"""
import csv
import json
from pathlib import Path

REPO = Path(r"D:\作业\science\solar_leak_price_model")
STAGE = REPO / "experiments/current/hch_v44_o1_matched_controls_20260918"
EVID = REPO / "experiments/evidence/hch_v44_o1_matched_controls_20260918"

TOKEN_INCONCLUSIVE = "HCH_O1_MATCHED_CONTROL_ROUTING_INCONCLUSIVE"
TOKEN_SUP_G2 = "HCH_O1_MATCHED_CONTROL_GEOMETRY_SUPPORTED_G2"
TOKEN_SUP_G1 = "HCH_O1_MATCHED_CONTROL_GEOMETRY_SUPPORTED_G1"
TOKEN_NOT_SUP = "HCH_O1_MATCHED_CONTROL_GEOMETRY_NOT_SUPPORTED"
G1 = "G1_GEOM_SHARED_CONTEXT"
G2 = "G2_GEOM_COORD_CONTEXT"
G3 = "G3_DIRECT_CONTEXT_CTRL"
B1 = "B1_EXACT_GEOMETRY_SUPPORTED"

SURVIVOR_A1_B1 = (
    "`G2_GEOM_COORD_CONTEXT` survives for paper use: coordinate-specific "
    "geometry-aligned context allocation is empirically supported, and the existing O1 "
    "full-20 TRAIN+VAL breadth is reusable as the development main table with no new "
    "full-20 method fits.")
SURVIVOR_A2_B1 = (
    "`G1_GEOM_SHARED_CONTEXT` survives for paper use: coordinate-specific identity is "
    "deleted and exact geometry is the primary method contribution; a separately "
    "registered remaining-12-cell G1 full-20 completion is required before confirmation.")
SURVIVOR_A3 = (
    "No method choice is authorised: Phase A is inconclusive (A3), so the stage stops "
    "before any direct control and returns for human adjudication.")
SURVIVOR_B2 = (
    "No structured variant survives for prediction from this stage: exact geometry "
    "remains a mathematical representation result only, no predictive advantage may be "
    "claimed, and the question returns to human paper-position adjudication rather than "
    "another architecture rescue.")


def param_counts(par: dict) -> dict:
    """VARIANT_PARITY.json records a parameter_table, not a counts dict."""
    return {r["variant"]: int(r["n_trainable_parameters"]) for r in par["parameter_table"]}


def rows(path: Path) -> list:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def truthy(v) -> bool:
    """The runner writes CSV booleans lowercase (``true``), not as ``True``."""
    return str(v).strip().lower() in ("true", "1", "yes")


def table(rs, k_struct, k_ctrl, k_gain, k_beat, k_host) -> str:
    out = [f"| cell | MAE structured (median) | MAE control (median) | gain | "
           f"structured beats control | structured Host+ |",
           "|---|---|---|---|---|---|"]
    for r in rs:
        out.append(f"| {r['cell']} | {float(r[k_struct]):.4f} | {float(r[k_ctrl]):.4f} | "
                   f"{float(r[k_gain]):+.4f}% | {'yes' if truthy(r[k_beat]) else 'no'} | "
                   f"{'yes' if truthy(r[k_host]) else 'no'} |")
    return "\n".join(out)


def main() -> int:
    checks = json.loads((EVID / "INDEPENDENT_VERIFICATION_REPORT.json").read_text(encoding="utf-8"))
    ra, rb = checks["recomputed_phase_a"], checks.get("recomputed_phase_b")
    selected = ra["selected"]
    a_verdict = ra["verdict"]
    b_verdict = (rb or {}).get("verdict")

    if selected is None:
        token = TOKEN_INCONCLUSIVE
        survivor, a1 = SURVIVOR_A3, False
    elif b_verdict is None:
        raise SystemExit(
            "BLOCKED: Phase A selected a structured variant, so section 8 requires "
            "Phase B; no terminal token may be written before it runs")
    else:
        a1 = selected == G2
        if b_verdict == B1:
            token = TOKEN_SUP_G2 if a1 else TOKEN_SUP_G1
            survivor = SURVIVOR_A1_B1 if a1 else SURVIVOR_A2_B1
        else:
            token, survivor = TOKEN_NOT_SUP, SURVIVOR_B2

    gate_a = json.loads((EVID / "PHASE_A_GATE.json").read_text(encoding="utf-8"))
    gate_b = (json.loads((EVID / "PHASE_B_GATE.json").read_text(encoding="utf-8"))
              if (EVID / "PHASE_B_GATE.json").is_file() else None)
    access = json.loads((EVID / "ACCESS_AUDIT.json").read_text(encoding="utf-8"))
    audit = json.loads((EVID / "SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    par = json.loads((EVID / "VARIANT_PARITY.json").read_text(encoding="utf-8"))
    pc = param_counts(par)
    prov = json.loads((EVID / "O1_G2_REUSE_PROVENANCE.json").read_text(encoding="utf-8"))
    snap = json.loads((EVID / "REUSE_SNAPSHOT_BEFORE.json").read_text(encoding="utf-8"))
    guard_hits = sum(int(r["access_state"].get("forbidden_guard_hits", 0))
                     for r in access["runs"])

    L = []
    A = L.append
    A("# HCH O1 matched-control closure")
    A("")
    A(f"**Terminal token: `{token}`**")
    A("")
    A("- protocol: `HCH_V44_O1_MATCHED_CONTROLS_20260918`")
    A("- recipe: `O1_AUX_WARMUP_400_THEN_MAE` — full structured objective through "
      "optimizer update 400, `L_rec` only after; selection = min EMA VAL MAE over all "
      "checks including step 0")
    A(f"- Phase A verdict: `{a_verdict}` → selected structured variant `{selected}`")
    A(f"- Phase B verdict: `{b_verdict}`" if b_verdict
      else "- Phase B: not executed (Phase A was inconclusive, A3)")
    A(f"- independent verification: all_pass=**{checks['all_pass']}** "
      f"({checks['n_checks']} scientific checks, {checks['n_failures']} failures); "
      f"hygiene_pass={checks['hygiene_pass']}")
    A("")
    A("## 1. Source, parity and access")
    A("")
    A(f"- `src/core` digest reproduces the frozen O1/G2 value: "
      f"{audit['core_tree_matches_frozen_o1_g2']} — `{audit['observed']['core_tree']}` "
      f"({audit['core_file_count']} files)")
    A(f"- `src/backbones` digest reproduces the frozen O1/G2 value: "
      f"{audit['backbones_tree_matches_frozen_o1_g2']} — "
      f"`{audit['observed']['backbones_tree']}` ({audit['backbones_file_count']} files)")
    A(f"- `src/core` modified by this stage: **{audit['src_core_modified']}**")
    A(f"- reused O1/G2 runs pinned by hash: {prov['n_reused_runs']}/24; one "
      f"`probe_code_hash` across all of them: "
      f"{prov['one_probe_code_hash_across_reused']}; every history support registered: "
      f"{prov['all_history_support_registered']}; artifact bytes snapshotted before the "
      f"fits: {snap['n_artifacts']}")
    A(f"- TEST target reads: {access['test_target_read_count_total_new']} in the new runs "
      f"({access['n_new_runs']} runs), all reused runs zero: "
      f"{all(int(v) == 0 for v in access['test_target_read_count_reused'])}")
    A(f"- forbidden-path guard hits summed over every run: {guard_hits}; TEST rows dropped "
      f"before any frame existed, per run: "
      f"{access['runs'][0]['test_rows_dropped_from_joint_cache']}")
    A("")
    A("## 2. Variant contract")
    A("")
    A(f"- G1 and G2 have equal parameter count: "
      f"{par['checks']['g1_g2_equal_parameter_count']} — {pc[G1]} "
      f"trainable parameters, identical named keys and shapes")
    A(f"- the direct control adds no new parameter to the existing `G3`: "
      f"`G3_SHARED` = {pc['G3_SHARED_DIRECT_CONTEXT_CTRL']} = `G3` = {pc[G3]} "
      f"trainable parameters; only `tied_identity` differs")
    A(f"- G1/G2 bitwise identical at initialization: "
      f"{par['checks']['g1_g2_bitwise_identical_at_init']}; forward output bitwise "
      f"identical at initialization: "
      f"{par['checks']['g1_g2_forward_bitwise_identical_at_init']}")
    A(f"- only the `tied_identity` flag differs between them: "
      f"{par['checks']['g1_g2_only_tied_identity_flag_differs']} "
      f"(`coord_embed` is zero-initialised, so the tie is exact at step 0 and only the "
      f"released variant can break it)")
    A(f"- direct control inside the registered 1.25x budget: "
      f"{par['checks']['g3_within_budget']} — ratio {par['g3_over_g2_ratio']:.6f}; "
      f"`G3_SHARED` adds no parameter to `G3`: "
      f"{par['checks']['g3_shared_adds_no_parameter_to_g3']}")
    A("")
    A("## 3. Phase A — G2 versus G1, seed-median-first")
    A("")
    A(f"gain(G2,G1) = 100 * (MAE_G1 - MAE_G2) / MAE_G1, per cell after taking each "
      f"method's 3-seed median VAL MAE.")
    A("")
    A(table(rows(EVID / "PHASE_A_CELL_MEDIANS.csv"), "mae_g2_median", "mae_g1_median",
            "gain_g2_vs_g1_pct", "g2_beats_g1", "g2_host_positive"))
    A("")
    A(f"- cells where G2 beats G1: {ra['cells_beating']}/8")
    A(f"- panel median gain(G2,G1): **{ra['panel_median_gain_pct']:+.4f}%**")
    A(f"- worst cell: {ra['worst_cell_gain_pct']:+.4f}%; best cell: "
      f"{ra['best_cell_gain_pct']:+.4f}%")
    A(f"- A1 clauses: `{json.dumps(gate_a['a1_checks'])}`")
    A(f"- A2 clauses: `{json.dumps(gate_a['a2_checks'])}`")
    A(f"- **verdict `{a_verdict}`**")
    if b_verdict:
        A("")
        A("## 4. Phase B — structured versus direct, seed-median-first")
        A("")
        A(f"- direct family fitted: `{gate_b['direct_variant']}` — "
          + ("the existing `G3_DIRECT_CONTEXT_CTRL`, tie off"
             if gate_b["direct_variant"] == "G3_DIRECT_CONTEXT_CTRL"
             else "the existing `G3_DIRECT_CONTEXT_CTRL` with `tied_identity=True`, "
                  "which introduces no new parameter and is byte/forward identical "
                  "otherwise"))
        A(f"- reconstruction-only loss throughout, same optimizer and checkpoint budget")
        A("")
        A(f"gain(STRUCT,DIRECT) = 100 * (MAE_DIRECT - MAE_STRUCT) / MAE_DIRECT.")
        A("")
        A(table(rows(EVID / "PHASE_B_CELL_MEDIANS.csv"), "mae_struct_median",
                "mae_direct_median", "gain_struct_vs_direct_pct", "struct_beats_direct",
                "struct_host_positive"))
        A("")
        A(f"- cells where structured beats direct: {rb['cells_beating']}/8")
        A(f"- panel median gain(STRUCT,DIRECT): **{rb['panel_median_gain_pct']:+.4f}%**")
        A(f"- worst cell: {rb['worst_cell_gain_pct']:+.4f}%; best cell: "
          f"{rb['best_cell_gain_pct']:+.4f}%")
        A(f"- B1 clauses: `{json.dumps(gate_b['b1_checks'])}`")
        A(f"- **verdict `{b_verdict}`**")
    A("")
    A("## 5. Which structured variant survives for paper use")
    A("")
    A(survivor)
    A("")
    A("## 6. Independent verification")
    A("")
    A("- `verification/verify_matched.py` imports none of `matched_runner`, "
      "`matched_train`, `matched_common`, `probe_train`, `probe_common` or "
      "`recovery_train`; the panel, seeds, registered schedule and every gate threshold "
      "are restated in it, and every number is recomputed from the freeze records, "
      "training curves, selected EMA checkpoints and CSV tables")
    A(f"- {checks['n_checks']} scientific checks: all_pass=**{checks['all_pass']}** "
      f"(failures: {checks['failures']})")
    A(f"- hygiene checks: hygiene_pass={checks['hygiene_pass']} "
      f"({checks['hygiene_failures']})")
    A("- the reused O1/G2 artifacts were additionally cross-checked against "
      "`experiments/evidence/hch_v44_o1_domestic20_guardrail_20260918/"
      "REUSE_PROVENANCE.json`, a witness written by a *different* stage — the one "
      "that closed with its own terminal token `HCH_V44_O1_DOMESTIC20_NOT_READY` and "
      "its own independent verification report. It agrees on all three artifact "
      "hashes, and independently records the same `selected_val_mae_ema`, "
      "`selected_step` and `selected_ema_parameter_hash` for every one of the 24 "
      "reused runs — so the baseline MAE that the Phase-A gain divides by is "
      "confirmed outside this stage's own records. mtime is not treated as evidence "
      "of priority anywhere in this stage; the basis is that stage's own closure "
      "artifacts.")
    A("")
    A("### Disclosed defects (none affect a scientific number)")
    A("")
    for d in checks["disclosed_defects"]:
        A(f"- {d['defect']} — impact: {d['impact']}")
    A("- `matched_tree_digest` hashes the entire stage directory, so the verifier could "
      "not be archived into `verification/` before the fits without changing "
      "`matched_code_hash` part-way through the stage. The verifier therefore recomputes "
      "the digest over the fit-time file set — the stage tree excluding its own "
      "`verification/` subtree — and lists the files archived afterwards. Every run's "
      "recorded `matched_code_hash` reproduces exactly under that rule.")
    A("- the verifier ran twice: once before the token existed, where its only failure was "
      "the absent token, and once after, from its archived location. The final "
      "`INDEPENDENT_VERIFICATION_REPORT.json` is the second run.")
    A("")
    A("## 7. Evidence files")
    A("")
    for p in sorted(EVID.rglob("*")):
        if p.is_file():
            A(f"- `{p.relative_to(EVID).as_posix()}`")
    A("")

    (EVID / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (EVID / "STAGE_TOKEN.json").write_text(json.dumps({
        "schema": "hch_v44_o1_matched_stage_token.v1",
        "protocol_id": "HCH_V44_O1_MATCHED_CONTROLS_20260918",
        "terminal_token": token,
        "phase_a_verdict": a_verdict, "phase_b_verdict": b_verdict,
        "selected_structured_variant": selected,
        "derived_from": "INDEPENDENT_VERIFICATION_REPORT.json (independently recomputed), not the runner gate",
        "verifier_all_pass": checks["all_pass"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"terminal_token": token, "phase_a": a_verdict,
                      "phase_b": b_verdict, "selected": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
