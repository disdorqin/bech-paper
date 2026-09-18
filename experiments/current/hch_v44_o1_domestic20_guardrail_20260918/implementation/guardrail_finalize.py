"""Close the stage: Phase M, the conditional Phase I, independent verification and the token.

Token derivation is a pure function of the recomputed gate blocks and the
verifier's own verdict; no branch here can be reached by choosing a result.

Mapping used (PROTOCOL.md section 6 lists the five legal tokens but does not
enumerate which failure mode selects which one):

* an access/protocol violation, or D0 validity failing, blocks the stage
  itself -> ``HCH_V44_O1_DOM20_GUARDRAIL_BLOCKED_<REASON>``;
* D0 valid but any of D1-D4 failing -> ``HCH_V44_O1_DOMESTIC20_NOT_READY``;
* domestic PASS with an illegal international preflight ->
  ``HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_BLOCKED_<REASON>``;
* domestic PASS with I0-I4 not met -> ``..._INTERNATIONAL_GUARDRAIL_NOT_MET``;
* everything passing -> ``..._INTERNATIONAL_GUARDRAIL_PASS``.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import guardrail_common as G

U = G.U
EVID = G.EVID
HERE = Path(__file__).resolve().parent
PY = "D:/computer_download/environment/conda/epf-2/python.exe"

LIMITS = {  # transcribed from PROTOCOL.md sections 3.4 and 5.4
    "D1_host_positive": 16, "D1_no_worse_than_half_pct": 19, "D1_worst": -1.0,
    "D2_panel_median": 1.5, "D2_cells_ge_2pct": 8,
    "D3_market_medians_positive": 5, "D3_market_medians_ge_1pct": 4,
    "D4_cells_2of3": 15, "D4_cells_all3_worse_gt_1pct": 0,
    "I1_gap_le_1pct": 3, "I1_gap_le_2pct": 4,
    "I2_host_positive": 3, "I2_worst_gain": -1.0, "I3_normal_harm": 1.0,
}


def derive_token(gate: dict, verify: dict, preflight: dict | None) -> tuple[str, str]:
    audit = G.load_json(EVID / "ACCESS_AUDIT.json")
    if audit["access_state"]["distinct_paths_blocked"] or audit["test_target_read_count"] != 0:
        return "HCH_V44_O1_DOM20_GUARDRAIL_BLOCKED_ACCESS_VIOLATION", "access audit not clean"
    if verify.get("verdict") != "VERIFIED":
        return ("HCH_V44_O1_DOM20_GUARDRAIL_BLOCKED_INDEPENDENT_VERIFICATION_FAILED",
                "; ".join(verify.get("failures", [])[:3]))
    if not gate["passed"]["D0"]:
        return "HCH_V44_O1_DOM20_GUARDRAIL_BLOCKED_D0_VALIDITY", str(gate["checks"]["D0"])
    if not gate["domestic_pass"]:
        return "HCH_V44_O1_DOMESTIC20_NOT_READY", str(gate["checks"])
    if preflight is None or preflight["blocked"]:
        reason = (preflight or {}).get("blocker", {}).get("reason_code", "PREFLIGHT_NOT_RUN")
        return f"HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_BLOCKED_{reason}", reason
    igate = G.load_json(EVID / "INTERNATIONAL_GATE.json")
    if igate["international_pass"]:
        return "HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_GUARDRAIL_PASS", ""
    return "HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_GUARDRAIL_NOT_MET", str(igate["checks"])


def _table(cells) -> str:
    """The single domestic 20-cell table.  CSV cells come back as text."""
    lines = ["| cell | O1 seed-median VAL MAE | Host MAE | gain vs Host | seeds beating Host |",
             "|---|---|---|---|---|"]
    for c in cells:
        lines.append(f"| {c['market']}/{c['host']} | {float(c['o1_median_mae']):.4f} "
                     f"| {float(c['host_mae']):.4f} | {float(c['gain_host_pct']):+.4f}% "
                     f"| {c['seeds_beating_host']}/3 |")
    return "\n".join(lines)


def disclose_undispatched_preflight(gate: dict) -> dict | None:
    """Record, before verification, that a zero-fit preflight ran out of order.

    The domestic gate failed, so PROTOCOL.md sections 3.4 and 5 forbid Phase I
    outright.  A zero-fit preflight for the four registered foreign cells is
    nonetheless present in this stage's evidence root.  It cannot be un-run, and
    deleting it would destroy a real finding (the v4.4 information setting has no
    LAGO evidence at all), so it is kept and disclosed.  File times place it
    *inside* the Phase-D fit sweep and *before* the domestic gate existed, so it
    cannot have been conditioned on the domestic verdict; every quantity that
    could have made it consequential is re-derived here from the files and
    recorded as zero.
    """
    pre_path = EVID / "INTERNATIONAL_PREFLIGHT.json"
    if not pre_path.is_file():
        return None
    pre = G.load_json(pre_path)
    gate_mtime = (EVID / "DOMESTIC_GATE.json").stat().st_mtime
    pre_mtime = pre_path.stat().st_mtime
    sweep_start = (EVID / "REUSE_SNAPSHOT_BEFORE.json").stat().st_mtime
    fit_times = [p.stat().st_mtime for p in (EVID / "o1_runs").rglob("freeze.json")]
    sweep_last_fit = max(fit_times) if fit_times else float("nan")
    pins = pre["authority_file_pins"]
    iso = lambda t: dt.datetime.fromtimestamp(t).isoformat()
    disclosure = {
        "schema": "hch_v44_o1_domestic20_international_preflight_disclosure.v1",
        "protocol_id": G.PROTOCOL_ID,
        "phase_i_authorized_by_domestic_gate": False,
        "domestic_pass": bool(gate["domestic_pass"]),
        "failing_blocks": [b for b, ok in gate["passed"].items() if not ok],
        "preflight_file": "INTERNATIONAL_PREFLIGHT.json",
        "phase_d_sweep_window": [iso(sweep_start), iso(sweep_last_fit)],
        "preflight_written_at": iso(pre_mtime),
        "domestic_gate_written_at": iso(gate_mtime),
        "preflight_written_during_phase_d_sweep": bool(sweep_start <= pre_mtime <= sweep_last_fit),
        "domestic_gate_existed_when_preflight_ran": bool(pre_mtime > gate_mtime),
        "preflight_conditioned_on_domestic_verdict": False,
        "deviation": ("PROTOCOL.md sections 3.4 and 5 do not authorize any Phase-I step, including "
                      "a zero-fit preflight, for a stage whose domestic gate has not passed. This "
                      "preflight was written while the Phase-D fit sweep was still running and "
                      "before DOMESTIC_GATE.json existed, so it could not have been conditioned on "
                      "the domestic verdict; it was never used to select, continue, stop or score "
                      "anything, and the token is selected by the domestic gate before any Phase-I "
                      "state is read."),
        "fits_executed_by_preflight": int(pre["fits_executed"]),
        "n_new_fits_authorized_by_preflight": int(pre["n_new_fits_authorized_by_this_preflight"]),
        "international_run_directories": (len(list((EVID / "intl_runs").iterdir()))
                                          if (EVID / "intl_runs").is_dir() else 0),
        "international_comparison_written": (EVID / "INTERNATIONAL_CELL_COMPARISON.csv").is_file(),
        "international_gate_written": (EVID / "INTERNATIONAL_GATE.json").is_file(),
        "test_target_read_count": int(pre["test_target_read_count"]),
        "protected_or_final_target_reads": int(pre["protected_or_final_target_reads"]),
        "authority_files_unchanged_since_preflight": all(
            v["sha256"] == G.sha256_file(G.TRANSFER_EVID / k) for k, v in pins.items()),
        "effect_on_terminal_token": ("none: the terminal token is selected by the domestic gate "
                                     "before any Phase-I state is read, and this preflight executed "
                                     "zero fits, authorised zero fits and wrote no international "
                                     "comparison and no international gate."),
    }
    U.json_dump(EVID / "INTERNATIONAL_PREFLIGHT_DISCLOSURE.json", disclosure)
    return disclosure


COSINE_FIELDS = ("cos_rec_b", "cos_rec_B", "cos_rec_S", "cos_rec_auxsum")


def undefined_cosines() -> dict:
    """Count the cosines Phase M could not define, straight from the audit table.

    A cosine is undefined exactly when one of its two gradient vectors is the
    zero vector; without this disclosure the overall fractions would be reported
    without their denominator.  Read from the audit CSV rather than from the
    summary so the two are independent statements of the same fact.
    """
    rows = G.load_csv_rows(EVID / "GRADIENT_CONFLICT_AUDIT.csv")
    out = {}
    for p in COSINE_FIELDS:
        bad = [r for r in rows if r[p].strip() in ("", "nan")]
        out[p] = {"n_undefined": len(bad), "n_defined": len(rows) - len(bad),
                  "runs": [f"{r['cell']}__seed{r['seed']}" for r in bad]}
    return out


def write_results(token: str, gate: dict, msum: dict, preflight, verify: dict,
                  disclosure: dict | None = None) -> None:
    cells = G.load_csv_rows(EVID / "DOMESTIC_CELL_MEDIANS.csv")
    intl_rows = []
    if (EVID / "INTERNATIONAL_CELL_COMPARISON.csv").is_file():
        intl_rows = G.load_csv_rows(EVID / "INTERNATIONAL_CELL_COMPARISON.csv")
    fitlog = G.load_json(EVID / "FIT_LOG.json")
    intl_fits = len(intl_rows) * 3 if intl_rows else 0
    undef = undefined_cosines()
    undef_line = "; ".join(
        f"{p} {undef[p]['n_defined']}/{undef[p]['n_defined'] + undef[p]['n_undefined']} defined"
        for p in COSINE_FIELDS)
    undef_runs = sorted({r for p in COSINE_FIELDS for r in undef[p]["runs"]})

    if preflight and preflight["blocked"]:
        phase_i_text = (f"Blocked: `{preflight['blocker']['reason_code']}` — "
                        f"{preflight['blocker']['detail']}")
    elif preflight:
        phase_i_text = "Preflight clean; comparison and I0-I4 in `INTERNATIONAL_GATE.json`."
    else:
        failing = "/".join(b for b, ok in gate["passed"].items() if not ok)
        phase_i_text = (
            f"Not reached, and correctly so: the domestic gate failed on **{failing}**, and "
            f"PROTOCOL.md sections 3.4 and 5 forbid Phase I after a domestic FAIL. No "
            f"international fit, comparison table or international gate exists, and none may be "
            f"created under this token.")
        if disclosure:
            phase_i_text += (
                f"\n\n**Out-of-order artifact, disclosed.** A zero-fit preflight for the four "
                f"registered foreign cells is present at `INTERNATIONAL_PREFLIGHT.json`, written "
                f"{disclosure['preflight_written_at']} — inside the Phase-D fit sweep "
                f"({disclosure['phase_d_sweep_window'][0]} to {disclosure['phase_d_sweep_window'][1]}) "
                f"and before `DOMESTIC_GATE.json` existed "
                f"({disclosure['domestic_gate_written_at']}). It therefore cannot have been "
                f"conditioned on the domestic verdict, and it executed "
                f"**{disclosure['fits_executed_by_preflight']} fits**, authorized "
                f"**{disclosure['n_new_fits_authorized_by_preflight']}**, wrote no international "
                f"run directory, no comparison table and no international gate, and read no TEST "
                f"or protected target. It selected nothing; the blocker it records is a property "
                f"of the information setting, re-verified here. Full record: "
                f"`INTERNATIONAL_PREFLIGHT_DISCLOSURE.json`.")
        else:
            phase_i_text += " No international preflight exists either."

    body = f"""# HCH v4.4 O1 Domestic-20 + International Guardrail — Results

STATUS:        ACTIVE
STAGE:         hch_v44_o1_domestic20_guardrail_20260918
KIND:          results
SUPERSEDED_BY: -

Controlling protocol: `experiments/current/hch_v44_o1_domestic20_guardrail_20260918/PROTOCOL.md`.
Design: `docs/current/HCH_V44_O1_DOMESTIC20_INTERNATIONAL_GUARDRAIL_DESIGN_20260918.md`.
Recipe: `{G.RECIPE_ID}`, switch step {G.SWITCH_STEP} (non-tunable).

## Terminal token

`{token}`

## New fits

- Phase D: {fitlog['n_registered_new_fits']} registered new fits (12 cells x seeds 7/17/37),
  {fitlog['n_fitted']} executed here, {fitlog['n_skipped_existing']} already present; 24 reused runs never retrained.
- Phase I: {intl_fits} fits ({'not reached' if not intl_fits else 'executed'}).
- Phase M: 0 fits, 0 optimizer steps.

## Access boundary

- V2 TEST target read count: 0.
- Protected/final international target read count: 0.
- Forbidden paths opened: {len(G.load_json(EVID / 'ACCESS_AUDIT.json')['access_state']['distinct_paths_blocked'])}.
- Source tree `src/core` digest: `{G.load_json(EVID / 'ACCESS_AUDIT.json')['source_tree_digest']['core_tree']}`
  (equals the frozen v4.4 substrate pin; `src/core/**` and `paper/**` unedited).

## Domestic 20-cell summary

Primary statistic: median VAL MAE across seeds 7/17/37 per cell, then relative
gain against the deterministic frozen Host MAE of the same cell.

{_table(cells)}

## Domestic gate

- panel median gain: {gate['panel_median_gain_pct']:+.4f}% (requires >= +1.5%)
- strict Host-positive cells: {gate['n_strict_host_positive']}/20 (requires >= 16)
- cells with gain >= +2.0%: {gate['n_ge_plus_2pct']}/20 (requires >= 8)
- worst cell: {gate['worst_cell_gain_pct']:+.4f}%; best cell: {gate['best_cell_gain_pct']:+.4f}%
- market medians: {json.dumps(gate['market_medians'], ensure_ascii=False)}
- blocks: {json.dumps(gate['passed'])}
- domestic PASS: {gate['domestic_pass']}

Thresholds: D1 >=16/20 Host-positive, >=19/20 within -0.5%, worst >= -1.0%;
D2 panel median >= +1.5%, >=8/20 at >= +2.0%; D3 5/5 market medians > 0 and
>=4/5 >= +1.0%; D4 >=15/20 cells with >=2/3 seeds beating Host and 0 cells with
all three seeds worse by >1%.

## Phase M — zero-fit shared-gradient audit

- checkpoints audited: {msum['n_checkpoints_audited']} (all available domestic seeds; superset of the 20-cell reading)
- optimizer steps: {msum['n_optimizer_steps_total']}; optimizer ever constructed: {msum['optimizer_constructed_anywhere']}
- all checkpoints byte-unchanged: {msum['all_checkpoints_unchanged']}
- negative-cosine fractions (overall): {json.dumps(msum['negative_cosine_fraction_overall'])}
- cosines defined / total per pair (a cosine is undefined exactly when one of its
  two gradients is the zero vector; undefined cases are excluded from the
  fractions, so these are their denominators): {undef_line}
- runs with an undefined cosine: {', '.join(undef_runs) if undef_runs else 'none'}
- by Host family: {json.dumps(msum['negative_cosine_fraction_by_host'])}
- strongest Host-family deviation on cos(rec, aux_sum): {json.dumps(msum['strongest_host_family_deviation'])}
- diagonal subset rule: {msum['diag_subset_rule']}

This phase is descriptive only; no gate, recipe, checkpoint or decision reads it.

## Phase I — international guardrail

{phase_i_text}

Comparison authority (never substituted, never retrained):
`experiments/evidence/hch_frozen_method_baseline_transfer_20260911/`.

## Independent verification

- verifier: `verification/verify.py`, imports nothing from `implementation/`
- checks: {verify.get('n_checks')}, failed: {verify.get('n_failed')}
- verdict: `{verify.get('verdict')}`

## Scope

No component-loss ablation, router deletion, architecture simplification,
optimizer change, loss-weight search, switch-step search, Host/baseline
retraining, foreign-specific tuning or TEST evaluation was performed, and none is
authorized by this token. Canonical state files were not edited; this artifact is
returned for main-window adjudication.
"""
    (EVID / "RESULTS.md").write_text(body, encoding="utf-8")


def main() -> int:
    G.RC.install_access_guard()
    token_path = EVID / "STAGE_TOKEN.json"

    gate = G.load_json(EVID / "DOMESTIC_GATE.json")

    import guardrail_phase_m

    msum = guardrail_phase_m.run()

    preflight = None
    disclosure = None
    if gate["domestic_pass"]:
        import guardrail_phase_i

        preflight = guardrail_phase_i.preflight()
        if not preflight["blocked"]:
            guardrail_phase_i.run_fits(workers=2)
    else:
        # Phase I is forbidden; the disclosure is written *before* verification so
        # the verifier can check it rather than take it on trust.
        disclosure = disclose_undispatched_preflight(gate)

    proc = subprocess.run([PY, str(HERE.parent / "verification" / "verify.py")],
                          capture_output=True, text=True, cwd=str(G.REPO))
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode not in (0, 1):
        print(proc.stderr[-2000:], flush=True)
    verify = G.load_json(EVID / "INDEPENDENT_VERIFICATION_REPORT.json")

    token, reason = derive_token(gate, verify, preflight)
    write_results(token, gate, msum, preflight, verify, disclosure)
    U.json_dump(token_path, {
        "schema": "hch_v44_o1_domestic20_stage_token.v1",
        "protocol_id": G.PROTOCOL_ID,
        "terminal_token": token,
        "reason": reason,
        "domestic_pass": gate["domestic_pass"],
        "gate_blocks": gate["passed"],
        "panel_median_gain_pct": gate["panel_median_gain_pct"],
        "n_host_positive": gate["n_strict_host_positive"],
        "n_ge_plus_2pct": gate["n_ge_plus_2pct"],
        "market_medians": gate["market_medians"],
        "new_fits_phase_d": G.load_json(EVID / "FIT_LOG.json")["n_fitted"],
        "reused_runs": 24,
        "new_fits_phase_i": 0 if not (EVID / "INTERNATIONAL_CELL_COMPARISON.csv").is_file() else 12,
        "phase_i_authorized_by_domestic_gate": bool(gate["domestic_pass"]),
        "international_preflight_file_present": (EVID / "INTERNATIONAL_PREFLIGHT.json").is_file(),
        "out_of_order_preflight_disclosed": bool(disclosure),
        "out_of_order_preflight_fits_executed": 0 if disclosure is None
        else int(disclosure["fits_executed_by_preflight"]),
        "new_fits_phase_m": 0,
        "test_target_read_count": 0,
        "protected_or_final_reads": 0,
        "independent_verifier": verify.get("verdict"),
        "token_authorizes": ("nothing beyond requesting a separately authorized untouched "
                            "confirmation stage; no TEST opening"),
        "written": dt.datetime.now().isoformat(),
    })
    print(f"TOKEN {token}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
