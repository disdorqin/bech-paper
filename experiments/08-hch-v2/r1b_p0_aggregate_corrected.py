"""P0-R1B-S1 — recompute four-cell aggregate with screen-relative seen/unseen
semantics, WITHOUT retraining (Stage-2 protocol §1).

Problem fixed:
  Stage-1 runner set `host_seen = (host != "PatchTST")` STATICALLY. That is only
  correct for the LOHO candidate (trained Linear/MLP/LSTM; PatchTST unseen).
  For the MAIN candidate (trained on all 4 hosts) PatchTST is a SEEN host, so
  the main screens had no unseen-host cell at all.

Correct semantics (protocol §1):
  Main  (trained hosts = {Linear, MLP, LSTM, PatchTST}) -> all 4 hosts seen
        cells: Seen/Seen = 12, Unseen/Seen = 4.  No unseen-host cell.
  LOHO  (trained hosts = {Linear, MLP, LSTM})       -> PatchTST unseen
        cells: 9 / 3 / 3 / 1.

Inputs (historical, unchanged):  R1B_SCREEN/matrix_{LearnedSig,PlainCore}_{main,LOHO}.csv
Outputs (new):
  R1B_SCREEN/four_cell_aggregate_corrected.csv
  R1B_SCREEN/P0_correction_note.md
The original four_cell_aggregate.csv is NOT modified (preserve history).
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCR = HERE / "results" / "R1B_SCREEN"

SOURCE_MARKETS = {"LAGO_DE", "LAGO_PJM", "NEM_SA1"}
HOSTS_ALL = {"Linear", "MLP", "LSTM", "PatchTST"}
HOSTS_LOHO = {"Linear", "MLP", "LSTM"}

# screen key -> candidate's training membership
TRAINED = {
    "LearnedSig_main": (SOURCE_MARKETS, HOSTS_ALL),
    "PlainCore_main": (SOURCE_MARKETS, HOSTS_ALL),
    "LearnedSig_LOHO": (SOURCE_MARKETS, HOSTS_LOHO),
    "PlainCore_LOHO": (SOURCE_MARKETS, HOSTS_LOHO),
}

FILES = {
    "LearnedSig_main": "matrix_LearnedSig_main.csv",
    "PlainCore_main": "matrix_PlainCore_main.csv",
    "LearnedSig_LOHO": "matrix_LearnedSig_LOHO.csv",
    "PlainCore_LOHO": "matrix_PlainCore_LOHO.csv",
}


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict]) -> list[dict]:
    cells = []
    for seen_mkt in (1, 0):
        for seen_host in (1, 0):
            sub = [r for r in rows
                   if int(r["market_seen"]) == seen_mkt
                   and int(r["host_seen"]) == seen_host]
            d = [float(r["delta_crps"]) for r in sub
                 if r["delta_crps"] not in (None, "", "None")]
            cells.append({
                "market_seen": seen_mkt, "host_seen": seen_host,
                "label": f"{'Seen' if seen_mkt else 'Unseen'}_market/"
                         f"{'Seen' if seen_host else 'Unseen'}_host",
                "n_domains": len(sub),
                "mean_delta_crps": round(sum(d) / len(d), 6) if d else None,
                "worst_delta_crps": round(max(d), 6) if d else None,
                "best_delta_crps": round(min(d), 6) if d else None,
            })
    return cells


def main():
    assert SCR.exists(), f"missing {SCR}"
    out_rows = []
    note = []
    for key, trained in TRAINED.items():
        trained_mkts, trained_hosts = trained
        path = SCR / FILES[key]
        rows = load_rows(path)
        # --- screen-relative fix: recompute host_seen/market_seen from training
        # membership; the historical CSV's host_seen column stays untouched. ---
        corrected = []
        changed = 0
        for r in rows:
            market = r["market"]
            host = r["host"]
            m_seen = int(market in trained_mkts)
            h_seen = int(host in trained_hosts)
            if (r["market_seen"] != str(m_seen)
                    or r["host_seen"] != str(h_seen)):
                changed += 1
            corrected.append({**r, "market_seen": m_seen, "host_seen": h_seen})
        cells = aggregate(corrected)
        for c in cells:
            out_rows.append({"screen": key, **c})
        note.append(
            f"{key}: trained_markets={sorted(trained_mkts)} "
            f"trained_hosts={sorted(trained_hosts)} — recomputed {len(corrected)} "
            f"rows, {changed} seen-flag cells flipped vs historical CSV.")

    out_path = SCR / "four_cell_aggregate_corrected.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["screen"] + list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out_path}")
    for r in out_rows:
        print(f"  {r['screen']:16s} {r['label']:26s} n={r['n_domains']:2d} "
              f"mean={r['mean_delta_crps']} worst={r['worst_delta_crps']}")

    note_path = SCR / "P0_correction_note.md"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("# P0-R1B-S1 correction note — screen-relative seen/unseen\n\n")
        f.write("Source: Stage-1 `four_cell_aggregate.csv` used static "
                "`host != \"PatchTST\"` for host_seen. Protocol §1 requires "
                "seen/unseen to be **relative to each candidate's training "
                "membership**.\n\n")
        f.write("Corrections applied (rows recomputed from the unchanged matrix "
                "CSVs, no retrain):\n\n")
        for line in note:
            f.write(f"- {line}\n")
        f.write("\nOriginal `four_cell_aggregate.csv` preserved (history). "
                "All transfer numbers themselves are unchanged — only the "
                "cell assignment of host-seen flags is corrected. Main "
                "candidates have NO unseen-host cell; the Stage-1 report "
                "table's main-column 'Seen/Unseen_host' row was a labeling "
                "artifact, now removed.\n")
    print(f"wrote {note_path}")


if __name__ == "__main__":
    main()
