"""P2 result deep-dive: action-level diagnosis of E0-E3.

Answers the experiment-design §9 questions:
  1. E1 == E0 day-by-day on real data?        (already known: yes, 0/437)
  2. On the days E2/E3 SWITCH the action, is the realized value net-better?
     (execute->identity should be bad, identity->execute should be good)
  3. Is the added execution in E2/E3 net-positive (mean A_true on execute days)?
  4. A_hat vs A_true calibration — does the head's own value estimate track reality?
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "phase4" / "p2_cavm_experiment_lago_de_linear.json"
r = json.load(open(OUT, encoding="utf-8"))
n = r["n_S4_days"]
e0, e1, e2, e3 = r["modes"]["E0_w1"], r["modes"]["E1_cavm_10"], \
                 r["modes"]["E2_cavm_01"], r["modes"]["E3_cavm_11"]

def act(s):  # executed day indices
    return [i for i in range(n) if s["actions"][i] == "execute"]

def block_stats(rows):
    """(n_exec, mean_A_true_exec, mean_A_true_id, mean_A_hat_exec, mean_lcb_exec)."""
    ex, idn = [], []
    for i, a in rows:
        (ex if a == "execute" else idn).append(i)
    def ma(idx, k):
        v = [rows[i][0][k] for i in idx]
        return sum(v) / len(v) if v else None
    return dict(
        n_exec=len(ex), n_identity=len(idn),
        mean_A_true_exec=ma(ex, "A_true"), mean_A_true_id=ma(idn, "A_true"),
        mean_A_hat_exec=ma(ex, "A_hat"), mean_lcb_exec=ma(ex, "lcb"),
        exec_true_pos=(sum(1 for i in ex if r["modes"].get("E0_w1")["A_true"][i] if False) or
                       sum(1 for i in ex if e0["A_true"][i] > 0) if len(ex) else 0),
    )

print("== Execution stats (realized value of executed days) ==")
for name, s in [("E0_w1", e0), ("E1_cavm_10", e1), ("E2_cavm_01", e2), ("E3_cavm_11", e3)]:
    ex = act(s)
    at = [s["A_true"][i] for i in ex if s["A_true"][i] == s["A_true"][i]]
    ah = [s["A_hat"][i] for i in ex]
    print(f"  {name}: exec {len(ex):3d} days | mean_A_true_exec={sum(at)/len(at):+.4f}"
          f" (pos {sum(1 for a in at if a>0)}/{len(at)}) | "
          f"mean_A_hat_exec={sum(ah)/len(ah):+.4f}")

print("\n== Action switches E2 vs E0 (identity->execute should add value) ==")
sw = [i for i in range(n) if e2["actions"][i] != e0["actions"][i]]
i2e = [i for i in sw if e2["actions"][i] == "execute"]
e2i = [i for i in sw if e0["actions"][i] == "execute"]
def deltas(idx):
    d = [e2["A_true"][i] - e0["A_true"][i] for i in idx if e2["A_true"][i] == e2["A_true"][i] and e0["A_true"][i] == e0["A_true"][i]]
    return sum(d)/len(d), sum(1 for x in d if x > 0), len(d)
da_i2e = deltas(i2e); da_e2i = deltas(e2i)
print(f"  total switched {len(sw)}/{n}; identity->execute {len(i2e)}, execute->identity {len(e2i)}")
print(f"  identity->execute: mean A_true gain {da_i2e[0]:+.4f} (better {da_i2e[1]}/{da_i2e[2]})")
print(f"  execute->identity: mean A_true gain {da_e2i[0]:+.4f} (better {da_e2i[1]}/{da_e2i[2]})")

print("\n== Action switches E3 vs E0 ==")
sw = [i for i in range(n) if e3["actions"][i] != e0["actions"][i]]
i2e = [i for i in sw if e3["actions"][i] == "execute"]
e2i = [i for i in sw if e0["actions"][i] == "execute"]
da_i2e = deltas(i2e); da_e2i = deltas(e2i)
print(f"  total switched {len(sw)}/{n}; identity->execute {len(i2e)}, execute->identity {len(e2i)}")
print(f"  identity->execute: mean A_true gain {da_i2e[0]:+.4f} (better {da_i2e[1]}/{da_i2e[2]})")
print(f"  execute->identity: mean A_true gain {da_e2i[0]:+.4f} (better {da_e2i[1]}/{da_e2i[2]})")

print("\n== A_hat vs A_true calibration (per executed day) ==")
for name, s in [("E0_w1", e0), ("E2_cavm_01", e2), ("E3_cavm_11", e3)]:
    ex = act(s)
    pairs = [(s["A_hat"][i], s["A_true"][i]) for i in ex if s["A_true"][i] == s["A_true"][i]]
    if not pairs:
        continue
    ah = [p[0] for p in pairs]; at = [p[1] for p in pairs]
    mh, mt = sum(ah)/len(ah), sum(at)/len(at)
    corr = sum((a-mh)*(b-mt) for a, b in zip(ah, at))
    den = (sum((a-mh)**2 for a in ah) * sum((b-mt)**2 for b in at)) ** 0.5
    print(f"  {name}: A_hat={mh:+.4f} vs A_true={mt:+.4f} on {len(pairs)} exec days,"
          f" corr={corr/den if den else float('nan'):+.2f}")

# Tail-hour analysis needs hourly y (not in this JSON) — covered in docs report.
print("\n== Note ==")
print("  Tail-hour (top-10% price) MAE needs hourly x_final vs y; the S4 hourly"
      " arrays are not serialized here. Documented as future work in the report.")
