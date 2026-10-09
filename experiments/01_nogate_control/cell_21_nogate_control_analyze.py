# ============================================================
# CELL 21 — CONTROL ANALYSIS: Stage 2 with gate  vs  Stage 2 without gate (same protocol)
# Prerequisite: Cell 16 (PRED_DIR, np, os) and Cell 20 must have run.
# "weak / strong branch" is defined by the Stage 1 accuracy (no gate, independent models).
# Differences are paired over the SAME 628 validation samples, paired bootstrap, 95% CI.
# ============================================================
import numpy as np, os

ARCHS = ["r2plus1d_18", "r3d_18", "mc3_18", "s3d", "mvit_v2_s", "swin3d_t"]

def _load(name, label):
    p = f"{PRED_DIR}/{name}_{label}_predictions.npz"
    if not os.path.exists(p): return None
    d = np.load(p); o = np.argsort(d["index"])
    return {"index": d["index"][o], "label": d["label"][o],
            "ir": d["probs_ir"][o].argmax(1), "dp": d["probs_depth"][o].argmax(1),
            "avg": ((d["probs_ir"][o] + d["probs_depth"][o]) / 2.0).argmax(1)}

_rng = np.random.default_rng(0)
def _paired(a_correct, b_correct, B=4000):
    """mean(a-b) in points + 95% bootstrap CI (paired over the same samples)."""
    diff = a_correct.astype(float) - b_correct.astype(float)
    n = len(diff)
    boots = diff[_rng.integers(0, n, size=(B, n))].mean(1) * 100
    return diff.mean() * 100, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

rows = []
print(f"{'arch':12s} | {'branch':7s} | {'S2 - S1':>8s} | {'S2nogate - S1':>14s} | {'S2 - S2nogate  [95% CI]':>28s}")
for name in ARCHS:
    s1, s2, nc = _load(name, "stage1"), _load(name, "stage2"), _load(name, "stage2_nogate")
    if s1 is None or s2 is None or nc is None:
        print(f"{name:12s} | (file missing, skipped)"); continue
    for d in (s2, nc):
        assert np.array_equal(d["index"], s1["index"]) and np.array_equal(d["label"], s1["label"]), "sample order mismatch"
    y = s1["label"]
    weak = "ir" if (s1["ir"] == y).mean() < (s1["dp"] == y).mean() else "dp"
    strong = "dp" if weak == "ir" else "ir"
    for tag, key in (("weak", weak), ("strong", strong), ("avg", "avg")):
        c1, c2, cn = (s1[key] == y), (s2[key] == y), (nc[key] == y)
        d21 = (c2.mean() - c1.mean()) * 100
        dn1 = (cn.mean() - c1.mean()) * 100
        m, lo, hi = _paired(c2, cn)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"{name:12s} | {tag:7s} | {d21:+8.2f} | {dn1:+14.2f} | {m:+8.2f}  [{lo:+6.2f}, {hi:+6.2f}] {sig}")
        rows.append((name, tag, d21, dn1, m, lo, hi))

print("\nSUMMARY (mean over the 6 architectures)  -- '*' = CI excludes zero")
for tag in ("weak", "strong", "avg"):
    r = [x for x in rows if x[1] == tag]
    if not r: continue
    print(f"  {tag:6s}: S2-S1 {np.mean([x[2] for x in r]):+.2f} | nogate-S1 {np.mean([x[3] for x in r]):+.2f} | "
          f"gate effect (S2 - nogate) {np.mean([x[4] for x in r]):+.2f}  "
          f"(CI excludes zero in {sum(1 for x in r if x[5] > 0 or x[6] < 0)}/{len(r)} architectures)")
print("\nReading: if the 'gate effect' column is ~0, the Stage 1 -> Stage 2 shift does not come from the gate; it comes from head retraining / epoch selection.")
print("         If 'nogate - S1' is also positive for the weak branch, most of the shift already happens without a gate.")
print("Caveat: one training run per configuration. The bootstrap covers validation-set sampling only, NOT training randomness.")
