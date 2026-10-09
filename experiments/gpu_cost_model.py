"""GPU-hour model behind docs/research-plan.md. Nothing here has been measured; every number is an estimate.

One 'SRE' (single-stream run equivalent) = fine-tuning ONE model for 25 epochs on ONE sensor stream.
  A100-hours(SRE) = clips * epochs * 3 * GFLOPs_forward * 1.4 / (throughput GFLOP/s) / 3600
  3  = forward + backward cost relative to forward
  1.4 = extra for the per-epoch evaluation passes
  throughput = 15 TFLOP/s effective, fp32 (an assumption; mixed precision is usually 2-3x faster)
Gated system, in SRE: Stage 1 = 2.0 (two models), full 3-stage run = 2.86, 3-stage run + Controls A, B, C = 9.54.
"""
THROUGHPUT = 15e3          # GFLOP/s effective
EPOCHS = 25
MODELS = {                 # approximate forward GFLOPs for one 16-frame clip
    "XS (~4M params, ~3 GFLOPs)": 3,
    "S (~10M, ~18 GFLOPs)": 18,
    "M (~30M, ~45 GFLOPs)": 45,
    "L (~85M, ~180 GFLOPs)": 180,
}
DATA = {"D1  2.3k clips": 2303, "D2  6k": 6000, "D3  38k": 38000, "D4  114k": 114000, "D5  240k": 240000}

def hours(gf, clips, epochs=EPOCHS):
    return clips * epochs * 3 * gf * 1.4 / THROUGHPUT / 3600

FULL = 0.35/0.175          # 2.86 SRE
CTRL = (0.35+0.2+0.45+0.2+0.45+0.02)/0.175   # 9.54 SRE
R = 1/0.175                # one full gated 3-stage run, in SRE (= 5.71 with the stage-cost ratios of the notebook)
# (SRE, dataset key) for each work package, per model class
WP = {
 "WP0a controls A,B,C (5 seeds)":      (5*CTRL, "D1  2.3k clips", 1),
 "WP0b control D (3 seeds)":           (3*2*R*1.0, "D1  2.3k clips", 2),
 "WP1 RGB ladder (8 cond x 3 seeds)":  (8*3, "D2  6k", 1),
 "WP2 input bridge (2 mod x 5 x 3)":   (2*5*3, "D1  2.3k clips", 2),
 "WP2b modality pretraining":          (None, None, 3),
 "WP3 depth encoding + gate":          (5*3 + 3*R*1.0, "D1  2.3k clips", 1),
 "WP4 probes and similarity":          (1*R, "D1  2.3k clips", 2),
 "WP5a stop-gradient (3 seeds)":       (3*0.45*R, "D1  2.3k clips", 1),
 "WP5b,c variants and corruption":     ((4*3*(0.2+0.45) + 0.5)*R, "D1  2.3k clips", 2),
 "WP6 RGB teacher to student":         (2*3*0.25*R/1.0, "D2  6k", 2),
 "WP7 gate with RGB partner":          (3*3*R, "D2  6k", 3),
 "WP8 other CUHK-X sensor pairs":      (2*3*R*0.5, "D1  2.3k clips", 3),
}
def wp_hours(name, gf):
    sre, d, tier = WP[name]
    if name.startswith("WP2b"):   # 3 pretrainings x 100 epochs on D3, plus 18 fine-tunes on D1
        return 3*hours(gf, DATA["D3  38k"], 100) + 18*hours(gf, DATA["D1  2.3k clips"])
    return sre * hours(gf, DATA[d])

if __name__ == "__main__":
    print("A. one single-stream 25-epoch run (A100 hours, fp32 estimate)")
    print("model class".ljust(30) + "".join(k.rjust(16) for k in DATA))
    for m, gf in MODELS.items():
        print(m.ljust(30) + "".join(f"{hours(gf, c):16.2f}" for c in DATA.values()))
    print("\nB. per work package, per model class (A100 hours)")
    print("work package".ljust(40) + "".join(m.split()[0].rjust(9) for m in MODELS))
    tot = {1: {}, 2: {}, 3: {}}
    for n, (sre, d, t) in WP.items():
        row = [wp_hours(n, gf) for gf in MODELS.values()]
        for m, v in zip(MODELS, row): tot[t][m] = tot[t].get(m, 0) + v
        print(n.ljust(40) + "".join(f"{v:9.1f}" for v in row))
    print("\nC. totals per model class by tier (A100 hours)")
    for t in (1, 2, 3):
        print(f"tier {t}: " + ", ".join(f"{m.split()[0]} {v:.0f}" for m, v in tot[t].items()))
    print("all: " + ", ".join(f"{m.split()[0]} {sum(tot[t][m] for t in tot):.0f}" for m in MODELS))
    print("\nMultipliers: sensible range of the estimate is 0.65x (mixed precision, good data loading) to 1.7x (fp32, slow loading).")
