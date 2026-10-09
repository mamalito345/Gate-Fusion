# ============================================================
# CELL 20 — CONTROL EXPERIMENT: "Stage 2 without a gate" (gate weights FROZEN at 0)
#
# Question: the Stage 1 -> Stage 2 shift we observed (weaker branch up,
#           stronger branch down) -- does it come from the gate's cross term
#           (W @ f_other), or simply from 25 more epochs of head training
#           plus picking the best epoch on the validation set?
#
# Everything is IDENTICAL to Stage 2: start from the Stage 1 backbone + head,
# backbone frozen, same lr / warmup / epochs / batch size, same "best epoch by
# mean validation accuracy" rule. The ONLY difference: the gate weight is 0 and
# frozen  ->  forward = feat_own + 0 = feat_own.
#
# Prerequisite: Cell 3 (shared infrastructure) and Cell 16 (ARCH_CONFIGS,
# shared_val_loader) must have run in the SAME Colab session.
# Does NOT touch existing files; it writes new ones:
#   *_stage2_nogate_full.pt / *_stage2_nogate_predictions.npz
# ============================================================

CTRL_HP = {   # (batch, backbone_lr, gate_lr, warmup_frac) -- same values as the per-architecture loader cells (4/6/8/10/12/14)
    "r2plus1d_18": (64, 1e-4, 2e-3, 0.05),
    "r3d_18":      (64, 1e-4, 2e-3, 0.05),
    "mc3_18":      (64, 1e-4, 2e-3, 0.05),
    "s3d":         (32, 1e-4, 2e-3, 0.05),
    "mvit_v2_s":   (16, 5e-5, 2e-3, 0.10),
    "swin3d_t":    (24, 5e-5, 2e-3, 0.10),
}
CTRL_ORDER = ["r2plus1d_18", "mc3_18", "r3d_18", "s3d", "swin3d_t", "mvit_v2_s"]  # fast to slow
CTRL_LABEL = "stage2_nogate"
CTRL_SEED = 0

def run_nogate_control(name):
    out_path = f"{PRED_DIR}/{name}_{CTRL_LABEL}_predictions.npz"
    if os.path.exists(out_path):
        log(f"[{name}] control already exists, skipping: {out_path}")
        return out_path
    ck_ir_p, ck_dp_p = f"{CKPT_DIR}/{name}_stage1_ir.pt", f"{CKPT_DIR}/{name}_stage1_depth.pt"
    if not (os.path.exists(ck_ir_p) and os.path.exists(ck_dp_p)):
        log(f"[{name}] Stage 1 checkpoint missing, skipping.")
        return None

    ctor, weights, resize_to = ARCH_CONFIGS[name]
    bs, bb_lr, g_lr, wf = CTRL_HP[name]
    torch.manual_seed(CTRL_SEED); np.random.seed(CTRL_SEED)
    train_loader, train_eval_loader, val_loader = make_loaders(bs)

    model = GatedModel(ctor, weights, resize_to, N_CLASSES).to(DEVICE)
    ck_ir = torch.load(ck_ir_p, map_location="cpu")
    ck_dp = torch.load(ck_dp_p, map_location="cpu")
    model.backbone_ir.load_state_dict(ck_ir["backbone"]);   model.backbone_depth.load_state_dict(ck_dp["backbone"])
    model.head_ir.load_state_dict(ck_ir["head"]);           model.head_depth.load_state_dict(ck_dp["head"])

    # freeze the gate at zero (FeatureLinearGate already starts at zero; zero it explicitly anyway, then freeze)
    for g in (model.gate_to_ir, model.gate_to_depth):
        nn.init.zeros_(g.lin.weight)
        g.lin.weight.requires_grad = False

    model, best = train_gated_stage(
        model, train_loader, train_eval_loader, val_loader,
        CTRL_LABEL, name, freeze_backbone=True,
        backbone_lr=bb_lr, gate_lr=g_lr, warmup_frac=wf,
    )

    # safety check: did the gate really stay at zero? (if not, this is not a control experiment)
    mx = max(float(model.gate_to_ir.lin.weight.abs().max()), float(model.gate_to_depth.lin.weight.abs().max()))
    assert mx == 0.0, f"[{name}] gate weight drifted from zero (max|W|={mx}) -- control INVALID"

    dump_predictions_gated(model, shared_val_loader, out_path)
    log(f"[{name}] CONTROL DONE: best val mean={best:.4f}  gate max|W|={mx}")
    del model; gpu_cleanup()
    return out_path

stage("20 - CONTROL: Stage 2 without a gate (gate=0, frozen)")
for _name in CTRL_ORDER:
    run_nogate_control(_name)
print("\n✓ Cell 20 finished. Now run Cell 21 (analysis).")
