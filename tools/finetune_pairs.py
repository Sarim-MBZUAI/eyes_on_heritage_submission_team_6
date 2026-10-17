"""Fine-tune the tail of a frozen encoder, siamese on scene pairs, with year-of-change supervision.

Same pair labels and point-level decision as tools/train_pairs.py, but the images go through the
encoder at every step, so augmentation is real (shared random flip/rotation per pair, independent
shift / gamma / colour / blur per side) and the last --blocks transformer blocks learn what change
looks like at 0.5 m. Everything before them stays frozen (no gradients, no activation memory).

GPU only; submit with slurm/finetune.sbatch.
python tools/finetune_pairs.py --model remoteclip:work/weights/RemoteCLIP-ViT-L-14.pt --cache work/cache \\
    --annotations train_dataset/train_annotations.geojson --out work/ft_remoteclip
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import CLASSES, backbones, pairs  # noqa: E402
from heritage_cd.images import blur, stretch  # noqa: E402

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="backbone spec, see heritage_cd/backbones.py")
ap.add_argument("--cache", required=True)
ap.add_argument("--annotations", required=True)
ap.add_argument("--train", default=os.path.join(root, "splits", "split_train.geojson"))
ap.add_argument("--test", default=os.path.join(root, "splits", "split_test.geojson"))
ap.add_argument("--out", required=True)
ap.add_argument("--blocks", type=int, default=4, help="trainable transformer blocks at the end of the encoder")
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=48, help="pairs per step (2 images each)")
ap.add_argument("--lr-backbone", type=float, default=2e-5)
ap.add_argument("--lr-head", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--topk", type=int, default=3)
ap.add_argument("--crop", type=int, default=96)
ap.add_argument("--min-quality", type=float, default=0.3)
ap.add_argument("--val-frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-reverse", action="store_true")
ap.add_argument("--limit-pairs", type=int, default=0, help="smoke test: train on this many random pairs only")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
assert torch.cuda.is_available(), "GPU only"
dev = "cuda"
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------- data
pos_of = pairs.positions(args.annotations)
tr_pts, te_pts = pairs.load_points(args.train), pairs.load_points(args.test)
chips, usable_of, years_of = {}, {}, {}
for site in sorted({s for _, s, _, _ in tr_pts + te_pts}):
    z = np.load(os.path.join(args.cache, f"{site}.npz"))
    chips[site] = torch.from_numpy(z["chips"][:, :, :3])  # (N, T, 3, 128, 128) uint8, kept on CPU
    years_of[site] = z["scene_years"]
    scenes = json.load(open(os.path.join(args.cache, f"{site}_scenes.json")))
    usable_of[site] = np.array([s["quality"] >= args.min_quality for s in scenes])

fit, val = pairs.stratified_val(tr_pts, args.val_frac, rng)
fit_pts, val_pts = [tr_pts[n] for n in fit], [tr_pts[n] for n in val]
rows, skipped = pairs.make_pairs(fit_pts, pos_of, usable_of, years_of, reverse=not args.no_reverse)
if args.limit_pairs:
    rows = [rows[i] for i in rng.permutation(len(rows))[:args.limit_pairs]]
labels = np.array([r[4] for r in rows])
w, counts = pairs.class_weights(labels)
print(f"fit points {len(fit_pts)} (skipped {skipped})  val {len(val_pts)}  test {len(te_pts)}  "
      f"pairs {len(rows)}  per class {dict(zip(CLASSES, counts.tolist()))}", flush=True)

# ---------------------------------------------------------------- model
bb = backbones.load(args.model, dev, half=False)
for p in (bb.m if hasattr(bb, "m") else bb.f).parameters():
    p.requires_grad_(False)
trainable = list(bb.blocks()[-args.blocks:].parameters()) + list(bb.final_norm().parameters())
for p in trainable:
    p.requires_grad_(True)
d = bb(torch.zeros(1, 3, bb.size, bb.size, device=dev)).shape[-1]
head = nn.Sequential(nn.Dropout(0.3), nn.Linear(3 * d, args.hidden), nn.GELU(),
                     nn.Dropout(0.3), nn.Linear(args.hidden, len(CLASSES))).to(dev)
opt = torch.optim.AdamW([{"params": trainable, "lr": args.lr_backbone},
                         {"params": head.parameters(), "lr": args.lr_head}], weight_decay=args.wd)
steps = args.epochs * (len(rows) // args.batch)
warm = max(1, steps // 10)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                          0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
print(f"encoder dim {d}, trainable encoder params {sum(p.numel() for p in trainable) / 1e6:.1f}M, "
      f"head {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M, {steps} steps", flush=True)


# ---------------------------------------------------------------- images
def prep(x, train):
    """x: (B, 3, 128, 128) uint8 on GPU -> (B, 3, size, size) float in [0, 1], centre crop with jitter"""
    B = len(x)
    x = x.float()
    o = (128 - args.crop) // 2
    if train:
        out = torch.empty(B, 3, args.crop, args.crop, device=x.device)
        dy, dx = rng.integers(-3, 4, size=(2, B))
        for b in range(B):
            out[b] = x[b, :, o + dy[b]:o + dy[b] + args.crop, o + dx[b]:o + dx[b] + args.crop]
        x = out
    else:
        x = x[:, :, o:o + args.crop, o:o + args.crop]
    x = stretch(x)
    if train:
        gamma = torch.tensor(rng.uniform(0.7, 1.4, B), device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
        gain = torch.tensor(rng.uniform(0.9, 1.1, (B, 3)), device=x.device, dtype=x.dtype).view(-1, 3, 1, 1)
        x = (x ** gamma * gain).clamp(0, 1)
        m = torch.tensor(rng.random(B) < 0.3, device=x.device)
        if m.any():
            x[m] = blur(x[m])
    return nn.functional.interpolate(x, size=bb.size, mode="bilinear", antialias=True)


def geom(x, k, flip):
    return torch.rot90(x.flip(3) if flip else x, k, (2, 3))


def embed(x):
    """(B, 3, size, size) -> (B, d) centre 4x4 of the feature map"""
    h = bb(x)
    c = h.shape[1] // 2
    return h[:, c - 2:c + 2, c - 2:c + 2].mean((1, 2)).float()


def logits(xa, xb):
    za, zb = embed(xa), embed(xb)
    return head(torch.cat([za, zb, zb - za], 1))


def gather(batch_rows):
    a = torch.stack([chips[s][p, i] for s, p, i, j, _ in batch_rows]).to(dev, non_blocking=True)
    b = torch.stack([chips[s][p, j] for s, p, i, j, _ in batch_rows]).to(dev, non_blocking=True)
    return a, b


@torch.no_grad()
def point_scores(pts, tta=False):
    bb_mode(False)
    S = np.zeros((len(pts), len(CLASSES)))
    for n, (pid, site, _, _) in enumerate(pts):
        _, pos = pos_of[pid]
        pr = pairs.scene_pairs(usable_of[site])
        a, b = gather([(site, pos, i, j, 0) for i, j in pr])
        xa, xb = prep(a, False), prep(b, False)
        views = [(k, f) for f in (False, True) for k in range(4)] if tta else [(0, False)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P = np.mean([torch.softmax(logits(geom(xa, k, f), geom(xb, k, f)), 1).float().cpu().numpy()
                         for k, f in views], 0)
        S[n] = pairs.aggregate(P, args.topk)
    return S


def bb_mode(train):
    head.train(train)
    for blk in bb.blocks()[-args.blocks:]:
        blk.train(train)


# ---------------------------------------------------------------- train
yv = np.array([c for _, _, c, _ in val_pts])
yte = np.array([c for _, _, c, _ in te_pts])
site_te = np.array([s for _, s, _, _ in te_pts])
best, best_t, best_ep, log = -1, 0.5, 0, []
t0 = time.time()
for ep in range(1, args.epochs + 1):
    bb_mode(True)
    perm = rng.permutation(len(rows))
    tl, nb = 0.0, 0
    for i in range(0, len(perm) - args.batch + 1, args.batch):
        br = [rows[j] for j in perm[i:i + args.batch]]
        a, b = gather(br)
        y = torch.tensor([r[4] for r in br], device=dev)
        k, f = int(rng.integers(4)), bool(rng.integers(2))  # one geometric transform for both sides
        xa, xb = geom(prep(a, True), k, f), geom(prep(b, True), k, f)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lossf(logits(xa, xb).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable + list(head.parameters()), 1.0)
        opt.step()
        sched.step()
        tl += loss.item()
        nb += 1
        if nb % 100 == 0:
            print(f"  ep {ep} step {nb}  loss {tl / nb:.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    Sv = point_scores(val_pts)
    t, vf1 = pairs.tune_threshold(Sv, yv)
    log.append({"epoch": ep, "train_loss": tl / nb, "val_macro_f1": vf1, "t": t})
    print(f"epoch {ep}  pair train loss {tl / nb:.3f}  val point macro-F1 {vf1:.3f} (t={t:.2f})  "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if vf1 > best:
        best, best_t, best_ep = vf1, t, ep
        enc = bb.m if hasattr(bb, "m") else bb.f
        torch.save({"encoder": {n: p.detach().cpu() for n, p in enc.named_parameters() if p.requires_grad},
                    "head": head.state_dict(), "args": vars(args), "t": t, "epoch": ep},
                   os.path.join(args.out, "best.pt"))

# ---------------------------------------------------------------- test with the best epoch
ck = torch.load(os.path.join(args.out, "best.pt"), map_location=dev)
enc = bb.m if hasattr(bb, "m") else bb.f
enc.load_state_dict({**enc.state_dict(), **{k: v.to(dev) for k, v in ck["encoder"].items()}})
head.load_state_dict(ck["head"])
print(f"\nbest epoch {best_ep} (val macro-F1 {best:.3f}, t={best_t:.2f})")
for tta in (False, True):
    St = point_scores(te_pts, tta=tta)
    p = pairs.decide(St, best_t)
    print(f"\n== TEST {'with 8-view TTA' if tta else 'single view'}:")
    pairs.report(yte, p, site_te)
    np.savez(os.path.join(args.out, f"test_scores{'_tta' if tta else ''}.npz"), S=St, y=yte, p=p, site=site_te, t=best_t)
json.dump({"log": log, "best_epoch": best_ep, "best_val": best, "t": best_t, "args": vars(args)},
          open(os.path.join(args.out, "log.json"), "w"), indent=1)
