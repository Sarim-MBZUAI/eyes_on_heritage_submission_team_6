"""Pair model on frozen per-year embeddings, trained with year-of-change supervision (heritage_cd.pairmodel).

Training examples are scene pairs (a, b) of a labelled point: a < Y <= b -> the point's class, otherwise
no_change; time reversal swaps construction/destruction. ~600 points become ~24k pairs. Inference uses no
year: every usable pair is scored and aggregated (top-k mean), with the change threshold tuned on validation.

python tools/train_pairs.py --embeddings work/emb_remoteclip_aug.npz --cache work/cache \\
    --annotations train_dataset/train_annotations.geojson --aug
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import metrics, pairmodel, pairs  # noqa: E402

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--embeddings", required=True)
ap.add_argument("--cache", required=True, help="chip cache folder (scene years and quality)")
ap.add_argument("--annotations", required=True)
ap.add_argument("--train", default=os.path.join(root, "splits", "split_train.geojson"))
ap.add_argument("--test", default=os.path.join(root, "splits", "split_test.geojson"))
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--aug", action="store_true", help="use Z_aug: random image-augmentation variant per pair side each epoch")
ap.add_argument("--no-reverse", action="store_true", help="disable time-reversal augmentation")
for k, v in pairmodel.DEFAULTS.items():
    if k not in ("aug", "reverse"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
args = ap.parse_args()

cfg = {k: getattr(args, k) for k in pairmodel.DEFAULTS if k not in ("aug", "reverse")}
cfg.update(aug=args.aug, reverse=not args.no_reverse)
site_data = pairmodel.load_site_data(args.embeddings, args.cache, aug=args.aug, min_quality=args.min_quality)
pos_of = pairs.positions(args.annotations)
tr_pts, te_pts = pairs.load_points(args.train), pairs.load_points(args.test)
yte = np.array([c for _, _, c, _ in te_pts])
site_te = np.array([s for _, s, _, _ in te_pts])

score, t, models = pairmodel.fit_ensemble(site_data, pos_of, tr_pts, seeds=args.seeds, verbose=True, dev=args.dev, **cfg)
per_seed = [metrics.macro_f1(yte, m.predict(te_pts)) for m in models]
print(f"\ntest macro-F1 per seed: {np.round(per_seed, 3).tolist()}  mean {np.mean(per_seed):.3f}")
print(f"ensemble of {args.seeds} seeds (t={t:.2f}): ", end="")
pairs.report(yte, pairs.decide(score(te_pts), t), site_te)
