"""Fit the submission model (stacked pair model + trees) on every labelled point, no held-out set.

Same recipe as the 'stacked' row of tools/eval_splits.py: pair-model ensemble on frozen per-year
embeddings, out-of-fold pair scores for the training points, trees on hand-made + PCA'd encoder
features + pair scores. Saves everything inference needs to --out (pair MLPs, threshold, tree models).

python tools/train_final.py --features work/features.npz --embeddings work/emb_dinov3sat_aug.npz \\
    --cache work/cache --annotations train_dataset/train_annotations.geojson --out work/final_model.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import CLASSES, models, pairmodel, pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--embeddings", required=True, help="embeddings.npz with Z and Z_aug stacks")
ap.add_argument("--cache", required=True)
ap.add_argument("--annotations", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pair-seeds", type=int, default=3)
ap.add_argument("--oof-folds", type=int, default=5)
ap.add_argument("--tree-seeds", type=int, default=3)
ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

pos_of = pairs.positions(args.annotations)
ann = json.load(open(args.annotations))["features"]
pts = [(pid, f["properties"]["site"], pairs.CLASS_TO_ID[f["properties"]["class"]],
        None if f["properties"]["year"] is None else int(f["properties"]["year"])) for pid, f in enumerate(ann)]
y = np.array([p[2] for p in pts])
print(f"{len(pts)} points, per class {dict(zip(CLASSES, np.bincount(y).tolist()))}", flush=True)
F, E = np.load(args.features), np.load(args.embeddings)
site_data = pairmodel.load_site_data(args.embeddings, args.cache, aug=True)

_, t, pair_models = pairmodel.fit_ensemble(site_data, pos_of, pts, seeds=args.pair_seeds, aug=True, dev=args.dev, verbose=True)
S = pairmodel.oof_scores(site_data, pos_of, pts, args.oof_folds, seed=0, aug=True, dev=args.dev)
X = np.hstack([models.design(F, E, pos_of, pts), S])
trees = models.fit(["lgbm", "logreg", "extratrees"], X, y, seeds=args.tree_seeds)
torch.save({"pair": [m.state() for m in pair_models], "t": t,
            "trees": trees, "n_features": X.shape[1], "embeddings": os.path.basename(args.embeddings), "model": str(E["model"])},
           args.out)
print(f"saved {args.out}: {len(pair_models)} pair models (t={t:.2f}), {len(trees)} tree models on {X.shape[1]} features")
