"""Trees on hand-made + encoder features: train on splits/split_train.geojson, score on splits/split_test.geojson.

python tools/train_test.py --features work/features.npz --embeddings work/emb_dinov3sat_aug.npz --annotations train_dataset/train_annotations.geojson
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import models, pairs  # noqa: E402

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True, help="features.npz built from the full train_annotations.geojson")
ap.add_argument("--annotations", required=True, help="the original train_annotations.geojson (defines point_id)")
ap.add_argument("--train", default=os.path.join(root, "splits", "split_train.geojson"))
ap.add_argument("--test", default=os.path.join(root, "splits", "split_test.geojson"))
ap.add_argument("--embeddings", nargs="*", default=[], help="embeddings.npz files; their PCA'd features are appended")
ap.add_argument("--models", nargs="+", default=["lgbm", "logreg", "extratrees"])
ap.add_argument("--view", default="both", choices=["raw", "std", "both"])
ap.add_argument("--seeds", type=int, default=5)
args = ap.parse_args()

F = np.load(args.features)
E = [np.load(e) for e in args.embeddings]
parts = {"raw": ["X"], "std": ["Xs"], "both": ["X", "Xs"]}[args.view]
pos_of = pairs.positions(args.annotations)


def matrix(pts):
    X = np.stack([np.concatenate([F[f"{s}/{p}"][pos_of[pid][1]] for p in parts] + [e[f"{s}/E"][pos_of[pid][1]] for e in E])
                  for pid, s, _, _ in pts])
    y = np.array([c for _, _, c, _ in pts])
    assert (y == np.array([F[f"{s}/y"][pos_of[pid][1]] for pid, s, _, _ in pts])).all(), "split / feature mismatch"
    return np.nan_to_num(X), y


tr_pts, te_pts = pairs.load_points(args.train), pairs.load_points(args.test)
Xtr, ytr = matrix(tr_pts)
Xte, yte = matrix(te_pts)
P = models.fit_predict_proba(args.models, Xtr, ytr, Xte, seeds=args.seeds)
print(f"train {len(ytr)}  test {len(yte)}  | {'+'.join(args.models)}, {args.view}, {[os.path.basename(e) for e in args.embeddings]}")
pairs.report(yte, P.argmax(1), np.array([s for _, s, _, _ in te_pts]))
