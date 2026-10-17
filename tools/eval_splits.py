"""Repeated random 80/20 splits of the 776 labelled points, same stratification as splits/, so model
choices rest on several test sets rather than one. Split 0 is the committed split; the rest are seeded.

Models (pair model on the GPU when available):
  trees    : LightGBM + logistic + extra-trees on hand-made features + PCA'd encoder features
  pairs    : pair model on frozen per-year embeddings (3 seeds)
  stacked  : trees on the same features + the pair model's change scores (out-of-fold on train points)

python tools/eval_splits.py --features work/features.npz --embeddings work/emb_remoteclip_aug.npz \\
    --cache work/cache --annotations train_dataset/train_annotations.geojson --splits 5
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import CLASSES, metrics, models, pairmodel, pairs  # noqa: E402

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--embeddings", required=True, help="embeddings.npz with Z stacks (and Z_aug for --aug)")
ap.add_argument("--cache", required=True)
ap.add_argument("--annotations", required=True)
ap.add_argument("--splits", type=int, default=5)
ap.add_argument("--aug", action="store_true")
ap.add_argument("--pair-seeds", type=int, default=3)
ap.add_argument("--oof-folds", type=int, default=5)
ap.add_argument("--tree-seeds", type=int, default=3)
ap.add_argument("--out", help="json with every split's numbers")
ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--encoder-only", action="store_true", help="trees on the PCA'd encoder features alone, no hand-made features")
ap.add_argument("--trees-only", action="store_true", help="skip the pair model and stacking")
args = ap.parse_args()

pos_of = pairs.positions(args.annotations)
ann = json.load(open(args.annotations))["features"]
all_pts = [(pid, f["properties"]["site"], pairs.CLASS_TO_ID[f["properties"]["class"]],
            None if f["properties"]["year"] is None else int(f["properties"]["year"])) for pid, f in enumerate(ann)]
F, E = np.load(args.features), np.load(args.embeddings)
site_data = None if args.trees_only else pairmodel.load_site_data(args.embeddings, args.cache, aug=args.aug)


def tree_matrix(pts):
    return models.design(F, E, pos_of, pts, ("E",) if args.encoder_only else ("X", "Xs", "E"))


def trees(Xtr, ytr, Xte):
    return models.fit_predict_proba(["lgbm", "logreg", "extratrees"], Xtr, ytr, Xte, seeds=args.tree_seeds).argmax(1)


def split(seed):
    if seed == 0:
        return pairs.load_points(os.path.join(root, "splits", "split_train.geojson")), \
            pairs.load_points(os.path.join(root, "splits", "split_test.geojson"))
    fit, te = pairs.stratified_val(all_pts, 0.2, np.random.default_rng(1000 + seed))
    return [all_pts[n] for n in fit], [all_pts[n] for n in te]


results = []
for k in range(args.splits):
    tr_pts, te_pts = split(k)
    ytr, yte = np.array([p[2] for p in tr_pts]), np.array([p[2] for p in te_pts])
    Xtr, Xte = tree_matrix(tr_pts), tree_matrix(te_pts)
    r = {"split": k, "n_test": len(te_pts)}
    pred = {"trees": trees(Xtr, ytr, Xte)}
    if not args.trees_only:
        score, t, _ = pairmodel.fit_ensemble(site_data, pos_of, tr_pts, seeds=args.pair_seeds, aug=args.aug, dev=args.dev)
        Ste = score(te_pts)
        pred["pairs"] = pairs.decide(Ste, t)
        Str = pairmodel.oof_scores(site_data, pos_of, tr_pts, args.oof_folds, seed=k, aug=args.aug, dev=args.dev)
        pred["stacked"] = trees(np.hstack([Xtr, Str]), ytr, np.hstack([Xte, Ste]))
    for m, p in pred.items():
        r[m] = metrics.macro_f1(yte, p)
        r[f"{m}_f1"] = metrics.f1_per_class(yte, p).tolist()
        r[f"{m}_pred"] = p.tolist()
    r["y"] = yte.tolist()
    r["site"] = [p[1] for p in te_pts]
    results.append(r)
    print(f"split {k}: " + "  ".join(f"{m} {r[m]:.3f}" for m in pred), flush=True)

print("\ntest macro-F1 over %d splits (mean +- sd):" % args.splits)
for m in [m for m in ("trees", "pairs", "stacked") if m in results[0]]:
    v = np.array([r[m] for r in results])
    print(f"  {m:<8} {v.mean():.3f} +- {v.std():.3f}   {np.round(v, 3).tolist()}")
    f1 = np.array([r[f"{m}_f1"] for r in results])
    print("           per-class F1 mean: " + "  ".join(f"{c} {f1[:, i].mean():.3f}+-{f1[:, i].std():.3f}" for i, c in enumerate(CLASSES)))
    y = np.concatenate([r["y"] for r in results])
    p = np.concatenate([r[f"{m}_pred"] for r in results])
    cm = metrics.confusion(y, p)
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    print("           pooled over splits: " + "  ".join(f"{c} P {prec[i]:.2f} R {rec[i]:.2f}" for i, c in enumerate(CLASSES)))
    print("           pooled confusion (rows = truth):", cm.tolist())
if args.out:
    json.dump(results, open(args.out, "w"), indent=1)
