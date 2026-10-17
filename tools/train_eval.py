"""Score feature sets and models on proxies of both hidden test conditions, with the rubric's S.

M_in  ~ 5-fold CV with 100 m grid cells withheld (all sites in training)
M_out ~ leave-one-site-out, predictions pooled over the five held-out sites
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import CLASSES, metrics, models, splits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True, help="features.npz from build_features.py")
ap.add_argument("--embeddings", help="optional embeddings.npz from embed_fm.py, appended as extra columns")
ap.add_argument("--models", nargs="+", default=["lgbm", "logreg", "extratrees"])
ap.add_argument("--view", default="std", choices=["raw", "std", "both"],
                help="raw features, site-standardised features, or both side by side")
ap.add_argument("--bootstrap", type=int, default=1000)
ap.add_argument("--report", help="write the full report as json here")
args = ap.parse_args()

F = np.load(args.features)
sites = sorted({k.split("/")[0] for k in F.files if "/" in k})


def view(site):
    if args.view == "raw":
        return F[f"{site}/X"]
    if args.view == "std":
        return F[f"{site}/Xs"]
    return np.hstack([F[f"{site}/X"], F[f"{site}/Xs"]])


X = np.vstack([view(s) for s in sites])
if args.embeddings:
    E = np.load(args.embeddings)
    X = np.hstack([X, np.vstack([E[f"{s}/E"] for s in sites])])
y = np.concatenate([F[f"{s}/y"] for s in sites])
site = np.concatenate([[s] * len(F[f"{s}/y"]) for s in sites])
cells = splits.cell_ids(site, np.vstack([F[f"{s}/rowcol"] for s in sites]))
X = np.nan_to_num(X)


def oof(folds):
    P = np.zeros((len(y), len(CLASSES)))
    for tr, te in folds:
        P[te] = models.fit_predict_proba(args.models, X[tr], y[tr], X[te])
    return P


P_in = oof(splits.blocked_folds(cells))
P_out = oof(splits.site_folds(site))
p_in, p_out = P_in.argmax(1), P_out.argmax(1)
rep = metrics.score_report(y, p_in, cells, y, p_out, cells, B=args.bootstrap)
rep["config"] = {"models": args.models, "view": args.view, "embeddings": args.embeddings, "n": int(len(y)),
                 "n_features": int(X.shape[1])}

print(f"S = {rep['S']:.3f}  [{rep['S_ci'][0]:.3f}, {rep['S_ci'][1]:.3f}]   "
      f"M_in = {rep['M_in']:.3f}   M_out = {rep['M_out']:.3f}   ({args.view}, {'+'.join(args.models)}"
      f"{', +emb' if args.embeddings else ''})")
print("  F1 in :", rep["F1_in"])
print("  F1 out:", rep["F1_out"])
print("  confusion out (rows = truth", CLASSES, "):")
for r in rep["confusion_out"]:
    print("   ", r)
if args.report:
    np.savez(args.report.replace(".json", "_oof.npz"), P_in=P_in, P_out=P_out, y=y, site=site, cells=cells)
    with open(args.report, "w") as f:
        json.dump(rep, f, indent=1)
