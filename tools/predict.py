"""Train on every labelled training point and classify the points of a (possibly unseen) site.

python tools/extract_chips.py --imagery <root> --points test.geojson --site damascus --out cache_test
python tools/build_features.py --cache cache_test --out test_features.npz
python tools/predict.py --train features.npz --test test_features.npz --out predictions.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import CLASSES, models  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--test", required=True)
ap.add_argument("--train-emb")
ap.add_argument("--test-emb")
ap.add_argument("--models", nargs="+", default=["lgbm", "logreg", "extratrees"])
ap.add_argument("--view", default="both", choices=["raw", "std", "both"])
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--cache", help="test cache folder, to copy point ids/coordinates into the output")
ap.add_argument("--out", required=True)
args = ap.parse_args()


def load(fpath, epath):
    F = np.load(fpath)
    sites = sorted({k.split("/")[0] for k in F.files if "/" in k})
    parts = {"raw": ["X"], "std": ["Xs"], "both": ["X", "Xs"]}[args.view]
    X = np.vstack([np.hstack([F[f"{s}/{p}"] for p in parts]) for s in sites])
    if epath:
        E = np.load(epath)
        X = np.hstack([X, np.vstack([E[f"{s}/E"] for s in sites])])
    y = np.concatenate([F[f"{s}/y"] for s in sites])
    site = np.concatenate([[s] * len(F[f"{s}/y"]) for s in sites])
    return np.nan_to_num(X), y, site


Xtr, ytr, _ = load(args.train, args.train_emb)
Xte, _, site = load(args.test, args.test_emb)
P = models.fit_predict_proba(args.models, Xtr, ytr, Xte, seeds=args.seeds)
df = pd.DataFrame(P, columns=[f"p_{c}" for c in CLASSES])
df.insert(0, "class", np.array(CLASSES)[P.argmax(1)])
df.insert(0, "site", site)
if args.cache:
    meta = pd.concat([pd.read_csv(os.path.join(args.cache, f"{s}_points.csv")) for s in sorted(set(site))],
                     ignore_index=True)
    df = pd.concat([meta.drop(columns=[c for c in ("class", "y", "site") if c in meta]), df], axis=1)
df.to_csv(args.out, index=False)
print(df["class"].value_counts().to_string())
