"""Classify the points of a site (or several) with a model saved by tools/train_final.py.

Inputs come from the same pipeline as training, run on the new imagery with the training projection:
  tools/extract_chips.py --imagery <root> --annotations points.geojson --out cache_test
  tools/build_features.py --cache cache_test --out features_test.npz
  tools/embed_fm.py --cache cache_test --out emb_test.npz --model <same encoder> --aug --projection <training emb.npz>
python tools/predict_final.py --model work/final_model.pt --annotations points.geojson --features features_test.npz \\
    --embeddings emb_test.npz --cache cache_test --out predictions.csv
Writes the organisers' submission format, one row per point in the order of the geojson:
site, location, construction_score, destruction_score, soil_sealing_score, no_change_score
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import CLASSES, models, pairmodel, pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--annotations", required=True, help="geojson with a site and an id per point")
ap.add_argument("--id-field", default="location", help="property holding the point id")
ap.add_argument("--features", required=True)
ap.add_argument("--embeddings", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dev", default="cpu")
args = ap.parse_args()

ck = torch.load(args.model, map_location=args.dev, weights_only=False)
F, E = np.load(args.features), np.load(args.embeddings)
assert str(E["model"]) == ck["model"], (str(E["model"]), ck["model"])
pos_of = pairs.positions(args.annotations)
ann = json.load(open(args.annotations))["features"]
pts = [(pid, f["properties"]["site"], -1, None) for pid, f in enumerate(ann)]
site_data = pairmodel.load_site_data(args.embeddings, args.cache, aug=True)
S = np.mean([pairmodel.PairModel.load(c, site_data, pos_of, args.dev).scores(pts) for c in ck["pair"]], 0)
X = np.hstack([models.design(F, E, pos_of, pts), S])
assert X.shape[1] == ck["n_features"], (X.shape, ck["n_features"])
P = models.predict_proba(ck["trees"], X)
df = pd.DataFrame(P.round(4), columns=[f"{c}_score" for c in CLASSES])
df.insert(0, "location", [f["properties"][args.id_field] for f in ann])
df.insert(0, "site", [p[1] for p in pts])
assert df["location"].is_unique and not df.isna().any().any()
df.to_csv(args.out, index=False)
print(f"{len(df)} points -> {args.out}; argmax " + ", ".join(f"{c} {n}" for c, n in pd.Series(np.array(CLASSES)[P.argmax(1)]).value_counts().items()))
