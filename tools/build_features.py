"""Hand-built change features for every cached site (labelled and background points).

python tools/build_features.py --cache cache --out features.npz
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

out = {}
for path in sorted(glob.glob(os.path.join(args.cache, "*.npz"))):
    site = os.path.basename(path)[:-4]
    z = np.load(path)
    with open(os.path.join(args.cache, f"{site}_scenes.json")) as f:
        scenes = json.load(f)
    X, names = features.stack_features(z["chips"], scenes)
    Xbg, _ = features.stack_features(z["bg_chips"], scenes)
    out[f"{site}/X"] = X
    out[f"{site}/X_bg"] = Xbg
    out[f"{site}/Xs"] = features.site_standardise(X, Xbg)
    for k in ("y", "year", "rowcol"):
        out[f"{site}/{k}"] = z[k]
    print(site, X.shape, "background", Xbg.shape)
out["names"] = np.array(names)
np.savez(args.out, **out)
