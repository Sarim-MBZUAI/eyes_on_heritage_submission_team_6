"""Turn a scored submission csv over grid points into a GeoJSON of detected changes (class + confidence) for QGIS.

python tools/changes_geojson.py --predictions herat_scores.csv --points herat_grid.geojson --out herat_changes.geojson
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import CLASSES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--predictions", required=True)
ap.add_argument("--points", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-confidence", type=float, default=0.0, help="drop changes whose winning score is below this")
args = ap.parse_args()

g = json.load(open(args.points))
df = pd.read_csv(args.predictions).set_index("location")
cols = [f"{c}_score" for c in CLASSES]
feats = []
for f in g["features"]:
    s = df.loc[f["properties"]["location"], cols]
    c = CLASSES[int(s.to_numpy().argmax())]
    if c == "no_change" or s.max() < args.min_confidence:
        continue
    feats.append({"type": "Feature", "geometry": f["geometry"],
                  "properties": {"location": f["properties"]["location"], "class": c, "confidence": round(float(s.max()), 4),
                                 **{k: round(float(v), 4) for k, v in s.items()}}})
json.dump({"type": "FeatureCollection", "name": "detected_changes", "crs": g["crs"], "features": feats}, open(args.out, "w"))
print(f"{len(feats)} changes of {len(g['features'])} points -> {args.out}: " + ", ".join(
    f"{c} {n}" for c, n in pd.Series([x['properties']['class'] for x in feats]).value_counts().items()))
