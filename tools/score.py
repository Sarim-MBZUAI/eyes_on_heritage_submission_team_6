"""Score a predictions csv against labelled points the way the rubric does: macro-F1 pooled over the
in_domain and out_of_domain points, S = their harmonic mean, bootstrap by spatial cell (B = 1000).
With --against, also the rubric's paired test of whether the first submission beats the second.

python tools/score.py --predictions a.csv --annotations test_annotations.geojson [--against b.csv]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import CLASSES, CLASS_TO_ID, metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--predictions", required=True)
ap.add_argument("--annotations", required=True, help="geojson with class and split (in_domain / out_of_domain)")
ap.add_argument("--against", help="second predictions csv, for the paired comparison")
ap.add_argument("--cell", type=float, default=100.0, help="bootstrap cell size in metres")
ap.add_argument("--B", type=int, default=1000)
args = ap.parse_args()

ann = json.load(open(args.annotations))["features"]
ids = [f["properties"]["point_id"] for f in ann]
y = np.array([CLASS_TO_ID[f["properties"]["class"]] for f in ann])
split = np.array([f["properties"]["split"] for f in ann])
lon, lat = np.array([f["geometry"]["coordinates"] for f in ann]).T
deg = args.cell / 111_320
cell = [f"{f['properties']['site']}/{int(la // deg)}/{int(lo * np.cos(np.radians(la)) // deg)}" for f, la, lo in zip(ann, lat, lon)]
cell = np.array(cell)


def preds(path):
    """argmax class of a submission csv (site, location, <class>_score x 4)"""
    df = pd.read_csv(path).set_index("location")
    assert set(ids) <= set(df.index), f"{path}: predictions missing for some points"
    return df.loc[ids, [f"{c}_score" for c in CLASSES]].to_numpy().argmax(1)


i, o = split == "in_domain", split == "out_of_domain"
assert i.sum() and o.sum() and (i | o).all(), np.unique(split)
p = preds(args.predictions)
r = metrics.score_report(y[i], p[i], cell[i], y[o], p[o], cell[o], B=args.B)
print(f"S = {r['S']:.3f}  (95% CI {r['S_ci'][0]:.3f}-{r['S_ci'][1]:.3f})   M_in = {r['M_in']:.3f} ({i.sum()} pts, "
      f"{len(set(cell[i]))} cells)   M_out = {r['M_out']:.3f} ({o.sum()} pts, {len(set(cell[o]))} cells)")
for k in ("in", "out"):
    print(f"  {k:<3} per-class F1 " + "  ".join(f"{c} {v:.2f}" for c, v in r[f"F1_{k}"].items()) +
          f"   confusion (rows = truth) {r[f'confusion_{k}']}")
if args.against:
    q = preds(args.against)
    r2 = metrics.score_report(y[i], q[i], cell[i], y[o], q[o], cell[o], B=args.B)
    d = metrics.bootstrap_s(y[i], p[i], cell[i], y[o], p[o], cell[o], B=args.B, p_in_b=q[i], p_out_b=q[o])
    ds = r["S"] - r2["S"]
    print(f"\nvs {args.against}: S {r2['S']:.3f}, dS = {ds:+.3f}, positive in {np.mean(d > 0):.0%} of paired resamples -> "
          + ("separated" if metrics.separated(ds, d) else "tied under the rubric"))
