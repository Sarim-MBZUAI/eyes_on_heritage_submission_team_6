"""Zoomed before / after chips of the strongest detections on an unlabelled site, with the predicted class, for the report.

python tools/detection_patches.py --cache work/herat/cache --points work/herat/grid.geojson --predictions work/herat/scores.csv --out report/figures/herat
"""
import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import CLASSES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--points", required=True)
ap.add_argument("--site", help="site to show when the point file covers several")
ap.add_argument("--predictions", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per-class", default="construction:5,destruction:3,soil_sealing:2")
ap.add_argument("--min-dist", type=float, default=80.0, help="metres between chosen points, to spread them over the site")
ap.add_argument("--box", type=int, default=40)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

pts = json.load(open(args.points))["features"]
site = args.site or pts[0]["properties"]["site"]
pts = [p for p in pts if p["properties"]["site"] == site]  # order = row order in the site's cache
df = pd.read_csv(args.predictions).set_index("location").loc[[p["properties"]["location"] for p in pts]]
S = df[[f"{c}_score" for c in CLASSES]].to_numpy()
z = np.load(os.path.join(args.cache, f"{site}.npz"))
chips, years = z["chips"], z["scene_years"]
assert len(chips) == len(pts), (len(chips), len(pts))
xy = z["rowcol"].astype(float) * 0.5  # metres
scenes = json.load(open(os.path.join(args.cache, f"{site}_scenes.json")))
usable = np.flatnonzero([s["quality"] >= 0.3 for s in scenes])
a, b = usable[0], usable[-1]


def rgb(chip):
    x = chip[:3].astype(np.float32)
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1).transpose(1, 2, 0)


chosen = []
for spec in args.per_class.split(","):
    c, k = spec.split(":")
    ci = CLASSES.index(c)
    for i in np.argsort(-S[:, ci]):
        if sum(cc == c for cc, _ in chosen) == int(k):
            break
        if S[i].argmax() != ci or any(np.hypot(*(xy[i] - xy[j])) < args.min_dist for _, j in chosen):
            continue
        chosen.append((c, i))
colour = {"construction": "lime", "destruction": "red", "soil_sealing": "cyan"}
h = args.box // 2
for n, (c, i) in enumerate(chosen, 1):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.4))
    for k, (t, lab) in enumerate(((a, "before"), (b, "after"))):
        ax[k].imshow(rgb(chips[i, t]))
        ax[k].add_patch(plt.Rectangle((64 - h, 64 - h), args.box, args.box, fill=False, color=colour[c], lw=2))
        ax[k].text(64 - h, 64 - h - 2, f"{c}  {S[i, CLASSES.index(c)]:.2f}", color="black", fontsize=8, fontweight="bold",
                   va="bottom", bbox=dict(facecolor=colour[c], edgecolor="none", pad=1.5))
        ax[k].set_title(f"{lab}: {years[t]}", fontsize=11)
        ax[k].axis("off")
    lon, lat = pts[i]["geometry"]["coordinates"]
    fig.suptitle(f"{site.capitalize()} test point, no label available: predicted {c}", fontsize=11)
    fig.text(0.5, 0.01, f"{pts[i]['properties']['location']} · {lat:.5f}, {lon:.5f} · 64 m × 64 m chip at 0.5 m · "
             + "  ".join(f"{cl} {S[i, j]:.2f}" for j, cl in enumerate(CLASSES)), ha="center", fontsize=7.5, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    name = f"{n:02d}_{c}_{pts[i]['properties']['location']}.png"
    fig.savefig(os.path.join(args.out, name), dpi=110)
    plt.close(fig)
    print(name, f"{S[i, CLASSES.index(c)]:.2f}")
