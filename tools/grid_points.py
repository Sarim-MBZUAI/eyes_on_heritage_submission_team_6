"""A regular grid of unlabelled points over a site's imagery, for dense change detection.

python tools/grid_points.py --imagery work/herat_imagery --site herat --stride 24 --out herat_grid.geojson
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.warp import transform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--imagery", required=True)
ap.add_argument("--site", required=True)
ap.add_argument("--stride", type=float, default=24.0, help="metres between points")
ap.add_argument("--margin", type=float, default=40.0, help="metres kept clear of the image edge")
ap.add_argument("--out", required=True)
args = ap.parse_args()

with rasterio.open(sorted(glob.glob(os.path.join(data.site_dir_for(args.imagery, args.site), "*.tif")))[0]) as r:
    b, crs = r.bounds, r.crs
xs = np.arange(b.left + args.margin, b.right - args.margin, args.stride)
ys = np.arange(b.bottom + args.margin, b.top - args.margin, args.stride)
X, Y = np.meshgrid(xs, ys)
lon, lat = transform(crs, "EPSG:4326", X.ravel().tolist(), Y.ravel().tolist())
feats = [{"type": "Feature", "properties": {"location": f"{args.site[0].upper()}{i:05d}", "site": args.site, "x": float(x), "y": float(y)},
          "geometry": {"type": "Point", "coordinates": [lo, la]}} for i, (x, y, lo, la) in enumerate(zip(X.ravel(), Y.ravel(), lon, lat))]
json.dump({"type": "FeatureCollection", "name": f"{args.site}_grid", "epsg": crs.to_epsg(),
           "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, "features": feats}, open(args.out, "w"))
print(f"{len(feats)} points ({len(xs)} x {len(ys)}, {args.stride} m) -> {args.out}")
