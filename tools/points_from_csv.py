"""The organisers' point list (location, site, lon, lat csv) as the GeoJSON the pipeline reads.

python tools/points_from_csv.py --csv test_points_ALL.csv --out test_points.geojson
"""
import argparse
import json

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

df = pd.read_csv(args.csv)
assert {"location", "site", "lon", "lat"} <= set(df.columns), df.columns
feats = [{"type": "Feature", "properties": {"location": r.location, "site": r.site},
          "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]}} for r in df.itertuples()]
json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
           "features": feats}, open(args.out, "w"))
print(f"{len(feats)} points -> {args.out}")
