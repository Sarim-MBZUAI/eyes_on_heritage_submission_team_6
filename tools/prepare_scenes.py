"""Bring scenes of a new site into the training format: <year>_<yyyymmdd>.tif, 4-band uint8 RGB + alpha,
projected CRS at 0.5 m. The acquisition date is read from the file name (yyyy-mm-dd or yyyymmdd).

python tools/prepare_scenes.py --epsg 32641 --out work/herat_imagery/herat-afghanistan <scene.tif ...>
"""
import argparse
import os
import re

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

ap = argparse.ArgumentParser()
ap.add_argument("scenes", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--epsg", type=int, required=True, help="projected CRS of the site, e.g. the UTM zone")
ap.add_argument("--res", type=float, default=0.5)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

for path in args.scenes:
    m = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", os.path.basename(path))
    assert m, f"no date in {path}"
    y, mo, d = m.groups()
    with rasterio.open(path) as src:
        assert src.count in (3, 4) and src.dtypes[0] == "uint8", (src.count, src.dtypes)
        dst_crs = f"EPSG:{args.epsg}"
        tr, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=args.res)
        out = np.zeros((4, h, w), np.uint8)
        rgb = src.read([1, 2, 3])
        alpha = src.read(4) if src.count == 4 else np.full(rgb.shape[1:], 255, np.uint8)
        for b in range(3):
            reproject(rgb[b], out[b], src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs=dst_crs,
                      resampling=Resampling.bilinear)
        reproject(alpha, out[3], src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs=dst_crs,
                  resampling=Resampling.nearest)
        dst = os.path.join(args.out, f"{y}_{y}{mo}{d}.tif")
        with rasterio.open(dst, "w", driver="GTiff", width=w, height=h, count=4, dtype="uint8", crs=dst_crs,
                           transform=tr, compress="deflate", tiled=True) as f:
            f.write(out)
    print(f"{os.path.basename(path)} -> {dst}  {w}x{h} px")
