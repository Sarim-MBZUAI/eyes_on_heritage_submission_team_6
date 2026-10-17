"""Cut co-registered chip stacks for labelled points and random background points, one .npz per site.

python tools/extract_chips.py --imagery <train_dataset>/imagery --annotations <train_dataset>/train_annotations.geojson --out cache
python tools/extract_chips.py --imagery <new_site_imagery_root> --points test_points.geojson --site damascus --out cache_test
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--imagery", required=True, help="folder holding one sub-folder of yearly .tif per site")
ap.add_argument("--annotations", help="training geojson with site/class/year columns")
ap.add_argument("--points", help="unlabelled points to predict (geojson); needs --site")
ap.add_argument("--site", nargs="*", help="restrict to these sites")
ap.add_argument("--n-background", type=int, default=300)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

assert bool(args.annotations) != bool(args.points), "give --annotations (labelled) or --points (unlabelled)"
g_all = data.load_points(args.annotations or args.points)
sites = args.site or sorted(g_all.site.unique())

for site in sites:
    sdir = data.site_dir_for(args.imagery, site)
    g = g_all[g_all.site == site].reset_index(drop=True)
    scenes = data.site_scene_table(sdir)
    ref = int(np.argmax([s["quality"] for s in scenes]))
    rc, inside = data.points_to_pixels(g, sdir)
    assert inside.all(), f"{site}: {(~inside).sum()} points fall outside the imagery"
    chips, shifts, peaks = data.extract_stacks(sdir, rc, ref)
    bg_rc = data.sample_background(sdir, args.n_background, seed=zlib.crc32(site.encode()))
    bg_chips, _, _ = data.extract_stacks(sdir, bg_rc, ref)
    lab = g.drop(columns="geometry")
    np.savez_compressed(
        os.path.join(args.out, f"{site}.npz"),
        chips=chips, shifts=shifts, peaks=peaks, bg_chips=bg_chips,
        rowcol=rc, bg_rowcol=bg_rc,
        lon=g.to_crs(4326).geometry.x.values, lat=g.to_crs(4326).geometry.y.values,
        y=lab["y"].values if "y" in lab else np.full(len(g), -1),
        year=lab["year"].fillna(-1).values if "year" in lab else np.full(len(g), -1.0),
        scene_years=np.array([s["year"] for s in scenes]),
        ref_index=ref,
    )
    with open(os.path.join(args.out, f"{site}_scenes.json"), "w") as f:
        json.dump(scenes, f, indent=1)
    lab.to_csv(os.path.join(args.out, f"{site}_points.csv"), index=False)
    q = ", ".join(f"{s['year']}:{s['quality']:.2f}" for s in scenes)
    moved = (np.abs(shifts).max(-1) > 0).mean()
    print(f"{site}: {len(g)} points, {len(bg_rc)} background, ref {scenes[ref]['year']}, "
          f"{moved:.0%} chip-years re-aligned | quality {q}")
