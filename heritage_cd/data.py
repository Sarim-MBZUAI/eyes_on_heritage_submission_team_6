"""Reading scenes and cutting co-registered chip stacks around points.

Every site folder holds one 4-band uint8 GeoTIFF per year (RGB + alpha), 0.5 m, UTM, all on the
same grid, named `<year>_<yyyymmdd>.tif`. Scenes differ a lot in sun angle, off-nadir view, haze and
sharpness, and a building top can sit several pixels away from where it was the year before. So each
chip is re-aligned to a reference year with phase correlation before anything compares years.
"""
import glob
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi

from . import CLASS_TO_ID, SITE_DIRS

CHIP = 128      # 64 m at 0.5 m
MARGIN = 16     # max re-alignment shift, 8 m


def list_scenes(site_dir):
    files = sorted(glob.glob(os.path.join(site_dir, "*.tif")))
    if not files:
        raise FileNotFoundError(f"no .tif scenes in {site_dir}")
    return files


def load_points(path):
    """points with a `site` column; labelled ones also carry `class` (-> `y`) and `year`"""
    g = gpd.read_file(path)
    assert "site" in g, f"{path}: no `site` column"
    if "class" in g:
        g["y"] = g["class"].map(CLASS_TO_ID)
        assert g["y"].notna().all(), f"unknown classes: {set(g['class']) - set(CLASS_TO_ID)}"
    return g.reset_index(drop=True)


def scene_stats(path, step=4):
    """Per-scene radiometry (valid-pixel mean/std per band), sharpness and acquisition geometry."""
    with rasterio.open(path) as r:
        a = r.read(out_shape=(4, r.height // step, r.width // step))
        full_h, full_w = r.height, r.width
        # sharpness on a full-resolution central block: blurry or upsampled scenes have little
        # high-frequency energy whatever their content
        s = 1024
        win = Window(max(0, full_w // 2 - s // 2), max(0, full_h // 2 - s // 2), min(s, full_w), min(s, full_h))
        blk = r.read(window=win).astype(np.float32)
    valid = a[3] == 255
    rgb = a[:3].astype(np.float32)
    mean = [float(rgb[c][valid].mean()) for c in range(3)]
    std = [float(rgb[c][valid].std()) + 1e-3 for c in range(3)]
    v = ndi.binary_erosion(blk[3] == 255, iterations=2)
    assert v.sum() > 1000, f"{path}: central block has almost no valid pixels"
    sharp = float(ndi.laplace(blk[:3].mean(0))[v].var())
    name = os.path.basename(path)
    return {
        "file": name,
        "year": int(name[:4]),
        "date": name[5:13],
        "mean": mean,
        "std": std,
        "valid_frac": float(valid.mean()),
        "sharpness": sharp,
        "contrast": float(np.mean(std)),
    }


def site_scene_table(site_dir):
    stats = [scene_stats(f) for f in list_scenes(site_dir)]
    sh = np.array([s["sharpness"] for s in stats])
    ct = np.array([s["contrast"] for s in stats])
    for s in stats:
        # 1.0 = typical scene of this site; well below 1 = blurred, hazy or noisy-smoothed scene
        s["quality"] = float(min(s["sharpness"] / np.median(sh), 1.5) * min(s["contrast"] / np.median(ct), 1.5))
    return stats


def valid_footprint(site_dir, step=8):
    """Boolean mask (decimated) of pixels valid in every scene, plus the decimation step."""
    m = None
    for f in list_scenes(site_dir):
        with rasterio.open(f) as r:
            a = r.read(4, out_shape=(r.height // step, r.width // step)) == 255
        m = a if m is None else (m & a)
    return ndi.binary_erosion(m, iterations=CHIP // (2 * step) + 1), step


def sample_background(site_dir, n, seed=0):
    """Random unlabeled locations inside the valid footprint, in pixel coordinates (row, col)."""
    m, step = valid_footprint(site_dir)
    rr, cc = np.nonzero(m)
    rng = np.random.default_rng(seed)
    k = rng.choice(len(rr), size=min(n, len(rr)), replace=False)
    return np.stack([rr[k] * step + step // 2, cc[k] * step + step // 2], 1)


def points_to_pixels(gdf, site_dir):
    with rasterio.open(list_scenes(site_dir)[0]) as r:
        pts = gdf.to_crs(r.crs).geometry
        rc = np.array([r.index(p.x, p.y) for p in pts])
        inside = (rc[:, 0] >= 0) & (rc[:, 0] < r.height) & (rc[:, 1] >= 0) & (rc[:, 1] < r.width)
    return rc, inside


def _phase_shift(ref, mov, max_shift):
    """Integer (dy, dx) such that mov[y + dy, x + dx] ~ ref[y, x]; (0, 0) when the peak is weak."""
    w = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    a = np.fft.fft2((ref - ref.mean()) * w)
    b = np.fft.fft2((mov - mov.mean()) * w)
    r = b * np.conj(a)
    r /= np.abs(r) + 1e-9
    c = np.fft.ifft2(r).real
    c = np.fft.fftshift(c)
    py, px = np.unravel_index(np.argmax(c), c.shape)
    dy, dx = py - ref.shape[0] // 2, px - ref.shape[1] // 2
    peak = c[py, px] / (c.std() + 1e-9)
    if abs(dy) > max_shift or abs(dx) > max_shift or peak < 6:
        return 0, 0, float(peak)
    return int(dy), int(dx), float(peak)


def extract_stacks(site_dir, rowcols, ref_index, chip=CHIP, margin=MARGIN, align=True):
    """Chip stacks (N, T, 4, chip, chip) uint8, re-aligned to scene `ref_index`, and shifts (N, T, 2)."""
    files = list_scenes(site_dir)
    big = chip + 2 * margin
    srcs = [rasterio.open(f) for f in files]
    n, t = len(rowcols), len(files)
    out = np.zeros((n, t, 4, chip, chip), np.uint8)
    shifts = np.zeros((n, t, 2), np.int16)
    peaks = np.zeros((n, t), np.float32)
    try:
        for i, (r0, c0) in enumerate(rowcols):
            win = Window(int(c0) - big // 2, int(r0) - big // 2, big, big)
            stack = [s.read(window=win, boundless=True, fill_value=0) for s in srcs]
            gray = [a[:3].astype(np.float32).mean(0) for a in stack]
            ref = gray[ref_index][margin:margin + chip, margin:margin + chip]
            for j, a in enumerate(stack):
                dy = dx = 0
                if align and j != ref_index:
                    dy, dx, peaks[i, j] = _phase_shift(ref, gray[j][margin:margin + chip, margin:margin + chip], margin)
                out[i, j] = a[:, margin + dy:margin + dy + chip, margin + dx:margin + dx + chip]
                shifts[i, j] = (dy, dx)
    finally:
        for s in srcs:
            s.close()
    return out, shifts, peaks


def site_dir_for(imagery_root, site):
    d = os.path.join(imagery_root, SITE_DIRS[site])
    assert os.path.isdir(d), d
    return d
