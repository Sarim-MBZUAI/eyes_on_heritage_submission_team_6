"""Hand-built change descriptors for one point, from its re-aligned yearly chip stack.

The idea that carries most of the weight: a real change is a *step* in time (the place looks one way
before year k and another way after), while shadow, haze, parallax and seasonal colour are *spikes*
that come and go. So besides comparing first vs last scene, we build the full year-by-year
dissimilarity matrix of the point and score how block-shaped it is.

Everything is computed from chip-normalised or chromaticity values so that a whole scene being darker,
hazier or warmer does not register as change. Scenes whose quality score is low (blurred, upsampled,
hazy) are skipped for that point.
"""
import numpy as np
from scipy import ndimage as ndi

C = 64  # chip centre

_yy, _xx = np.mgrid[:128, :128]
_rr = np.hypot(_yy - C + 0.5, _xx - C + 0.5)
DISK8, DISK16, DISK28 = _rr < 8, _rr < 16, _rr < 28
RING = (_rr >= 32) & (_rr < 60)

STATE_NAMES = ["bright", "exg", "sat", "grad16", "grad28", "std16", "dark16", "dark28", "white16", "edge_dir"]


def _prep(chip, mean, std):
    rgb = chip[:3].astype(np.float32)
    valid = chip[3] == 255
    assert valid.any()
    scene_z = (rgb - np.array(mean, np.float32)[:, None, None]) / np.array(std, np.float32)[:, None, None]
    g = rgb.mean(0)
    m, s = g[valid].mean(), g[valid].std() + 1e-3
    cz = (rgb - m) / s                         # chip-normalised: kills local haze / exposure
    gz = cz.mean(0)
    gx, gy = ndi.sobel(gz, 1), ndi.sobel(gz, 0)
    grad = np.hypot(gx, gy)
    tot = rgb.sum(0) + 1e-3
    chrom = rgb / tot                          # chromaticity: illumination-invariant colour
    return {"valid": valid, "sz": scene_z, "cz": cz, "gz": gz, "grad": grad, "gx": gx, "gy": gy, "chrom": chrom}


def _state(p):
    r, g, b = p["chrom"]
    exg = 2 * g - r - b
    sat = p["chrom"].max(0) - p["chrom"].min(0)
    gz, grad = p["gz"], p["grad"]
    # dominant-orientation strength: man-made edges are aligned, rubble and trees are not
    gx, gy = p["gx"][DISK28], p["gy"][DISK28]
    j = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])
    ev = np.linalg.eigvalsh(j)
    coh = (ev[1] - ev[0]) / (ev[1] + ev[0] + 1e-6)
    return np.array([
        p["sz"].mean(0)[DISK16].mean(),
        exg[DISK16].mean(),
        sat[DISK16].mean(),
        grad[DISK16].mean(),
        grad[DISK28].mean(),
        gz[DISK16].std(),
        (gz[DISK16] < -1.0).mean(),
        (gz[DISK28] < -1.0).mean(),
        (gz[DISK16] > 1.0).mean(),
        coh,
    ], np.float32)


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def _pair(pa, pb):
    d = np.abs(pb["cz"] - pa["cz"]).mean(0)
    d16, ring = d[DISK16].mean(), d[RING].mean()
    return {
        "d8": d[DISK8].mean(),
        "d16": d16,
        "loc": d16 / (ring + 1e-3),
        "corr16": _corr(pa["gz"][DISK16], pb["gz"][DISK16]),
        "corr28": _corr(pa["gz"][DISK28], pb["gz"][DISK28]),
        "gcorr": _corr(pa["grad"][DISK28], pb["grad"][DISK28]),
    }


PAIR_NAMES = ["d8", "d16", "loc", "corr16", "corr28", "gcorr"]


def _best_step(D):
    """Split k maximising mean across-split dissimilarity minus mean within-side dissimilarity."""
    t = len(D)
    best, kbest = -np.inf, 1
    for k in range(1, t):
        across = D[:k, k:].mean()
        w = [D[:k, :k][np.triu_indices(k, 1)], D[k:, k:][np.triu_indices(t - k, 1)]]
        w = np.concatenate(w)
        within = w.mean() if len(w) else D[np.triu_indices(t, 1)].mean()
        if across - within > best:
            best, kbest = across - within, k
    return best, kbest


def usable_scenes(stack, scenes, min_quality=0.3):
    """scenes of acceptable quality whose chip is mostly inside the imagery (points at the AOI edge lose
    a constant part of the disk in every year, which is harmless for year-to-year comparison)"""
    use = [t for t, s in enumerate(scenes) if s["quality"] >= min_quality and (stack[t, 3][DISK28] == 255).mean() > 0.5]
    assert len(use) >= 2, f"only {len(use)} usable scenes at this point"
    return use


def point_features(stack, scenes, min_quality=0.3):
    """stack: (T, 4, 128, 128) uint8; scenes: list of scene dicts. Returns (feature vector, names)."""
    use = usable_scenes(stack, scenes, min_quality)
    preps = [_prep(stack[t], scenes[t]["mean"], scenes[t]["std"]) for t in use]
    states = np.stack([_state(p) for p in preps])
    T = len(use)
    pairs = {}
    D = np.zeros((T, T))
    Dc = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            pr = _pair(preps[i], preps[j])
            pairs[i, j] = pr
            D[i, j] = D[j, i] = pr["d16"]
            Dc[i, j] = Dc[j, i] = 1 - pr["corr28"]

    f, names = [], []

    def add(vals, prefix, labels):
        f.extend(np.asarray(vals, np.float32).ravel().tolist())
        names.extend(f"{prefix}_{n}" for n in labels)

    # consecutive pairs: largest jump and how much it stands out from the typical year-to-year noise
    cons = [D[i, i + 1] for i in range(T - 1)]
    k = int(np.argmax(cons))
    add([max(cons), np.median(cons), max(cons) - np.median(cons)], "cons", ["max", "med", "excess"])
    add([pairs[k, k + 1][n] for n in PAIR_NAMES], "cons_at", PAIR_NAMES)
    add(states[k + 1] - states[k], "cons_dstate", STATE_NAMES)

    # first vs last usable scene
    add([pairs[0, T - 1][n] for n in PAIR_NAMES], "fl", PAIR_NAMES)
    add(states[-1] - states[0], "fl_dstate", STATE_NAMES)

    # step structure of the dissimilarity matrices
    s1, k1 = _best_step(D)
    s2, _ = _best_step(Dc)
    dstate = states[k1:].mean(0) - states[:k1].mean(0)
    add([s1, s2, D[np.triu_indices(T, 1)].mean(), np.linalg.norm(dstate / (states.std(0) + 1e-3))],
        "step", ["d16", "corr", "dmean", "zstate"])
    add(dstate, "step_dstate", STATE_NAMES)

    # what was there at the start and at the end
    add(states[0], "first", STATE_NAMES)
    add(states[-1], "last", STATE_NAMES)
    add([T], "n", ["scenes"])
    return np.array(f, np.float32), names


def stack_features(stacks, scenes, **kw):
    rows = [point_features(s, scenes, **kw) for s in stacks]
    return np.stack([r[0] for r in rows]), rows[0][1]


def site_standardise(X, X_bg):
    """Robust z-score of each feature against random unlabelled locations of the same site.

    Needs no labels, so it is computed the same way on a site the model has never seen.
    """
    med = np.median(X_bg, 0)
    iqr = np.percentile(X_bg, 75, 0) - np.percentile(X_bg, 25, 0)
    scale = np.where(iqr > 1e-6, iqr, X_bg.std(0) + 1e-6)
    return np.clip((X - med) / scale, -20, 20)
