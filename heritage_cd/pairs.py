"""Year-of-change supervision as scene pairs, and the point-level decision on top of pair scores.

A change point with year Y gives, for scenes a < b:  a < Y <= b -> its class, otherwise no_change.
A no_change point gives no_change for every pair. Reversing a construction pair gives a destruction
pair and vice versa; no_change is symmetric.
"""
import itertools
import json

import numpy as np

from . import CLASS_TO_ID, CLASSES, metrics

NC = CLASS_TO_ID["no_change"]
CON, DES = CLASS_TO_ID["construction"], CLASS_TO_ID["destruction"]
CHANGE = [i for i in range(len(CLASSES)) if i != NC]
REVERSED = {CON: DES, DES: CON, NC: NC}


def load_points(path):
    """[(point_id, site, class id, year or None)] from a split geojson"""
    out = []
    for f in json.load(open(path))["features"]:
        p = f["properties"]
        y = p["year"]
        out.append((p["point_id"], p["site"], CLASS_TO_ID[p["class"]], None if y is None else int(y)))
    return out


def positions(annotations_path):
    """point_id -> (site, row within that site's cached arrays)"""
    ann = json.load(open(annotations_path))["features"]
    seen, out = {}, {}
    for pid, f in enumerate(ann):
        s = f["properties"]["site"]
        out[pid] = (s, seen.get(s, 0))
        seen[s] = seen.get(s, 0) + 1
    return out


def scene_pairs(usable):
    return list(itertools.combinations(np.flatnonzero(usable), 2))


def make_pairs(pts, pos_of, usable_of, years_of, reverse=True):
    """training rows (site, pos, i, j, label); points whose change no pair straddles are skipped"""
    rows, skipped = [], 0
    for pid, site, c, year in pts:
        _, pos = pos_of[pid]
        years = years_of[site]
        pr = scene_pairs(usable_of[site])
        if c != NC:
            assert year is not None, pid
            if not any(years[i] < year <= years[j] for i, j in pr):
                skipped += 1
                continue
        for i, j in pr:
            lab = c if (c != NC and years[i] < year <= years[j]) else NC
            rows.append((site, pos, i, j, lab))
            if reverse and lab in REVERSED:
                rows.append((site, pos, j, i, REVERSED[lab]))
    return rows, skipped


def class_weights(labels):
    counts = np.bincount(labels, minlength=len(CLASSES))
    return (len(labels) / (len(CLASSES) * counts)) ** 0.5, counts


def aggregate(P, topk=3):
    """pair probabilities (n_pairs, K) of one point -> per-class score: top-k mean for change classes"""
    s = np.zeros(len(CLASSES))
    k = min(topk, len(P))
    for c in CHANGE:
        s[c] = np.sort(P[:, c])[-k:].mean()
    return s


def decide(S, t):
    best = np.array(CHANGE)[S[:, CHANGE].argmax(1)]
    return np.where(S[:, CHANGE].max(1) >= t, best, NC)


def tune_threshold(S, y):
    grid = np.linspace(0.05, 0.95, 19)
    f = [metrics.macro_f1(y, decide(S, t)) for t in grid]
    return float(grid[int(np.argmax(f))]), max(f)


def stratified_val(pts, frac, rng):
    strata = {}
    for n, (_, s, c, _) in enumerate(pts):
        strata.setdefault((s, c), []).append(n)
    val = sorted(sum((list(rng.permutation(v)[:max(1, round(frac * len(v)))]) for v in strata.values()), []))
    fit = [n for n in range(len(pts)) if n not in set(val)]
    return fit, val


def report(yte, p, site_te):
    print(f"macro-F1 = {metrics.macro_f1(yte, p):.3f}   accuracy = {(p == yte).mean():.3f}")
    for c, f in zip(CLASSES, metrics.f1_per_class(yte, p)):
        print(f"  F1 {c:<13} {f:.3f}   (n={np.sum(yte == CLASSES.index(c))})")
    print("confusion (rows = truth, cols = predicted):", CLASSES)
    for c, r in zip(CLASSES, metrics.confusion(yte, p)):
        print(f"  {c:<13}", r)
    # per-site accuracy only: a per-site macro-F1 would count classes absent from that site as F1 = 0
    print("accuracy per site:", {str(s): round(float((p[site_te == s] == yte[site_te == s]).mean()), 3)
                                 for s in np.unique(site_te)})
