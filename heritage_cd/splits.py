"""Validation splits that mimic the two hidden test sets.

in-domain  (T_in)  : whole spatial grid cells withheld, every site present in training
out-of-domain (T_out): one whole site withheld (leave-one-site-out), predictions pooled over sites
"""
import numpy as np

PIXEL_M = 0.5


def cell_ids(sites, rowcol, cell_m=100.0):
    """One id per (site, grid cell); the rubric bootstraps and withholds by these cells."""
    k = int(round(cell_m / PIXEL_M))
    rc = np.asarray(rowcol) // k
    return np.array([f"{s}:{r}:{c}" for s, (r, c) in zip(sites, rc)])


def blocked_folds(cells, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(cells)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % n_folds))
    f = np.array([fold_of[c] for c in cells])
    return [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(n_folds)]


def site_folds(sites):
    sites = np.asarray(sites)
    return [(np.flatnonzero(sites != s), np.flatnonzero(sites == s)) for s in np.unique(sites)]
