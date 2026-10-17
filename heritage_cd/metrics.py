"""The ranking metric from Evaluation_Metric_Rubric.pdf, implemented exactly.

M     = unweighted mean of per-class F1 over the 4 classes (pooled predictions)
S     = 2 * M_in * M_out / (M_in + M_out)
CI    = 2.5 / 97.5 percentiles of S over B bootstrap resamples drawn by spatial cell
ties  = dS >= 0.02 and positive in >= 95% of paired resamples
"""
import numpy as np

from . import CLASSES

K = len(CLASSES)


def f1_per_class(y, p, k=K):
    y, p = np.asarray(y), np.asarray(p)
    out = np.zeros(k)
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        # a class with no true and no predicted examples counts as F1 = 0 here; the organisers pool
        # predictions precisely so that every class is present in each test set
        out[c] = 2 * tp / d if d else 0.0
    return out


def macro_f1(y, p, k=K):
    return float(f1_per_class(y, p, k).mean())


def harmonic(m_in, m_out):
    return 0.0 if m_in + m_out == 0 else 2 * m_in * m_out / (m_in + m_out)


def confusion(y, p, k=K):
    m = np.zeros((k, k), int)
    np.add.at(m, (np.asarray(y), np.asarray(p)), 1)
    return m


def _cell_index(cells):
    uniq, inv = np.unique(np.asarray(cells), return_inverse=True)
    return [np.flatnonzero(inv == i) for i in range(len(uniq))]


def bootstrap_s(y_in, p_in, cells_in, y_out, p_out, cells_out, B=1000, seed=0, p_in_b=None, p_out_b=None):
    """Spatial-cell bootstrap of S. With p_*_b also given, returns paired S differences (a - b)."""
    rng = np.random.default_rng(seed)
    gi, go = _cell_index(cells_in), _cell_index(cells_out)
    y_in, p_in, y_out, p_out = map(np.asarray, (y_in, p_in, y_out, p_out))
    s, d = np.empty(B), np.empty(B)
    for b in range(B):
        ii = np.concatenate([gi[j] for j in rng.integers(0, len(gi), len(gi))])
        oo = np.concatenate([go[j] for j in rng.integers(0, len(go), len(go))])
        s[b] = harmonic(macro_f1(y_in[ii], p_in[ii]), macro_f1(y_out[oo], p_out[oo]))
        if p_in_b is not None:
            sb = harmonic(macro_f1(y_in[ii], np.asarray(p_in_b)[ii]), macro_f1(y_out[oo], np.asarray(p_out_b)[oo]))
            d[b] = s[b] - sb
    if p_in_b is not None:
        return d
    return s


def score_report(y_in, p_in, cells_in, y_out, p_out, cells_out, B=1000):
    m_in, m_out = macro_f1(y_in, p_in), macro_f1(y_out, p_out)
    s = harmonic(m_in, m_out)
    bs = bootstrap_s(y_in, p_in, cells_in, y_out, p_out, cells_out, B=B)
    return {
        "S": s,
        "S_ci": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
        "M_in": m_in,
        "M_out": m_out,
        "F1_in": dict(zip(CLASSES, f1_per_class(y_in, p_in).round(3).tolist())),
        "F1_out": dict(zip(CLASSES, f1_per_class(y_out, p_out).round(3).tolist())),
        "confusion_in": confusion(y_in, p_in).tolist(),
        "confusion_out": confusion(y_out, p_out).tolist(),
    }


def separated(delta_s, paired_deltas):
    """Rubric section 6: A beats B only if dS >= 0.02 and dS > 0 in >= 95% of paired resamples."""
    return delta_s >= 0.02 and np.mean(np.asarray(paired_deltas) > 0) >= 0.95
