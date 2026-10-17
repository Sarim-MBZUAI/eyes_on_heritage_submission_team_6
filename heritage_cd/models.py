"""Small tabular classifiers. Shallow and heavily regularised on purpose: with ~800 labels from five
sites, the risk is fitting a site's look, not the change."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import CLASSES

K = len(CLASSES)


MAKE = {
    "logreg": lambda seed: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
    "extratrees": lambda seed: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, max_features=0.3,
                                                    class_weight="balanced", random_state=seed, n_jobs=8),
    "lgbm": lambda seed: LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=8,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0,
                                        class_weight="balanced", random_state=seed, verbose=-1),
}


def design(F, E, pos_of, pts, blocks=("X", "Xs", "E")):
    """one row per point: hand-made feature blocks from F (features.npz) and PCA'd encoder block E"""
    return np.nan_to_num(np.stack([np.concatenate([(E if b == "E" else F)[f"{s}/{b}"][pos_of[p][1]] for b in blocks])
                                   for p, s, _, _ in pts]))


def fit(names, Xtr, ytr, seeds=1):
    return [MAKE[n](k).fit(Xtr, ytr) for k in range(seeds) for n in names]


def predict_proba(fitted, Xte):
    """Average of class probabilities over fitted models (a class missing from training gets 0)."""
    P = np.zeros((len(Xte), K))
    for m in fitted:
        P[:, m.classes_] += m.predict_proba(Xte)
    return P / len(fitted)


def fit_predict_proba(names, Xtr, ytr, Xte, seeds=1):
    return predict_proba(fit(names, Xtr, ytr, seeds), Xte)
