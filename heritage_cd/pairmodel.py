"""The pair model as a reusable object: MLP on [z_a, z_b, z_b - z_a] of frozen per-year embeddings,
trained on year-of-change pairs (heritage_cd.pairs), scored per point by top-k pair aggregation."""
import json
import os

import numpy as np
import torch
import torch.nn as nn

from . import CLASSES, pairs

DEFAULTS = dict(hidden=256, dropout=0.3, lr=3e-4, wd=1e-2, epochs=60, patience=12, batch=256, topk=3,
                reverse=True, aug=False, min_quality=0.3)


def load_site_data(embeddings_path, cache_dir, aug=False, min_quality=0.3):
    """site -> dict(z (N,T,d) site-centred, za (N,T,A,d) or None, usable (T,), years (T,))"""
    E = np.load(embeddings_path)
    out = {}
    for key in E.files:
        if not key.endswith("/Z"):
            continue
        s = key[:-2]
        centre = E[f"{s}/Z_bg"].astype(np.float32).mean((0, 1))
        scenes = json.load(open(os.path.join(cache_dir, f"{s}_scenes.json")))
        years = np.load(os.path.join(cache_dir, f"{s}.npz"))["scene_years"]
        assert len(years) == len(scenes) == E[key].shape[1]
        out[s] = dict(z=E[key].astype(np.float32) - centre,
                      za=E[f"{s}/Z_aug"].astype(np.float32) - centre if aug else None,
                      usable=np.array([q["quality"] >= min_quality for q in scenes]), years=years)
    return out


class PairModel:
    def __init__(self, site_data, pos_of, seed=0, dev="cpu", **cfg):
        unknown = set(cfg) - set(DEFAULTS)
        assert not unknown, unknown
        self.cfg = {**DEFAULTS, **cfg}
        self.dev = dev
        self.sd_, self.pos_of, self.seed = site_data, pos_of, seed
        self.usable_of = {s: d["usable"] for s, d in site_data.items()}
        self.years_of = {s: d["years"] for s, d in site_data.items()}
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

    def _emb(self, s, p, i, which):
        d = self.sd_[s]
        return d["z"][p, i] if which is None else d["za"][p, i, which]

    def _n_aug(self, s):
        return self.sd_[s]["za"].shape[2]

    def _X(self, rows, variant=None):
        """variant None: plain; int: that augmentation on both sides; 'random': independent draws"""
        def pick(s):
            return self.rng.integers(self._n_aug(s)) if variant == "random" else variant
        X = np.stack([np.concatenate([a, b, b - a]) for a, b in
                      ((self._emb(s, p, i, pick(s)), self._emb(s, p, j, pick(s))) for s, p, i, j, _ in rows)])
        return torch.tensor(X, device=self.dev)

    def scores(self, pts):
        """per-point change scores; all pairs of all points in one batched forward per augmentation view"""
        c = self.cfg
        rows, owner = [], []
        for n, (pid, site, _, _) in enumerate(pts):
            _, pos = self.pos_of[pid]
            pr = pairs.scene_pairs(self.usable_of[site])
            rows += [(site, pos, i, j, 0) for i, j in pr]
            owner += [n] * len(pr)
        owner = np.array(owner)
        n_aug = min(self._n_aug(s) for s in {r[0] for r in rows}) if c["aug"] else 0
        self.model.eval()
        with torch.no_grad():
            P = np.mean([torch.softmax(self.model((self._X(rows, v) - self.mu) / self.sd), 1).cpu().numpy()
                         for v in [None] + list(range(n_aug))], 0)
        return np.stack([pairs.aggregate(P[owner == n], c["topk"]) for n in range(len(pts))])

    def fit(self, fit_pts, val_pts, verbose=False):
        c = self.cfg
        rows, skipped = pairs.make_pairs(fit_pts, self.pos_of, self.usable_of, self.years_of, c["reverse"])
        y = torch.tensor([r[4] for r in rows], device=self.dev)
        X0 = self._X(rows)
        self.mu, self.sd = X0.mean(0), X0.std(0) + 1e-6
        Xplain = (X0 - self.mu) / self.sd
        w, counts = pairs.class_weights(y.cpu().numpy())
        yv = np.array([p[2] for p in val_pts])
        if verbose:
            print(f"fit points {len(fit_pts)} (skipped {skipped})  val {len(val_pts)}  pairs {len(y)}  "
                  f"per class {dict(zip(CLASSES, counts.tolist()))}")
        self.model = self._mlp(X0.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=c["lr"], weight_decay=c["wd"])
        lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=self.dev))
        best, best_state, best_ep, self.t = -1, None, 0, 0.5
        for ep in range(1, c["epochs"] + 1):
            X = (self._X(rows, "random") - self.mu) / self.sd if c["aug"] else Xplain
            self.model.train()
            perm = torch.randperm(len(y), device=self.dev)
            tl = 0.0
            for i in range(0, len(y), c["batch"]):
                b = perm[i:i + c["batch"]]
                opt.zero_grad()
                loss = lossf(self.model(X[b]), y[b])
                loss.backward()
                opt.step()
                tl += loss.item() * len(b)
            t, vf1 = pairs.tune_threshold(self.scores(val_pts), yv)
            if verbose and (ep % 5 == 0 or ep == 1):
                print(f"  seed {self.seed} ep {ep:3d}  pair train loss {tl / len(y):.3f}  val point macro-F1 {vf1:.3f} (t={t:.2f})")
            if vf1 > best:
                best, best_ep, self.t = vf1, ep, t
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            if ep - best_ep >= c["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.best_val, self.best_epoch = best, best_ep
        return self

    def predict(self, pts):
        return pairs.decide(self.scores(pts), self.t)

    def _mlp(self, d_in):
        c = self.cfg
        return nn.Sequential(nn.Dropout(c["dropout"]), nn.Linear(d_in, c["hidden"]), nn.GELU(),
                             nn.Dropout(c["dropout"]), nn.Linear(c["hidden"], len(CLASSES))).to(self.dev)

    def state(self):
        return dict(state=self.model.state_dict(), mu=self.mu, sd=self.sd, cfg=self.cfg)

    @classmethod
    def load(cls, ck, site_data, pos_of, dev="cpu"):
        m = cls(site_data, pos_of, dev=dev, **ck["cfg"])
        m.model = m._mlp(ck["state"]["1.weight"].shape[1])
        m.model.load_state_dict(ck["state"])
        m.mu, m.sd = ck["mu"].to(dev), ck["sd"].to(dev)
        return m


def fit_ensemble(site_data, pos_of, tr_pts, seeds=3, val_frac=0.2, verbose=False, dev="cpu", **cfg):
    """seeds models, each with its own stratified val split; returns (mean scores fn, mean threshold)"""
    models = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        fit, val = pairs.stratified_val(tr_pts, val_frac, rng)
        m = PairModel(site_data, pos_of, seed=seed, dev=dev, **cfg).fit([tr_pts[n] for n in fit], [tr_pts[n] for n in val], verbose)
        if verbose:
            print(f"  seed {seed}: best epoch {m.best_epoch} (val macro-F1 {m.best_val:.3f}, t={m.t:.2f})")
        models.append(m)
    t = float(np.mean([m.t for m in models]))
    return (lambda pts: np.mean([m.scores(pts) for m in models], 0)), t, models


def oof_scores(site_data, pos_of, pts, folds=5, seed=0, dev="cpu", **cfg):
    """pair-model change scores for every point, each from a model that never saw it (site x class folds)"""
    S = np.zeros((len(pts), len(CLASSES)))
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(pts), int)
    for key in {(p[1], p[2]) for p in pts}:
        idx = rng.permutation([n for n, p in enumerate(pts) if (p[1], p[2]) == key])
        fold[idx] = np.arange(len(idx)) % folds
    for k in range(folds):
        score, _, _ = fit_ensemble(site_data, pos_of, [p for n, p in enumerate(pts) if fold[n] != k], seeds=1, dev=dev, **cfg)
        S[fold == k] = score([p for n, p in enumerate(pts) if fold[n] == k])
    return S
