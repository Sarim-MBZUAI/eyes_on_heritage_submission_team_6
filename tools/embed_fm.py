"""Foundation-model change features: embed every year of every chip with a frozen encoder, then
describe the point by how its embedding moves over time.

Needs no GDAL, so it runs in any GPU env on the cached chips. Submit through SLURM
(slurm/embed.sbatch): bare GPU processes are reaped on this cluster.

python tools/embed_fm.py --cache cache --out embeddings.npz --model hf:facebook/dinov2-base
python tools/embed_fm.py --cache cache --out embeddings.npz --model satlas:weights/aerial_swinb_si.pth
python tools/embed_fm.py --cache cache --out embeddings.npz --model remoteclip:weights/RemoteCLIP-ViT-L-14.pt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heritage_cd import backbones  # noqa: E402
from heritage_cd.images import blur, stretch  # noqa: E402
from heritage_cd.features import _best_step, usable_scenes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="hf:facebook/dinov2-base", help="backbone spec, see heritage_cd/backbones.py")
ap.add_argument("--crop", type=int, default=96, help="centre crop in pixels (96 px = 48 m) before resizing")
ap.add_argument("--pca", type=int, default=24, help="dimensions kept for each embedding delta")
ap.add_argument("--min-quality", type=float, default=0.3)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--projection", help="embeddings.npz of the training sites: reuse its PCA axes (for a new site)")
ap.add_argument("--aug", action="store_true",
                help="also store every labelled chip-year under 12 image augmentations (Z_aug), for pair training")
args = ap.parse_args()

assert torch.cuda.is_available(), "run on a GPU node (sbatch slurm/embed.sbatch)"
dev = "cuda"
model = backbones.load(args.model, dev)


# dihedral group (rotation k, flip) and photometric variants; the first 4 are what the plain embedding averages
GEOM = [(k, f) for f in (False, True) for k in range(4)]
PHOTO = [lambda x: blur(x), lambda x: x ** 0.7, lambda x: x ** 1.4,
         lambda x: (x * torch.tensor([0.9, 1.05, 1.1], device=x.device).view(1, 3, 1, 1)).clamp(0, 1)]
N_AUG = len(GEOM) + len(PHOTO)


def centre(h):
    c = h.shape[1] // 2
    return h[:, c - 2:c + 2, c - 2:c + 2].mean((1, 2)).float()


def view(x, k, flip):
    """embedding of one geometric view, mapped back to the chip frame"""
    v = torch.rot90(x.flip(3) if flip else x, k, (2, 3))
    h = torch.rot90(model(v), -k, (1, 2))
    return centre(h.flip(2) if flip else h)


@torch.no_grad()
def embed(chips, aug=False):
    """chips: (N, 4, 128, 128) uint8 -> (N, d) centre embedding averaged over 4 rotations, and with aug
    also (N, N_AUG, d): one embedding per augmentation, each L2-normalised."""
    out, out_aug = [], []
    o = (128 - args.crop) // 2
    for i in range(0, len(chips), args.batch):
        x = stretch(torch.from_numpy(chips[i:i + args.batch, :3, o:o + args.crop, o:o + args.crop]).to(dev).float())
        x = torch.nn.functional.interpolate(x, size=model.size, mode="bilinear", antialias=True)
        views = [view(x, k, f) for k, f in (GEOM if aug else GEOM[:4])]
        out.append(torch.nn.functional.normalize(sum(views[:4]), dim=-1).cpu().numpy())
        if aug:
            views += [view(p(x), 0, False) for p in PHOTO]
            out_aug.append(torch.nn.functional.normalize(torch.stack(views, 1), dim=-1).cpu().numpy())
    return (np.concatenate(out), np.concatenate(out_aug)) if aug else np.concatenate(out)


def describe(stack_emb, chips, scenes):
    """stack_emb: (T, d) for one point -> scalars + (step delta, first-last delta)."""
    Z = stack_emb[usable_scenes(chips, scenes, args.min_quality)]
    D = 1 - Z @ Z.T
    s, k = _best_step(D)
    cons = [D[i, i + 1] for i in range(len(Z) - 1)]
    scal = [s, D[np.triu_indices(len(Z), 1)].mean(), max(cons), max(cons) - np.median(cons), D[0, -1]]
    return np.array(scal, np.float32), Z[k:].mean(0) - Z[:k].mean(0), Z[-1] - Z[0]


raw, stacks = {}, {}
for path in sorted(glob.glob(os.path.join(args.cache, "*.npz"))):
    site = os.path.basename(path)[:-4]
    z = np.load(path)
    with open(os.path.join(args.cache, f"{site}_scenes.json")) as f:
        scenes = json.load(f)
    stacks[f"{site}/usable"] = np.array([s["quality"] >= args.min_quality for s in scenes])
    for key in ("chips", "bg_chips"):
        C = z[key]
        n, t = C.shape[:2]
        if args.aug and key == "chips":
            e, ea = embed(C.reshape(n * t, *C.shape[2:]), aug=True)
            stacks[f"{site}/Z_aug"] = ea.reshape(n, t, N_AUG, -1).astype(np.float16)
            e = e.reshape(n, t, -1)
        else:
            e = embed(C.reshape(n * t, *C.shape[2:])).reshape(n, t, -1)
        rows = [describe(e[i], C[i], scenes) for i in range(n)]
        raw[site, key] = [np.stack([r[j] for r in rows]) for j in range(3)]
        # full per-year embeddings, for heads trained on top of the frozen encoder (tools/train_head.py)
        stacks[f"{site}/Z" if key == "chips" else f"{site}/Z_bg"] = e.astype(np.float16)
        print(site, key, e.shape, flush=True)

# deltas are high-dimensional; project them on principal axes of *background* deltas (unlabelled),
# after removing each site's background mean, so a new site is treated the same way at test time
out, proj = {}, {}
for j, name in ((1, "step"), (2, "fl")):
    bg = np.vstack([raw[s, "bg_chips"][j] - raw[s, "bg_chips"][j].mean(0) for s, k in raw if k == "bg_chips"])
    if args.projection:
        W = proj[f"W_{name}"] = np.load(args.projection)[f"W_{name}"]
    else:
        _, _, vt = np.linalg.svd(bg - bg.mean(0), full_matrices=False)
        W = proj[f"W_{name}"] = vt[:args.pca].T
    for s, k in raw:
        if k == "chips":
            out.setdefault(s, []).append((raw[s, k][j] - raw[s, "bg_chips"][j].mean(0)) @ W)
np.savez(args.out, **{f"{s}/E": np.hstack([raw[s, "chips"][0]] + v) for s, v in out.items()},
         model=args.model, **proj, **stacks)
print("saved", args.out, {s: np.hstack([raw[s, "chips"][0]] + v).shape for s, v in out.items()})
