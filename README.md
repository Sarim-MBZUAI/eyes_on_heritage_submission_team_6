# Eyes on Heritage — Team submission

Point-level classification of `construction`, `destruction`, `soil_sealing`, `no_change` from yearly 50 cm satellite scenes. This repository holds the training and inference code, the trained weights and the files we submitted. The method, the experiments and the numbers are in [METHOD.md](METHOD.md).

| what | where |
|---|---|
| trained model (stacked pair model + trees on frozen DINOv3-sat, fitted on all 776 training points) | `weights/final_model.pt` |
| PCA axes of the training embeddings, needed to embed a new site the same way | `weights/projection_dinov3sat.npz` |
| scored-task submission, 433 test points | `submission/team6_submission.csv` |
| Herat change layer for QGIS | `submission/team6_herat_changes.geojson` |
| code | `heritage_cd/` (library), `tools/` (one script per step), `slurm/` (batch wrappers we used) |

## 1. Setup

```bash
python3.13 -m venv env && env/bin/pip install -r requirements.txt
```

The encoder `facebook/dinov3-vitl16-pretrain-sat493m` is gated on Hugging Face: request access once, then export `HUGGINGFACE_TOKEN=<your token>` (the `slurm/*.sbatch` wrappers read it from `~/.env`). The embedding step needs a GPU (about 2 min for 433 points, 7 min for an 8 000-point grid on one shared GPU); everything else runs on CPU.

Imagery layout expected by every tool: one folder per site, named as in `heritage_cd/__init__.py` (`agadez-niger`, `aleppo-syria`, `derna-libya`, `marrakesh-morocco`, `timbuktu-mali`, `damascus-syria`, `herat-afghanistan`), holding one 4-band uint8 GeoTIFF per year named `<year>_<yyyymmdd>.tif`, projected CRS, 0.5 m. Symlinks are fine. Scenes in another format (3-band, geographic CRS, other names) go through `tools/prepare_scenes.py` first.

## 2. Inference on test points (how `submission/team6_submission.csv` was made)

```bash
# imagery root: one sub-folder per site (symlinks to the organisers' folders)
mkdir -p work/sub_imagery && ln -s <package>/02_imagery_damascus work/sub_imagery/damascus-syria
for d in <package>/03_imagery_seen_sites/*; do ln -s $d work/sub_imagery/$(basename $d); done

python tools/points_from_csv.py --csv <package>/01_test_points/test_points_ALL.csv --out work/test_points.geojson
python tools/extract_chips.py --imagery work/sub_imagery --annotations work/test_points.geojson --out work/cache_sub
python tools/build_features.py --cache work/cache_sub --out work/features_sub.npz
python tools/embed_fm.py --cache work/cache_sub --out work/emb_sub.npz --aug \
    --model hf:facebook/dinov3-vitl16-pretrain-sat493m --projection weights/projection_dinov3sat.npz      # GPU
python tools/predict_final.py --model weights/final_model.pt --annotations work/test_points.geojson \
    --features work/features_sub.npz --embeddings work/emb_sub.npz --cache work/cache_sub --out team6_submission.csv
```

The output has exactly the requested columns, one row per input point in input order: `site,location,construction_score,destruction_score,soil_sealing_score,no_change_score`, scores summing to 1. No labels from the new site are needed: site normalisation uses 300 random background locations drawn from the imagery.

## 3. Herat: dense change detection without a point list

```bash
mkdir -p work/herat/imagery && ln -s <package>/04_imagery_herat work/herat/imagery/herat-afghanistan
python tools/grid_points.py --imagery work/herat/imagery --site herat --stride 24 --out work/herat/grid.geojson
python tools/extract_chips.py --imagery work/herat/imagery --annotations work/herat/grid.geojson --out work/herat/cache
python tools/build_features.py --cache work/herat/cache --out work/herat/features.npz
python tools/embed_fm.py --cache work/herat/cache --out work/herat/emb.npz --aug \
    --model hf:facebook/dinov3-vitl16-pretrain-sat493m --projection weights/projection_dinov3sat.npz      # GPU
python tools/predict_final.py --model weights/final_model.pt --annotations work/herat/grid.geojson \
    --features work/herat/features.npz --embeddings work/herat/emb.npz --cache work/herat/cache --out work/herat/scores.csv
python tools/changes_geojson.py --predictions work/herat/scores.csv --points work/herat/grid.geojson \
    --out team6_herat_changes.geojson        # --min-confidence 0.6 for a stricter layer
```

Every grid point (24 m spacing) is scored like a test point; points whose winning class is not `no_change` are written as a WGS84 point feature with `class`, `confidence` (winning score) and the four scores.

## 4. Training from scratch

```bash
python tools/extract_chips.py --imagery train_dataset/imagery --annotations train_dataset/train_annotations.geojson --out work/cache
python tools/build_features.py --cache work/cache --out work/features.npz
python tools/embed_fm.py --cache work/cache --out work/emb_dinov3sat_aug.npz --aug --model hf:facebook/dinov3-vitl16-pretrain-sat493m   # GPU
python tools/train_final.py --annotations train_dataset/train_annotations.geojson --features work/features.npz \
    --cache work/cache --embeddings work/emb_dinov3sat_aug.npz --out weights/final_model.pt                         # GPU, ~3 min
python -c "import numpy as np; E=np.load('work/emb_dinov3sat_aug.npz'); np.savez('weights/projection_dinov3sat.npz', W_step=E['W_step'], W_fl=E['W_fl'], model=E['model'])"
```

`train_final.py` fits the three pair-model seeds, computes out-of-fold pair scores for every training point, and fits the tree ensemble on hand-made + encoder + pair-score features; the saved file carries everything `predict_final.py` needs. The `slurm/` wrappers (`prep`, `embed`, `train_final`, `eval`) run the same commands under SLURM; edit the account/QOS lines for another cluster.

## 5. Evaluation code

- `tools/eval_splits.py`: the five-split model selection reported in METHOD.md (trees / pair model / stacked, per-class F1, pooled confusion).
- `tools/score.py`: the organisers' metric on a labelled point file with a `split` column (macro-F1 in / out of domain, harmonic mean S, spatial-cell bootstrap, paired comparison of two submissions).
