#!/usr/bin/env bash
# 5-split evaluation (trees / pair model / stacked) on the GPU. Usage: sbatch slurm/eval.sbatch <embeddings.npz> <out.json> [extra args]
#SBATCH --job-name=heritage-eval
#SBATCH --account=nils
#SBATCH --qos=guaranteed-nils
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=${PY:-$SLURM_SUBMIT_DIR/../work/env/bin/python}
emb=$1; out=$2; shift 2
$PY tools/eval_splits.py --features ../work/features.npz --embeddings "$emb" --cache ../work/cache \
    --annotations ../train_dataset/train_annotations.geojson --splits 5 --aug --out "$out" "$@"
