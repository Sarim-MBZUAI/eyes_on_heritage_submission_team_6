#!/usr/bin/env bash
# Frozen-ViT embeddings of all cached chips. Usage: EXTRA=--aug sbatch --export=ALL,EXTRA slurm/embed.sbatch <cache> <out.npz> [backbone spec]
#SBATCH --job-name=heritage-embed
#SBATCH --account=nils
#SBATCH --qos=guaranteed-nils
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HUGGINGFACE_TOKEN=$(grep -m1 '^HUGGINGFACE_TOKEN=' ~/.env | cut -d= -f2- | tr -d "\"'\r")
PY=${PY:-$SLURM_SUBMIT_DIR/../work/env/bin/python}
$PY tools/embed_fm.py --cache "$1" --out "$2" --model "${3:-hf:facebook/dinov2-base}" ${EXTRA:-}
