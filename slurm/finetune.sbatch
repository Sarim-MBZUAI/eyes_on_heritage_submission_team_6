#!/usr/bin/env bash
# Fine-tune an encoder's tail on scene pairs. Usage: sbatch slurm/finetune.sbatch <backbone spec> <out dir> [extra args]
#SBATCH --job-name=heritage-ft
#SBATCH --account=nils
#SBATCH --qos=guaranteed-nils
#SBATCH --gres=shard:6
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HUGGINGFACE_TOKEN=$(grep -m1 '^HUGGINGFACE_TOKEN=' ~/.env | cut -d= -f2- | tr -d "\"'\r")
PY=${PY:-$SLURM_SUBMIT_DIR/../work/env/bin/python}
spec=$1; out=$2; shift 2
$PY tools/finetune_pairs.py --model "$spec" --cache ../work/cache --annotations ../train_dataset/train_annotations.geojson --out "$out" "$@"
