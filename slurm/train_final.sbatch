#!/usr/bin/env bash
# Fit the submission model on all labelled points. Usage: sbatch slurm/train_final.sbatch <annotations.geojson> <features.npz> <cache dir> <embeddings.npz> <out.pt>
#SBATCH --job-name=heritage-final
#SBATCH --account=nils
#SBATCH --qos=guaranteed-nils
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=${PY:-$SLURM_SUBMIT_DIR/../work/env/bin/python}
[ $# -eq 5 ] || { echo "need 5 args: annotations features cache embeddings out" >&2; exit 2; }
$PY tools/train_final.py --annotations "$1" --features "$2" --cache "$3" --embeddings "$4" --out "$5"
