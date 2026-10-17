#!/usr/bin/env bash
# Chips + hand-made features for an annotation file. Usage: sbatch slurm/prep.sbatch <imagery root> <annotations.geojson> <cache dir> <features.npz>
#SBATCH --job-name=heritage-prep
#SBATCH --account=nils
#SBATCH --qos=guaranteed-nils
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=${PY:-$SLURM_SUBMIT_DIR/../work/env/bin/python}
[ $# -eq 4 ] || { echo "need 4 args: imagery annotations cache features" >&2; exit 2; }
$PY tools/extract_chips.py --imagery "$1" --annotations "$2" --out "$3"
$PY tools/build_features.py --cache "$3" --out "$4"
