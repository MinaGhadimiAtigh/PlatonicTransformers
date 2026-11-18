#!/bin/bash
#
# Diagnose a trained LIBERO flow checkpoint on the IvI cluster (no training; a short job).
#   sbatch /home/mghadim/PlatonicTransformers/slurm/libero_diagnose.sbatch <config> <checkpoint> <label> [extra libero_diagnose.py options]
# example:
#   sbatch /home/mghadim/PlatonicTransformers/slurm/libero_diagnose.sbatch configs/nvs_fm_libero.yaml \
#     /home/mghadim/PlatonicTransformers/mains/logs/Platonic-NVS-FM/pke5ihgd/checkpoints/snapshot-epoch0499-step64000.ckpt baseline
#
# The CPU, memory and time requests below are ESTIMATES, not measurements:
#   memory 32G: the 0.9 GB checkpoint, the 3.6 GB autoencoder folder (0.8 GB of weights), a few hundred decoded scenes.
#   time 1 h: model loading a few minutes, about 160 scenes read from the shards, a few thousand network calls, 100 scenes rendered.
#   any GPU type is fine (the diagnoser does not compile the network); it needs about 5 GB of GPU memory.
# After the run read the real numbers with: sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State
#
#SBATCH --job-name=libero-diag
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/mghadim/logs/%x-%j.out

set -e
set -x

BASE=/home/mghadim
SHARDS=/ivi/zfs/s0/original_homes/mghadim/libero-256-shards
AE=$BASE/ae_libero_80k_trivial_bias_n256
CONFIG=${1:?usage: sbatch libero_diagnose.sbatch <config> <checkpoint> <label> [options]}
CKPT=${2:?missing checkpoint path}
LABEL=${3:?missing label}
shift 3

source /home/mghadim/miniconda3/etc/profile.d/conda.sh
conda activate /ivi/zfs/s0/original_homes/mghadim/envs/nvsfm

cd $BASE/PlatonicTransformers
export PYTHONPATH="$BASE/PlatonicTransformers:$BASE/PlatonicSplats${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS="ignore:The given NumPy array is not writable:UserWarning"

test -f "$SHARDS/index.json" || { echo "missing $SHARDS/index.json"; exit 1; }
test -f "$AE/config.yaml" || { echo "missing $AE/config.yaml"; exit 1; }
test -f "$CONFIG" || { echo "missing config $CONFIG"; exit 1; }
test -f "$CKPT" || { echo "missing checkpoint $CKPT"; exit 1; }
test -f slurm/libero_diagnose.py || { echo "missing slurm/libero_diagnose.py"; exit 1; }

hostname
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import transformers; print('transformers', transformers.__version__)" || echo "no transformers: the image metrics will use the low-level descriptor only"
git -C $BASE/PlatonicTransformers log -1 --oneline
git -C $BASE/PlatonicSplats log -1 --oneline

python -u slurm/libero_diagnose.py --config "$CONFIG" --checkpoint "$CKPT" --label "$LABEL" \
  --out-dir $BASE/PlatonicTransformers/diagnose_out "$@"
