#!/bin/bash
#
# Flow matching on the LIBERO autoencoder latents, on the IvI cluster.
#   sbatch --export=ALL,WARM=0 --time=01:30:00 --mem=24G /home/mghadim/PlatonicTransformers/slurm/libero_fm.sbatch configs/nvs_fm_libero_smoke_any.yaml   (short test, no cache warm-up)
#   sbatch /home/mghadim/PlatonicTransformers/slurm/libero_fm.sbatch configs/nvs_fm_libero.yaml                                                       (full run)
#
# The CPU, memory and time requests below are ESTIMATES, not measurements.
#   memory 80G: about 57 GB of shard files kept in the node's file cache + about 15 GB for the training process and 8 data workers.
#   time 6 h: about 10 min warm-up + about 3 h of training (gravity ran about 6 steps/s on a 3090; LIBERO not measured)
#             + validation every 10 epochs + start-up, with margin.
# After a run read the real numbers with: sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,TotalCPU,State
#
#SBATCH --job-name=libero-fm
#SBATCH --partition=all
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/home/mghadim/logs/%x-%j.out

set -e
set -x

BASE=/home/mghadim
SHARDS=/ivi/zfs/s0/original_homes/mghadim/libero-256-shards
AE=$BASE/ae_libero_80k_trivial_bias_n256
CONFIG=${1:?usage: sbatch libero_fm.sbatch <config path relative to PlatonicTransformers>}

source /home/mghadim/miniconda3/etc/profile.d/conda.sh
conda activate /ivi/zfs/s0/original_homes/mghadim/envs/nvsfm
echo "conda env activated"

cd $BASE/PlatonicTransformers
export PYTHONPATH="$BASE/PlatonicTransformers:$BASE/PlatonicSplats${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS="ignore:The given NumPy array is not writable:UserWarning"

# fail early, with a clear message, if an input is missing
test -f "$SHARDS/index.json" || { echo "missing $SHARDS/index.json"; exit 1; }
test -f "$AE/config.yaml" || { echo "missing $AE/config.yaml"; exit 1; }
test -f "$BASE/PlatonicSplats/psplat/data/libero_shards.py" || { echo "PlatonicSplats has no libero_shards.py: switch it to the branch libero-loader (commit 58a994b)"; exit 1; }
test -f "$CONFIG" || { echo "missing config $CONFIG"; exit 1; }

hostname
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
git -C $BASE/PlatonicTransformers log -1 --oneline
git -C $BASE/PlatonicSplats log -1 --oneline

# Read every shard once, in sequence, so the node keeps them in its file cache.
# Random reads from the network disk are about 10x slower than from the cache (benchmark of 10 Nov:
# 34 against 332 samples/s with 8 workers). Skip it for short tests with WARM=0.
if [ "${WARM:-1}" = "1" ]; then
  echo "warming the file cache: $(date)"
  ls "$SHARDS"/shard_*.h5 | xargs -P 4 -I{} cat {} > /dev/null
  echo "warm-up done: $(date)"
  free -g
fi

python -u mains/main_nvs_fm_libero.py --config "$CONFIG"
