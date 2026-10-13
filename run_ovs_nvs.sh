#!/usr/bin/env bash
# Launch Platonic-Transformer novel view synthesis on an objascenes split.
#   bash run_ovs_nvs.sh smoke        # 5-minute pipeline check on 8 scenes (should overfit)
#   bash run_ovs_nvs.sh regression   # deterministic, LVSM-style
#   bash run_ovs_nvs.sh flow         # flow matching
#   bash run_ovs_nvs.sh ablation     # same as regression with the trivial group (= plain transformer)
# Extra flags are passed through, e.g.  bash run_ovs_nvs.sh regression --rot_aug --steps 200000
set -euo pipefail

TRAIN=${TRAIN:-/media/mina/hdd/amlab/nvsm/free512v16_dense/train/manifest.json}   # <- set these (or export TRAIN=... VAL=...)
VAL=${VAL:-/media/mina/hdd/amlab/nvsm/free512v16_dense/val/manifest.json}                                        # optional; empty = hold out 5% of train scenes
MODE=${1:-smoke}; shift || true
VAL_ARG=(); [ -n "$VAL" ] && VAL_ARG=(--val "$VAL")

# 12 layers, octahedral group: 24 frames x 48 ch = 1152 hidden (the paper's QM9 width), 48 heads (2 per frame)
MODEL=(--solid octahedron --layers 12 --dim_per_g 48 --heads_per_g 2 --patch_size 8)
COMMON=(--train "$TRAIN" "${VAL_ARG[@]}" --num_context 2 --num_target 1 --batch_size 16 --workers 8)

case "$MODE" in
  smoke)
    python mains/main_ovs_nvs_v3.py --train "$TRAIN" --val "$TRAIN" --scene_limit 8 --val_limit 8 \
      "${MODEL[@]}" --batch_size 8 --steps 1500 --warmup 100 --lr 1e-3 \
      --log_every 100 --eval_every 500 --out runs/smoke "$@" ;;
  regression)
    python mains/main_ovs_nvs_v3.py "${COMMON[@]}" "${MODEL[@]}" --mode regression \
      --steps 100000 --lr 4e-4 --out runs/nvs_reg_octa "$@" ;;
  flow)
    # flow matching needs clearly more steps than regression before samples get sharp
    python mains/main_ovs_nvs_v3.py "${COMMON[@]}" "${MODEL[@]}" --mode flow \
      --steps 300000 --lr 4e-4 --sample_steps 25 --out runs/nvs_flow_octa "$@" ;;
  ablation)
    # trivial group = plain transformer. Same width (1152), heads (48) and ~same compute per step;
    # it has many more parameters, because the equivariant model shares weights across its 24 frames.
    python mains/main_ovs_nvs_v3.py "${COMMON[@]}" --solid trivial --layers 12 --dim_per_g 1152 \
      --heads_per_g 48 --patch_size 8 --mode regression --steps 100000 --lr 4e-4 \
      --out runs/nvs_reg_trivial "$@" ;;
  *) echo "usage: bash run_ovs_nvs.sh {smoke|regression|flow|ablation} [extra flags]"; exit 1 ;;
esac
