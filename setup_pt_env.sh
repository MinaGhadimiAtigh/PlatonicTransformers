#!/usr/bin/env bash
# Environment for PlatonicTransformers flow matching (branch FM-qm9-gen), Python 3.12 + torch 2.4.1/cu124.
#
# Run from the repo root, with NO conda env active:
#     conda deactivate            # repeat until the prompt no longer shows (base)
#     bash setup_pt_env.sh
#
# Every install goes to venv/bin/python explicitly, so an active conda env can't hijack pip
# (that is what caused "No matching distribution found for torch==2.4.0": pip was running
# conda-base's Python 3.13, and torch 2.4 has no Python 3.13 builds).
set -euo pipefail

VENV=venv
PY="$VENV/bin/python"
TORCH_IDX=https://download.pytorch.org/whl/cu124
PYG_LINKS=https://data.pyg.org/whl/torch-2.4.0+cu124.html
FLASH_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

command -v uv >/dev/null || { echo "uv not found: install it with  curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# 1. fresh venv with a real Python 3.12 (uv downloads one if needed; never uses conda's)
rm -rf "$VENV"
uv venv --python 3.12 --python-preference only-managed "$VENV"
"$PY" --version

# 2. pin torch first; the constraints file stops any later package from upgrading it
cat > .pt_constraints.txt <<EOF
torch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1
numpy<2
EOF
uv pip install --python "$PY" torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url "$TORCH_IDX"

# 3. what the flow-matching path imports (no fairchem / dali / lightning needed for it)
uv pip install --python "$PY" -c .pt_constraints.txt \
    "numpy<2" pytorch-lightning torchmetrics ml-collections pyyaml wandb pandas psutil requests tqdm \
    matplotlib Pillow scikit-learn rdkit posebusters torch_geometric

# 4. optional compiled extras (the code runs without them; failures here are not fatal)
uv pip install --python "$PY" -c .pt_constraints.txt torch_cluster torch_scatter -f "$PYG_LINKS" \
    || echo "WARNING: torch_cluster/torch_scatter skipped (only needed for kNN/EdgeConv modes)"
uv pip install --python "$PY" "$FLASH_WHL" \
    || echo "WARNING: flash-attn skipped -> set model.attention_backend: scatter in the config"

# 5. check
"$PY" - <<'EOF'
import sys, torch, numpy, pytorch_lightning, ml_collections, torch_geometric, rdkit
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpu", torch.cuda.is_available(), "| numpy", numpy.__version__)
for mod in ("flash_attn", "torch_cluster"):
    try:
        __import__(mod); print(mod, "OK")
    except Exception as e:
        print(mod, "not available:", type(e).__name__)
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
print("PlatonicTransformer import OK")
EOF
rm -f .pt_constraints.txt
echo "Done. Activate with:  conda deactivate; source $VENV/bin/activate"
