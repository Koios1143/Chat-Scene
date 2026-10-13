#!/bin/bash
# Chat-Scene runtime env (python 3.9, torch 2.2.1+cu121, transformers 4.39.3, peft 0.9.0,
# flash-attn 2.5.6 prebuilt wheel) as a uv venv, plus the extras the Beacon3D scripts need
# (open3d for the ScanNet extractor, openai for the QA evaluator).
#   sbatch -A <account> -p <partition> beacon3d/build_env.sbatch
# Needs a node with internet access. Overridable:
#   CHATSCENE_ENV   where the venv goes            (default .venv, under the repo root)
#   UV              the uv binary                  (default: uv on PATH)
#   UV_CACHE_DIR / TMPDIR   point these at a disk with room if $HOME is full
#SBATCH --job-name=chatscene-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=log/build_env_%j.out
#SBATCH --error=log/build_env_%j.err
set -euo pipefail
unset CONDA_PREFIX CONDA_DEFAULT_ENV
cd "${SLURM_SUBMIT_DIR:-.}"
UV="${UV:-uv}"
ENV="${CHATSCENE_ENV:-.venv}"
$UV python install 3.9
$UV venv "$ENV" -p 3.9 --seed
PIP="$UV pip install --python $ENV/bin/python"
$PIP torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
grep -v flash_attn requirements.txt > "${TMPDIR:-/tmp}/req_$$.txt"
$PIP -r "${TMPDIR:-/tmp}/req_$$.txt" pyyaml wandb tqdm timm plyfile open3d openai
$PIP "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
$ENV/bin/python -c "
import torch, transformers, peft, flash_attn
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('transformers', transformers.__version__, 'peft', peft.__version__, 'flash_attn', flash_attn.__version__)
import models.chat3d, dataset, tasks.shared_utils; print('chat-scene imports ok')
"
echo ENV_DONE
