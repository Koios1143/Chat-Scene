#!/bin/bash
# Chat-Scene inference on the Beacon3D val sets (8 GPUs, torchrun), evaluate-only.
#   sbatch -A <account> -p <partition> beacon3d/run_beacon3d.sbatch
#   VAL_TAG="beacon3d_3rscan_grounding#beacon3d_3rscan_qa" sbatch ...   # a subset
# Overridable (paths relative to the repo root):
#   CHATSCENE_ENV  venv from build_env.sbatch                         (default .venv)
#   LLAMA          Vicuna-7B v1.5, HF id or local path                (default lmsys/vicuna-7b-v1.5)
#   CKPT           Chat-Scene checkpoint from its HF release          (default pretrained/ckpt_01_3446.pth)
#   OFFICIAL_DIR   Chat-Scene's HF annotations/ (the ScanNet .pt files) (default annotations)
#   FEATS_DIR      our Uni3D GT features                              (default beacon3d/features)
#   BEACON_ROOT    Beacon3D data                                      (default beacon3d/Beacon3D)
#   OUT            output dir                                         (default outputs/beacon3d_<timestamp>)
#SBATCH --job-name=chatscene-beacon3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --output=log/beacon3d_%j.out
#SBATCH --error=log/beacon3d_%j.err
set -uo pipefail
unset CONDA_PREFIX CONDA_DEFAULT_ENV
cd "${SLURM_SUBMIT_DIR:-.}"
ENV="${CHATSCENE_ENV:-.venv}"
export PYTHONPATH=.:${PYTHONPATH:-}
export MASTER_PORT=$((54000 + RANDOM % 10000)) MASTER_ADDR=localhost
LLAMA="${LLAMA:-lmsys/vicuna-7b-v1.5}"
CKPT="${CKPT:-pretrained/ckpt_01_3446.pth}"
VAL_TAG="${VAL_TAG:-beacon3d_scannet_grounding#beacon3d_scannet_qa#beacon3d_scannet_myfeat_grounding#beacon3d_scannet_myfeat_qa#beacon3d_scannet_no2d_grounding#beacon3d_scannet_no2d_qa#beacon3d_3rscan_grounding#beacon3d_3rscan_qa#beacon3d_multiscan_grounding#beacon3d_multiscan_qa}"
OUT="${OUT:-outputs/beacon3d_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
# Annotations + config (cheap, regenerated every run so they track the feature files).
$ENV/bin/python beacon3d/build_annos.py --beacon_root "${BEACON_ROOT:-beacon3d/Beacon3D}" \
    --official_dir "${OFFICIAL_DIR:-annotations}" \
    --uni3d_dir "${FEATS_DIR:-beacon3d/features}" || exit 1
echo "output_dir=$OUT val_tag=$VAL_TAG ckpt=$CKPT"
$ENV/bin/torchrun --nproc_per_node=8 tasks/train.py scripts/beacon3d_config.py \
    output_dir "$OUT" evaluate True pretrained_path "$CKPT" val_tag "$VAL_TAG" train_tag "" \
    wandb.enable False gpu_num 8 do_save False batch_size "${BATCH_SIZE:-32}" num_workers 4 \
    model.llama_model_path "$LLAMA" model.add_scene_token False model.add_img_token True \
    model.train_emb True model.train_img_proj True model.no_obj False model.input_dim 1024 \
    model.bidirection False model.max_obj_num 100 lora.lora_r 16 lora.lora_alpha 16 \
    model.add_pos_emb False model.feat_fusion False model.fuse_with_id False \
    model.use_location_token False seed 42 debug False
echo "CHATSCENE_EXIT=$?"
ls "$OUT"
echo CHATSCENE_DONE
