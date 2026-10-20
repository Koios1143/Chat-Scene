#!/bin/bash
# Chat-Scene inference on ScanRefer, Multi3DRefer, Scan2Cap, ScanQA (val), SQA3D (test) and
# Real-3DQA (torchrun over all visible GPUs), evaluate-only with the released checkpoint.
#   bash extra_eval/run_extra_eval.sh
#   VAL_TAG="scanqa#sqa3d_test" bash extra_eval/run_extra_eval.sh   # a subset
# Then score with extra_eval/score.sh (CPU only).
# Paths come from env.sh (see env.sh.example); relative ones are under the repo root:
#   CHATSCENE_ENV  venv from beacon3d/build_env.sh                    (default .venv)
#   LLAMA          Vicuna-7B v1.5, HF id or local path                (default lmsys/vicuna-7b-v1.5)
#   CKPT           Chat-Scene checkpoint from its HF release          (default pretrained/ckpt_01_3446.pth)
#   OFFICIAL_DIR   Chat-Scene's HF annotations/ dir                   (default annotations)
#   REAL3DQA_DIR   Real-3DQA <split>.jsonl files (HF Oliver-Ma/Real-3DQA)
#   OUT            output dir                                         (default outputs/extra_eval_<timestamp>)
#   NGPU           GPUs for torchrun                                  (default: all visible)
#   BATCH_SIZE     per-GPU batch size                                 (default 32)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
ENV="${CHATSCENE_ENV:-.venv}"
export PYTHONPATH=.:${PYTHONPATH:-}
# torch>=2.6 (the cu128 env) defaults torch.load to weights_only=True, which rejects the
# official checkpoint (it pickles its EasyDict config); restore the old default
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MASTER_PORT=$((54000 + RANDOM % 10000)) MASTER_ADDR=localhost
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
LLAMA="${LLAMA:-lmsys/vicuna-7b-v1.5}"
CKPT="${CKPT:-pretrained/ckpt_01_3446.pth}"
REAL3DQA="real3dqa_debiased_test#real3dqa_rotation_0#real3dqa_rotation_90#real3dqa_rotation_180#real3dqa_rotation_270"
VAL_TAG="${VAL_TAG:-scanrefer#multi3dref#scan2cap#scanqa#sqa3d_test#$REAL3DQA}"
OUT="${OUT:-outputs/extra_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
$ENV/bin/python extra_eval/build_annos.py --official_dir "${OFFICIAL_DIR:-annotations}" \
    --real3dqa_dir "${REAL3DQA_DIR:?set REAL3DQA_DIR}" || exit 1
echo "output_dir=$OUT val_tag=$VAL_TAG ckpt=$CKPT ngpu=$NGPU"
# Model options = scripts/run.sh (the released checkpoint's training run).
$ENV/bin/torchrun --nproc_per_node="$NGPU" tasks/train.py scripts/extra_eval_config.py \
    output_dir "$OUT" evaluate True pretrained_path "$CKPT" val_tag "$VAL_TAG" train_tag "" \
    wandb.enable False gpu_num "$NGPU" do_save False batch_size "${BATCH_SIZE:-32}" num_workers 4 \
    model.llama_model_path "$LLAMA" model.add_scene_token False model.add_img_token True \
    model.train_emb True model.train_img_proj True model.no_obj False model.input_dim 1024 \
    model.bidirection False model.max_obj_num 100 lora.lora_r 16 lora.lora_alpha 16 \
    model.add_pos_emb False model.feat_fusion False model.fuse_with_id False \
    model.use_location_token False seed 42 debug False
echo "CHATSCENE_EXIT=$?"
ls "$OUT"
echo CHATSCENE_DONE
