#!/bin/bash
# Finetune Chat-Scene's released checkpoint on XR-QA + XR-SceneCaption train (LSceneLLM's
# "finetuned" Chat-Scene baseline; its recipe is unpublished, so this is a guess: Chat-Scene's
# own hyper-parameters from scripts/run.sh, starting from the released checkpoint), evaluating
# on val after every epoch. Then score with xr_eval/score.sh.
#   bash xr_eval/run_xr_finetune.sh
#   SELECT=big EPOCH=3 bash xr_eval/run_xr_finetune.sh
# Knobs: SELECT (trunc | rand | big, see build_xr.py), PROMPT (short | raw; XR-QA prompt),
#   EXTRA_TRAIN ("#"-joined extra train tags, e.g. ScanNet ones from scripts/config.py),
#   EPOCH (3), LR (5e-6), BATCH_SIZE (32 per GPU), MAX_TXT_LEN (256: captions are ~120 tokens),
#   OUT, NGPU, PREFIX_CACHE / GEN_BATCH / SHARED_PREFIX (eval speed-ups, as run_xr_eval.sh)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
ENV="${CHATSCENE_ENV:-.venv}"
export PYTHONPATH=.:${PYTHONPATH:-}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MASTER_PORT=$((54000 + RANDOM % 10000)) MASTER_ADDR=localhost
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
LLAMA="${LLAMA:-lmsys/vicuna-7b-v1.5}"
CKPT="${CKPT:-pretrained/ckpt_01_3446.pth}"
SELECT="${SELECT:-trunc}"
PROMPT="${PROMPT:-short}"
TRAIN_TAG="xrqa_train_${SELECT}_${PROMPT}#xrcap_train_${SELECT}${EXTRA_TRAIN:+#$EXTRA_TRAIN}"
VAL_TAG="${VAL_TAG:-xrqa_val_${SELECT}_${PROMPT}#xrcap_val_${SELECT}}"
CACHE_ARGS=()
[ "${PREFIX_CACHE:-0}" = 1 ] && CACHE_ARGS=(eval_scene_perm True eval_prefix_cache True eval_gen_batch "${GEN_BATCH:-16}")
[ "${PREFIX_CACHE:-0}" = 1 ] && [ "${SHARED_PREFIX:-1}" = 1 ] && CACHE_ARGS+=(eval_shared_prefix True)
OUT="${OUT:-outputs/xr_ft_${SELECT}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
$ENV/bin/python xr_eval/build_xr.py --xr_dir "${XR_DIR:?}" --scan_data "${XR_SCAN_DATA:?}" \
    --feats_dir "${XR_FEATS_DIR:-xr_eval/features}" --splits train val || exit 1
echo "output_dir=$OUT train_tag=$TRAIN_TAG val_tag=$VAL_TAG ckpt=$CKPT ngpu=$NGPU epoch=${EPOCH:-3} lr=${LR:-5e-6}"
# Model and optimisation options = scripts/run.sh (the released checkpoint's training run).
$ENV/bin/torchrun --nproc_per_node="$NGPU" tasks/train.py scripts/xr_config.py \
    output_dir "$OUT" evaluate False pretrained_path "$CKPT" train_tag "$TRAIN_TAG" val_tag "$VAL_TAG" \
    scheduler.epochs "${EPOCH:-3}" optimizer.lr "${LR:-5e-6}" optimizer.max_grad_norm 0.01 \
    optimizer.different_lr.enable False wandb.enable False gpu_num "$NGPU" do_save True \
    batch_size "${BATCH_SIZE:-32}" num_workers 4 \
    model.llama_model_path "$LLAMA" model.add_scene_token False model.add_img_token True \
    model.train_emb True model.train_img_proj True model.no_obj False model.input_dim 1024 \
    model.bidirection False model.max_obj_num 100 lora.lora_r 16 lora.lora_alpha 16 \
    model.add_pos_emb False model.feat_fusion False model.fuse_with_id False \
    model.use_location_token False model.max_txt_len "${MAX_TXT_LEN:-256}" seed 42 debug False "${CACHE_ARGS[@]}"
echo "CHATSCENE_EXIT=$?"
ls "$OUT"
echo CHATSCENE_DONE
