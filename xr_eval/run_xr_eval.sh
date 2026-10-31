#!/bin/bash
# Chat-Scene zero-shot inference on XR-QA / XR-SceneCaption val (LSceneLLM, CVPR 2025), released
# checkpoint, evaluate only. Then score with xr_eval/score.sh (CPU only).
#   bash xr_eval/run_xr_eval.sh
#   VAL_TAG="xrqa_val_trunc_raw#xrcap_val_trunc" bash xr_eval/run_xr_eval.sh   # a subset
# Needs xr_eval/features/regions_{uni3d,videofeats}.*.pt (extract_uni3d.sh, render_videofeats.sh).
# val tags: xrqa_val_<select>_<prompt>[_no2d], xrcap_val_<select>[_no2d] with
#   select = trunc | rand | big (objects that get Chat-Scene's 100 identifiers, see build_xr.py)
#   prompt = raw | short (question as is / + ScanQA's "Answer the question using a single word or phrase.")
# Paths come from env.sh (see env.sh.example); the other knobs are as in extra_eval/run_extra_eval.sh:
#   OUT, NGPU, BATCH_SIZE, PREFIX_CACHE, GEN_BATCH, SHARED_PREFIX
#   MAX_TXT_LEN    max new tokens (default 64 = scripts/config.py)
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
VAL_TAG="${VAL_TAG:-xrqa_val_trunc_raw#xrqa_val_trunc_short#xrcap_val_trunc}"
CACHE_ARGS=()
[ "${PREFIX_CACHE:-0}" = 1 ] && CACHE_ARGS=(eval_scene_perm True eval_prefix_cache True eval_gen_batch "${GEN_BATCH:-16}")
[ "${PREFIX_CACHE:-0}" = 1 ] && [ "${SHARED_PREFIX:-1}" = 1 ] && CACHE_ARGS+=(eval_shared_prefix True)
OUT="${OUT:-outputs/xr_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
$ENV/bin/python xr_eval/build_xr.py --xr_dir "${XR_DIR:?}" --scan_data "${XR_SCAN_DATA:?}" \
    --feats_dir "${XR_FEATS_DIR:-xr_eval/features}" || exit 1
echo "output_dir=$OUT val_tag=$VAL_TAG ckpt=$CKPT ngpu=$NGPU max_txt_len=${MAX_TXT_LEN:-64} prefix_cache=${PREFIX_CACHE:-0}"
# Model options = scripts/run.sh (the released checkpoint's training run).
$ENV/bin/torchrun --nproc_per_node="$NGPU" tasks/train.py scripts/xr_config.py \
    output_dir "$OUT" evaluate True pretrained_path "$CKPT" val_tag "$VAL_TAG" train_tag "" \
    wandb.enable False gpu_num "$NGPU" do_save False batch_size "${BATCH_SIZE:-32}" num_workers 4 \
    model.llama_model_path "$LLAMA" model.add_scene_token False model.add_img_token True \
    model.train_emb True model.train_img_proj True model.no_obj False model.input_dim 1024 \
    model.bidirection False model.max_obj_num 100 lora.lora_r 16 lora.lora_alpha 16 \
    model.add_pos_emb False model.feat_fusion False model.fuse_with_id False \
    model.use_location_token False model.max_txt_len "${MAX_TXT_LEN:-64}" seed 42 debug False "${CACHE_ARGS[@]}"
echo "CHATSCENE_EXIT=$?"
ls "$OUT"
echo CHATSCENE_DONE
