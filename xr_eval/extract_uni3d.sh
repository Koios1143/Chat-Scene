#!/bin/bash
# Uni3D GT-object features of every SceneVerse HM3D region used by XR-QA / XR-SceneCaption.
#   bash xr_eval/extract_uni3d.sh            # NGPU shards in parallel
# Paths come from env.sh (see env.sh.example): XR_DIR, XR_SCAN_DATA, UNI3D_CKPT.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${EVAL_PY:-${CHATSCENE_ENV:-.venv}/bin/python}"
OUT="${XR_FEATS_DIR:-xr_eval/features}"
N="${NGPU:-2}"
mkdir -p "$OUT"
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i $PY -u xr_eval/extract_uni3d_regions.py --xr_dir "${XR_DIR:?}" \
      --scan_data "${XR_SCAN_DATA:?}" --ckpt "${UNI3D_CKPT:?}" --out_dir "$OUT" --shard "$i/$N" &
done
wait
echo UNI3D_DONE
