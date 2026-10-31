#!/bin/bash
# Chat-Scene's 2D token for the XR regions: render RGB-D views from the HM3D v0.2 glb, DINOv2-fuse
# them per object (xr_eval/render_videofeats.py). Needs xr_eval/data/region_offsets.json
# (register_regions.sh).
#   bash xr_eval/render_videofeats.sh                       # all XR regions, NGPU shards
#   REGIONS="00009-vLpv2VX547B_sub002" DUMP_DIR=/tmp/views bash xr_eval/render_videofeats.sh   # debug
# Paths come from env.sh (see env.sh.example): XR_DIR, XR_SCAN_DATA, HM3D_GLB, RENDER_ENV, DINOV2_HUB.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${RENDER_ENV:?}/bin/python"
[ -n "${DINOV2_HUB:-}" ] && export HF_HUB_CACHE="$DINOV2_HUB"
OUT="${XR_FEATS_DIR:-xr_eval/features}"
N="${NGPU:-2}"
EXTRA=()
[ -n "${REGIONS:-}" ] && EXTRA+=(--regions $REGIONS) && N=1 && OUT="${OUT}_debug"
[ -n "${DUMP_DIR:-}" ] && EXTRA+=(--dump_dir "$DUMP_DIR")
mkdir -p "$OUT"
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i EGL_DEVICE_ID=$i PYOPENGL_PLATFORM=egl $PY -u xr_eval/render_videofeats.py \
      --xr_dir "${XR_DIR:?}" --scan_data "${XR_SCAN_DATA:?}" --glb_root "${HM3D_GLB:?}" \
      --offsets "${REGION_OFFSETS:-xr_eval/data/region_offsets.json}" --out_dir "$OUT" --shard "$i/$N" "${EXTRA[@]}" &
done
wait
echo RENDER_DONE
