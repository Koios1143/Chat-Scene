#!/bin/bash
# DINOv2 multi-view features (Chat-Scene's 2D token) for the Beacon3D 3RScan / MultiScan
# objects -> beacon3d/features/<domain>_gt_videofeats.pt. One GPU.
#   bash beacon3d/extract_videofeats.sh
# Paths come from env.sh (see env.sh.example):
#   RSCAN_RAW         raw 3RScan scans/ (<scan>/sequence.zip, labels.instances.annotated.v2.ply)
#   MULTISCAN_RAW     raw MultiScan scans/ (<scene>.zip)
#   MULTISCAN_SCAN_DATA  SceneVerse MultiScan scan_data/ (instance ids)
#   FEATS_DIR         feature dir with <domain>_scene_map.json (default beacon3d/features)
#   EVAL_PY           python with torch + transformers (default $CHATSCENE_ENV/bin/python); needs ffmpeg on PATH
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${EVAL_PY:-${CHATSCENE_ENV:-.venv}/bin/python}"
OUT="${FEATS_DIR:-beacon3d/features}"
$PY beacon3d/extract_videofeats_gt.py --domain 3rscan --raw_dir "${RSCAN_RAW:?set RSCAN_RAW}" --feats_dir "$OUT" || exit 1
$PY beacon3d/extract_videofeats_gt.py --domain multiscan --raw_dir "${MULTISCAN_RAW:?set MULTISCAN_RAW}" \
    --scan_data "${MULTISCAN_SCAN_DATA:?set MULTISCAN_SCAN_DATA}" --feats_dir "$OUT" || exit 1
echo VIDEOFEATS_DONE
