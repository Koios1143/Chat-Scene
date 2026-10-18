#!/bin/bash
# Uni3D GT-object features for the Beacon3D scenes. Only needed to REBUILD
# beacon3d/features/ (the committed files are what the reported run used).
#   bash beacon3d/extract_uni3d.sh          # one GPU, ~1.5 h
# Paths come from env.sh (see env.sh.example).
# Required:
#   UNI3D_CKPT           uni3d-g model.pt (https://huggingface.co/BAAI/Uni3D)
#   RSCAN_SCAN_DATA      SceneVerse 3RScan scan_data/ (pcd_with_global_alignment, instance_id_to_label)
#   MULTISCAN_SCAN_DATA  SceneVerse MultiScan scan_data/
# Optional:
#   SCANNET_SCANS   raw ScanNet scans/ (<scene>_vh_clean_2.ply, <scene>.aggregation.json and
#                   one <objectId>.npy GT mask per object). Only used to validate the extractor
#                   against Chat-Scene's official ScanNet features; skipped when unset.
#   OFFICIAL_DIR    Chat-Scene's HF annotations/ (default annotations)
#   FEATS_DIR       output dir (default beacon3d/features)
#   BEACON_ROOT     Beacon3D data dir (default beacon3d/Beacon3D)
#   EVAL_PY         python with torch + timm (+ open3d for ScanNet) (default $CHATSCENE_ENV/bin/python)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${EVAL_PY:-${CHATSCENE_ENV:-.venv}/bin/python}"
OUT="${FEATS_DIR:-beacon3d/features}"
CK="${UNI3D_CKPT:?set UNI3D_CKPT to the uni3d-g model.pt}"
B="${BEACON_ROOT:-beacon3d/Beacon3D}"
OFFICIAL_DIR="${OFFICIAL_DIR:-annotations}"
if [ -n "${SCANNET_SCANS:-}" ]; then
  $PY beacon3d/extract_uni3d_gt.py --domain scannet --beacon_root "$B" --ckpt "$CK" \
      --scans_root "$SCANNET_SCANS" --align_from_attrs "$OFFICIAL_DIR/scannet_val_attributes.pt" --out_dir "$OUT" || exit 1
  $PY beacon3d/compare_uni3d_official.py --mine "$OUT/scannet_gt_uni3d_feats.pt" \
      --official "$OFFICIAL_DIR/scannet_gt_uni3d_feats.pt"
else
  echo "SCANNET_SCANS unset: skipping the ScanNet validation extraction"
fi
$PY beacon3d/extract_uni3d_gt.py --domain 3rscan --beacon_root "$B" --ckpt "$CK" \
    --scan_data "${RSCAN_SCAN_DATA:?set RSCAN_SCAN_DATA}" --out_dir "$OUT" || exit 1
$PY beacon3d/extract_uni3d_gt.py --domain multiscan --beacon_root "$B" --ckpt "$CK" \
    --scan_data "${MULTISCAN_SCAN_DATA:?set MULTISCAN_SCAN_DATA}" --out_dir "$OUT" || exit 1
echo UNI3D_DONE
