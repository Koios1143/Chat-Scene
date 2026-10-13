#!/bin/bash
# Convert Chat-Scene preds to Beacon3D infer files and score (grounding + GPT-4o QA).
#   PREDS_DIR=outputs/beacon3d_<ts> OPENAI_API_KEY=... sbatch -A <account> -p <partition> beacon3d/score.sbatch
# Overridable:
#   DOMAINS      comma list (default scannet,scannet_myfeat,scannet_no2d,3rscan,multiscan)
#   RESULT_DIR   where infer_*.json / score_*.txt go (default beacon3d/results)
#   BEACON_ROOT  Beacon3D data + evaluators (default beacon3d/Beacon3D)
#   FEATS_DIR    scene maps for 3rscan/multiscan (default beacon3d/features)
#   EVAL_PY      python with the openai package (default $CHATSCENE_ENV/bin/python)
# OPENAI_API_KEY must be in the environment (sbatch exports it by default); QA is skipped without it.
#SBATCH --job-name=chatscene-score
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --output=log/score_%j.out
#SBATCH --error=log/score_%j.err
set -uo pipefail
unset CONDA_PREFIX CONDA_DEFAULT_ENV
cd "${SLURM_SUBMIT_DIR:-.}"
PY="${EVAL_PY:-${CHATSCENE_ENV:-.venv}/bin/python}"
PREDS_DIR="${PREDS_DIR:?set PREDS_DIR}"
ARGS=(--preds_dir "$PREDS_DIR" --beacon_root "${BEACON_ROOT:-beacon3d/Beacon3D}"
      --uni3d_dir "${FEATS_DIR:-beacon3d/features}" --result_dir "${RESULT_DIR:-beacon3d/results}"
      --domains "${DOMAINS:-scannet,scannet_myfeat,scannet_no2d,3rscan,multiscan}"
      --shim beacon3d/beacon_grounding_shim.py --score)
$PY beacon3d/to_beacon3d.py "${ARGS[@]}" --tasks grounding || exit 1
$PY beacon3d/to_beacon3d.py "${ARGS[@]}" --tasks qa || exit 1
echo SCORE_DONE
