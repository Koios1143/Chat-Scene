#!/bin/bash
# Score an extra_eval/run_extra_eval.sh output dir (CPU only; needs Java for pycocoevalcap).
#   PREDS_DIR=outputs/extra_eval_<ts> bash extra_eval/score.sh
# RESULT_DIR defaults to extra_eval/results; BENCHES limits which benchmarks are (re)scored.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${CHATSCENE_ENV:-.venv}/bin/python"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # as in run_extra_eval.sh
$PY extra_eval/score.py --pred_dir "${PREDS_DIR:?set PREDS_DIR}" --result_dir "${RESULT_DIR:-extra_eval/results}" \
    --benches "${BENCHES:-scanrefer,multi3dref,scan2cap,scanqa,sqa3d_test,real3dqa}"
