#!/bin/bash
# Re-score the saved predictions with the common aligned normalisation (CPU only, no inference).
#   bash extra_eval/aligned/run_score_aligned.sh [--only scan2cap,scanqa,sqa3d,xr_scenecaption,xr_qa]
# Writes extra_eval/results/aligned/<bench>__<variant>.json + summary.json.
# Java for PTB / METEOR: JAVA_HOME from env.sh, else the repo's deps/jre11.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
JAVA_HOME="${JAVA_HOME:-deps/jre11}"
export PATH="$(cd "$JAVA_HOME" && pwd)/bin:$PATH"
java -version 2>&1 | head -1
PY="${CHATSCENE_ENV:-deps/venv-cu128}/bin/python"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # attribute .pt files, as in extra_eval/score.sh
"$PY" extra_eval/aligned/score_aligned.py "$@"
