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
# XR-SceneCaption SPICE, same setup as just3d: SPICE 1.0 jars in deps/spice, JDK 17 in deps/jdk17
if [ -f deps/spice/spice-1.0.jar ] && [ -x deps/jdk17/bin/java ]; then
    export ALIGNED_SPICE_DIR="$PWD/deps/spice" ALIGNED_SPICE_JAVA="$PWD/deps/jdk17/bin/java"
    export ALIGNED_SPICE_CACHE="${ALIGNED_SPICE_CACHE:-$PWD/deps/spice_cache}"
    export ALIGNED_SPICE_SHARDS="${ALIGNED_SPICE_SHARDS:-6}" ALIGNED_SPICE_HEAP="${ALIGNED_SPICE_HEAP:-6G}"
else
    echo "[warn] deps/spice or deps/jdk17 missing: XR-SceneCaption scored without SPICE" >&2
fi
PY="${CHATSCENE_ENV:-deps/venv-cu128}/bin/python"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # attribute .pt files, as in extra_eval/score.sh
"$PY" extra_eval/aligned/score_aligned.py "$@"
