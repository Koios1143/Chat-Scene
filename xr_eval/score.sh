#!/bin/bash
# Score every XR prediction file in PREDS_DIR (LSceneLLM's metric, xr_eval/score.py). CPU only.
#   PREDS_DIR=outputs/xr_eval_<ts> bash xr_eval/score.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${EVAL_PY:-${CHATSCENE_ENV:-.venv}/bin/python}"
D="${PREDS_DIR:?set PREDS_DIR}"
for f in "$D"/preds_epoch*_xr*.json; do
  case "$f" in *.scores.json) continue ;; esac
  tag=$(basename "$f" .json); tag=${tag#preds_epoch*_step*_}           # e.g. xrqa_val_trunc_raw_no2d
  task=${tag%%_*}; split=$(echo "$tag" | cut -d_ -f2)
  case "$tag" in *_raw*) p=_raw ;; *_short*) p=_short ;; *) p= ;; esac
  echo "== $tag"
  $PY xr_eval/score.py --preds "$f" --anno "annotations/xr/${task}_${split}${p}.json" \
      --scene_map "annotations/xr/${task}_${split}_scene_map.json" | python3 -c "
import json,sys;d=json.load(sys.stdin)
for n in ('raw','chatscene'):
  for sp,v in d.get(n,{}).items(): print(f'  {n:9} {sp:5}', {m:round(x,2) for m,x in v.items()})
print('  mean_pred_words', round(d['mean_pred_words'],1))"
done
