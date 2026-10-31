#!/bin/bash
# Exact SceneVerse-region -> HM3D glb offsets (translation only) for every XR scan.
#   bash xr_eval/register_regions.sh            # ~CPU only; NSHARD parallel processes
# Paths come from env.sh (see env.sh.example): XR_SCAN_DATA, HM3D_GLB.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${EVAL_PY:-${CHATSCENE_ENV:-.venv}/bin/python}"
OUT="${REGION_OFFSETS:-xr_eval/data/region_offsets.json}"
N="${NSHARD:-4}"
mkdir -p "$(dirname "$OUT")"
for i in $(seq 0 $((N - 1))); do
  $PY -u xr_eval/register_regions.py --scan_data "${XR_SCAN_DATA:?}" --glb_root "${HM3D_GLB:?}" --xr_dir "${XR_DIR:?}" \
      --out "$OUT.shard$i" --shard "$i/$N" &
done
wait
$PY - "$OUT" "$N" <<'PY'
import json, sys
out, n = sys.argv[1], int(sys.argv[2])
res = {}
for i in range(n):
    res.update(json.load(open(f"{out}.shard{i}")))
json.dump(res, open(out, "w"), indent=1, sort_keys=True)
PY
# second pass: regions whose median point-to-mesh distance is > 2 cm, searched over a wider range
$PY -u xr_eval/register_regions.py --scan_data "${XR_SCAN_DATA:?}" --glb_root "${HM3D_GLB:?}" --xr_dir "${XR_DIR:?}" \
    --out "$OUT" --redo_bad
$PY - "$OUT" <<'PY'
import json, sys
res = json.load(open(sys.argv[1]))
bad = [(s, r, v["med"], v["inl3cm"]) for s, d in res.items() for r, v in d.items() if v["med"] > 0.02]
print(len(res), "scans,", sum(map(len, res.values())), "regions; median > 2 cm:", bad)
PY
echo REGISTER_DONE
