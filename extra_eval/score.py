#!/usr/bin/env python3
"""Score a ``run_extra_eval.sh`` output dir (CPU only; pycocoevalcap needs Java).

* ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, SQA3D: Chat-Scene's own scorers in
  ``utils/eval.py``, the same calls ``tasks/train.py`` makes during evaluation (SQA3D is
  scored on the test split, which ``train.py`` saves but does not score).
* Real-3DQA: the official ``evaluate_debiased_exact_match.py`` (EM, refined EM) on
  ``debiased_test`` and on each rotation split, and ``evaluate_rotation_robustness.py``
  (One..Four, VRS) on the four rotation splits. An empty generation is replaced by
  ``<empty>`` first; otherwise the refined match counts it as correct, because the empty
  string is a substring of every answer.

Writes ``<result_dir>/<bench>/scores.json`` (+ Real-3DQA prediction files) and
``<result_dir>/summary.json``.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / "real3dqa"))

REAL3DQA_SPLITS = {"debiased_test": 1485, "rotation_0": 750, "rotation_90": 750,
                   "rotation_180": 750, "rotation_270": 750}
ROTATIONS = ["rotation_0", "rotation_90", "rotation_180", "rotation_270"]
EXPECTED = {"scanrefer": 9508, "multi3dref": 11120, "scan2cap": 2007, "scanqa": 4675, "sqa3d_test": 3519}
EMPTY = "<empty>"


def item_key(scene_id, obj_id, qid, prompt):
    return (scene_id, int(obj_id), str(qid), prompt)


def load_preds(pred_dir, name, config):
    """Merged predictions of one val set, minus DistributedSampler's padding: with N GPUs
    the sampler repeats up to N-1 items so every rank gets the same count, and
    ``train.py`` keeps those repeats. Only rows beyond an item's count in the annotation
    file are dropped."""
    files = glob.glob(os.path.join(pred_dir, f"preds_epoch*_step*_{name}.json"))
    files = [f for f in files if "_rank" not in os.path.basename(f)]
    if not files:
        return None
    assert len(files) == 1, f"several prediction files for {name}: {files}"
    preds = json.load(open(files[0]))
    left = Counter(item_key(a["scene_id"], a.get("obj_id", 0), a.get("qid", 0), a["prompt"])
                   for a in json.load(open(config.val_file_dict[name][3])))
    kept = []
    for p in preds:
        k = item_key(p["scene_id"], p["gt_id"], p["qid"], p["prompt"])
        if left[k] > 0:
            left[k] -= 1
            kept.append(p)
    assert not +left, f"{name}: {sum(left.values())} annotation items have no prediction"
    if len(kept) != len(preds):
        print(f"[{name}] dropped {len(preds) - len(kept)} sampler-padding duplicate(s)")
    return kept


def coco_scorers():
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    # as in tasks/train.py
    return PTBTokenizer(), [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Meteor(), "METEOR"),
                            (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]


def score_chatscene(name, preds, config):
    from utils import eval as ev
    if name in ("scanqa", "scan2cap", "sqa3d_test"):
        tokenizer, scorers = coco_scorers()
    if name == "scanrefer":
        s = ev.calc_scanrefer_score(preds, config)
    elif name == "multi3dref":
        s = ev.calc_multi3dref_score(preds, config)
    elif name == "scan2cap":
        s = ev.calc_scan2cap_score(preds, tokenizer, scorers, config)
    elif name == "scanqa":
        s = ev.calc_scanqa_score(preds, tokenizer, scorers, config)
    elif name == "sqa3d_test":
        s = ev.calc_sqa3d_score(preds, tokenizer, scorers, config)
        s = {k.replace("[sqa3d]", "[sqa3d_test]"): v for k, v in s.items()}
    # fractions -> percent, as papers report them
    return {k: 100 * float(v) for k, v in s.items()}


def score_real3dqa(pred_dir, out, name, config):
    import evaluate_debiased_exact_match as dem
    import evaluate_rotation_robustness as rot
    res, files = {"empty_preds": {}}, {}
    for split, n in REAL3DQA_SPLITS.items():
        preds = load_preds(pred_dir, f"real3dqa_{split}", config)
        if preds is None:
            return None
        assert len(preds) == n, f"{split}: expected {n}, got {len(preds)}"
        rows = []
        for p in preds:
            empty = not dem.clean_answer(p["pred"]).strip()
            res["empty_preds"][split] = res["empty_preds"].get(split, 0) + empty
            # int ids: the rotation script int()s the suffix-stripped 90/180/270 ids
            # but not the 0-degree ones, so string ids would never intersect
            rows.append({"question_id": int(p["qid"]), "scene_id": p["scene_id"],
                         "response_gt": p["ref_captions"], "response_pred": EMPTY if empty else p["pred"],
                         "prompt": p["prompt"]})
        files[split] = out / f"pred_{split}.json"
        json.dump(rows, open(files[split], "w"), indent=1)
        res[split] = dem.evaluate_em(str(files[split]))
    res["rotation"] = rot.quick_consistency_score([str(files[s]) for s in ROTATIONS])
    # the same numbers through the unmodified scripts' CLIs, for the record
    log = []
    for cmd in [[sys.executable, str(HERE / "real3dqa" / "evaluate_debiased_exact_match.py"),
                 str(files["debiased_test"]), "--name", name],
                [sys.executable, str(HERE / "real3dqa" / "evaluate_rotation_robustness.py")]
                + [str(files[s]) for s in ROTATIONS] + ["--name", name]]:
        log.append(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    (out / "official_scripts.txt").write_text("\n".join(log).replace(str(REPO) + "/", ""))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", required=True, help="run_extra_eval.sh output dir")
    ap.add_argument("--result_dir", default="extra_eval/results")
    ap.add_argument("--config", default="scripts/extra_eval_config.py")
    ap.add_argument("--benches", default="scanrefer,multi3dref,scan2cap,scanqa,sqa3d_test,real3dqa")
    ap.add_argument("--name", default="Chat-Scene")
    args = ap.parse_args()
    os.chdir(REPO)  # utils/eval.py reads annotations/ relative to the repo root
    cfg = {}
    exec(open(args.config).read(), cfg)
    config = SimpleNamespace(val_file_dict=cfg["val_file_dict"])
    result_dir = Path(args.result_dir)
    summary_file = result_dir / "summary.json"
    summary = json.load(open(summary_file)) if summary_file.exists() else {}
    for bench in args.benches.split(","):
        out = result_dir / bench
        out.mkdir(parents=True, exist_ok=True)
        if bench == "real3dqa":
            res = score_real3dqa(args.pred_dir, out, args.name, config)
        else:
            preds = load_preds(args.pred_dir, bench, config)
            if preds is None:
                res = None
            else:
                assert len(preds) == EXPECTED[bench], f"{bench}: expected {EXPECTED[bench]}, got {len(preds)}"
                res = score_chatscene(bench, preds, config)
        if res is None:
            print(f"[{bench}] no predictions in {args.pred_dir}, skipped")
            continue
        res = {"pred_dir": os.path.relpath(args.pred_dir, REPO), **res}
        json.dump(res, open(out / "scores.json", "w"), indent=1)
        summary[bench] = res
        print(f"[{bench}]", json.dumps(res, indent=1))
    json.dump(summary, open(summary_file, "w"), indent=1)


if __name__ == "__main__":
    main()
