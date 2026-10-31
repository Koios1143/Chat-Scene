#!/usr/bin/env python3
"""Score Chat-Scene predictions on XR-QA / XR-SceneCaption the way LSceneLLM does.

LSceneLLM (``tools/runner_Adaptive_LLM.py: validate / score_captions``) scores the raw
decoded text, only collapsing repeated spaces (no lower-casing, no punctuation or PTB
tokenisation), with LL3DA's ``capeval`` (vendored in ``xr_eval/capeval``), and keys both
predictions and references by ``episode_id`` (``HM3D#<scan>#<region_id>#<question>``), so
repeated keys keep the last item. Scores are x100.

``--norm chatscene`` instead scores like Chat-Scene's own ScanQA metric (``utils/eval.py:
calc_scanqa_score``): the prediction loses a trailing '.', gets its first letter lower-cased and
``clean_answer``; predictions and references then go through the PTB tokenizer (lower-case, no
punctuation). ``--norm both`` (default) writes both, under "raw" and "chatscene".

    python xr_eval/score.py --preds <out>/preds_epoch-1_step0_<val_tag>.json \
        --anno annotations/xr/xrqa_val_raw.json [--scene_map annotations/xr/xrqa_val_scene_map.json]
Writes ``<preds>.scores.json`` next to the predictions.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capeval.bleu.bleu import Bleu  # noqa: E402
from capeval.cider.cider import Cider  # noqa: E402
from capeval.meteor.meteor import Meteor  # noqa: E402
from capeval.rouge.rouge import Rouge  # noqa: E402


def clean(s):
    return " ".join(w for w in s.split(" ") if w)


def score(corpus, cands):
    keys = list(cands)
    gts = OrderedDict((k, corpus[k]) for k in keys)
    res = OrderedDict((k, cands[k]) for k in keys)
    bleu = Bleu(4).compute_score(gts, res)
    cider = Cider().compute_score(gts, res)
    rouge = Rouge().compute_score(gts, res)
    meteor = Meteor().compute_score(gts, res)
    return {"n": len(keys), "CIDEr": 100 * cider[0], "BLEU-4": 100 * bleu[0][3],
            "METEOR": 100 * meteor[0], "ROUGE-L": 100 * rouge[0]}


def chatscene_norm(corpus, cands):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from utils.helper import clean_answer

    def pred(p):
        if len(p) > 1:
            p = p[:-1] if p[-1] == "." else p
            p = p[0].lower() + p[1:]
        return clean_answer(p)

    tok = PTBTokenizer()
    return (tok.tokenize({k: [{"caption": c} for c in v] for k, v in corpus.items()}),
            tok.tokenize({k: [{"caption": pred(v[0])}] for k, v in cands.items()}))


def score_splits(corpus, cands, scene, n_obj):
    out = {"all": score(corpus, cands)}
    for name, keep in (("le100", lambda n: n <= 100), ("gt100", lambda n: n > 100)):
        sub = {k: v for k, v in cands.items() if n_obj and keep(n_obj[scene[k]])}
        if sub:
            out[name] = score(corpus, sub)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", required=True)
    ap.add_argument("--anno", required=True)
    ap.add_argument("--scene_map", default="", help="adds scores split by object count (<=100 / >100)")
    ap.add_argument("--norm", default="both", choices=["raw", "chatscene", "both"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    anno = json.load(open(args.anno))
    preds = {}
    for p in json.load(open(args.preds)):
        preds[int(p["qid"])] = p["pred"]  # DistributedSampler padding repeats items
    missing = [i for i in range(len(anno)) if i not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(anno)} items have no prediction")
    corpus, cands, scene = {}, {}, {}
    for i, a in enumerate(anno):
        corpus[a["key"]] = list(a["ref_captions"])
        cands[a["key"]] = [clean(preds[i])]
        scene[a["key"]] = a["scene_id"]
    n_obj = {k: v["n_obj"] for k, v in json.load(open(args.scene_map)).items()} if args.scene_map else {}
    out = {}
    if args.norm in ("raw", "both"):
        out["raw"] = score_splits(corpus, cands, scene, n_obj)
    if args.norm in ("chatscene", "both"):
        out["chatscene"] = score_splits(*chatscene_norm(corpus, cands), scene, n_obj)
    out["mean_pred_words"] = sum(len(v[0].split()) for v in cands.values()) / len(cands)
    path = args.out or os.path.splitext(args.preds)[0] + ".scores.json"
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
