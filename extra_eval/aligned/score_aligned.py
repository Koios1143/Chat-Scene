#!/usr/bin/env python3
"""Re-score Chat-Scene's saved predictions with the common aligned normalisation.

No inference: reads the prediction files listed in ``SOURCES`` / ``XR_SOURCES``, strips only
Chat-Scene's own special tokens from each prediction (see ``strip_special``) and hands
refs/preds to ``aligned_eval.py`` (identical in every baseline repo), which applies the
text normalisation and the metrics.

* Scan2Cap: Chat-Scene's protocol (``utils/eval.py: calc_scan2cap_score``): one caption per GT
  object of ``scan2cap_mask3d_val.json`` for its assigned Mask3D proposal; the caption counts
  when IoU(proposal box, GT box) >= 0.25 / 0.5, otherwise and for corpus objects without an
  annotation item the prediction is "" (-> ``sos eos``). References: the common untruncated
  corpus ``deps/aligned/scan2cap_val_corpus.json``. Mode ``s2c``.
* ScanQA val: all saved references per question; modes ``clean_ptb``, ``raw`` (+ EM / EM-R).
* SQA3D test: EM / EM-R only (``em_scores``), compared with the existing Chat-Scene EM.
* XR-SceneCaption (modes ``ptb``, ``raw``) and XR-QA (``clean_ptb``, ``raw``): keyed per ITEM
  (all 2994 XR-QA items, all 283 captions), not de-duplicated by episode id the way
  ``xr_eval/score.py`` / LSceneLLM do (2994 -> 2981 unique episodes, recorded as
  ``n_unique_episodes``). Headline runs only (Mask3D proposals, ``PC_TYPE=pred``); the older
  GT-object runs are not re-scored.

Writes ``extra_eval/results/aligned/<bench>__<variant>.json`` and ``summary.json``.
Run via ``extra_eval/aligned/run_score_aligned.sh`` (CPU, needs java for PTB / METEOR).
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "extra_eval"))
sys.path.insert(0, str(REPO))

import aligned_eval as ae  # noqa: E402

EXTRA = "outputs/extra_eval_20261021_025847"  # headline paper-benchmark run (extra_eval/results)
XR_ZS = "outputs/xr_eval_20261101_171552"      # headline XR zero-shot (Mask3D proposals)
XR_FT = "outputs/xr_ft_pred_20261101_172455"   # headline XR finetune (Mask3D proposals)
XR_HEADLINE_EPOCH = 2                          # README "epoch 3" = preds_epoch2_step909
CORPUS = "deps/aligned/scan2cap_val_corpus.json"
EXISTING_SQA3D = "extra_eval/results/sqa3d_test/scores.json"

# (bench, variant, prediction file, annotation file, headline)
XR_SOURCES = [
    ("xr_scenecaption", "zs_pred", f"{XR_ZS}/preds_epoch-1_step0_xrcap_val_pred.json", "xrcap_val", True),
    ("xr_scenecaption", "zs_pred_no2d", f"{XR_ZS}/preds_epoch-1_step0_xrcap_val_pred_no2d.json", "xrcap_val", False),
    ("xr_qa", "zs_pred_short", f"{XR_ZS}/preds_epoch-1_step0_xrqa_val_pred_short.json", "xrqa_val_short", True),
    ("xr_qa", "zs_pred_short_no2d", f"{XR_ZS}/preds_epoch-1_step0_xrqa_val_pred_short_no2d.json", "xrqa_val_short", False),
    ("xr_qa", "zs_pred_raw", f"{XR_ZS}/preds_epoch-1_step0_xrqa_val_pred_raw.json", "xrqa_val_raw", False),
    ("xr_qa", "zs_pred_raw_no2d", f"{XR_ZS}/preds_epoch-1_step0_xrqa_val_pred_raw_no2d.json", "xrqa_val_raw", False),
]
for _ep, _step in ((0, 303), (1, 606), (2, 909)):
    XR_SOURCES += [
        ("xr_scenecaption", f"ft_pred_epoch{_ep}", f"{XR_FT}/preds_epoch{_ep}_step{_step}_xrcap_val_pred.json",
         "xrcap_val", _ep == XR_HEADLINE_EPOCH),
        ("xr_qa", f"ft_pred_short_epoch{_ep}", f"{XR_FT}/preds_epoch{_ep}_step{_step}_xrqa_val_pred_short.json",
         "xrqa_val_short", _ep == XR_HEADLINE_EPOCH),
    ]
XR_EXPECTED = {"xr_scenecaption": 283, "xr_qa": 2994}
EXPECTED = {"scan2cap": 2007, "scanqa": 4675, "sqa3d_test": 3519}
N_CORPUS = 2068

# Chat-Scene's tokenizer: Vicuna's <s> </s> <unk> (pad = unk) plus the <OBJ000>..<OBJ199>
# object-id tokens it adds with special_tokens=True (models/chat3d.py). Generation is decoded
# without skip_special_tokens and cut at end_sym "</s>", so object tokens can survive.
SPECIAL = re.compile(r"<OBJ\d{3}>|</?s>|<unk>|<pad>")


def strip_special(text):
    return " ".join(SPECIAL.sub(" ", text or "").split())


def rel(p):
    return os.path.relpath(p, REPO)


def load_config():
    cfg = {}
    exec(open(REPO / "scripts" / "extra_eval_config.py").read(), cfg)  # written by build_annos.py
    return SimpleNamespace(val_file_dict=cfg["val_file_dict"])


def load_extra(name, config):
    """De-duplicated predictions of one extra_eval benchmark (sampler padding removed)."""
    from score import load_preds  # extra_eval/score.py
    preds = load_preds(EXTRA, name, config)
    assert preds is not None, f"no {name} predictions in {EXTRA}"
    assert len(preds) == EXPECTED[name], f"{name}: expected {EXPECTED[name]}, got {len(preds)}"
    return preds


def strip_count(preds):
    return sum(strip_special(p["pred"]) != " ".join(p["pred"].split()) for p in preds)


def scan2cap(config):
    import torch
    from utils.box_utils import box3d_iou, construct_bbox_corners
    preds = load_extra("scan2cap", config)
    corpus = json.load(open(REPO / CORPUS))
    assert len(corpus) == N_CORPUS, f"corpus: expected {N_CORPUS} keys, got {len(corpus)}"
    inst = torch.load(config.val_file_dict["scan2cap"][2], map_location="cpu")
    gt = torch.load("annotations/scannet_val_attributes.pt", map_location="cpu")
    anno_iou = {f"{a['scene_id']}|{a['obj_id']}": a.get("iou")
                for a in json.load(open(config.val_file_dict["scan2cap"][3]))}
    ious = {}
    texts = {}
    for p in preds:
        key = f"{p['scene_id']}|{p['gt_id']}"
        assert key in corpus, f"scan2cap: {key} not in the common corpus"
        assert key not in ious, f"scan2cap: duplicate object {key}"
        pl = inst[p["scene_id"]]["locs"][p["pred_id"]].tolist()
        gl = gt[p["scene_id"]]["locs"][p["gt_id"]].tolist()
        ious[key] = box3d_iou(construct_bbox_corners(pl[:3], pl[3:]), construct_bbox_corners(gl[:3], gl[3:]))
        texts[key] = strip_special(p["pred"])
    iou_mismatch = sum(1 for k, v in ious.items()
                       if anno_iou.get(k) is not None and abs(anno_iou[k] - v) > 1e-4)
    out = []
    for thr, tag in ((0.25, "iou0.25"), (0.5, "iou0.50")):
        res = {k: (texts[k] if ious.get(k, -1) >= thr else "") for k in corpus}
        matched = sum(1 for v in ious.values() if v >= thr)
        out.append(("scan2cap", tag, ae.score_captions(corpus, res, modes=("s2c",)), {
            "n": len(res), "n_protocol_items": len(preds), "n_matched": matched,
            "n_corpus_without_item": len(corpus) - len(preds),
            "n_iou_differs_from_annotation": iou_mismatch, "n_special_stripped": strip_count(preds),
            "refs": CORPUS, "source": rel(REPO / EXTRA / "preds_epoch-1_step0_scan2cap.json")}))
    return out


def scanqa(config):
    preds = load_extra("scanqa", config)
    refs, res = OrderedDict(), OrderedDict()
    for i, p in enumerate(preds):
        k = f"{p['scene_id']}|{p['gt_id']}|{p['qid']}|{i}"
        refs[k] = list(p["ref_captions"])
        res[k] = strip_special(p["pred"])
    return [("scanqa", "val", ae.score_qa(refs, res, modes=("clean_ptb", "raw")), {
        "n": len(res), "n_special_stripped": strip_count(preds),
        "source": rel(REPO / EXTRA / "preds_epoch-1_step0_scanqa.json")})]


def sqa3d(config):
    preds = load_extra("sqa3d_test", config)
    refs, res = OrderedDict(), OrderedDict()
    for i, p in enumerate(preds):
        k = f"{p['scene_id']}|{p['qid']}|{i}"
        refs[k] = list(p["ref_captions"])
        res[k] = strip_special(p["pred"])
    em = ae.em_scores(res, refs)
    old = json.load(open(REPO / EXISTING_SQA3D))
    existing = {"EM": old["[sqa3d_test] EM1"], "EM-R": old["[sqa3d_test] EM1_refined"]}
    diff = {m: em[m] - existing[m] for m in em}
    print(f"[sqa3d_test] aligned EM {em} vs existing {existing} (diff {diff})")
    return [("sqa3d", "test", {"em": em}, {
        "n": len(res), "n_special_stripped": strip_count(preds), "existing_em": existing,
        "existing_em_source": EXISTING_SQA3D, "diff_vs_existing": diff,
        "source": rel(REPO / EXTRA / "preds_epoch-1_step0_sqa3d_test.json")})]


def xr(bench, variant, pred_file, anno_name, headline):
    anno_file = REPO / "annotations" / "xr" / f"{anno_name}.json"  # written by xr_eval/build_xr.py
    anno = json.load(open(anno_file))
    rows = json.load(open(REPO / pred_file))
    by_qid = {}
    for p in rows:
        by_qid[int(p["qid"])] = p  # DistributedSampler padding repeats items
    missing = [i for i in range(len(anno)) if i not in by_qid]
    assert not missing, f"{pred_file}: {len(missing)} of {len(anno)} items have no prediction"
    assert len(by_qid) == len(anno), f"{pred_file}: {len(by_qid)} qids vs {len(anno)} annotation items"
    refs, res = OrderedDict(), OrderedDict()
    stripped = 0
    for i, a in enumerate(anno):
        p = by_qid[i]
        assert list(p["ref_captions"]) == list(a["ref_captions"]), f"{pred_file}: qid {i} refs differ from {anno_file}"
        k = f"{a['key']}#item{i}"  # per item: repeated episode keys are kept, not de-duplicated
        refs[k] = list(a["ref_captions"])
        res[k] = strip_special(p["pred"])
        stripped += res[k] != " ".join(p["pred"].split())
    assert len(res) == XR_EXPECTED[bench], f"{pred_file}: expected {XR_EXPECTED[bench]} items, got {len(res)}"
    if bench == "xr_qa":
        metrics = ae.score_qa(refs, res, modes=("clean_ptb", "raw"))
    else:
        metrics = ae.score_captions(refs, res, modes=("ptb", "raw"))
    return [(bench, variant, metrics, {
        "n": len(res), "n_unique_episodes": len({a["key"] for a in anno}), "n_special_stripped": stripped, "headline": headline,
        "refs": rel(anno_file), "source": pred_file})]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out_dir", default="extra_eval/results/aligned")
    ap.add_argument("--only", default="", help="comma list of benches to run (default: all)")
    args = ap.parse_args()
    os.chdir(REPO)  # annotations/ and outputs/ are repo-relative
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    want = lambda b: not only or b in only  # noqa: E731
    config = load_config()
    jobs = []
    if want("scan2cap"):
        jobs.append(lambda: scan2cap(config))
    if want("scanqa"):
        jobs.append(lambda: scanqa(config))
    if want("sqa3d"):
        jobs.append(lambda: sqa3d(config))
    for src in XR_SOURCES:
        if want(src[0]):
            jobs.append(lambda src=src: xr(*src))
    summary_file = out_dir / "summary.json"
    summary = json.load(open(summary_file)) if summary_file.exists() else {}
    for job in jobs:
        for bench, variant, metrics, info in job():
            rec = {"bench": bench, "variant": variant, **info, "metrics": metrics}
            name = f"{bench}__{variant}"
            json.dump(rec, open(out_dir / f"{name}.json", "w"), indent=1)
            summary[name] = rec
            print(f"[{name}] n={info['n']}", json.dumps(metrics))
    json.dump(dict(sorted(summary.items())), open(summary_file, "w"), indent=1)


if __name__ == "__main__":
    main()
