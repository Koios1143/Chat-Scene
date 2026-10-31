#!/usr/bin/env python3
"""XR-QA / XR-SceneCaption (LSceneLLM) -> Chat-Scene val files + ``scripts/xr_config.py``.

An XR scene is a set of SceneVerse HM3D regions (XR-QA ``qa_pairs[].region_id``,
XR-SceneCaption ``room_id``, e.g. ``sub002-sub012-sub014``). Its objects are the GT objects
of those regions in that order (ascending SceneVerse instance id inside a region), i.e. the
order LSceneLLM concatenates the regions in. Each scene gets an alias ``xq<NNNN>_00`` /
``xc<NNNN>_00`` because Chat-Scene recovers the scene of a feature key as
``'_'.join(key.split('_')[:2])``.

Chat-Scene has 100 object identifiers (``<OBJ000>``..``<OBJ099>``); LSceneLLM does not say
how its Chat-Scene baseline handled larger scenes. One attributes file per hypothesis
(``--select``) picks the objects that get identifiers:
  trunc  stock code: no ``obj_ids``, so ``prepare_scene_features`` reads columns 0..99
         (the first 100 objects) and pads missing ones with zeros
  rand   100 objects drawn at random (seeded per scene)
  big    drop Chat-Scene's ``unwanted_words`` classes (wall/ceiling/floor/object/item), then
         the 100 with the most points
Scenes with <= 100 objects are identical under all three.

Prompts (Chat-Scene's user turn; the object list goes in front as usual):
  XR-QA          raw: the question; short: + " Answer the question using a single word or
                 phrase." (Chat-Scene's ScanQA prompt)
  XR-SceneCaption  the XR question without its "### human:" / "### assistant:" wrapper

Train splits (``--splits train val``) get Chat-Scene train items (``caption`` = one reference
answer; a caption with several references becomes several items) and are registered in
``train_file_dict`` under the same names (with 2D features).

Writes ``<anno_dir>/xr/``: ``xr{qa,cap}_<split>_{uni3d,video}feats.pt``,
``xr{qa,cap}_<split>_attributes_<select>.pt``, ``xr{qa,cap}_<split>[_<prompt>].json`` and
``xr{qa,cap}_<split>_scene_map.json``; ``scripts/xr_config.py`` registers
``xrqa_<split>_<select>_<prompt>`` and ``xrcap_<split>_<select>``.
"""

import argparse
import glob
import json
import os
import random
import re
import zlib

import torch

QA_SUFFIX = " Answer the question using a single word or phrase."
UNWANTED = ["wall", "ceiling", "floor", "object", "item"]
SELECTS = ["trunc", "rand", "big"]
MAX_OBJ = 100


def load_shards(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        out.update(torch.load(f, map_location="cpu"))
    return out


def qa_items(xr_dir, split):
    """LSceneLLM's HD_Hm3dQADataset order; key = its episode_id."""
    items = []
    for scan, eps in json.load(open(os.path.join(xr_dir, "XR-QA", f"{split}.json"))).items():
        for ep in eps:
            for qa in ep["qa_pairs"]:
                items.append({
                    "scan": scan,
                    "regions": qa["region_id"].split("-"),
                    "question": qa["question"],
                    "answers": [qa["answer"]],
                    "type": qa["type"],
                    "key": "HM3D#{}#{}#{}".format(scan, qa["region_id"], qa["question"]),
                })
    return items


def cap_items(xr_dir, split):
    items = []
    for i, x in enumerate(json.load(open(os.path.join(xr_dir, "XR-SceneCaption", f"{split}.json")))):
        q = re.sub(r"^\s*###\s*human:\s*|\s*###\s*assistant:\s*$", "", x["question"]).strip()
        items.append({
            "scan": x["scan_id"],
            "regions": x["room_id"].split("-"),
            "question": q,
            "answers": x["answers"],
            "key": "HM3D#{}#{}#{}#{}".format(x["scan_id"], x["room_id"], q, i),
        })
    return items


def select(objs, how, seed_key):
    n = len(objs)
    if how == "trunc" or n <= MAX_OBJ:
        return None if how == "trunc" else list(range(n)) + list(range(n, MAX_OBJ))
    if how == "rand":
        rng = random.Random(zlib.crc32(seed_key.encode()))
        cols = sorted(rng.sample(range(n), MAX_OBJ))
    elif how == "big":
        keep = [c for c, o in enumerate(objs) if not any(w in o["label"] for w in UNWANTED)]
        keep = sorted(keep, key=lambda c: -objs[c]["npts"])[:MAX_OBJ]
        cols = sorted(keep)
    else:
        raise ValueError(how)
    return cols + list(range(n, n + MAX_OBJ - len(cols)))


def build(task, split, items, uni3d, video, room_center, anno_dir, prefix):
    scenes = {}
    for it in items:
        scenes.setdefault((it["scan"], tuple(it["regions"])), len(scenes))
    feats, vfeats, scene_map = {}, {}, {}
    attrs = {s: {} for s in SELECTS}
    for (scan, regs), k in scenes.items():
        alias = f"{prefix}{k:04d}_00"
        objs = []
        for sub in regs:
            r = uni3d[f"{scan}_{sub}"]
            rv = video.get(f"{scan}_{sub}", {})
            c = torch.as_tensor(room_center[scan][sub]["center"], dtype=torch.float32)
            for j, oid in enumerate(r["ids"]):
                col = len(objs)
                feats[f"{alias}_{col:02}"] = r["feats"][j]
                if oid in rv:
                    vfeats[f"{alias}_{col:02}"] = rv[oid]
                loc = r["locs"][j].clone()
                loc[:3] += c
                objs.append({"sub": sub, "id": oid, "label": r["labels"][j], "npts": r["npts"][j], "loc": loc})
        if not objs:  # all regions empty in SceneVerse: one zero feature so the scene exists
            feats[f"{alias}_00"] = torch.zeros(next(iter(uni3d.values()))["feats"].shape[1])
        locs = torch.stack([o["loc"] for o in objs]) if objs else torch.zeros(0, 6)
        for how in SELECTS:
            a = {"objects": [o["label"] for o in objs], "locs": locs}
            cols = select(objs, how, alias)
            if cols is not None:
                a["obj_ids"] = cols
            attrs[how][alias] = a
        scene_map[alias] = {"scan": scan, "regions": list(regs), "n_obj": len(objs),
                            "objects": [[o["sub"], o["id"]] for o in objs]}
    base = os.path.join(anno_dir, f"{task}_{split}")
    torch.save(feats, f"{base}_uni3dfeats.pt")
    torch.save(vfeats, f"{base}_videofeats.pt")
    for how in SELECTS:
        torch.save(attrs[how], f"{base}_attributes_{how}.pt")
    json.dump(scene_map, open(f"{base}_scene_map.json", "w"))
    n = [m["n_obj"] for m in scene_map.values()]
    print(f"{task}_{split}: {len(items)} items, {len(scenes)} scenes, objects mean {sum(n) / len(n):.1f} "
          f"max {max(n)}, >100: {sum(x > MAX_OBJ for x in n)}; 2D seen {len(vfeats)}/{len(feats)}")
    return {(it["scan"], tuple(it["regions"])): f"{prefix}{scenes[(it['scan'], tuple(it['regions']))]:04d}_00"
            for it in items}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xr_dir", required=True)
    ap.add_argument("--scan_data", required=True, help="SceneVerse HM3D scan_data (room_center/)")
    ap.add_argument("--feats_dir", default="xr_eval/features")
    ap.add_argument("--anno_dir", default="annotations/xr")
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--config_out", default="scripts/xr_config.py")
    args = ap.parse_args()
    os.makedirs(args.anno_dir, exist_ok=True)
    uni3d = load_shards(os.path.join(args.feats_dir, "regions_uni3d.*.pt"))
    video = load_shards(os.path.join(args.feats_dir, "regions_videofeats.*.pt"))
    print(f"uni3d regions {len(uni3d)}, 2D regions {len(video)}")
    rc_dir = os.path.join(args.scan_data, "room_center")
    room_center = {f[:-4]: torch.load(os.path.join(rc_dir, f), weights_only=False) for f in os.listdir(rc_dir)}

    cfg = ["\n\n# ---- XR-QA / XR-SceneCaption (generated by xr_eval/build_xr.py) ----"]
    for split in args.splits:
        for task, items, prefix in (("xrqa", qa_items(args.xr_dir, split), "xq"),
                                    ("xrcap", cap_items(args.xr_dir, split), "xc")):
            alias = build(task, split, items, uni3d, video, room_center, args.anno_dir, prefix)
            prompts = {"raw": "", "short": QA_SUFFIX} if task == "xrqa" else {"": ""}
            for pname, suffix in prompts.items():
                out = []
                for i, it in enumerate(items):
                    if split == "train":
                        out += [{"scene_id": alias[(it["scan"], tuple(it["regions"]))], "obj_id": 0,
                                 "prompt": it["question"] + suffix, "caption": ans} for ans in it["answers"]]
                        continue
                    out.append({
                        "scene_id": alias[(it["scan"], tuple(it["regions"]))],
                        "obj_id": 0,
                        "prompt": it["question"] + suffix,
                        "ref_captions": it["answers"],
                        "qid": i,
                        "key": it["key"],
                        **({"type": it["type"]} if "type" in it else {}),
                    })
                tag = f"_{pname}" if pname else ""
                json.dump(out, open(os.path.join(args.anno_dir, f"{task}_{split}{tag}.json"), "w"), indent=1)
                base = f"{{anno_root}}/xr/{task}_{split}"
                for how in SELECTS:
                    if split == "train":
                        cfg.append(f"train_file_dict['{task}_{split}_{how}{tag}'] = [f'{base}_uni3dfeats.pt', "
                                   f"f'{base}_videofeats.pt', f'{base}_attributes_{how}.pt', f'{base}{tag}.json']")
                        continue
                    cfg.append(f"val_file_dict['{task}_{split}_{how}{tag}'] = [f'{base}_uni3dfeats.pt', "
                               f"f'{base}_videofeats.pt', f'{base}_attributes_{how}.pt', f'{base}{tag}.json']")
                    cfg.append(f"val_file_dict['{task}_{split}_{how}{tag}_no2d'] = [f'{base}_uni3dfeats.pt', "
                               f"None, f'{base}_attributes_{how}.pt', f'{base}{tag}.json']")
    open(args.config_out, "w").write(open("scripts/config.py").read() + "\n".join(cfg) + "\n")
    print(f"config -> {args.config_out}")


if __name__ == "__main__":
    main()
