#!/usr/bin/env python3
"""Beacon3D -> Chat-Scene val annotations + a config file that registers them.

For each domain and task this writes ``annotations/beacon3d_<domain>_<task>.json`` in
Chat-Scene's val format (``scene_id`` / ``obj_id`` / ``prompt`` / ``ref_captions`` /
``qid``) and ``scripts/beacon3d_config.py`` = ``scripts/config.py`` plus
``val_file_dict`` entries ``beacon3d_<domain>_<task>`` (paths relative to the repo root):

* scannet   -> Chat-Scene's official GT-mask files (``scannet_gt_uni3d_feats.pt``,
               ``scannet_gt_videofeats.pt``, ``scannet_val_attributes.pt``); object id =
               ScanNet objectId, which is what those files are keyed by.
* 3rscan / multiscan -> ``extract_uni3d_gt.py`` output (Uni3D over GT masks, no 2D
               feature file so the image token is zeros); scenes are aliased
               (``<domain>_scene_map.json``) and object ids are columns.

``qid`` is the item's index in Beacon3D's ``<domain>_{grounding,qa}.json`` so the
predictions (which Chat-Scene sorts) can be put back in Beacon3D order.
Prompts: grounding uses the template below; QA uses Chat-Scene's
ScanQA suffix ``Answer the question using a single word or phrase.``.
"""

import argparse
import json
import os

import torch

GROUNDING_PROMPT = "What's the ID of the object that corresponds to the description \"{}\"?"
QA_SUFFIX = " Answer the question using a single word or phrase."


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--beacon_root", default="beacon3d/Beacon3D", help="Beacon3D data dir")
    ap.add_argument("--official_dir", default="annotations", help="Chat-Scene HF annotations/ dir")
    ap.add_argument("--uni3d_dir", default="beacon3d/features", help="extract_uni3d_gt.py --out_dir")
    ap.add_argument("--anno_dir", default="annotations")
    ap.add_argument("--config_out", default="scripts/beacon3d_config.py")
    ap.add_argument("--domains", default="scannet,3rscan,multiscan")
    ap.add_argument(
        "--scannet_variants",
        default="myfeat,no2d",
        help="Extra ScanNet controls: 'myfeat' = this repo's Uni3D extraction + official 2D "
        "(isolates the 3D-feature approximation used for 3RScan/MultiScan); 'no2d' = official "
        "Uni3D with the image token zeroed (the 3RScan/MultiScan input condition).",
    )
    args = ap.parse_args()
    os.makedirs(args.anno_dir, exist_ok=True)
    entries = {}
    official_uni3d = os.path.join(args.official_dir, "scannet_gt_uni3d_feats.pt")
    official_video = os.path.join(args.official_dir, "scannet_gt_videofeats.pt")
    official_attrs = os.path.join(args.official_dir, "scannet_val_attributes.pt")
    variants = {
        "scannet": [official_uni3d, official_video, official_attrs],
        "scannet_myfeat": [os.path.join(args.uni3d_dir, "scannet_gt_uni3d_feats.pt"), official_video, official_attrs],
        "scannet_no2d": [official_uni3d, None, official_attrs],
    }
    domains = [d for d in args.domains.split(",") if d]
    if "scannet" in domains:
        domains += [f"scannet_{v}" for v in args.scannet_variants.split(",") if v]
    for domain in domains:
        bdir = os.path.join(args.beacon_root, "data", domain.split("_")[0])
        if domain.startswith("scannet"):
            attrs = torch.load(official_attrs, map_location="cpu")
            files = variants[domain]
            alias_of = {}
            col_of = {}
            for x in json.load(open(os.path.join(bdir, "scannet_grounding.json"))):  # noqa: E501
                s = x["scan_id"]
                alias_of[s] = s
                col_of[s] = {i: i for i in range(attrs[s]["locs"].shape[0])}
        else:
            smap = json.load(open(os.path.join(args.uni3d_dir, f"{domain}_scene_map.json")))
            files = [
                os.path.join(args.uni3d_dir, f"{domain}_gt_uni3d_feats.pt"),
                None,
                os.path.join(args.uni3d_dir, f"{domain}_gt_attributes.pt"),
            ]
            alias_of = {v["scene_id"]: a for a, v in smap.items()}
            col_of = {v["scene_id"]: {oid: c for c, oid in enumerate(v["object_ids"])} for v in smap.values()}

        src_domain = domain.split("_")[0]
        g = json.load(open(os.path.join(bdir, f"{src_domain}_grounding.json")))
        annos = []
        for i, it in enumerate(g):
            s = it["scan_id"]
            cols = [col_of[s][t] for t in it["target_id"]]
            annos.append(
                {
                    "scene_id": alias_of[s],
                    "obj_id": cols[0],
                    "ref_captions": [f"<OBJ{c:03}>." for c in cols],
                    "prompt": GROUNDING_PROMPT.format(it["utterance"]),
                    "qid": i,
                }
            )
        gpath = os.path.join(args.anno_dir, f"beacon3d_{domain}_grounding.json")
        json.dump(annos, open(gpath, "w"), indent=1)
        entries[f"beacon3d_{domain}_grounding"] = files + [gpath]

        q = json.load(open(os.path.join(bdir, f"{src_domain}_qa.json")))
        annos = []
        for i, it in enumerate(q):
            s = it["scene_id"]
            annos.append(
                {
                    "scene_id": alias_of[s],
                    "obj_id": col_of[s][it["object_ids"][0]],
                    "prompt": it["question"] + QA_SUFFIX,
                    "ref_captions": list(it["answers"]),
                    "qid": i,
                }
            )
        qpath = os.path.join(args.anno_dir, f"beacon3d_{domain}_qa.json")
        json.dump(annos, open(qpath, "w"), indent=1)
        entries[f"beacon3d_{domain}_qa"] = files + [qpath]
        print(f"[{domain}] {len(g)} grounding, {len(q)} qa -> {gpath}, {qpath}")

    base = open("scripts/config.py").read()
    lines = ["\n\n# ---- Beacon3D (generated by beacon3d/build_annos.py) ----", "val_file_dict.update({"]
    for k, v in entries.items():
        lines.append(f"    {k!r}: {v!r},")
    lines.append("})\n")
    with open(args.config_out, "w") as f:
        f.write(base + "\n".join(lines))
    print(f"config -> {args.config_out} ({len(entries)} val entries)")


if __name__ == "__main__":
    main()
