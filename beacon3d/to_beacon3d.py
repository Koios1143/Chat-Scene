#!/usr/bin/env python3
"""Chat-Scene ``preds_*_beacon3d_<domain>_<task>.json`` -> Beacon3D infer files + scores.

Writes ``<result_dir>/<domain>/infer_{grounding,qa}.json`` aligned 1:1 with
``data/<domain>/<domain>_{grounding,qa}.json`` (via ``qid``), with ``correct`` for
grounding (first ``<OBJxxx>`` -> column -> GT id, in ``target_id``) and ``pred`` for QA,
then runs Beacon3D's evaluators from ``--beacon_root`` (the grounding one through
``beacon_grounding_shim.py``, which skips knowledge tags with no cases).
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

OBJ = re.compile(r"<OBJ(\d{3})>")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds_dir", required=True, help="Chat-Scene output_dir")
    ap.add_argument("--beacon_root", default="beacon3d/Beacon3D", help="Beacon3D data + evaluators")
    ap.add_argument("--uni3d_dir", default="beacon3d/features", help="scene maps for 3rscan/multiscan")
    ap.add_argument("--result_dir", default="beacon3d/results")
    ap.add_argument("--domains", default="scannet,3rscan,multiscan")
    ap.add_argument("--tasks", default="grounding,qa")
    ap.add_argument("--shim", default="beacon3d/beacon_grounding_shim.py",
                    help="wrapper around Beacon3D's evaluate_grounding.py")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--qa_model", default="gpt-4o-mini-2024-07-18",
                    help="Beacon3D QA judge (upstream default gpt-4o-2024-11-20)")
    args = ap.parse_args()

    for domain in [d for d in args.domains.split(",") if d]:
        src_domain = domain.split("_")[0]  # scannet_myfeat / scannet_no2d are ScanNet controls
        bdir = os.path.join(args.beacon_root, "data", src_domain)
        out_dir = os.path.abspath(os.path.join(args.result_dir, domain))
        os.makedirs(out_dir, exist_ok=True)
        if src_domain == "scannet":
            ids_of = None
        else:
            smap = json.load(open(os.path.join(args.uni3d_dir, f"{domain}_scene_map.json")))
            ids_of = {v["scene_id"]: v["object_ids"] for v in smap.values()}
        for task in [t for t in args.tasks.split(",") if t]:
            paths = [p for p in glob.glob(os.path.join(args.preds_dir, f"preds_*_beacon3d_{domain}_{task}.json"))
                     if "_rank" not in os.path.basename(p)]
            if len(paths) != 1:
                raise SystemExit(f"{domain}/{task}: expected one merged preds file, got {paths}")
            preds = {p["qid"]: p for p in json.load(open(paths[0]))}
            src = json.load(open(os.path.join(bdir, f"{src_domain}_{task}.json")))
            if sorted(preds) != list(range(len(src))):
                raise SystemExit(f"{domain}/{task}: {len(preds)} preds for {len(src)} items")
            infer = []
            for i, item in enumerate(src):
                p = preds[i]
                if task == "grounding":
                    scene = item["scan_id"]
                    n_obj = len(ids_of[scene]) if ids_of else None
                    m = OBJ.search(p["pred"])
                    col = int(m.group(1)) if m else None
                    if col is not None and ids_of is not None:
                        obj = ids_of[scene][col] if col < n_obj else None
                    else:
                        obj = col
                    infer.append({
                        "item_id": item["item_id"], "scan_id": scene, "pred": p["pred"],
                        "pred_object_id": obj, "target_id": item["target_id"],
                        "correct": obj is not None and obj in item["target_id"],
                    })
                else:
                    infer.append({
                        "question_id": item["question_id"], "scene_id": item["scene_id"],
                        "question": item["question"], "pred": p["pred"], "answers": item["answers"],
                    })
            infer_path = os.path.join(out_dir, f"infer_{task}.json")
            json.dump(infer, open(infer_path, "w"), indent=1)
            if task == "grounding":
                acc = sum(r["correct"] for r in infer) / len(infer)
                print(f"[{domain}] grounding case acc {acc * 100:.2f} ({len(infer)} items)")
            if not args.score:
                continue
            if task == "grounding":
                cmd = [sys.executable, os.path.abspath(args.shim), "--infer", infer_path, "--domain", src_domain]
            else:
                if not os.environ.get("OPENAI_API_KEY"):
                    print(f"[{domain}] qa: OPENAI_API_KEY unset, skipping")
                    continue
                # evaluate_qa.py silently reuses an existing processed file instead of re-judging
                processed = os.path.splitext(infer_path)[0] + "_processed.json"
                if os.path.exists(processed):
                    os.remove(processed)
                cmd = [sys.executable, "evaluate_qa.py", "--infer", infer_path, "--domain", src_domain,
                       "--model", args.qa_model, "--grounding",
                       os.path.join(out_dir, "infer_grounding_processed.json")]
            print(f"[{domain}] $ {' '.join(cmd)}", flush=True)
            res = subprocess.run(cmd, cwd=args.beacon_root, capture_output=True, text=True)
            with open(os.path.join(out_dir, f"score_{task}.txt"), "w") as f:
                f.write(res.stdout + ("" if res.returncode == 0 else "\n[stderr]\n" + res.stderr))
            print(res.stdout)
            if res.returncode != 0:
                print(res.stderr[-2000:])
                raise SystemExit(f"{domain}/{task}: evaluator failed")


if __name__ == "__main__":
    main()
