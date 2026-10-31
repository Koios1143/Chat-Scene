"""Helpers shared by the xr_eval scripts (no heavy imports)."""

import json
import os


def xr_regions(xr_dir):
    regs = set()
    for split in ("train", "val"):
        for x in json.load(open(os.path.join(xr_dir, "XR-SceneCaption", f"{split}.json"))):
            regs |= {f"{x['scan_id']}_{r}" for r in x["room_id"].split("-")}
        for scan, eps in json.load(open(os.path.join(xr_dir, "XR-QA", f"{split}.json"))).items():
            for ep in eps:
                for qa in ep["qa_pairs"]:
                    regs |= {f"{qa['scan_name']}_{r}" for r in qa["region_id"].split("-")}
    return sorted(regs)
