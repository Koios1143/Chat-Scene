#!/usr/bin/env python3
"""Uni3D (uni3d-g) features of every GT object in every SceneVerse HM3D region used by XR.

Objects are encoded per region with Beacon3D's extractor (``beacon3d/extract_uni3d_gt.py``:
object points centred + max-radius scaled, colour / 255, ``Uni3D.encode_pc``), since a
Uni3D feature only sees the object itself. XR scenes are unions of regions and are put
together later (``xr_eval/build_xr.py``).

Output ``<out_dir>/regions_uni3d.<shard>.pt``:
    {"<scan>_<sub>": {"ids": [inst id], "labels": [str], "npts": [int],
                      "locs": float32[N, 6] (centre, size; region-local coords),
                      "feats": float32[N, 1024]}}
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "beacon3d"))
from extract_uni3d_gt import encode_object, load_sceneverse, load_uni3d  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xr_common import xr_regions  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xr_dir", required=True)
    ap.add_argument("--scan_data", required=True)
    ap.add_argument("--ckpt", required=True, help="uni3d-g model.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    i, n = map(int, args.shard.split("/"))
    regions = xr_regions(args.xr_dir)[i::n]
    out = os.path.join(args.out_dir, f"regions_uni3d.{i}.pt")
    res = torch.load(out) if os.path.exists(out) else {}
    torch.manual_seed(0)
    model = load_uni3d(args.ckpt, args.device)
    for k, reg in enumerate(regions):
        if reg in res:
            continue
        coords, colors, masks, labels = load_sceneverse(args.scan_data, reg)
        ids = sorted(masks)
        feats, locs, npts = [], [], []
        for oid in ids:
            m = masks[oid]
            npts.append(int(m.sum()))
            if m.sum() == 0:
                feats.append(torch.zeros(1024))
                locs.append(np.zeros(6, dtype=np.float32))
                continue
            lo, hi = coords[m].min(0), coords[m].max(0)
            # a single point (or coincident points) has radius 0 -> NaN after Uni3D's scaling;
            # such objects get zeros, like objects without points
            feats.append(encode_object(model, coords[m], colors[m], args.device) if (hi > lo).any()
                         else torch.zeros(1024))
            locs.append(np.concatenate([(lo + hi) / 2.0, hi - lo]).astype(np.float32))
        res[reg] = {
            "ids": ids,
            "labels": [labels[o] for o in ids],
            "npts": npts,
            "locs": torch.from_numpy(np.stack(locs)) if locs else torch.zeros(0, 6),
            "feats": torch.stack(feats) if feats else torch.zeros(0, 1024),
        }
        print(f"[{k + 1}/{len(regions)}] {reg}: {len(ids)} objects", flush=True)
        if (k + 1) % 50 == 0:
            torch.save(res, out + ".tmp")
            os.replace(out + ".tmp", out)
    torch.save(res, out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"wrote {len(res)} regions to {out}")


if __name__ == "__main__":
    main()
