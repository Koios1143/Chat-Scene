#!/usr/bin/env python3
"""Translation of every SceneVerse HM3D region into its HM3D v0.2 glb frame.

SceneVerse stores each region self-centred (xy mean 0, z min 0) and unrotated; LSceneLLM's
``room_center`` only places regions approximately (it is a room centre, not the offset
SceneVerse subtracted). To project SceneVerse objects into views rendered from the glb we
need the exact offset, found here by a coarse grid search around ``room_center`` (median
point-to-surface distance) followed by translation-only ICP.

Output (``--out``): {scan: {sub: {"T": [x, y, z], "med": m, "p90": m}}}, glb = pcd + T.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


def grid_search(q, tree, c, half, step, dz_lo, dz_hi, tol, chunk=4_000_000):
    """Offset on a grid around ``c`` with the largest fraction of ``q`` within ``tol`` of the
    surface; all offsets are queried in batched KD-tree calls."""
    g = np.arange(-half, half + 1e-6, step)
    Ts = np.array([c + (dx, dy, dz) for dz in np.arange(dz_lo, dz_hi + 1e-6, step) for dx in g for dy in g])
    per = max(1, chunk // len(q))
    score = []
    for j in range(0, len(Ts), per):
        pts = (q[None] + Ts[j:j + per, None]).reshape(-1, 3)
        d = tree.query(pts, distance_upper_bound=tol, workers=-1)[0].reshape(-1, len(q))
        score.append((d < tol).mean(1))
    return Ts[int(np.argmax(np.concatenate(score)))]


def refine(p, tree, surf, T):
    for _ in range(30):
        d, i = tree.query(p + T, workers=-1)
        k = d < max(0.03, np.percentile(d, 60))
        step = np.median(surf[i[k]] - (p + T)[k], 0)
        T = T + step
        if np.abs(step).max() < 1e-4:
            break
    return T


BAD_MED = 0.02  # median point-to-mesh distance above which --redo_bad searches wider


def register(p, tree, surf, c, half=2.5, dz=(-2.5, 0.5)):
    rng = np.random.default_rng(0)
    q = p[rng.choice(len(p), min(len(p), 800), replace=False)]
    T = grid_search(q, tree, c, half, 0.25, dz[0], dz[1], tol=0.15)  # coarse
    T = grid_search(q, tree, T, 0.2, 0.05, -0.2, 0.2, tol=0.05)    # fine
    T = refine(p, tree, surf, T)
    d, _ = tree.query(p + T, workers=-1)
    return T, float(np.median(d)), float(np.percentile(d, 90)), float((d < 0.03).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_data", required=True)
    ap.add_argument("--glb_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scans", nargs="*")
    ap.add_argument("--xr_dir", default="", help="only regions used by XR-QA / XR-SceneCaption")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th scan starting at i")
    ap.add_argument("--redo_bad", action="store_true",
                    help=f"re-register regions of --out with median > {BAD_MED} m over xy +-5 m, z -4..+1.5 m")
    args = ap.parse_args()
    rc_dir = os.path.join(args.scan_data, "room_center")
    need = None
    if args.xr_dir:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from xr_common import xr_regions
        need = set(xr_regions(args.xr_dir))
    scans = args.scans or sorted(f[:-4] for f in os.listdir(rc_dir))
    i, n = map(int, args.shard.split("/"))
    scans = scans[i::n]
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for scan in scans:
        redo = [sub for sub, v in out.get(scan, {}).items() if v["med"] > BAD_MED] if args.redo_bad else []
        if scan in out and not redo:
            continue
        h = scan.split("-")[1]
        mesh = trimesh.load(os.path.join(args.glb_root, scan, f"{h}.glb"), force="mesh")
        surf = trimesh.sample.sample_surface(mesh, 3_000_000, seed=0)[0]
        tree = cKDTree(surf)
        rc = torch.load(os.path.join(rc_dir, f"{scan}.pth"), weights_only=False)
        res = dict(out.get(scan, {}))
        for sub in redo or sorted(rc):
            f = os.path.join(args.scan_data, "pcd_with_global_alignment", f"{scan}_{sub}.pth")
            if not os.path.exists(f) or (need is not None and f"{scan}_{sub}" not in need):
                continue
            p = torch.load(f, weights_only=False)[0][::5].astype(np.float64)
            wide = dict(half=5.0, dz=(-4.0, 1.5)) if redo else {}
            T, med, p90, inl = register(p, tree, surf, np.asarray(rc[sub]["center"], float), **wide)
            new = {"T": T.round(4).tolist(), "med": round(med, 4), "p90": round(p90, 4), "inl3cm": round(inl, 4)}
            if redo and new["inl3cm"] <= res[sub]["inl3cm"]:
                print(scan, sub, "wide search no better:", new, flush=True)
                continue
            res[sub] = new
            print(scan, sub, res[sub], flush=True)
        out[scan] = res
        json.dump(out, open(args.out + ".tmp", "w"))
        os.replace(args.out + ".tmp", args.out)


if __name__ == "__main__":
    main()
