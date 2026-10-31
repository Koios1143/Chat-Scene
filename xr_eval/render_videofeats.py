#!/usr/bin/env python3
"""Chat-Scene's 2D token (DINOv2 multi-view object features) for SceneVerse HM3D regions.

SceneVerse's HM3D release has no images, so RGB-D views are rendered from the HM3D v0.2
glb (textured mesh, lighting baked into the textures -> flat shading) and fused exactly as
for 3RScan / MultiScan (``beacon3d/extract_videofeats_gt.py``: DINOv2-large patch grid,
OpenScene projection + depth test, convex-hull crop, area-weighted mean).

Views (per region, in the glb frame; region points = SceneVerse points + the offset from
``xr_eval/register_regions.py``):
* 640 x 480, ScanNet's intrinsics (fx = fy = 577.87, ~58 deg horizontal FOV);
* camera at 1.5 m above the region floor (0.5th percentile of its point heights; kept 0.3 m
  below its top), on a 0.75 m grid over the region, 8 yaws, 20 deg downward pitch;
* a position is kept if it is >= 0.25 m from every region point, a straight-down view hits
  the mesh 1-2 m below (floor, not furniture or outside), and >= 5 of its 8 views have a
  median depth >= 0.6 m (not facing a wall at arm's length; in small rooms where no position
  reaches 5, the positions with the most such views); at most MAX_POS positions
  (farthest-point subset, starting nearest the region centre).

Output ``<out_dir>/regions_videofeats.<shard>.pt`` = {"<scan>_<sub>": {inst id: float32[1024]}}
(objects never seen get no entry; Chat-Scene then feeds zeros, as upstream).
"""

import argparse
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "beacon3d"))
sys.path.insert(0, HERE)
from xr_common import xr_regions  # noqa: E402
from extract_videofeats_gt import Dinov2, fuse  # noqa: E402

W, H, F = 640, 480, 577.87
K = np.array([[F, 0, (W - 1) / 2], [0, F, (H - 1) / 2], [0, 0, 1]])
CAM_H, GRID_STEP, CLEAR, MAX_POS = 1.5, 0.75, 0.25, 12
YAWS = np.deg2rad(np.arange(0, 360, 45))
PITCH = np.deg2rad(-20)
CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])


def camera_positions(pts, render):
    """render(c2w) -> (color, depth). Returns [(position, [(c2w, color, depth) x 8])]."""
    zf, z1 = np.percentile(pts[:, 2], 0.5), pts[:, 2].max()
    zc = min(zf + CAM_H, z1 - 0.3)
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    xs = np.arange(lo[0] + GRID_STEP / 2, hi[0], GRID_STEP)
    ys = np.arange(lo[1] + GRID_STEP / 2, hi[1], GRID_STEP)
    cand = np.array([(x, y, zc) for x in xs for y in ys]).reshape(-1, 3)
    if len(cand):
        cand = cand[cKDTree(pts).query(cand)[0] >= CLEAR]
    scored = []
    for pos in cand:
        down = render(look_c2w(pos, 0.0, np.deg2rad(-89.9)))[1]
        d = down[H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20]
        d = d[d > 0]
        if not len(d) or not 1.0 <= np.median(d) <= 2.0:
            continue
        views = [(c2w, *render(c2w)) for c2w in (look_c2w(pos, yaw, PITCH) for yaw in YAWS)]
        open_ = sum(bool((dep > 0).any()) and np.median(dep[dep > 0]) >= 0.6 for _, _, dep in views)
        scored.append((open_, pos, views))
    # >= 5 open views; in small rooms where no position has that many, the most open ones
    best = max((o for o, _, _ in scored), default=0)
    good = [(pos, views) for o, pos, views in scored if o >= min(5, best) and o > 0]
    if not good:  # fall back to the region centre
        pos = np.r_[pts[:, :2].mean(0), zc]
        return [(pos, [(c2w, *render(c2w)) for c2w in (look_c2w(pos, yaw, PITCH) for yaw in YAWS)])]
    P = np.array([g[0] for g in good])
    sel = [int(np.argmin(np.linalg.norm(P[:, :2] - pts[:, :2].mean(0), axis=1)))]
    dist = np.linalg.norm(P - P[sel[0]], axis=1)
    while len(sel) < min(MAX_POS, len(P)):
        j = int(np.argmax(dist))
        sel.append(j)
        dist = np.minimum(dist, np.linalg.norm(P - P[j], axis=1))
    return [good[j] for j in sel]


def look_c2w(pos, yaw, pitch):
    """OpenCV camera-to-world (x right, y down, z forward), world z up."""
    z = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, pos
    return c2w


def load_region(scan_data, reg, T):
    pcd = torch.load(os.path.join(scan_data, "pcd_with_global_alignment", f"{reg}.pth"), weights_only=False)
    lab = torch.load(os.path.join(scan_data, "instance_id_to_label", f"{reg}.pth"), weights_only=False)
    pts = np.asarray(pcd[0], np.float64) + np.asarray(T)
    inst = np.asarray(pcd[2]).astype(np.int64)
    inst = np.where(np.isin(inst, [int(k) for k in lab]), inst, -1)
    return pts, inst


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xr_dir", required=True)
    ap.add_argument("--scan_data", required=True)
    ap.add_argument("--glb_root", required=True)
    ap.add_argument("--offsets", required=True, help="xr_eval/register_regions.py output")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--regions", nargs="*", help="only these <scan>_<sub> (debug)")
    ap.add_argument("--dump_dir", default="", help="save the rendered views (debug)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    offsets = json.load(open(args.offsets))
    regions = args.regions or xr_regions(args.xr_dir)
    by_scan = {}
    for reg in regions:
        scan, sub = reg.rsplit("_", 1)
        by_scan.setdefault(scan, []).append(sub)
    i, n = map(int, args.shard.split("/"))
    scans = sorted(by_scan)[i::n]
    out = os.path.join(args.out_dir, f"regions_videofeats.{i}.pt")
    res = torch.load(out) if os.path.exists(out) else {}
    dino = Dinov2(args.device)
    renderer = pyrender.OffscreenRenderer(W, H)
    for k, scan in enumerate(scans):
        todo = [s for s in by_scan[scan] if f"{scan}_{s}" not in res]
        if not todo:
            continue
        h = scan.split("-")[1]
        scene = pyrender.Scene.from_trimesh_scene(trimesh.load(os.path.join(args.glb_root, scan, f"{h}.glb")),
                                                  bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
        cam = scene.add(pyrender.IntrinsicsCamera(F, F, K[0, 2], K[1, 2], znear=0.05, zfar=60.0))
        for sub in todo:
            reg = f"{scan}_{sub}"
            pts, inst = load_region(args.scan_data, reg, offsets[scan][sub]["T"])
            def render(c2w):
                scene.set_pose(cam, c2w @ CV2GL)
                color, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
                return color[..., :3], depth.astype(np.float64)

            frames = [(Image.fromarray(color), c2w, K, depth, K, False)
                      for _, views in camera_positions(pts, render) for c2w, color, depth in views]
            if args.dump_dir:
                os.makedirs(os.path.join(args.dump_dir, reg), exist_ok=True)
                for f, fr in enumerate(frames):
                    fr[0].save(os.path.join(args.dump_dir, reg, f"{f:03d}.jpg"))
                    np.save(os.path.join(args.dump_dir, reg, f"{f:03d}_c2w.npy"), fr[1])
            feats = fuse(frames, pts, inst, dino)
            res[reg] = {int(o): v for o, v in feats.items()}
            n_obj = len(set(inst.tolist()) - {-1})
            print(f"[{k + 1}/{len(scans)}] {reg}: {len(frames)} views, {len(feats)}/{n_obj} objects seen", flush=True)
        scene.clear()
        torch.save(res, out + ".tmp")
        os.replace(out + ".tmp", out)
    renderer.delete()
    print(f"wrote {len(res)} regions to {out}")


if __name__ == "__main__":
    main()
