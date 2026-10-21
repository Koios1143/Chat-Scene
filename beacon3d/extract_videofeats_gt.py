#!/usr/bin/env python3
"""DINOv2 multi-view object features (Chat-Scene's 2D token) for Beacon3D 3RScan / MultiScan.

Reproduces Chat-Scene's ``scannet_videofeat.py`` (ZzZZCHS/openscene fork,
``scripts/feature_fusion``), which made the official ``scannet_gt_videofeats.pt``:

* every frame -> ``facebook/dinov2-large`` with its default processor (shortest side 256,
  centre crop 224), patch tokens as a 16 x 16 x 1024 grid;
* every GT object's points are projected into every frame (OpenScene's
  ``PointCloudToImageMapper``: rounded pixel, 10-px border cut, visible if
  ``|sensor depth - point depth| <= 0.25 * sensor depth``);
* per frame, the 2D convex hull of the object's visible pixels: if its area is at least
  one grid cell (``H/16 * W/16``), the grid cells covering the hull's bounding box are
  averaged (the box is mapped onto the 16 x 16 grid as if the grid spanned the whole
  image, exactly as upstream does, although the processor centre-crops);
* frames are fused by an area-weighted mean; objects never seen get no entry (Chat-Scene
  then feeds zeros, as for the official file).

What differs from ScanNet, and why:

* Sensor depth has its own resolution / intrinsics here (ScanNet's export resampled both to
  640 x 480), so the depth test projects with the depth intrinsics and the pixel position
  uses the colour intrinsics; colour and depth share the camera centre in both datasets.
* 3RScan stores frames rotated by 90 deg (world-up points to image-left in every frame), so
  colour frames are rotated clockwise to upright before DINOv2, and pixel positions with
  them; ScanNet and MultiScan frames are upright.
* Frame rate: ScanNet used every 20th frame of 30 Hz (1.5 Hz). MultiScan (60 Hz video) uses
  every 40th frame; 3RScan's sequences are already sparse keyframes and are used in full.

Geometry (checked by back-projecting sensor depth onto the mesh, median error per frame):
* 3RScan: ``frame-*.pose.txt`` is camera-to-world in the mesh frame (2.7 cm);
  ``labels.instances.annotated.v2.ply`` vertices are SceneVerse's points, ``objectId`` its
  instance id (= Beacon3D object id).
* MultiScan: camera-to-mesh = ``inv(align^T) @ transform^T`` with the ARKit (OpenGL)
  camera axes flipped to OpenCV (0.9 cm). SceneVerse's cloud is the ``.ply`` vertices (or,
  for large scenes, a subsample of them) shifted by a constant, so SceneVerse's points and
  instance ids are used, shifted back (``sceneverse_to_mesh``).

Output: ``<out_dir>/<domain>_gt_videofeats.pt`` = {f"{alias}_{col:02}": float32[1024]},
keyed like ``<domain>_gt_uni3d_feats.pt`` through ``<domain>_scene_map.json``.
"""

import argparse
import io
import json
import os
import subprocess
import tempfile
import zipfile
import zlib
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from scipy.spatial import ConvexHull

VIS_THR, CUT_BOUND, GRID = 0.25, 10, 16


class Dinov2:
    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images, bz=64):
        out = []
        for i in range(0, len(images), bz):
            inputs = self.processor(images=images[i:i + bz], return_tensors="pt").to(self.device)
            h = self.model(**inputs).last_hidden_state
            out.append(h[:, 1:].float().cpu().reshape(-1, GRID, GRID, 1024))
        return torch.cat(out, 0)


def visible_pixels(pts, c2w, K_col, color_hw, depth, K_depth):
    """OpenScene's compute_mapping, with separate colour / depth intrinsics.
    Returns (row, col) in the colour image and a visibility mask."""
    w2c = np.linalg.inv(c2w)
    p = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = p[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    u = np.round(p[:, 0] * K_col[0, 0] / zs + K_col[0, 2]).astype(int)
    v = np.round(p[:, 1] * K_col[1, 1] / zs + K_col[1, 2]).astype(int)
    H, W = color_hw
    inside = (z > 0) & (u >= CUT_BOUND) & (v >= CUT_BOUND) & (u < W - CUT_BOUND) & (v < H - CUT_BOUND)
    ud = np.round(p[:, 0] * K_depth[0, 0] / zs + K_depth[0, 2]).astype(int)
    vd = np.round(p[:, 1] * K_depth[1, 1] / zs + K_depth[1, 2]).astype(int)
    Hd, Wd = depth.shape
    inside &= (ud >= 0) & (vd >= 0) & (ud < Wd) & (vd < Hd)
    idx = np.nonzero(inside)[0]
    d = depth[vd[idx], ud[idx]]
    ok = np.abs(d - z[idx]) <= VIS_THR * d
    vis = np.zeros(len(pts), bool)
    vis[idx[ok]] = True
    return v, u, vis


def fuse(frames, pts, inst, dino):
    """frames: list of (PIL image, c2w, K_col, depth, K_depth, rotate_cw). -> {inst_id: feat}."""
    images = []
    for img, *_, rot in frames:
        images.append(img.rotate(-90, expand=True) if rot else img)
    feats = dino(images)
    ids = [i for i in np.unique(inst) if i >= 0]
    members = {i: np.nonzero(inst == i)[0] for i in ids}
    acc = defaultdict(list)
    for f, (img, c2w, K_col, depth, K_depth, rot) in enumerate(frames):
        W, H = img.size
        row, col, vis = visible_pixels(pts, c2w, K_col, (H, W), depth, K_depth)
        if rot:  # clockwise rotation: (row, col) -> (col, H - 1 - row); image becomes W x H
            row, col, H, W = col, H - 1 - row, W, H
        dH, dW = H // GRID, W // GRID
        for i in ids:
            m = members[i][vis[members[i]]]
            if len(m) < 3:
                continue
            xy = np.stack([row[m], col[m]], 1)
            try:
                hull = ConvexHull(xy)
            except Exception:
                continue
            if hull.volume < dH * dW:
                continue
            hv = xy[hull.vertices]
            x0, y0 = hv.min(0)
            x1, y1 = hv.max(0)
            crop = feats[f, x0 // dH:(x1 + dH - 1) // dH, y0 // dW:(y1 + dW - 1) // dW]
            acc[i].append((hull.volume, crop.flatten(0, 1).mean(0)))
    out = {}
    for i, lst in acc.items():
        tot = sum(w for w, _ in lst)
        out[int(i)] = sum((w / tot) * f for w, f in lst).float()
    return out


def frames_3rscan(scan_dir):
    z = zipfile.ZipFile(os.path.join(scan_dir, "sequence.zip"))
    info = dict(l.split(" = ") for l in z.read("_info.txt").decode().strip().splitlines())
    K_col = np.array(info["m_calibrationColorIntrinsic"].split(), float).reshape(4, 4)[:3, :3]
    K_dep = np.array(info["m_calibrationDepthIntrinsic"].split(), float).reshape(4, 4)[:3, :3]
    shift = float(info["m_depthShift"])
    frames = []
    for i in range(int(info["m_frames.size"])):
        c2w = np.loadtxt(io.BytesIO(z.read(f"frame-{i:06d}.pose.txt")))
        if not np.isfinite(c2w).all():
            continue
        img = Image.open(io.BytesIO(z.read(f"frame-{i:06d}.color.jpg"))).convert("RGB")
        depth = np.array(Image.open(io.BytesIO(z.read(f"frame-{i:06d}.depth.pgm"))), float) / shift
        frames.append((img, c2w, K_col, depth, K_dep, True))
    return frames


def frames_multiscan(zip_path, scene, step, tmp):
    zf = zipfile.ZipFile(zip_path)
    name = lambda ext: f"{scene}/{scene}.{ext}"  # noqa: E731
    cams = [json.loads(l) for l in zf.read(name("jsonl")).decode().splitlines() if l.strip()]
    meta = json.loads(zf.read(name("json")))
    dstream = next(s for s in meta["streams"] if s["type"] == "lidar_sensor")
    dh, dw = dstream["resolution"]
    T = np.array(json.loads(zf.read(name("align.json")))["coordinate_transform"]).reshape(4, 4).T
    raw = zlib.decompressobj(-15).decompress(zf.read(name("depth.zlib")))
    depth_all = np.frombuffer(raw, np.float16).reshape(len(cams), dh, dw)
    keep = list(range(0, len(cams), step))
    video = os.path.join(tmp, "v.mp4")
    with open(video, "wb") as f:
        f.write(zf.read(name("mp4")))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-vf", f"select=not(mod(n\\,{step}))",
                    "-vsync", "0", "-q:v", "2", os.path.join(tmp, "f%05d.jpg")], check=True)
    n_dec = len([f for f in os.listdir(tmp) if f.endswith(".jpg")])
    assert n_dec >= len(keep) - 1, f"{scene}: decoded {n_dec} frames, expected {len(keep)}"
    keep = keep[:n_dec]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL camera axes -> OpenCV
    frames = []
    for j, k in enumerate(keep):
        img = Image.open(os.path.join(tmp, f"f{j + 1:05d}.jpg")).convert("RGB")
        W, H = img.size
        K_col = np.array(cams[k]["intrinsics"]).reshape(3, 3).T
        K_dep = K_col.copy()
        K_dep[0] *= dw / W
        K_dep[1] *= dh / H
        c2w = np.linalg.inv(T) @ np.array(cams[k]["transform"]).reshape(4, 4).T @ flip
        frames.append((img, c2w, K_col, depth_all[k].astype(np.float64), K_dep, False))
    return frames


def sceneverse_to_mesh(sv, raw, scene, iters=10):
    """SceneVerse's MultiScan cloud = the mesh vertices (all, or a subsample for large
    scenes) shifted by a constant. Recover the shift by translation-only ICP and return the
    SceneVerse points in the mesh frame; they must land on mesh vertices."""
    from scipy.spatial import cKDTree
    tree = cKDTree(raw)
    t = sv.mean(0) - raw.mean(0)
    for _ in range(iters):
        _, nn = tree.query(sv - t)
        t = (sv - raw[nn]).mean(0)
    dist, _ = tree.query(sv - t)
    assert np.median(dist) < 1e-3 and np.percentile(dist, 99) < 1e-2, (scene, np.median(dist), t)
    return sv - t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", required=True, choices=["3rscan", "multiscan"])
    ap.add_argument("--raw_dir", required=True, help="3RScan scans/ (per-scan dirs) or MultiScan scans/ (zips)")
    ap.add_argument("--scan_data", help="SceneVerse MultiScan scan_data/ (instance ids, multiscan only)")
    ap.add_argument("--feats_dir", default="beacon3d/features", help="has <domain>_scene_map.json; output dir")
    ap.add_argument("--multiscan_step", type=int, default=40, help="every n-th frame of the 60 Hz video")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = Dinov2(device)
    smap = json.load(open(os.path.join(args.feats_dir, f"{args.domain}_scene_map.json")))
    out_path = os.path.join(args.feats_dir, f"{args.domain}_gt_videofeats.pt")
    all_feats = {}
    for alias, v in sorted(smap.items()):
        scene = v["scene_id"]
        if args.domain == "3rscan":
            from plyfile import PlyData
            sd = os.path.join(args.raw_dir, scene)
            vert = PlyData.read(os.path.join(sd, "labels.instances.annotated.v2.ply"))["vertex"]
            pts = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
            inst = np.asarray(vert["objectId"]).astype(int)
            frames = frames_3rscan(sd)
        else:
            from plyfile import PlyData
            zp = os.path.join(args.raw_dir, f"{scene}.zip")
            with zipfile.ZipFile(zp) as zf:
                vert = PlyData.read(io.BytesIO(zf.read(f"{scene}/{scene}.ply")))["vertex"]
            raw = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
            sv = torch.load(os.path.join(args.scan_data, "pcd_with_global_alignment", f"{scene}.pth"),
                            weights_only=False)
            pts, inst = sceneverse_to_mesh(np.asarray(sv[0], np.float64), raw, scene), np.asarray(sv[2]).astype(int)
            with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
                frames = frames_multiscan(zp, scene, args.multiscan_step, tmp)
        feats = fuse(frames, pts, inst, dino)
        n_seen = 0
        for col, oid in enumerate(v["object_ids"]):
            if oid in feats:
                all_feats[f"{alias}_{col:02}"] = feats[oid]
                n_seen += 1
        print(f"{args.domain} {alias} ({scene}): {len(frames)} frames, {n_seen}/{len(v['object_ids'])} objects seen",
              flush=True)
    torch.save(all_feats, out_path)
    print(f"-> {out_path} ({len(all_feats)} objects)")


if __name__ == "__main__":
    main()
