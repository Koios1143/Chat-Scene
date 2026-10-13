#!/usr/bin/env python3
"""Uni3D (uni3d-g) object features for Beacon3D scenes over GT instance masks.

Reproduces Chat-Scene's ``extract_3d_feat`` (ZzZZCHS/Uni3D fork, main.py): every point
of the object, centred on its mean, scaled by the max radius, colour /255, duplicated
up to >= 64 points, then ``Uni3D.encode_pc`` (512 FPS groups x 64 kNN, EVA-giant
transformer, 1024-d output, NOT normalised -- Chat-Scene normalises at load time).
The point encoder is vendored below with a pure-torch farthest point sampling in
place of ``pointnet2_ops`` (same greedy rule, seeded at point 0, so the groups match).

Sources
-------
* ``scannet``: ``<scans>/<scene>/<scene>_vh_clean_2.ply`` (+ ``<scene>.txt``
  axisAlignment, applied as Chat-Scene does) and GT masks ``<scans>/<scene>/<id>.npy``.
  Used only to validate this extractor against Chat-Scene's official
  ``scannet_gt_uni3d_feats.pt``; the ScanNet eval itself uses the official file.
* ``3rscan`` / ``multiscan``: SceneVerse ``pcd_with_global_alignment/<scene>.pth`` =
  ``(coord, color, instance_id)`` + ``instance_id_to_label``.

Outputs (``--out_dir``)
-----------------------
``<domain>_gt_uni3d_feats.pt``   {f"{alias}_{col:02}": float32[1024]}
``<domain>_gt_attributes.pt``    {alias: {"objects": [label], "locs": float32[N, 6]}}
``<domain>_scene_map.json``      {alias: {"scene_id": ..., "object_ids": [gt id per col]}}

``alias`` is a scene id of the form ``<prefix>NNNN_00`` because Chat-Scene recovers the
scene id of a feature key with ``'_'.join(key.split('_')[:2])``; ScanNet ids already
have that shape and keep their name. Columns are ascending GT id, the same order
"""

import argparse
import json
import os

import numpy as np
import timm
import torch
import torch.nn as nn

PREFIX = {"scannet": "", "3rscan": "rscan", "multiscan": "mscan"}


# ---------------------------------------------------------------------------
# Uni3D point encoder (vendored from ZzZZCHS/Uni3D models/point_encoder.py)
# ---------------------------------------------------------------------------


def fps_torch(xyz: torch.Tensor, n: int) -> torch.Tensor:
    """Farthest point sampling, [B, N, 3] -> [B, n, 3]; greedy from index 0 like pointnet2."""
    b, npts, _ = xyz.shape
    idx = torch.zeros(b, n, dtype=torch.long, device=xyz.device)
    dist = torch.full((b, npts), float("inf"), device=xyz.device)
    far = torch.zeros(b, dtype=torch.long, device=xyz.device)
    ar = torch.arange(b, device=xyz.device)
    for i in range(n):
        idx[:, i] = far
        centroid = xyz[ar, far].unsqueeze(1)
        d = ((xyz - centroid) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return xyz[ar.unsqueeze(1), idx]


def knn_point(nsample, xyz, new_xyz):
    dist = -2 * torch.matmul(new_xyz, xyz.permute(0, 2, 1))
    dist += torch.sum(new_xyz**2, -1).unsqueeze(-1)
    dist += torch.sum(xyz**2, -1).unsqueeze(1)
    return torch.topk(dist, nsample, dim=-1, largest=False, sorted=False)[1]


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group, self.group_size = num_group, group_size

    def forward(self, xyz, color):
        b, n, _ = xyz.shape
        center = fps_torch(xyz, self.num_group)
        idx = knn_point(self.group_size, xyz, center)
        idx = (idx + torch.arange(0, b, device=xyz.device).view(-1, 1, 1) * n).view(-1)
        nb = xyz.view(b * n, -1)[idx].view(b, self.num_group, self.group_size, 3)
        nc = color.view(b * n, -1)[idx].view(b, self.num_group, self.group_size, 3)
        nb = nb - center.unsqueeze(2)
        return nb, center, torch.cat((nb, nc), dim=-1)


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        pg = point_groups.reshape(bs * g, n, 6)
        feature = self.first_conv(pg.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        return torch.max(feature, dim=2, keepdim=False)[0].reshape(bs, g, self.encoder_channel)


class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, pc_feat_dim=1408, embed_dim=1024, group_size=64,
                 num_group=512, pc_encoder_dim=512):
        super().__init__()
        self.group_divider = Group(num_group, group_size)
        self.encoder = Encoder(pc_encoder_dim)
        self.encoder2trans = nn.Linear(pc_encoder_dim, pc_feat_dim)
        self.trans2embed = nn.Linear(pc_feat_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, pc_feat_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, pc_feat_dim))
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, pc_feat_dim))
        self.visual = point_transformer

    def forward(self, pts, colors):
        _, center, features = self.group_divider(pts, colors)
        tokens = self.encoder2trans(self.encoder(features))
        cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(tokens.size(0), -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1) + torch.cat((cls_pos, self.pos_embed(center)), dim=1)
        x = self.visual.pos_drop(x)
        for blk in self.visual.blocks:
            x = blk(x)
        x = self.visual.norm(x[:, 0, :])
        x = self.visual.fc_norm(x)
        return self.trans2embed(x)


class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc):
        return self.point_encoder(pc[:, :, :3].contiguous(), pc[:, :, 3:].contiguous())


def load_uni3d(ckpt_path, device):
    vit = timm.create_model("eva_giant_patch14_560.m30m_ft_in22k_in1k", pretrained=False, drop_path_rate=0.0)
    model = Uni3D(PointcloudEncoder(vit))
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["module"] if "module" in ck else ck
    if next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("point_encoder.visual.")]
    if missing or unexpected:
        raise RuntimeError(f"uni3d checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Scene loaders -> (coords [N,3] float32 aligned, colors [N,3] 0-255, {gt id: mask}, {gt id: label})
# ---------------------------------------------------------------------------


def fit_alignment(src_centers, dst_centers):
    """Rigid transform (R, t) with dst ~ R src + t, by Kabsch over matched object centres.

    ScanNet's ``axisAlignment`` is a rigid transform (a rotation about z plus a
    translation). The raw scans here carry no ``<scene>.txt``, but Chat-Scene's official
    ``scannet_val_attributes.pt`` holds every GT object's box in the aligned frame, so
    the transform is recovered from the box centres (residual ~1 cm over 30-50 objects).
    """
    src = np.asarray(src_centers, dtype=np.float64)
    dst = np.asarray(dst_centers, dtype=np.float64)
    sc, dc = src.mean(0), dst.mean(0)
    h = (src - sc).T @ (dst - dc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    t = dc - r @ sc
    resid = np.sqrt((((src @ r.T + t) - dst) ** 2).sum(1)).mean()
    return r.astype(np.float32), t.astype(np.float32), float(resid)


def load_scannet(scans_root, scene, align_attrs=None):
    import open3d as o3d

    d = os.path.join(scans_root, scene)
    mesh = o3d.io.read_triangle_mesh(os.path.join(d, f"{scene}_vh_clean_2.ply"))
    coords = np.asarray(mesh.vertices, dtype=np.float32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32) * 255.0
    align = np.eye(4, dtype=np.float32)
    txt = os.path.join(d, f"{scene}.txt")
    if os.path.exists(txt):
        for line in open(txt):
            if line.startswith("axisAlignment"):
                align = np.array([float(x) for x in line.strip().split()[-16:]], dtype=np.float32).reshape(4, 4)
    else:
        print(f"  [warn] {scene}: no {scene}.txt, coordinates left unaligned", flush=True)
    pts = np.concatenate([coords, np.ones((coords.shape[0], 1), dtype=np.float32)], 1)
    coords = (pts @ align.T)[:, :3]
    agg = json.load(open(os.path.join(d, f"{scene}.aggregation.json")))
    labels = {int(g["objectId"]): g["label"] for g in agg["segGroups"]}
    ids = sorted(int(f[:-4]) for f in os.listdir(d) if f.endswith(".npy") and f[:-4].isdigit())
    masks = {i: np.load(os.path.join(d, f"{i}.npy")) for i in ids}
    if align_attrs is not None and not os.path.exists(txt):
        ref = align_attrs[scene]["locs"][:, :3].numpy()
        src = np.stack([(coords[masks[i]].min(0) + coords[masks[i]].max(0)) / 2 for i in ids])
        r, t, resid = fit_alignment(src, ref[: len(ids)])
        coords = coords @ r.T + t
        print(f"  {scene}: alignment fitted from official boxes, residual {resid * 100:.1f} cm", flush=True)
    return coords, colors, masks, {i: labels.get(i, "object") for i in ids}


def load_sceneverse(scan_data, scene):
    pcd = torch.load(os.path.join(scan_data, "pcd_with_global_alignment", f"{scene}.pth"),
                     map_location="cpu", weights_only=False)
    lab = torch.load(os.path.join(scan_data, "instance_id_to_label", f"{scene}.pth"),
                     map_location="cpu", weights_only=False)
    coords = np.asarray(pcd[0], dtype=np.float32)
    colors = np.asarray(pcd[1], dtype=np.float32)
    if colors.max() <= 1.0:
        colors = colors * 255.0
    inst = np.asarray(pcd[2]).astype(np.int64)
    ids = sorted(int(k) for k in lab)
    return coords, colors, {i: inst == i for i in ids}, {i: str(lab[i]) for i in ids}


@torch.no_grad()
def encode_object(model, coords, colors, device):
    pc = coords - coords.mean(0)
    pc = pc / np.max(np.sqrt(np.sum(pc**2, axis=1)))
    x = torch.cat([torch.from_numpy(pc), torch.from_numpy(colors / 255.0)], dim=-1).float().unsqueeze(0).to(device)
    while x.shape[1] < 64:
        x = torch.cat([x, x], dim=1)
    return model.encode_pc(x).squeeze(0).float().cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", choices=list(PREFIX), required=True)
    ap.add_argument("--beacon_root", required=True)
    ap.add_argument("--ckpt", required=True, help="uni3d-g model.pt")
    ap.add_argument("--scans_root", default="", help="scannet: raw scans dir")
    ap.add_argument("--scan_data", default="", help="3rscan/multiscan: SceneVerse scan_data dir")
    ap.add_argument(
        "--align_from_attrs",
        default="",
        help="scannet only: official scannet_*_attributes.pt used to recover the axis "
        "alignment when the raw scan carries no <scene>.txt",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    bdir = os.path.join(args.beacon_root, "data", args.domain)
    g = json.load(open(os.path.join(bdir, f"{args.domain}_grounding.json")))
    q = json.load(open(os.path.join(bdir, f"{args.domain}_qa.json")))
    scenes = sorted({x["scan_id"] for x in g} | {x["scene_id"] for x in q})
    torch.manual_seed(0)
    align_attrs = torch.load(args.align_from_attrs, map_location="cpu") if args.align_from_attrs else None
    model = load_uni3d(args.ckpt, args.device)
    feats, attrs, scene_map = {}, {}, {}
    for k, scene in enumerate(scenes):
        if args.domain == "scannet":
            coords, colors, masks, labels = load_scannet(args.scans_root, scene, align_attrs)
            alias = scene
        else:
            coords, colors, masks, labels = load_sceneverse(args.scan_data, scene)
            alias = f"{PREFIX[args.domain]}{k:04d}_00"
        ids = sorted(masks)
        locs, objs = [], []
        for col, oid in enumerate(ids):
            m = masks[oid]
            if m.sum() < 10:
                print(f"  [warn] {scene} object {oid}: {int(m.sum())} points", flush=True)
            if m.sum() == 0:
                feats[f"{alias}_{col:02}"] = torch.zeros(1024)
                locs.append(np.zeros(6, dtype=np.float32))
            else:
                feats[f"{alias}_{col:02}"] = encode_object(model, coords[m], colors[m], args.device)
                pc = coords[m]
                lo, hi = pc.min(0), pc.max(0)
                locs.append(np.concatenate([(lo + hi) / 2.0, hi - lo]).astype(np.float32))
            objs.append(labels[oid])
        attrs[alias] = {"objects": objs, "locs": torch.from_numpy(np.stack(locs))}
        scene_map[alias] = {"scene_id": scene, "object_ids": ids}
        print(f"{scene} -> {alias}: {len(ids)} objects, {coords.shape[0]} points", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(feats, os.path.join(args.out_dir, f"{args.domain}_gt_uni3d_feats.pt"))
    torch.save(attrs, os.path.join(args.out_dir, f"{args.domain}_gt_attributes.pt"))
    with open(os.path.join(args.out_dir, f"{args.domain}_scene_map.json"), "w") as f:
        json.dump(scene_map, f, indent=1)
    print(f"wrote {len(feats)} object features for {len(scenes)} scenes to {args.out_dir}")


if __name__ == "__main__":
    main()
