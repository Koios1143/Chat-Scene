#!/usr/bin/env python3
"""Cosine between this extractor's ScanNet GT Uni3D features and Chat-Scene's official file."""

import argparse

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--mine", required=True)
ap.add_argument("--official", required=True)
args = ap.parse_args()
mine = torch.load(args.mine, map_location="cpu")
off = torch.load(args.official, map_location="cpu")
cos = [float(F.cosine_similarity(v, off[k], dim=0)) for k, v in mine.items() if k in off]
miss = [k for k in mine if k not in off]
cos_t = torch.tensor(cos)
print(f"{len(cos)} shared keys, {len(miss)} not in official (e.g. {miss[:3]})")
print(f"cosine mean {cos_t.mean():.4f}  median {cos_t.median():.4f}  min {cos_t.min():.4f}  "
      f"p10 {cos_t.quantile(0.1):.4f}  frac>0.99 {(cos_t > 0.99).float().mean():.3f}")
