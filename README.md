# Chat-Scene Benchmark Evaluation

This is a fork of [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene) (NeurIPS 2024)
whose purpose is to **evaluate the released Chat-Scene checkpoint on a range of
benchmarks** under one reproducible setup, so its scores can be compared with other
models on equal footing. The first benchmark wired in is
[Beacon3D](https://github.com/beacon-3d/Beacon3D) (CVPR 2025), an object-centric
grounding + QA benchmark over ScanNet, 3RScan and MultiScan. The upstream model,
training and data code are untouched; everything this fork adds lives under
[`beacon3d/`](beacon3d/), and the run that produced the numbers below is committed in
full (`outputs/`, `beacon3d/results/`).

Upstream Chat-Scene's own README is in the git history (`git show 475e106:README.md`).

## Results: Beacon3D

Beacon3D, GT-mask setting (one object token per GT instance, the protocol the Beacon3D
leaderboard uses). *Case* = per-question accuracy; *Obj* = all questions about an object
answered correctly.

| Setting | 3D feature | 2D token | Grounding Case / Obj | QA Case / Obj |
| --- | --- | --- | --- | --- |
| ScanNet (official features) | official Uni3D | official DINOv2 | 62.03 / 41.56 | 50.09 / 9.87 |
| ScanNet, our Uni3D extraction | ours | official DINOv2 | 62.16 / 41.30 | 52.47 / 10.91 |
| ScanNet, **no 2D token** | official Uni3D | zeros | 17.36 / 6.75 | 40.93 / 6.75 |
| 3RScan | ours | zeros | 12.18 / 5.15 | 47.55 / 5.51 |
| MultiScan | ours | zeros | 11.42 / 3.14 | 41.98 / 5.66 |

The ScanNet row reproduces the Beacon3D leaderboard entry for Chat-Scene (61.6 / 42.3,
49.8 / 10.9). Per-item outputs, the official evaluators' full reports (per knowledge
type, grounding→QA chain analysis) and the chain figures are in
[`beacon3d/results/<setting>/`](beacon3d/results/).

### Read this before quoting the 3RScan / MultiScan numbers

Chat-Scene gives every object a 3D token (Uni3D) **and** a 2D token (DINOv2 features
projected from multi-view RGB). The 2D token needs RGB frames + camera poses, which are
not part of the SceneVerse point-cloud release used here, so on 3RScan and MultiScan the
2D token is all zeros. The *ScanNet, no 2D token* control shows that this alone drops
grounding from 62 to 17. The 3RScan / MultiScan grounding scores therefore measure the
missing-2D condition, not domain transfer; compare them with other models only on
ScanNet, or only under the same no-2D condition. QA is less affected because many
questions are answerable from the object label and language prior alone, which is
exactly what Beacon3D's chain analysis exposes (the G✗Q✓ share is 45% on 3RScan).

## Repository layout

| Path | What |
| --- | --- |
| `beacon3d/Beacon3D/` | Vendored copy of the Beacon3D repo at `bda1203` (Apache-2.0): its two evaluators, `utils.py`, and `data/` with the grounding/QA annotations, metadata and GPT system prompt for all three domains. Unmodified; see `VENDORED.md`. |
| `beacon3d/features/` | Our Uni3D-g features over GT masks for the Beacon3D scenes of all three domains (`<domain>_gt_uni3d_feats.pt`), their box attributes, and the scene-id map (below). 5.7 MB. |
| `beacon3d/results/` | Beacon3D infer files, evaluator reports and chain figures for the run in the table. |
| `outputs/beacon3d_20260929_230515/` | The raw Chat-Scene predictions and config of that run. |
| `beacon3d/build_env.sh` | Builds the Chat-Scene runtime as a uv venv (py3.9, torch 2.2.1+cu121, transformers 4.39.3, peft 0.9.0, flash-attn 2.5.6 wheel) plus `open3d` and `openai`. |
| `beacon3d/extract_uni3d_gt.py`, `extract_uni3d.sh` | Rebuilds `beacon3d/features/`. Vendors the Uni3D point encoder with a pure-torch farthest-point sampling so `pointnet2_ops` is not needed. |
| `beacon3d/compare_uni3d_official.py` | Cosine between our ScanNet extraction and Chat-Scene's official file (sanity check of the extractor). |
| `beacon3d/build_annos.py` | Beacon3D JSON → Chat-Scene val annotations (`annotations/beacon3d_<setting>_<task>.json`) + `scripts/beacon3d_config.py`. Also builds the two ScanNet controls. Regenerated on every run. |
| `beacon3d/run_beacon3d.sh` | Evaluate-only inference with the official checkpoint (torchrun over all visible GPUs; ~8 min for all ten val sets on 8 GPUs). |
| `beacon3d/to_beacon3d.py`, `score.sh`, `beacon_grounding_shim.py` | Predictions → Beacon3D infer files → official evaluators. The shim only skips knowledge tags with zero cases (3RScan / MultiScan have no *functionality* questions, which makes upstream `evaluate_grounding.py` divide by zero). |

## Setup

### External dependencies (not in this repo)

| Needed for | What | Where | Env var (default) |
| --- | --- | --- | --- |
| inference | Chat-Scene checkpoint `ckpt_01_3446.pth` (1.3 GB) | [HF `ZzZZCHS/Chat-Scene`](https://huggingface.co/datasets/ZzZZCHS/Chat-Scene) | `CKPT` (`pretrained/ckpt_01_3446.pth`) |
| inference | Chat-Scene's ScanNet GT-mask files: `scannet_gt_uni3d_feats.pt` (205 MB), `scannet_gt_videofeats.pt` (203 MB), `scannet_val_attributes.pt` | same HF release, `annotations/` | `OFFICIAL_DIR` (`annotations`) |
| inference | Vicuna-7B v1.5 | [HF `lmsys/vicuna-7b-v1.5`](https://huggingface.co/lmsys/vicuna-7b-v1.5) | `LLAMA` (`lmsys/vicuna-7b-v1.5`) |
| QA scoring | an OpenAI API key (GPT-4o judge, `gpt-4o-2024-11-20`) | | `OPENAI_API_KEY` |
| rebuilding features | Uni3D-g weights `model.pt` | [HF `BAAI/Uni3D`](https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-g/model.pt) | `UNI3D_CKPT` |
| rebuilding features | 3RScan and MultiScan scene point clouds with GT instance ids, in the [SceneVerse](https://scene-verse.github.io/) `scan_data/` layout (`pcd_with_global_alignment/<scene>.pth`, `instance_id_to_label/<scene>.pth`) | SceneVerse release | `RSCAN_SCAN_DATA`, `MULTISCAN_SCAN_DATA` |
| validating the extractor (optional) | raw ScanNet scans (`<scene>_vh_clean_2.ply`, `<scene>.aggregation.json`) with one `<objectId>.npy` GT mask per object | ScanNet | `SCANNET_SCANS` |

Only the first three rows are needed to reproduce the table: the Beacon3D data and
evaluators, and our 3RScan / MultiScan features, are committed. Chat-Scene ships
ScanNet features only, which is why the other two domains needed an extractor.

### Environment

```shell
cp env.sh.example env.sh      # fill in your paths / API key; env.sh is gitignored
bash beacon3d/build_env.sh    # -> $CHATSCENE_ENV (default .venv)
```

Every script in `beacon3d/*.sh` sources `env.sh` from the repo root and reads all paths
from environment variables, with defaults relative to the repo root; the header of each
script lists them. A value given on the command line overrides `env.sh`. The scripts are
plain bash, so on a cluster wrap them in your own job script (e.g. an sbatch file that just
calls `bash beacon3d/run_beacon3d.sh`). The env build needs a GPU node with internet
access. If `$HOME` is full, set `UV_CACHE_DIR` and `TMPDIR`.

## Running

```shell
# 1. inference, all ten val sets (five settings x grounding/qa); VAL_TAG="a#b" for a subset
bash beacon3d/run_beacon3d.sh                    # NGPU=... to limit GPUs

# 2. convert + score; grounding is deterministic, QA calls GPT-4o (~1 h for five sets)
PREDS_DIR=outputs/beacon3d_<ts> bash beacon3d/score.sh
#    -> beacon3d/results/<setting>/{infer_*.json,score_*.txt}; set RESULT_DIR to keep the committed ones

# 3. (only to rebuild beacon3d/features/) Uni3D over GT masks, ~1.5 h on one GPU
bash beacon3d/extract_uni3d.sh                   # needs UNI3D_CKPT, RSCAN_SCAN_DATA, MULTISCAN_SCAN_DATA
```

Step 1 regenerates `annotations/beacon3d_*.json` and `scripts/beacon3d_config.py` from
the committed Beacon3D data before launching, so neither is tracked. The run and the
checkpoint are deterministic; re-scoring QA moves the numbers by well under a point
because of the GPT judge.

## How the evaluation is wired, and what to be careful about

- **Settings.** `scannet` = Chat-Scene's official 3D + 2D features (the leaderboard
  condition). `scannet_myfeat` = our Uni3D extraction + official 2D, isolating the
  extractor. `scannet_no2d` = official Uni3D with the 2D token zeroed, the input condition
  3RScan / MultiScan run under. `3rscan`, `multiscan` = our Uni3D, no 2D.
- **No 2D token means a zero vector, not a missing one.** `val_file_dict`'s second entry
  is the video-feature file; it is `None` for the no-2D settings and Chat-Scene's
  `base_dataset.py` then feeds `torch.zeros(1024)`. The model never saw that at training
  time, which is the whole story behind the caveat above.
- **Scene ids are aliased.** Chat-Scene recovers the scene id of a feature key with
  `'_'.join(key.split('_')[:2])`, so 3RScan scenes are stored as `rscanNNNN_00` and
  MultiScan as `mscanNNNN_00` (N = index in the sorted scene list); object columns are GT
  ids in ascending order. `beacon3d/features/<domain>_scene_map.json` is the map back,
  and `to_beacon3d.py` uses it to turn a predicted `<OBJxxx>` into a GT id. The alias
  depends on the Beacon3D JSONs, so if those change, re-extract; do not only rerun
  `build_annos.py`.
- **Our Uni3D features are not bit-identical to the official ones.** Mean cosine 0.82
  against Chat-Scene's ScanNet file (same preprocessing: centre, scale by max radius,
  colour /255, pad to 64 points, 512 FPS groups × 64 kNN; pure-torch FPS instead of
  `pointnet2_ops`). The `scannet_myfeat` row shows this does not move the scores, but the
  cause of the residual difference was not tracked down.
- **ScanNet axis alignment.** The extractor expects `<scene>.txt` next to the scan; when
  it is absent it recovers the alignment from the official attribute boxes with a Kabsch
  fit (residual 1–4 cm). This only affects the optional ScanNet validation extraction.
- **Prompts.** Grounding: `What's the ID of the object that corresponds to the
  description "..."?`, target `<OBJxxx>.`, every valid target listed. QA: Chat-Scene's
  ScanQA suffix `Answer the question using a single word or phrase.`. Predictions carry
  Beacon3D's item index as `qid` so they are put back in Beacon3D order; a count mismatch
  is an error, never silently skipped.
- **Order of scoring.** The QA evaluator reads `infer_grounding_processed.json` for the
  chain analysis, so grounding is always scored first (`score.sh` does this).

## Citation

This fork evaluates the model of

```BibTeX
@article{huang2024chat,
  title={Chat-scene: Bridging 3d scene and large language models with object identifiers},
  author={Huang, Haifeng and Chen, Yilun and Wang, Zehan and Huang, Rongjie and Xu, Runsen and Wang, Tai and Liu, Luping and Cheng, Xize and Zhao, Yang and Pang, Jiangmiao and others},
  journal={Proceedings of the Advances in Neural Information Processing Systems, Vancouver, BC, Canada},
  year={2024}
}
```

using the data and evaluators of

```BibTeX
@inproceedings{huang2025unveiling,
  title={Unveiling the Mist over 3D Vision-Language Understanding: Object-centric Evaluation with Chain-of-Analysis},
  author={Huang, Jiangyong and Jia, Baoxiong and Wang, Yan and Zhu, Ziyu and Linghu, Xiongkun and Li, Qing and Zhu, Song-Chun and Huang, Siyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## License

Chat-Scene's code is MIT (see `LICENSE`). The vendored Beacon3D evaluators and data under
`beacon3d/Beacon3D/` are Apache-2.0 (`beacon3d/Beacon3D/LICENSE`). The 3D features in
`beacon3d/features/` are derived from the 3RScan, MultiScan (via SceneVerse) and ScanNet
datasets and are subject to their terms.
