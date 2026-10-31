# Chat-Scene Benchmark Evaluation

This is a fork of [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene) (NeurIPS 2024)
whose purpose is to **evaluate the released Chat-Scene checkpoint on a range of
benchmarks** under one reproducible setup, so its scores can be compared with other
models on equal footing:

- [Beacon3D](https://github.com/beacon-3d/Beacon3D) (CVPR 2025), an object-centric
  grounding + QA benchmark over ScanNet, 3RScan and MultiScan, in [`beacon3d/`](beacon3d/);
- Chat-Scene's own five benchmarks (ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, SQA3D),
  re-run as a check that the setup reproduces the paper, plus
  [Real-3DQA](https://github.com/xianzhengma/Real-3DQA), in [`extra_eval/`](extra_eval/);
- [XR-QA and XR-SceneCaption](https://huggingface.co/datasets/Hoyard/XR-Scene), the
  large-scene benchmarks of [LSceneLLM](https://github.com/Hoyyyaard/LSceneLLM) (CVPR 2025),
  zero-shot and finetuned, in [`xr_eval/`](xr_eval/).

The upstream model, training and data code are untouched (except a cache-key fix in
`dataset/dataset_val.py`, see *XR-QA / XR-SceneCaption*); the runs that produced the
numbers below are committed in full (`outputs/`, `beacon3d/results/`, `extra_eval/results/`;
finetuned checkpoints are not).

Upstream Chat-Scene's own README is in the git history (`git show 475e106:README.md`).

## Results: Chat-Scene's benchmarks and Real-3DQA

Released checkpoint `ckpt_01_3446.pth`, Mask3D proposals, run of 2026-10-21
(`outputs/extra_eval_20261021_025847/`, scores in `extra_eval/results/`). The *paper*
column is the upstream README (Chat-Scene) and, for Real-3DQA, Table 2/3 of
[Ma et al. 2026](https://arxiv.org/abs/2603.23523), which also evaluated the released weights.

| Benchmark (split) | Metric | Ours | Paper |
| --- | --- | --- | --- |
| ScanRefer (val) | Acc@0.25 / Acc@0.5 | 56.02 / 50.45 | 55.5 / 50.2 |
| Multi3DRefer (val) | F1@0.25 / F1@0.5 | 57.92 / 53.23 | 57.1 / 52.4 |
| Scan2Cap (val) | CIDEr@0.5 / B-4@0.5 | 79.59 / 36.04 | 77.1 / 36.3 |
| ScanQA (val) | CIDEr / B-4 | 87.26 / 14.03 | 87.7 / 14.3 |
| ScanQA (val) | EM / EM-R | 21.35 / 37.13 | – |
| SQA3D (test) | EM / EM-R | 54.28 / 56.98 | 54.6 / – |
| Real-3DQA (debiased test) | EM / EM-R | 17.31 / 22.69 | 17.0 / 22.1 |
| Real-3DQA (rotation) | One / Two / Three / Four / VRS | 44.3 / 6.8 / 1.7 / 0.0 / 13.2 | 43.3 / 7.1 / 1.2 / 0.1 / 12.9 |

Every number is within about one point of the published one (Scan2Cap CIDEr@0.5 +2.5).
Per-type breakdowns (ScanRefer unique/multiple, Multi3DRefer per eval type, SQA3D per
question type, Real-3DQA per rotation) are in `extra_eval/results/<bench>/scores.json`.

## Results: Beacon3D

Beacon3D, GT-mask setting (one object token per GT instance, the protocol the Beacon3D
leaderboard uses). *Case* = per-question accuracy; *Obj* = all questions about an object
answered correctly. QA is judged by **GPT-4o-mini** (`gpt-4o-mini-2024-07-18`), the judge
used for every baseline in this project; the upstream evaluator defaults to
`gpt-4o-2024-11-20`.

| Setting | 3D feature | 2D token | Grounding Case / Obj | QA Case / Obj |
| --- | --- | --- | --- | --- |
| ScanNet (official features) | official Uni3D | official DINOv2 | 62.03 / 41.56 | 49.89 / 9.09 |
| ScanNet, our Uni3D extraction | ours | official DINOv2 | 62.16 / 41.30 | 52.32 / 10.39 |
| ScanNet, **no 2D token** | official Uni3D | zeros | 17.36 / 6.75 | 40.93 / 6.49 |
| 3RScan | ours | zeros | 12.18 / 5.15 | 46.97 / 5.88 |
| MultiScan | ours | zeros | 11.42 / 3.14 | 42.35 / 5.66 |
| 3RScan, **with 2D token** | ours | ours (DINOv2, raw RGB-D) | 45.63 / 27.94 | 46.51 / 2.21 |
| MultiScan, **with 2D token** | ours | ours (DINOv2, raw RGB-D) | 43.68 / 27.67 | 29.93 / 1.26 |

The two *with 2D token* rows (run `outputs/beacon3d_2d_20261021/`) give 3RScan and
MultiScan the ScanNet input condition: their 2D token is extracted from the raw RGB-D
sequences with Chat-Scene's own recipe (`beacon3d/extract_videofeats_gt.py`, below).

Judged by GPT-4o (`gpt-4o-2024-11-20`) the same predictions score QA 50.09 / 9.87,
52.47 / 10.91, 40.93 / 6.75, 47.55 / 5.51 and 41.98 / 5.66 (reports kept as
`score_qa_gpt-4o-2024-11-20.txt` / `infer_qa_processed_gpt-4o-2024-11-20.json`), and the
ScanNet row then reproduces the Beacon3D leaderboard entry for Chat-Scene (61.6 / 42.3,
49.8 / 10.9); compare with the leaderboard only under that judge. Per-item outputs, the official evaluators' full reports (per knowledge
type, grounding→QA chain analysis) and the chain figures are in
[`beacon3d/results/<setting>/`](beacon3d/results/).

### Read this before quoting the 3RScan / MultiScan numbers

Chat-Scene gives every object a 3D token (Uni3D) **and** a 2D token (DINOv2 features
projected from multi-view RGB). The SceneVerse point clouds have no RGB frames, so the
first 3RScan / MultiScan rows run with the 2D token zeroed; the *ScanNet, no 2D token*
control shows that this alone drops grounding from 62 to 17. The *with 2D token* rows
restore it from the raw RGB-D sequences, and grounding recovers to 45.6 / 43.7 — these are
the rows to compare with other models. Two things to know about them:

- **The 2D features are ours, not official.** They follow Chat-Scene's ScanNet script
  exactly (DINOv2-large patch grid, OpenScene's depth-checked projection, convex-hull crop,
  area-weighted fusion) with only the dataset-specific changes listed in the docstring
  of `extract_videofeats_gt.py` (separate colour / depth intrinsics, 3RScan frames rotated
  upright, MultiScan sampled at ScanNet's 1.5 Hz). They could not be validated against the
  official ScanNet file (no ScanNet RGB-D frames were available), but they sit in the same
  space: similar norms (24.5 / 23.8 vs 26.1) and, for ~40% of objects, the nearest
  official ScanNet feature has the same class label (2–5% with shuffled labels).
- **With the 2D token, Chat-Scene answers "yes" to almost every yes/no question** —
  93% on ScanNet with the official features, 94% on 3RScan, 98% on MultiScan, against
  23–26% with the token zeroed. MultiScan's yes/no questions are 70% "no", so its QA drops
  (42.4 → 29.9, yes/no accuracy 0.64 → 0.32) while grounding and the *class* questions
  improve; this is the model's behaviour, not a feature artefact. Without the 2D token the
  QA scores lean on the opposite ("no") prior and the language prior generally (the G✗Q✓
  share is 44% on 3RScan without the token, 33% with it).

## Results: XR-QA / XR-SceneCaption (LSceneLLM)

LSceneLLM's Table 1 reports Chat-Scene# (point clouds + images) zero-shot and finetuned on
its XR-Scene benchmarks without saying how Chat-Scene was run on them. XR scenes are unions of
SceneVerse HM3D regions (3 per QA, 5 per caption), often with more than Chat-Scene's 100
object identifiers, and SceneVerse ships no images. `xr_eval/` makes every unknown an explicit
choice and runs the alternatives:

- **3D token**: Uni3D-g per GT object (Beacon3D's extractor), regions concatenated in
  LSceneLLM's order.
- **2D token**: views rendered from the HM3D v0.2 glb (SceneVerse region → glb is a pure
  translation, recovered by grid search + ICP: 2032 of 2035 regions at ~1 cm median), a
  0.75 m camera grid at 1.5 m height, 8 yaws, fused with DINOv2-large exactly as for
  3RScan / MultiScan. 77% of objects get a 2D feature.
- **>100 objects** (35% of XR-QA val questions, 54% of captions): `trunc` (stock code, first
  100), `big` (drop wall/floor/ceiling, keep the 100 with most points), `rand` (100 at random).
- **QA prompt**: the question as is (`raw`) or with ScanQA's "Answer the question using a
  single word or phrase." (`short`).
- **Scoring**: LSceneLLM's capeval (vendored), keyed by its episode ids, on the raw text
  (LSceneLLM's validation) and after Chat-Scene's own ScanQA normalisation (`clean_answer` +
  PTB tokenizer). `xr_eval/score.py` writes both.

Columns are CIDEr / ROUGE-L / METEOR: the paper labels them C / M / R, but its values only fit
this order. Scores below use Chat-Scene's normalisation, the only one that fits the paper
in every setting (raw-text scoring gives zero-shot XR-QA CIDEr 17–30).

| Setting | XR-QA (val) | XR-SceneCaption (val) |
| --- | --- | --- |
| Paper, zero-shot | 69.55 / 26.63 / 10.06 | 0.01 / 5.94 / 1.52 |
| Ours, zero-shot, `short` + 2D (`trunc` / `big` / `rand`) | 65.51 / 26.78 / 8.72 · 65.64 / 26.93 / 8.74 · 65.88 / 26.96 / 8.77 | 0.02 / 6.07 / 1.50 · 0.01 / 6.17 / 1.54 · 0.02 / 6.09 / 1.48 |
| Ours, zero-shot, `short`, no 2D (`trunc`) | 59.07 / 24.38 / 8.09 | 0.02 / 6.13 / 1.57 |
| Ours, zero-shot, `raw` + 2D (`trunc`) | 56.89 / 24.72 / 8.60 | – |
| Paper, finetuned | 114.10 / 35.93 / 14.32 | 3.58 / 17.49 / 11.59 |
| Ours, finetuned 3 ep, `trunc` | 97.19 / 36.30 / 13.73 | 6.99 / 17.51 / 12.21 |
| Ours, finetuned 3 ep, `big` | 98.28 / 36.52 / 13.99 | 5.74 / 16.86 / 11.87 |
| Ours, finetuned 3 ep, `rand` | 97.49 / 36.31 / 13.91 | 6.55 / 17.49 / 12.42 |

Finetuning starts from the released checkpoint on XR-QA train (`short`) + XR-SceneCaption
train only, with `scripts/run.sh`'s hyper-parameters (3 epochs, lr 5e-6 × batch × GPUs,
cosine) and 256 new tokens at evaluation (captions are ~120 tokens); the paper also used LL3DA
data, which is not included here. Zero-shot matches the paper (XR-QA CIDEr −4); finetuned
ROUGE-L and METEOR match, CIDEr does not (XR-QA −16, captions +2–3). The object selection
barely matters: < 0.5 CIDEr zero-shot; finetuned, `trunc` is 1–3 CIDEr lower than the other
two on XR-QA scenes with >100 objects, against ≤ 1.1 run-to-run spread on the ≤100-object
scenes where the three inputs are identical. Every epoch and the ≤100 / >100 split are in
`outputs/xr_*/*.scores.json`: zero-shot `outputs/xr_eval_20261030_232754/` (all 18
variants), finetuned `outputs/xr_ft_{trunc,big,rand}_*/`.

Running (paths from `env.sh`, see `env.sh.example`):

```bash
bash xr_eval/register_regions.sh; bash xr_eval/extract_uni3d.sh; bash xr_eval/render_videofeats.sh
PREFIX_CACHE=1 bash xr_eval/run_xr_eval.sh        # VAL_TAG=xrqa_val_<trunc|big|rand>_<raw|short>[_no2d]#xrcap_val_<sel>[_no2d]
PREFIX_CACHE=1 SELECT=big bash xr_eval/run_xr_finetune.sh
PREDS_DIR=outputs/xr_<...> bash xr_eval/score.sh
```

The only change to upstream code: `dataset/dataset_val.py` cached scene features by feature
file, so two val sets sharing features but not attributes (the three object selections) got
the first one's objects; the cache key now includes the attribute file.

## Repository layout

| Path | What |
| --- | --- |
| `beacon3d/Beacon3D/` | Vendored copy of the Beacon3D repo at `bda1203` (Apache-2.0): its two evaluators, `utils.py`, and `data/` with the grounding/QA annotations, metadata and GPT system prompt for all three domains. Unmodified; see `VENDORED.md`. |
| `beacon3d/features/` | Our Uni3D-g features over GT masks for the Beacon3D scenes of all three domains (`<domain>_gt_uni3d_feats.pt`), their box attributes, and the scene-id map (below). 5.7 MB. |
| `beacon3d/results/` | Beacon3D infer files, evaluator reports and chain figures for the run in the table. |
| `outputs/beacon3d_20260929_230515/` | The raw Chat-Scene predictions and config of that run. |
| `beacon3d/build_env.sh` | Builds the Chat-Scene runtime as a uv venv (py3.9, torch 2.2.1+cu121, transformers 4.39.3, peft 0.9.0, flash-attn 2.5.6 wheel) plus `open3d` and `openai`. |
| `beacon3d/extract_videofeats_gt.py`, `extract_videofeats.sh` | Chat-Scene's 2D token (DINOv2 multi-view object features) for 3RScan / MultiScan from the raw RGB-D sequences → `beacon3d/features/<domain>_gt_videofeats.pt`. ~5 min per domain on one GPU. |
| `beacon3d/extract_uni3d_gt.py`, `extract_uni3d.sh` | Rebuilds `beacon3d/features/`. Vendors the Uni3D point encoder with a pure-torch farthest-point sampling so `pointnet2_ops` is not needed. |
| `beacon3d/compare_uni3d_official.py` | Cosine between our ScanNet extraction and Chat-Scene's official file (sanity check of the extractor). |
| `beacon3d/build_annos.py` | Beacon3D JSON → Chat-Scene val annotations (`annotations/beacon3d_<setting>_<task>.json`) + `scripts/beacon3d_config.py`. Also builds the two ScanNet controls. Regenerated on every run. |
| `beacon3d/run_beacon3d.sh` | Evaluate-only inference with the official checkpoint (torchrun over all visible GPUs; ~8 min for all ten val sets on 8 GPUs). |
| `beacon3d/to_beacon3d.py`, `score.sh`, `beacon_grounding_shim.py` | Predictions → Beacon3D infer files → official evaluators. The shim only skips knowledge tags with zero cases (3RScan / MultiScan have no *functionality* questions, which makes upstream `evaluate_grounding.py` divide by zero). |
| `extra_eval/build_annos.py` | Links Chat-Scene's HF `annotations/` into `annotations/`, converts Real-3DQA to Chat-Scene's SQA3D format (`annotations/real3dqa_<split>.json`), writes `scripts/extra_eval_config.py`. Regenerated on every run. |
| `extra_eval/run_extra_eval.sh` | Evaluate-only inference on ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, SQA3D test and the five Real-3DQA splits (~1.7 h on 2 RTX PRO 6000). |
| `extra_eval/score.py`, `score.sh` | Scores the predictions with Chat-Scene's own `utils/eval.py` scorers and the official Real-3DQA scripts (vendored, unmodified, in `extra_eval/real3dqa/`). CPU only. |
| `extra_eval/check_prefix_cache.py` | Agreement, time and memory of the `PREFIX_CACHE=1` modes against upstream on a few scenes of one val set (see *Faster evaluation*). |
| `extra_eval/build_env_cu128.sh` | The runtime for Blackwell GPUs: py3.10, torch 2.7.1+cu128, flash-attn 2.8.3 wheel, upstream `requirements.txt` otherwise. |
| `beacon3d/plot_gqa_chain.py` | Grounding→QA chain shares (as in Beacon3D's `eval_chain`) and the Sankey figures in `beacon3d/results/figures/`, from the evaluators' processed files. |
| `xr_eval/register_regions.py`, `.sh` | SceneVerse HM3D region → HM3D v0.2 glb translation (`xr_eval/data/region_offsets.json`, committed). CPU only. |
| `xr_eval/extract_uni3d_regions.py`, `extract_uni3d.sh` | Uni3D-g features of every GT object of every XR region → `xr_eval/features/regions_uni3d.*.pt`. |
| `xr_eval/render_videofeats.py`, `.sh` | Renders the regions from the glb (pyrender, EGL) and fuses DINOv2 features per object → `xr_eval/features/regions_videofeats.*.pt`. ~2.6 h on 2 GPUs. |
| `xr_eval/build_xr.py` | Region features → Chat-Scene scenes, attributes per object selection, annotations (`annotations/xr/`) and `scripts/xr_config.py`. Regenerated on every run. |
| `xr_eval/run_xr_eval.sh`, `run_xr_finetune.sh` | Zero-shot inference / finetuning with per-epoch evaluation. |
| `xr_eval/score.py`, `score.sh`, `capeval/` | LSceneLLM's scorers (vendored, see `VENDORED.md`), raw and Chat-Scene-normalised, with ≤100 / >100-object splits. CPU only. |

## Setup

### External dependencies (not in this repo)

| Needed for | What | Where | Env var (default) |
| --- | --- | --- | --- |
| inference | Chat-Scene checkpoint `ckpt_01_3446.pth` (1.3 GB) | [HF `ZzZZCHS/Chat-Scene`](https://huggingface.co/datasets/ZzZZCHS/Chat-Scene) | `CKPT` (`pretrained/ckpt_01_3446.pth`) |
| inference | Chat-Scene's ScanNet GT-mask files: `scannet_gt_uni3d_feats.pt` (205 MB), `scannet_gt_videofeats.pt` (203 MB), `scannet_val_attributes.pt` | same HF release, `annotations/` | `OFFICIAL_DIR` (`annotations`) |
| Real-3DQA | `debiased_test.jsonl`, `rotation_{0,90,180,270}.jsonl` | [HF `Oliver-Ma/Real-3DQA`](https://huggingface.co/datasets/Oliver-Ma/Real-3DQA) | `REAL3DQA_DIR` |
| Scan2Cap / ScanQA / SQA3D scoring | Java 8+ on `PATH` (pycocoevalcap's tokenizer and METEOR) | e.g. a Temurin JRE | `JAVA_HOME` |
| inference | Vicuna-7B v1.5 | [HF `lmsys/vicuna-7b-v1.5`](https://huggingface.co/lmsys/vicuna-7b-v1.5) | `LLAMA` (`lmsys/vicuna-7b-v1.5`) |
| QA scoring | an OpenAI API key (judge `gpt-4o-mini-2024-07-18`; `QA_MODEL` to change) | | `OPENAI_API_KEY` |
| rebuilding features | Uni3D-g weights `model.pt` | [HF `BAAI/Uni3D`](https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-g/model.pt) | `UNI3D_CKPT` |
| rebuilding features | 3RScan and MultiScan scene point clouds with GT instance ids, in the [SceneVerse](https://scene-verse.github.io/) `scan_data/` layout (`pcd_with_global_alignment/<scene>.pth`, `instance_id_to_label/<scene>.pth`) | SceneVerse release | `RSCAN_SCAN_DATA`, `MULTISCAN_SCAN_DATA` |
| 2D token for 3RScan / MultiScan | raw RGB-D: 3RScan `<scan>/sequence.zip` + `labels.instances.annotated.v2.ply`; MultiScan `<scene>.zip` (video, depth, camera jsonl, mesh) | [3RScan](https://github.com/WaldJohannaU/3RScan), [MultiScan](https://3dlg-hcvc.github.io/multiscan/) (ToS) | `RSCAN_RAW`, `MULTISCAN_RAW` |
| validating the extractor (optional) | raw ScanNet scans (`<scene>_vh_clean_2.ply`, `<scene>.aggregation.json`) with one `<objectId>.npy` GT mask per object | ScanNet | `SCANNET_SCANS` |

Only the first three rows are needed to reproduce the table: the Beacon3D data and
evaluators, and our 3RScan / MultiScan features, are committed. Chat-Scene ships
ScanNet features only, which is why the other two domains needed an extractor.

`deps/` (gitignored) is the place for this project's own copy of every external dependency,
with `env.sh` pointing only there. Use hard links (`cp -al`, `ln`) when a file is already
somewhere else on the same filesystem: they cost no space, and the copy survives when the
other path is deleted (e.g. a shared Hugging Face cache or another project's data dir).
A symlink does not protect anything, so use one only for directories this project owns.
The venv's interpreter is part of this: a uv venv links to uv's managed Python
(`pyvenv.cfg` `home`), so keep a hard-linked copy of that too.

### Environment

```shell
cp env.sh.example env.sh      # fill in your paths / API key; env.sh is gitignored
bash beacon3d/build_env.sh    # -> $CHATSCENE_ENV (default .venv)
# or, on Blackwell GPUs (sm_100/sm_120), which torch 2.2.1 / flash-attn 2.5.6 predate:
bash extra_eval/build_env_cu128.sh
```

The `extra_eval/` numbers above were produced with the cu128 env. torch >= 2.6 loads
checkpoints with `weights_only=True` by default, which rejects the official checkpoint
(it pickles an EasyDict config), so the run scripts set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

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

# 2. convert + score; grounding is deterministic, QA calls the GPT judge (QA_MODEL, default gpt-4o-mini-2024-07-18)
PREDS_DIR=outputs/beacon3d_<ts> bash beacon3d/score.sh
#    -> beacon3d/results/<setting>/{infer_*.json,score_*.txt}; set RESULT_DIR to keep the committed ones

# 3. (only to rebuild beacon3d/features/) Uni3D over GT masks, ~1.5 h on one GPU
bash beacon3d/extract_uni3d.sh                   # needs UNI3D_CKPT, RSCAN_SCAN_DATA, MULTISCAN_SCAN_DATA
#    2D token for 3RScan / MultiScan, ~10 min on one GPU
bash beacon3d/extract_videofeats.sh              # needs RSCAN_RAW, MULTISCAN_RAW, MULTISCAN_SCAN_DATA, ffmpeg

# 4. the *with 2D token* settings (registered by build_annos.py once those files exist)
VAL_TAG="beacon3d_3rscan_2d_grounding#beacon3d_3rscan_2d_qa#beacon3d_multiscan_2d_grounding#beacon3d_multiscan_2d_qa" \
    OUT=outputs/beacon3d_2d bash beacon3d/run_beacon3d.sh
DOMAINS=3rscan_2d,multiscan_2d PREDS_DIR=outputs/beacon3d_2d bash beacon3d/score.sh
python beacon3d/plot_gqa_chain.py                # chain JSONs + figures, incl. *_2d
```

Chat-Scene's benchmarks and Real-3DQA:

```shell
bash extra_eval/run_extra_eval.sh                          # VAL_TAG="scanqa#sqa3d_test" for a subset
PREDS_DIR=outputs/extra_eval_<ts> bash extra_eval/score.sh  # -> extra_eval/results/
```

Notes on `extra_eval/`:

- **Real-3DQA prompt** = Chat-Scene's SQA3D format, `<situation> <question> Answer the
  question using a single word or phrase.`. The rotation splits only rewrite the
  situation text (the viewpoint), so the scene input is the same as for SQA3D. The same
  Mask3D feature files as Chat-Scene's `sqa3d_test` are used.
- **SQA3D is scored on test** (the paper's split). `train.py` saves `sqa3d_test`
  predictions but does not score them, so `score.py` calls `calc_sqa3d_score` on them.
- **Sampler padding.** With N GPUs, `DistributedSampler` repeats up to N-1 items so the
  ranks get equal shares, and `train.py`'s merged prediction file keeps those repeats
  (its in-run scores include them). `score.py` drops rows beyond each item's count in
  the annotation file, and it is an error if any annotation item has no prediction.
- **Empty generations** are replaced by `<empty>` before Real-3DQA scoring; otherwise
  the refined match counts them as correct (none occurred in this run).

Step 1 of the Beacon3D run regenerates `annotations/beacon3d_*.json` and `scripts/beacon3d_config.py` from
the committed Beacon3D data before launching, so neither is tracked. The run and the
checkpoint are deterministic; re-scoring QA moves the numbers by well under a point
because of the GPT judge.

## Faster evaluation: `PREFIX_CACHE=1`

Every question on a scene feeds the LLM the same prefix (system prompt + 100 objects ×
3 tokens) and only the question differs. With `PREFIX_CACHE=1` the run scripts set four
config switches. All of them are off by default, which keeps the upstream behaviour above:

- `eval_scene_perm`: the object order (`assigned_ids`) is drawn once per scene from
  `(seed, scene_id)` instead of per question in the DataLoader workers. The other eval
  randomness (prompt choice, `recover_caption`'s fallback id) is drawn from
  `(seed, question)`. Results do not depend on the GPU count, worker count or item order.
  `SceneShardSampler` gives each GPU whole scenes, items sorted by scene.
- `eval_prefix_cache`: `Chat3D.generate_with_prefix_cache` runs the prefix through the LLM
  once per scene and starts every question's beam search from that KV cache. The cache is
  reused only while the prefix embedding is bit-identical. `prepare_inputs_for_generation`
  in `models/modeling_llama.py` is patched so the first step can take a cache plus
  `inputs_embeds`.
- `eval_gen_batch` (`GEN_BATCH`, default 16): up to this many consecutive questions of a
  scene run as one batched beam search. Each row is `[prefix][padding][question]`; the
  padding is masked and skipped by the position ids, which makes it left padding of the
  question.
- `eval_shared_prefix` (`SHARED_PREFIX`, default 1; flash-attn only): one copy of the prefix
  KV for all rows. Without it, `generate()` copies the prefix into every beam, and every
  step copies it twice more (cache `cat` + beam reorder), so batching stops paying off
  beyond 4 questions. `LlamaFlashAttention2._shared_prefix_attention` attends to the prefix
  through `flash_attn_with_kvcache(cache_batch_idx=0)` and to the row's own keys causally
  (`cache_leftpad` for the padding), then merges the two by log-sum-exp.

```shell
PREFIX_CACHE=1 bash extra_eval/run_extra_eval.sh
PREFIX_CACHE=1 bash beacon3d/run_beacon3d.sh
# PREFIX_CACHE=1 GEN_BATCH=1 SHARED_PREFIX=0: cache only, one question per generate()
```

Full runs on 2 RTX PRO 6000, against the runs in the tables above. The cache-only column is
`outputs/{extra_eval,beacon3d}_prefixcache_20261027_171135/`, the default column
`outputs/{extra_eval,beacon3d}_prefixcache_b16s1_20261027_202255/`:

| | upstream | cache only | `PREFIX_CACHE=1` (batch 16, shared prefix) |
| --- | --- | --- | --- |
| extra_eval inference (6 benchmarks) | 1 h 37 min | 1 h 03 min | 19 min (5.1×) |
| Beacon3D inference (14 val sets) | 34.6 min | 20.9 min | 5.2 min (6.7×) |
| ScanRefer Acc@0.25 / @0.5 | 56.0 / 50.5 | 56.1 / 50.7 | 56.1 / 50.7 |
| Multi3DRefer F1@0.25 / @0.5 | 57.9 / 53.2 | 58.2 / 53.4 | 58.2 / 53.4 |
| Scan2Cap CIDEr@0.5 | 79.6 | 80.0 | 79.7 |
| ScanQA CIDEr | 87.3 | 87.9 | 88.1 |
| SQA3D test EM | 54.3 | 53.9 | 54.2 |
| Real-3DQA EM / EM-R / VRS | 17.3 / 22.7 / 13.2 | 17.3 / 22.8 / 12.6 | 17.2 / 22.6 / 12.6 |
| Beacon3D grounding, case overall (scannet / myfeat / no2d / 3rscan / multiscan / 3rscan_2d / multiscan_2d) | 62.0 / 62.2 / 17.4 / 12.2 / 11.4 / 45.6 / 43.7 | 62.6 / 62.9 / 18.3 / 11.3 / 10.5 / 44.2 / 44.3 | 62.5 / 62.7 / 18.1 / 11.2 / 10.4 / 44.2 / 44.3 |
| Beacon3D QA, case (same order; gpt-4o-mini judge) | 49.9 / 52.3 / 40.9 / 47.0 / 42.4 / 46.5 / 29.9 | 50.2 / 52.5 / 40.5 / 46.2 / 42.3 / 46.8 / 29.6 | 50.2 / 52.5 / 40.3 / 46.1 / 42.6 / 47.2 / 29.7 |

Per benchmark the default is 5.0–6.7× faster (ScanRefer 18.2 → 3.2 min, Multi3DRefer 35.5 →
7.1, Scan2Cap 11.1 → 1.9, ScanQA 12.3 → 1.8). Peak GPU memory at batch 16 is about 25 GB
(17 GB for the cache alone), measured by the check script below.

`extra_eval/check_prefix_cache.py` compares the modes on a few scenes per val set (one GPU;
`CHECK_SET`, `CHECK_SCENES`, `CHECK_MODES`, e.g. `"orig perm cache batch4 shared16"`), with
time and peak memory. Same inputs, different computation: cache vs no cache, batched vs
not, and shared vs copied prefix each give identical outputs for 94–100% of the questions
(ScanRefer and SQA3D 99–100%). The rest are bf16 near-ties in the beam search, which
compound over long outputs (Scan2Cap: 76–84%). Changing from per-question to per-scene
order changes many grounding outputs (only 32–38% are identical), but not the accuracy;
the per-question order already varied this much.

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
- **`evaluate_qa.py` caches.** It silently reuses an existing `infer_qa_processed.json`
  instead of calling the judge again, so `to_beacon3d.py` deletes that file before every
  QA scoring run; without this a re-score with another judge returns the old numbers.
- **Figures.** After re-scoring, `python beacon3d/plot_gqa_chain.py` rebuilds the chain
  JSONs and figures.

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
