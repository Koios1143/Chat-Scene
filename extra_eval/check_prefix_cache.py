"""Check and time the scene-prefix KV cache (config eval_scene_perm / eval_prefix_cache /
eval_gen_batch).

Runs one val set, restricted to its first CHECK_SCENES scenes (default 4), once per mode
in CHECK_MODES (default "orig perm cache") with the same model:
  orig    upstream: a fresh object order per question, full prompt per generate()
  perm    one object order per scene, full prompt per generate()
  cache   one object order per scene, scene prefix encoded once, KV cache reused
  batchN  as cache, up to N questions of a scene in one batched beam search
  sharedN as batchN, one copy of the prefix KV for all rows (eval_shared_prefix)
`cache` should reproduce `perm` (same inputs; only the attention split differs), and
`batchN` should reproduce `cache` (only padding and batch shape differ). Prints every
mode's agreement with `perm`, `cache`, `orig` (when they ran) and the first mode, its time and peak GPU
memory. Same arguments as tasks/train.py (config file + overrides); launch with torchrun:
  CHECK_SET=scanqa torchrun --nproc_per_node=1 extra_eval/check_prefix_cache.py <config> <overrides>
Writes <output_dir>/check_<set>.json.
"""
import json
import os
import re
import time

import torch
from torch.utils.data import ConcatDataset, DataLoader

from dataset import SceneShardSampler
from dataset.dataset_val import ValDataset, val_collate_fn
from models.chat3d import Chat3D
from tasks.shared_utils import setup_model
from utils.basic_utils import setup_seed
from utils.config_utils import setup_main


def run(model, loader, device):
    preds = []
    torch.cuda.synchronize()
    t0 = time.time()
    for batch in loader:
        for k in batch:
            if type(batch[k]) == torch.Tensor:
                batch[k] = batch[k].to(device)
        with torch.no_grad():
            preds += model(**batch, is_eval=True)
    torch.cuda.synchronize()
    return preds, time.time() - t0


def main(config):
    name = os.environ.get("CHECK_SET", "scanqa")
    n_scenes = int(os.environ.get("CHECK_SCENES", 4))
    setup_seed(config.seed)
    device = torch.device(config.device)
    config.scheduler.num_training_steps = config.scheduler.num_warmup_steps = 0  # as main() with no train set
    _, model, *_ = setup_model(config, model_cls=Chat3D, find_unused_parameters=True)
    model.eval()

    v = config.val_file_dict[name]
    ds = ValDataset(ann_list=v[0] if type(v[0]) == list else v, dataset_name=name, config=config)
    scenes = sorted({a["scene_id"] for a in ds.anno})[:n_scenes]
    ds.anno = [a for a in ds.anno if a["scene_id"] in scenes]
    sampler = SceneShardSampler(ConcatDataset([ds]), 1, 0)  # scene order, as in the real runs
    loader = DataLoader(ds, batch_size=config.batch_size, sampler=sampler, num_workers=2, collate_fn=val_collate_fn)

    out = {"set": name, "scenes": scenes, "n": len(ds.anno)}
    preds = {}
    modes = os.environ.get("CHECK_MODES", "orig perm cache").split()
    for mode in modes:
        perm, cache = mode != "orig", mode not in ("orig", "perm")
        ds.scene_perm_seed = config.seed if perm else None
        model.config.eval_scene_perm, model.config.eval_prefix_cache = perm, cache
        model.config.eval_gen_batch = int(re.sub(r"\D", "", mode)) if mode.startswith(("batch", "shared")) else 1
        model.config.eval_shared_prefix = mode.startswith("shared")
        model._prefix_cache = None
        torch.cuda.reset_peak_memory_stats()
        preds[mode], out[f"sec_{mode}"] = run(model, loader, device)
        out[f"gb_{mode}"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"[{name}] {mode}: {out[f'sec_{mode}']:.1f}s for {len(ds.anno)} questions, "
              f"peak {out[f'gb_{mode}']:.1f} GB", flush=True)

    def agree(a, b):
        return sum(x == y for x, y in zip(preds[a], preds[b])) / len(preds[a])

    items = [ds.anno[i] for i in sampler.indices]
    for mode in modes:
        for ref in ("perm", "cache", "orig", modes[0]):
            if ref in preds and ref != mode and f"agree_{ref}_{mode}" not in out:
                out[f"agree_{mode}_{ref}"] = agree(mode, ref)
    ref = "cache" if "cache" in preds else modes[0]
    out["diffs"] = {mode: [{"prompt": a["prompt"], ref: r, mode: m} for a, r, m in
                           zip(items, preds[ref], preds[mode]) if r != m][:20]
                    for mode in modes if mode != ref and ref in preds}
    print(json.dumps({k: v for k, v in out.items() if k != "diffs"}, indent=1))
    json.dump(out, open(os.path.join(config.output_dir, f"check_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(setup_main())
