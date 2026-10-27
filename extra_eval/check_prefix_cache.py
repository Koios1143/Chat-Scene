"""Check and time the scene-prefix KV cache (config eval_scene_perm / eval_prefix_cache).

Runs one val set, restricted to its first CHECK_SCENES scenes (default 4), three times
with the same model:
  orig    upstream: a fresh object order per question, full prompt per generate()
  perm    one object order per scene, full prompt per generate()
  cache   one object order per scene, scene prefix encoded once, KV cache reused
`cache` should reproduce `perm` (same inputs; only the attention split differs), so this
prints their agreement, the agreement of both with `orig`, and the time of each mode.
Same arguments as tasks/train.py (config file + overrides); launch with torchrun:
  CHECK_SET=scanqa torchrun --nproc_per_node=1 extra_eval/check_prefix_cache.py <config> <overrides>
Writes <output_dir>/check_<set>.json.
"""
import json
import os
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
    for mode, perm, cache in [("orig", False, False), ("perm", True, False), ("cache", True, True)]:
        ds.scene_perm_seed = config.seed if perm else None
        model.config.eval_scene_perm, model.config.eval_prefix_cache = perm, cache
        model._prefix_cache = None
        preds[mode], out[f"sec_{mode}"] = run(model, loader, device)
        print(f"[{name}] {mode}: {out[f'sec_{mode}']:.1f}s for {len(ds.anno)} questions", flush=True)

    def agree(a, b):
        return sum(x == y for x, y in zip(preds[a], preds[b])) / len(preds[a])

    out.update(agree_cache_perm=agree("cache", "perm"), agree_perm_orig=agree("perm", "orig"),
               agree_cache_orig=agree("cache", "orig"),
               diffs=[{"prompt": a["prompt"], "perm": p, "cache": c} for a, p, c in
                      zip((ds.anno[i] for i in sampler.indices), preds["perm"], preds["cache"]) if p != c][:20])
    print(json.dumps({k: v for k, v in out.items() if k != "diffs"}, indent=1))
    json.dump(out, open(os.path.join(config.output_dir, f"check_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(setup_main())
