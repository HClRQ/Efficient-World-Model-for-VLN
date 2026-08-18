"""val_unseen 特征缓存生成器 (08-10, 用户令: TSSM rollout 上真未见场景).

habitat 按 GT 动作回放 R2R val_unseen 10 集 → 逐帧 SigLIP lm2p9 特征(与训练缓存
siglip8f_p9_lm2 同口径: navila_8f 塔 fp32 / BILINEAR / hidden_states[-2] / 27²→3×3块均值)
→ 存 {ep: {feats[T,10368], actions[T](语义词表0-9), scene}}。

动作口径: GT 是 habitat 原语(0停/1前25cm/2左15°/3右15°); TSSM 训练吃语义步
(data_navila.ACTION_VOCAB), 故把原语同类连跑打包成 ≤3 的组: 3×前=前75(id3) 等,
帧取组首(决策时刻), 与实弹桥 step(rgb, last_sem_action) 的节奏一致。
"""
import gzip
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

NAVILA = "/cluster/project/rsl/ruiqzhang/codes/NaVILA"
SIGLIP_DIR = os.environ.get("TSSM_ONLINE_SIGLIP", f"{NAVILA}/navila_8f/vision_tower")
OUT = os.environ.get("UNSEEN_FEAT_OUT", "/cluster/scratch/ruiqzhang/tssm_unseen10_feats.pt")
N_TARGET = int(os.environ.get("N_EPS", "10"))
PER_SCENE = int(os.environ.get("PER_SCENE", "3"))
MIN_SEM = int(os.environ.get("MIN_SEM", "16"))   # 语义步下限(burn5+滚≥11)

sys.path.insert(0, f"{NAVILA}/RSSM_v4")


class SigTower:
    def __init__(self, device="cuda"):
        from transformers import SiglipVisionModel, SiglipImageProcessor
        self.sig = SiglipVisionModel.from_pretrained(SIGLIP_DIR).eval().to(device)
        ip = SiglipImageProcessor.from_pretrained(SIGLIP_DIR)
        self.mean = torch.tensor(ip.image_mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(ip.image_std).view(1, 3, 1, 1).to(device)
        self.size = ip.size["height"]
        self.dev = device

    @torch.no_grad()
    def encode(self, rgb_np):
        img = Image.fromarray(np.uint8(rgb_np)).convert("RGB")
        rgb = img.resize((self.size, self.size), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(rgb).copy()).float().permute(2, 0, 1) / 255.0
        arr = ((arr.unsqueeze(0).to(self.dev)) - self.mean) / self.std
        hs = self.sig(pixel_values=arr, output_hidden_states=True).hidden_states[-2]
        feat = (hs.view(1, 27, 27, 1152).view(1, 3, 9, 3, 9, 1152)
                .mean(dim=(2, 4)).reshape(1, 9 * 1152)).float()
        return feat[0].cpu()


def to_semantic(prims):
    """habitat原语串 → [(语义id, [原语...])]; 同类连跑≤3合并."""
    groups, i = [], 0
    while i < len(prims):
        p = prims[i]
        if p == 0:
            groups.append((0, [0])); i += 1; continue
        j = i
        while j < len(prims) and prims[j] == p and j - i < 3:
            j += 1
        base = {1: 0, 2: 3, 3: 6}[p]                 # 前→1..3 左→4..6 右→7..9
        groups.append((base + (j - i), prims[i:j]))
        i = j
    return groups


def main():
    os.chdir(f"{NAVILA}/evaluation")
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
    from habitat_baselines.common.environments import get_env_class

    with gzip.open("data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/"
                   "val_unseen_gt.json.gz", "rt") as f:
        gt = json.load(f)

    config = get_config("vlnce_baselines/config/r2r_baselines/navila.yaml", [])
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    split = config.EVAL.SPLIT
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.DATASET.ROLES = ["guide"]
    config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.NUM_CHUNKS = 1
    config.TASK_CONFIG.DATASET.CHUNK_IDX = 0
    _sc = os.environ.get("SCENES", "").strip()
    if _sc:
        config.TASK_CONFIG.DATASET.CONTENT_SCENES = _sc.split(",")
    config.freeze()
    envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
    tower = SigTower()

    num_eps = sum(envs.number_of_episodes)
    print(f"[cache] candidate episodes: {num_eps}", flush=True)
    cache, scene_count, seen_cnt = {}, {}, 0
    obs = envs.reset()
    while len(cache) < N_TARGET and seen_cnt < num_eps:
        ep = envs.current_episodes()[0]
        seen_cnt += 1
        sc = str(getattr(ep, "scene_id", "?")).split("/")[-1].split(".")[0]
        eid = str(ep.episode_id)
        g = gt.get(eid, {}).get("actions", None)
        groups = to_semantic([int(a) for a in g]) if g else []
        if (g is None or len(groups) < MIN_SEM
                or scene_count.get(sc, 0) >= PER_SCENE):
            if len(cache) < N_TARGET and seen_cnt < num_eps:
                obs = envs.reset_at(0)
            continue
        scene_count[sc] = scene_count.get(sc, 0) + 1
        feats, acts, frames, done = [], [], [], False
        for sem_id, prims in groups:
            feats.append(tower.encode(obs[0]["rgb"]))
            frames.append(np.asarray(Image.fromarray(np.uint8(obs[0]["rgb"]))
                                     .convert("RGB").resize((224, 224))))
            acts.append(sem_id)
            for prim in prims:
                o, r, d, info = envs.step([prim])[0]
                obs = [o]
                done = d
                if done:
                    break
            if done:
                break
        cache[eid] = {"feats": torch.stack(feats).half(),
                      "actions": acts, "scene": sc,
                      "frames": np.stack(frames)}
        print(f"[{len(cache)}/{N_TARGET}] ep{eid} scene={sc} "
              f"T={len(acts)} done={done}", flush=True)
        if len(cache) < N_TARGET and seen_cnt < num_eps:
            obs = envs.reset_at(0)

    torch.save(cache, OUT)
    print(f"saved {len(cache)} eps -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
