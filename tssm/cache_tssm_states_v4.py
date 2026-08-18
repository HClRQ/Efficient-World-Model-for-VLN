"""
Pre-cache TSSM v4 hidden states for every (episode, step) in NaVILA's R2R
annotations.

差异 (vs cache_rssm_states.py):
  - 用 NavigationTSSM (TransDreamer 变体, RSSM_v4) 替代 NavigationRSSM。
  - inference_step(state, a_prev, feat_t) 内部把 (z_{t-1}, a_emb) 拼成一个
    transformer token 喂进 history，符合 v4 的训练流程。
  - 输出格式不变：单文件 dict {episode_id: tensor[T, state_dim] fp16}。

输入 (Inputs):
  --tssm_ckpt   : 训练好的 TSSM v4 checkpoint (e.g. tssm_feat_v4_ckpt/checkpoint-final.pt)
  --feat_cache  : 已经预计算好的 SigLIP scene-vector cache (features.pt)
  --annotation  : NaVILA R2R annotations.json
  --out_path    : where to save the state cache

state[t] = posterior state AFTER observing frame t with action a_{t-1}
         = "summary of trajectory up to and including step t"
state[0] uses action_prev = 0 (no previous action).

Usage:
    python RSSM_v4/cache_tssm_states_v4.py \\
        --tssm_ckpt  /cluster/scratch/ruiqzhang/tssm_feat_v4_ckpt/checkpoint-final.pt \\
        --feat_cache /cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/siglip \\
        --annotation /cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json \\
        --out_path   /cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/tssm_states_v4.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_navila import build_episode_trajectories                       # noqa: E402
from navigation_tssm_feat import NavigationTSSM, TSSMConfig, TSSMState   # noqa: E402


@torch.no_grad()
def filter_episode(model: NavigationTSSM, feats: torch.Tensor,
                   actions: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Run posterior filtering over one episode using TSSM v4 inference_step.

    feats   : [T, feat_dim] fp16  — SigLIP scene-vector for each frame
    actions : [T] long            — NaVILA compound action taken at each step

    Returns : [T, model.state_dim] fp16
              state[t] = posterior state AFTER observing frame t with a_{t-1}.
    """
    T = feats.shape[0]
    feats   = feats.to(device).float()                                # [T, D]
    actions = actions.to(device).long()                               # [T]

    state = model.get_initial_state(1, device, dtype=torch.float32)
    states_per_step = []

    for t in range(T):
        a_prev = (torch.zeros(1, dtype=torch.long, device=device)
                  if t == 0 else actions[t - 1:t])
        state = model.inference_step(state, a_prev, feats[t:t + 1])
        states_per_step.append(state.full.squeeze(0).to(torch.float16).cpu())

    return torch.stack(states_per_step, dim=0)                        # [T, state_dim] fp16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tssm_ckpt",  required=True)
    p.add_argument("--feat_cache", required=True,
                   help="directory or file with SigLIP scene-vector features")
    p.add_argument("--annotation", required=True)
    p.add_argument("--out_path",   required=True,
                   help="output .pt file (single dict)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load TSSM v4 ----
    print(f"loading TSSM checkpoint: {args.tssm_ckpt}")
    blob = torch.load(args.tssm_ckpt, map_location="cpu", weights_only=False)
    cfg_dict = blob.get("config", {})
    valid = {k: v for k, v in cfg_dict.items() if k in TSSMConfig.__dataclass_fields__}
    cfg = TSSMConfig(**valid)
    model = NavigationTSSM(cfg).to(device).eval()
    model.load_state_dict(blob["model"])
    state_dim = model.state_dim
    print(f"  ckpt step  : {blob.get('step', '?')}")
    print(f"  feat_dim   : {cfg.feat_dim}")
    print(
        f"  state_dim  : {state_dim}  "
        f"(deter_type={cfg.deter_type}, h={model.deter_dim}, "
        f"z={cfg.stoch}*{cfg.classes})"
    )

    # ---- load SigLIP feature cache ----
    print(f"loading feature cache: {args.feat_cache}")
    if os.path.isfile(args.feat_cache):
        feat_paths = [args.feat_cache]
    else:
        cands = [
            os.path.join(args.feat_cache, "features.pt"),
            *sorted(glob.glob(os.path.join(args.feat_cache, "features_rank*.pt"))),
            os.path.join(args.feat_cache, "features.pt.partial"),
        ]
        feat_paths = [c for c in cands if os.path.exists(c)]
        if not feat_paths:
            raise FileNotFoundError(f"no features file under {args.feat_cache}")
    features: dict = {}
    for p_ in feat_paths:
        d = torch.load(p_, map_location="cpu", weights_only=True)
        features.update(d)
    print(f"  episodes in feature cache: {len(features)}")

    # ---- load episodes ----
    print(f"loading episodes: {args.annotation}")
    episodes = build_episode_trajectories(args.annotation)
    print(f"  episodes parsed: {len(episodes)}")

    # ---- filter and cache ----
    out: dict = {}
    skipped = 0
    t0 = time.time()
    for ep in tqdm(episodes, desc="filtering"):
        if ep.episode_id not in features:
            skipped += 1
            continue
        feats   = features[ep.episode_id]                              # [T, feat_dim] or [T, P, D]
        if feats.dim() == 3:
            # non-pooled patch cache (v5 lm2-p9): flatten [T, P, 1152] -> [T, P*1152]
            feats = feats.reshape(feats.shape[0], -1)
        actions = torch.tensor(ep.actions, dtype=torch.long)
        if feats.shape[0] != len(ep.actions):
            print(f"  size mismatch for ep {ep.episode_id}: "
                  f"feats T={feats.shape[0]} vs ann T={len(ep.actions)}, skipping")
            skipped += 1
            continue
        states = filter_episode(model, feats, actions, device)         # [T, state_dim]
        out[ep.episode_id] = states

    elapsed = time.time() - t0
    print(f"\nfiltered {len(out)} episodes (skipped {skipped}) in {elapsed/60:.1f} min")

    # ---- save ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    print(f"saving cache: {args.out_path}")
    tmp = args.out_path + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, args.out_path)
    sz_mb = os.path.getsize(args.out_path) / (1024 ** 2)
    print(f"  done. file size: {sz_mb:.1f} MB")

    # ---- provenance sidecar (lesson from the kl04 online-ckpt mismatch):
    # record exactly which ckpt + feature cache built this state cache, so any
    # online eval can be checked against it before trusting the pairing.
    import json, hashlib
    with open(args.tssm_ckpt, "rb") as f:
        ckpt_sha = hashlib.sha256(f.read(1 << 24)).hexdigest()[:16]  # first 16MB
    meta = {
        "tssm_ckpt": os.path.abspath(args.tssm_ckpt),
        "tssm_ckpt_sha256_16mb": ckpt_sha,
        "feat_cache": os.path.abspath(args.feat_cache),
        "annotation": os.path.abspath(args.annotation),
        "state_dim": state_dim,
        "feat_dim": cfg.feat_dim,
        "feat_scale": getattr(cfg, "feat_scale", 1.0),
        "episodes": len(out),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.out_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  meta -> {args.out_path}.meta.json")


if __name__ == "__main__":
    main()
