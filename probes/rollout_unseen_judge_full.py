"""seen vs unseen rollout 大判决·绝对数版 (08-18 用户令).

规模: seen = R2R 训练 heldout 全部合格集; unseen = val_unseen 全部合格集(habitat 重放缓存).
口径: burn-in 5 → GT 动作开环; 报**绝对 cos**(想象解码 vs 真值 / 帧地板 vs 真值)与 NN 命中.
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from navigation_tssm_feat import NavigationTSSM, TSSMConfig  # noqa: E402
from data_navila import build_episode_trajectories, split_episodes  # noqa: E402

CKPT = os.environ.get("TSSM_CKPT", "/cluster/scratch/ruiqzhang/tssm_rssmpub.pt")
UNSEEN_GLOB = os.environ.get("UNSEEN_GLOB",
                             "/cluster/scratch/ruiqzhang/tssm_unseen_full_sigpub_g*.pt")
SEEN_FEATS = os.environ.get("SEEN_FEATS",
    "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/sigpub_p9_lm2/features.pt")
R2R_ANN = "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json"
BURN = 5
HORIZONS = [1, 3, 5, 8, 12]
MIN_LEN = BURN + max(HORIZONS) + 1          # 双池同一合格线

blob = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = TSSMConfig(**{k: v for k, v in blob["config"].items()
                    if k in TSSMConfig.__dataclass_fields__})
model = NavigationTSSM(cfg).eval()
model.load_state_dict(blob["model"])
torch.set_num_threads(8)


@torch.no_grad()
def rollout_metrics(feats, acts):
    T = feats.shape[0]
    st = model.get_initial_state(1, torch.device("cpu"), dtype=torch.float32)
    for t in range(BURN):
        a = torch.zeros(1, dtype=torch.long) if t == 0 else torch.tensor([acts[t - 1]])
        st = model.inference_step(st, a, feats[t:t + 1])
    floor = feats[BURN - 1]
    out = {}
    for t in range(BURN, T):
        st = model.imagine_step(st, torch.tensor([acts[t - 1]]))
        h = t - BURN + 1
        if h > max(HORIZONS):
            break
        if h in HORIZONS:
            sv = torch.cat([st.h, st.z_flat], dim=-1)
            pred = model.decode_feats_raw(sv)[0]
            cm = float(F.cosine_similarity(pred, feats[t], dim=0))
            cf = float(F.cosine_similarity(floor, feats[t], dim=0))
            sims = F.cosine_similarity(pred.unsqueeze(0), feats, dim=1)
            out[h] = (cm, cf, float(int(sims.argmax()) == t))
    return out


def pool_stats(eps_iter):
    agg = {h: [] for h in HORIZONS}
    n = 0
    for feats, acts in eps_iter:
        if feats.shape[0] < MIN_LEN:
            continue
        m = rollout_metrics(feats.float(), acts)
        n += 1
        if n % 200 == 0:
            print(f"  ...{n}", flush=True)
        for h in HORIZONS:
            if h in m:
                agg[h].append(m[h])
    return n, {h: np.array(v) for h, v in agg.items() if v}


def unseen_iter():
    for p in sorted(glob.glob(UNSEEN_GLOB)):
        cache = torch.load(p, map_location="cpu", weights_only=False)
        for eid, d in cache.items():
            yield d["feats"], d["actions"]


def seen_iter():
    feats = torch.load(SEEN_FEATS, map_location="cpu", weights_only=True)
    eps = build_episode_trajectories(R2R_ANN)
    _, te = split_episodes(eps, val_ratio=0.10, seed=0)
    for e in te:
        if e.episode_id in feats and feats[e.episode_id].shape[0] == len(e.actions):
            f = feats[e.episode_id]
            yield (f.reshape(f.shape[0], -1) if f.dim() > 2 else f), list(e.actions)


print(f"ckpt={os.path.basename(CKPT)} 合格线: 长度≥{MIN_LEN}", flush=True)
n_u, U = pool_stats(unseen_iter())
print(f"unseen 合格 {n_u} 集", flush=True)
n_s, S = pool_stats(seen_iter())
print(f"seen 合格 {n_s} 集", flush=True)

print(f"\n===== 绝对数对照 (seen n={n_s} | unseen n={n_u}) =====")
print("视距 | seen想象cos | seen地板cos | seen NN | unseen想象cos | unseen地板cos | unseen NN")
for h in HORIZONS:
    s = S.get(h)
    u = U.get(h)
    if s is None or u is None:
        continue
    print(f"{h:3d}  |   {s[:,0].mean():.3f}    |   {s[:,1].mean():.3f}    |  {s[:,2].mean():.2f}  |"
          f"    {u[:,0].mean():.3f}     |    {u[:,1].mean():.3f}     |  {u[:,2].mean():.2f}")
