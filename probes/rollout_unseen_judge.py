"""seen vs unseen rollout 判决 (08-10): 同协议同代码路径, 双池对拍.

协议 = eval_imagine_dreamer 同款: burn-in 5 步 posterior → 纯 GT 动作开环 imagine
→ decode_feats_raw(state) 对比真值特征。
  - cos: 想象解码 vs 真值帧特征
  - 帧地板: 直接拿 burn-in 最后一帧特征硬扛(不动模型), 想象必须赢它才算"在推演"
  - NN命中: 解码特征在本集全部真帧里最近邻 == 目标帧
两池: unseen = habitat 回放的 R2R val_unseen 10 集(cache_unseen_feats.py 产物);
      seen   = R2R 训练 heldout 同长度过滤同数量(场景见过, 轨迹没见过)。
"""
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from navigation_tssm_feat import NavigationTSSM, TSSMConfig  # noqa: E402
from data_navila import build_episode_trajectories, split_episodes  # noqa: E402

CKPT = os.environ.get("TSSM_CKPT", "/cluster/scratch/ruiqzhang/tssm_ncev5.pt")
UNSEEN = os.environ.get("UNSEEN_FEATS", "/cluster/scratch/ruiqzhang/tssm_unseen10_feats.pt")
R2R_FEAT = os.environ.get("SEEN_FEATS",
    "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/siglip8f_p9_lm2/features.pt")
R2R_ANN = "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json"
BURN = 5
HORIZONS = [1, 3, 5, 8, 12]

blob = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = TSSMConfig(**{k: v for k, v in blob["config"].items()
                    if k in TSSMConfig.__dataclass_fields__})
model = NavigationTSSM(cfg).eval()
model.load_state_dict(blob["model"])


@torch.no_grad()
def rollout_metrics(feats, acts):
    """feats [T,D] fp32, acts [T]. 返回 {h: (cos_model, cos_floor, nn_hit)}."""
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
        sv = torch.cat([st.h, st.z_flat], dim=-1)
        pred = model.decode_feats_raw(sv)[0]
        cm = float(F.cosine_similarity(pred, feats[t], dim=0))
        cf = float(F.cosine_similarity(floor, feats[t], dim=0))
        sims = F.cosine_similarity(pred.unsqueeze(0), feats, dim=1)
        out[h] = (cm, cf, float(int(sims.argmax()) == t))
    return out


def pool_report(name, eps_iter):
    agg = {h: [] for h in HORIZONS}
    n = 0
    for feats, acts in eps_iter:
        m = rollout_metrics(feats.float(), acts)
        n += 1
        for h in HORIZONS:
            if h in m:
                agg[h].append(m[h])
    print(f"\n===== {name} (n={n}) =====")
    print("视距h | 想象cos | 帧地板cos | 想象-地板 | NN命中")
    for h in HORIZONS:
        if not agg[h]:
            continue
        a = np.array(agg[h])
        print(f"{h:4d}  | {a[:,0].mean():.3f}  | {a[:,1].mean():.3f}    | "
              f"{a[:,0].mean()-a[:,1].mean():+.3f}    | {a[:,2].mean():.2f}")


def unseen_iter():
    cache = torch.load(UNSEEN, map_location="cpu", weights_only=False)
    for eid, d in cache.items():
        yield d["feats"], d["actions"]


def seen_iter(n_take, min_len):
    feats = torch.load(R2R_FEAT, map_location="cpu", weights_only=True)
    eps = build_episode_trajectories(R2R_ANN)
    _, te = split_episodes(eps, val_ratio=0.10, seed=0)
    random.seed(0)
    pool = [e for e in te if e.episode_id in feats
            and len(e.actions) >= min_len
            and feats[e.episode_id].shape[0] == len(e.actions)]
    for e in random.sample(pool, min(n_take, len(pool))):
        f = feats[e.episode_id]
        f = f.reshape(f.shape[0], -1) if f.dim() == 3 else f
        yield f, list(e.actions)


cache = torch.load(UNSEEN, map_location="cpu", weights_only=False)
lens = [d["feats"].shape[0] for d in cache.values()]
print(f"unseen池: {len(cache)}集 长度{min(lens)}-{max(lens)} "
      f"场景{sorted(set(d['scene'] for d in cache.values()))}")
pool_report("UNSEEN val_unseen(场景真没见过)", unseen_iter())
pool_report("SEEN   R2R heldout(场景见过,轨迹没见过)", seen_iter(len(cache), min(lens)))
print("\n判读: 若 unseen 的(想象-地板)增益显著低于 seen → 动力学推演依赖场景熟悉度;")
print("      若两者接近 → TSSM 学到的是可迁移的空间动力学, 不是场景背诵。")
