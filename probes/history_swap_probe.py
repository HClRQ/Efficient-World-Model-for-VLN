"""因果干预探针 (08-08): 证明检索读出由 state 携带的历史决定, 而非当前观测捷径.

三条件, 同一批 heldout 集对(A,B), 同一查询时刻 t, 同一读出头(ncev5 自带 NCE 头):
  1. true    : 正常喂 A[0:t+1]                    → 应检索到 A[t-k] (基线, 候选翻倍口径)
  2. swap    : 喂 B[0:t] + 当前帧 A[t]            → 若检索命中 B[t-k] 而非 A[t-k],
               则读出跟着"喂进去的历史"走 —— 历史内容真在 state 里, 与当前帧无关
  3. shuffle : 喂 A 的历史乱序 perm + 当前帧 A[t] → 若命中"喂入序位 t-k"的帧 A[perm[t-k]]
               而非原序 A[t-k], 则 state 存的是**有序**历史, 不是词袋

候选池: true/swap 用 [喂入历史 t 帧 ‖ 反事实历史 t 帧] 共 2t 帧 (随机=1/2t);
shuffle 用 A 的 t 帧历史 (随机=1/t)。
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

CKPT = "/cluster/scratch/ruiqzhang/tssm_ncev5.pt"
FEAT = "/cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/feat/siglip8f_p9_lm2/features.pt"
ANN = "/cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/annotations.json"
T_Q = 24          # 查询时刻(历史长度)
N_PAIRS = 40

blob = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = TSSMConfig(**{k: v for k, v in blob["config"].items()
                    if k in TSSMConfig.__dataclass_fields__})
model = NavigationTSSM(cfg).eval()
model.load_state_dict(blob["model"])
LAGS = [int(x) for x in str(cfg.nce_lags).split(",") if x.strip()]
print(f"ckpt={os.path.basename(CKPT)} lags={LAGS} t_query={T_Q}", flush=True)

feats = torch.load(FEAT, map_location="cpu", weights_only=True)
eps = build_episode_trajectories(ANN)
_, te = split_episodes(eps, val_ratio=0.10, seed=0)
random.seed(0)
pool = [e for e in te if e.episode_id in feats and len(e.actions) >= T_Q + 2
        and feats[e.episode_id].shape[0] == len(e.actions)]
random.shuffle(pool)
pairs = [(pool[2 * i], pool[2 * i + 1]) for i in range(N_PAIRS)]


def flat(f):
    return (f.reshape(f.shape[0], -1) if f.dim() == 3 else f).float()


@torch.no_grad()
def run_state(frames, acts):
    """逐步喂入, 返回末位置 state 向量 [3072] (h‖z, 与训练 forward 同序)."""
    st = model.get_initial_state(1, torch.device("cpu"), dtype=torch.float32)
    for t in range(frames.shape[0]):
        a_prev = torch.zeros(1, dtype=torch.long) if t == 0 else acts[t - 1:t]
        st = model.inference_step(st, a_prev, frames[t:t + 1])
    return torch.cat([st.h, st.z_flat], dim=-1)[0]


@torch.no_grad()
def query(state, acts, k, cand_frames):
    """用模型自己的 NCE 头查 lag=k, 返回候选池上的 argmax 下标."""
    li = LAGS.index(k)
    a_emb = model.action_embed(acts)                       # [T,64]
    a_ctx = a_emb[T_Q - k:T_Q].mean(0)                     # 缺口内动作均值(同训练口径)
    lg = model.nce_lag_emb(torch.tensor([li]))[0]
    q = F.normalize(model.nce_q_head(
        torch.cat([state, lg.to(state.dtype), a_ctx])), dim=-1)
    Kf = F.normalize(model.nce_k_head(cand_frames), dim=-1)
    return int((Kf @ q).argmax())


res = {c: {k: [0, 0] for k in LAGS} for c in
       ["true", "swap_fed", "swap_orig", "shuf_fed", "shuf_orig"]}
for A, B in pairs:
    FA, FB = flat(feats[A.episode_id]), flat(feats[B.episode_id])
    AA = torch.tensor(A.actions, dtype=torch.long)
    AB = torch.tensor(B.actions, dtype=torch.long)
    perm = torch.randperm(T_Q)

    # 三种喂法: (帧序列[T_Q+1, D], 动作序列)
    feeds = {
        "true": (torch.cat([FA[:T_Q], FA[T_Q:T_Q + 1]]), AA[:T_Q + 1]),
        "swap": (torch.cat([FB[:T_Q], FA[T_Q:T_Q + 1]]),
                 torch.cat([AB[:T_Q], AA[T_Q:T_Q + 1]])),
        "shuf": (torch.cat([FA[perm], FA[T_Q:T_Q + 1]]), AA[:T_Q + 1]),
    }
    cands = {
        "true": torch.cat([FA[:T_Q], FB[:T_Q]]),   # [喂入 ‖ 反事实]
        "swap": torch.cat([FB[:T_Q], FA[:T_Q]]),
        "shuf": FA[:T_Q],
    }
    for cond, (fr, ac) in feeds.items():
        state = run_state(fr, ac)
        for k in LAGS:
            hit = query(state, ac, k, cands[cond])
            if cond == "true":
                res["true"][k][0] += int(hit == T_Q - k)
                res["true"][k][1] += 1
            elif cond == "swap":
                res["swap_fed"][k][0] += int(hit == T_Q - k)          # B 的帧
                res["swap_orig"][k][0] += int(hit == T_Q + (T_Q - k))  # A 的原帧
                res["swap_fed"][k][1] += 1
                res["swap_orig"][k][1] += 1
            else:
                res["shuf_fed"][k][0] += int(hit == int(perm[T_Q - k]))  # 喂入序位
                res["shuf_orig"][k][0] += int(hit == T_Q - k)            # 原时序
                res["shuf_fed"][k][1] += 1
                res["shuf_orig"][k][1] += 1

print(f"\n===== 因果干预: {N_PAIRS} 对集, 查询位 t={T_Q}, 候选=双池2t/单池t =====")
print(f"{'条件':28s} | " + " | ".join(f"lag{k:2d}" for k in LAGS) + " | 随机水平")
rows = [
    ("true    正常喂A→检索A[t-k]", "true", 1 / (2 * T_Q)),
    ("swap    换史喂B→命中B[t-k]", "swap_fed", 1 / (2 * T_Q)),
    ("swap    换史喂B→仍命中A原帧", "swap_orig", 1 / (2 * T_Q)),
    ("shuffle 乱序喂→命中喂入序位", "shuf_fed", 1 / T_Q),
    ("shuffle 乱序喂→命中原时序帧", "shuf_orig", 1 / T_Q),
]
for label, key, chance in rows:
    accs = [res[key][k][0] / max(res[key][k][1], 1) for k in LAGS]
    print(f"{label:28s} | " + " | ".join(f"{a:.2f} " for a in accs)
          + f" | {chance:.3f}")
print("\n判读: swap_fed 高 + swap_orig 低 → 读出由喂入历史(经 state)决定, 当前帧无捷径;")
print("      shuf_fed 高 + shuf_orig 低 → state 存的是有序历史, 非词袋。")
