"""未见场景·历史检索大考 (08-18 用户令: 保住"历史编码器"论据).

问题: 想象在unseen失效已实证; 那记忆(后验编码+检索)在unseen还成立吗?
协议A(检索): 睁眼滚完整集(正常记忆工作模式), 末位置用模型自带NCE头考 lag 3/5/10
             检索(候选=本集全部帧), seen/unseen 双池同口径对照。
协议B(因果, unseen池): 换史(喂B史+A当前帧)与乱序对照, 证明unseen下读出仍由喂入历史决定。
"""
import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from navigation_tssm_feat import NavigationTSSM, TSSMConfig  # noqa: E402
from data_navila import build_episode_trajectories, split_episodes  # noqa: E402

UNSEEN_GLOB = "/cluster/scratch/ruiqzhang/tssm_unseen_full_sigpub_g*.pt"
SEEN_FEATS = "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/sigpub_p9_lm2/features.pt"
R2R_ANN = "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json"
CKPTS = {"TSSM(ncev5pub)": "/cluster/scratch/ruiqzhang/tssm_ncev5pub.pt",
         "RSSM(rssmpub)": "/cluster/scratch/ruiqzhang/tssm_rssmpub.pt"}
N_MAX = int(os.environ.get("N_MAX", "500"))       # 每池抽样上限(检索)
N_PAIRS = int(os.environ.get("N_PAIRS", "60"))    # 因果对数
T_Q = 20
torch.set_num_threads(8)

print("加载特征池...", flush=True)
unseen = {}
for p in sorted(glob.glob(UNSEEN_GLOB)):
    for k, d in torch.load(p, map_location="cpu", weights_only=False).items():
        unseen[k] = (d["feats"].float(), list(d["actions"]))
sfeats = torch.load(SEEN_FEATS, map_location="cpu", weights_only=True)
eps = build_episode_trajectories(R2R_ANN)
_, te = split_episodes(eps, val_ratio=0.10, seed=0)
seen = {}
for e in te:
    if e.episode_id in sfeats and sfeats[e.episode_id].shape[0] == len(e.actions):
        f = sfeats[e.episode_id]
        seen[e.episode_id] = ((f.reshape(f.shape[0], -1) if f.dim() > 2 else f).float(),
                              list(e.actions))
print(f"unseen {len(unseen)} | seen {len(seen)}", flush=True)


@torch.no_grad()
def run_state(model, frames, acts, upto=None):
    st = model.get_initial_state(1, torch.device("cpu"), dtype=torch.float32)
    T = frames.shape[0] if upto is None else upto
    for t in range(T):
        a = torch.zeros(1, dtype=torch.long) if t == 0 else torch.tensor([acts[t - 1]])
        st = model.inference_step(st, a, frames[t:t + 1])
    return torch.cat([st.h, st.z_flat], dim=-1)[0]


@torch.no_grad()
def query(model, lags, state, acts, t_q, k, cand):
    li = lags.index(k)
    a_emb = model.action_embed(torch.tensor(acts[:t_q], dtype=torch.long))
    a_ctx = a_emb[t_q - k:t_q].mean(0)
    lg = model.nce_lag_emb(torch.tensor([li]))[0]
    q = F.normalize(model.nce_q_head(
        torch.cat([state, lg.to(state.dtype), a_ctx])), dim=-1)
    Kf = F.normalize(model.nce_k_head(cand), dim=-1)
    return int((Kf @ q).argmax())


for name, ck in CKPTS.items():
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = TSSMConfig(**{k: v for k, v in blob["config"].items()
                        if k in TSSMConfig.__dataclass_fields__})
    model = NavigationTSSM(cfg).eval()
    model.load_state_dict(blob["model"])
    LAGS = [int(x) for x in str(cfg.nce_lags).split(",") if x.strip()]
    print(f"\n########## {name} lags={LAGS} ##########", flush=True)

    # ---- 协议A: 双池检索 ----
    for pool_name, pool in [("seen", seen), ("unseen", unseen)]:
        keys = [k for k, (f, a) in pool.items() if f.shape[0] >= max(LAGS) + 6]
        random.seed(0)
        random.shuffle(keys)
        keys = keys[:N_MAX]
        hits = {k: [0, 0] for k in LAGS}
        for kk in keys:
            f, a = pool[kk]
            T = f.shape[0]
            state = run_state(model, f, a)
            tq = T - 1                       # state停在最后处理的位置T-1 (off-by-one修正)
            for lag in LAGS:
                pred = query(model, LAGS, state, a, tq, lag, f)
                hits[lag][0] += int(pred == tq - lag)
                hits[lag][1] += 1
        line = " | ".join(f"lag{k}: {v[0]/max(v[1],1):.3f}" for k, v in hits.items())
        print(f"[检索] {pool_name:6s} n={len(keys)}  {line}  (随机≈{1/np.mean([pool[k][0].shape[0] for k in keys]):.3f})", flush=True)

    # ---- 协议B: unseen 因果(换史/乱序) ----
    ukeys = [k for k, (f, a) in unseen.items() if f.shape[0] >= T_Q + 2]
    random.seed(1)
    random.shuffle(ukeys)
    res = {c: {k: [0, 0] for k in LAGS} for c in ["swap_fed", "swap_orig", "shuf_fed"]}
    for i in range(min(N_PAIRS, len(ukeys) // 2)):
        A, B = unseen[ukeys[2 * i]], unseen[ukeys[2 * i + 1]]
        FA, AA = A; FB, AB = B
        perm = torch.randperm(T_Q)
        fr_sw = torch.cat([FB[:T_Q], FA[T_Q:T_Q + 1]])
        ac_sw = AB[:T_Q] + [AA[T_Q] if T_Q < len(AA) else 0]
        st_sw = run_state(model, fr_sw, ac_sw)
        cand_sw = torch.cat([FB[:T_Q], FA[:T_Q]])
        fr_sh = torch.cat([FA[perm], FA[T_Q:T_Q + 1]])
        st_sh = run_state(model, fr_sh, AA[:T_Q + 1])
        for k in LAGS:
            if k >= T_Q:
                continue
            hit = query(model, LAGS, st_sw, ac_sw, T_Q, k, cand_sw)
            res["swap_fed"][k][0] += int(hit == T_Q - k); res["swap_fed"][k][1] += 1
            res["swap_orig"][k][0] += int(hit == T_Q + (T_Q - k)); res["swap_orig"][k][1] += 1
            hit2 = query(model, LAGS, st_sh, AA[:T_Q + 1], T_Q, k, FA[:T_Q])
            res["shuf_fed"][k][0] += int(hit2 == int(perm[T_Q - k])); res["shuf_fed"][k][1] += 1
    for cname, lbl in [("swap_fed", "换史→命中喂入的B帧"), ("swap_orig", "换史→仍命中A原帧(捷径检验)"),
                       ("shuf_fed", "乱序→命中喂入序位")]:
        line = " | ".join(f"lag{k}: {v[0]/max(v[1],1):.2f}" for k, v in res[cname].items())
        print(f"[因果-unseen] {lbl}: {line}", flush=True)
print("\n判读: unseen检索若与seen同档且换史100/0分离 → 历史编码器跨场景成立, 论据保住。")
