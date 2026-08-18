"""readnce 钥匙缓存 (07-29): 用 ncev5 冻结 nce_k_head 把 p9 特征投成 L2 归一钥匙.

产物: {"keys": {"r2r:<ep>"|"rxr:<ep>": fp16 [T, 256]}, "meta": {...}}
LoRA 阶段读出检索损失(TSSM_READNCE)用它当负样本池 —— 钥匙空间冻结,
LLM 的查询头只能去 state token 里取信息, 没有别处可抄。
"""
import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "RSSM_v4")
from navigation_tssm_feat import build_mlp

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/cluster/scratch/ruiqzhang/tssm_ncev5_ckpt/checkpoint-final.pt")
ap.add_argument("--feat_r2r", default="/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/siglip8f_p9_lm2/features.pt")
ap.add_argument("--feat_rxr", default="/cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/feat/siglip8f_p9_lm2/features.pt")
ap.add_argument("--out", default="/cluster/scratch/ruiqzhang/tssm_readnce_keys_ncev5.pt")
args = ap.parse_args()

blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
sd = blob["model"]
cfg = blob["config"]
feat_dim = sd["nce_k_head.0.weight"].shape[1]
nce_dim = sd["nce_k_head.6.weight"].shape[0]
k_head = build_mlp(feat_dim, 512, nce_dim, 3)
k_head.load_state_dict({k[len("nce_k_head."):]: v for k, v in sd.items()
                        if k.startswith("nce_k_head.")})
k_head.eval().float()
print(f"k_head rebuilt: {feat_dim} -> {nce_dim} (ckpt step {blob.get('step')})", flush=True)

keys, meta_T = {}, []
with torch.no_grad():
    for dom, path in [("r2r", args.feat_r2r), ("rxr", args.feat_rxr)]:
        feats = torch.load(path, map_location="cpu", weights_only=True)
        print(f"{dom}: {len(feats)} eps loaded", flush=True)
        for i, (ep, f) in enumerate(feats.items()):
            f = f.float()
            if f.dim() == 3:
                f = f.reshape(f.shape[0], -1)
            assert f.shape[1] == feat_dim, f"{dom}:{ep} feat dim {f.shape} != {feat_dim}"
            k = F.normalize(k_head(f), dim=-1)
            keys[f"{dom}:{ep}"] = k.half()
            meta_T.append(f.shape[0])
            if i % 2000 == 0:
                print(f"  {dom} {i}", flush=True)
        del feats

out = {"keys": keys,
       "meta": {"ckpt": args.ckpt, "feat_dim": feat_dim, "nce_dim": nce_dim,
                "n_eps": len(keys), "mean_T": sum(meta_T) / len(meta_T)}}
torch.save(out, args.out)
print(f"saved {len(keys)} eps -> {args.out} (mean T {out['meta']['mean_T']:.1f})", flush=True)
