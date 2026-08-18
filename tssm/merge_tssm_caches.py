"""
============================================================================
 Script 4 / 5  — merge R2R + RxR TSSM state caches into one dict
----------------------------------------------------------------------------
 做什么: 把 tssm_states_v4.pt (R2R) 和 tssm_states_rxr.pt (RxR) 合并成
         单个 dict[episode_id_str -> Tensor[T, 1536] fp16]，让
         RSSMMemoryAdapter 一个 cache_path 就能 cover 两个数据集。

 上游依赖:
   - /cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/tssm_states_v4.pt  (已有)
   - /cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/tssm_states_v4.pt  (cache_tssm_states_rxr.sh 跑完后才有)

 下游使用:
   - sft_tssm_v4_r2r_rxr.sh 通过 --rssm_v4_cache_path 指向输出文件

 防呆:
   1) 检查 episode_id 是否在两个 cache 里冲突 (R2R 用整数 id, RxR 也用整数,
      mp3d 数据集不同 split 通常 id 不重叠, 但保险起见会显式报错)
   2) 检查 state_dim 一致 (都是 h_dim + stoch*classes = 1536)

 用法:
   python RSSM_v4/merge_tssm_caches.py \\
       --r2r  /cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/tssm_states_v4.pt \\
       --rxr  /cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/tssm_states_v4.pt \\
       --out  /cluster/scratch/ruiqzhang/NaVILA-Dataset/tssm_states_v4_r2r_rxr.pt
============================================================================
"""
from __future__ import annotations

import argparse
import os

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r2r", required=True, help="R2R TSSM state cache .pt")
    p.add_argument("--rxr", required=True, help="RxR TSSM state cache .pt")
    p.add_argument("--out", required=True, help="merged output .pt")
    args = p.parse_args()

    # ---- load both ----
    print(f"[load] R2R: {args.r2r}")
    r2r = torch.load(args.r2r, map_location="cpu", weights_only=True)
    print(f"  episodes: {len(r2r)}")

    print(f"[load] RxR: {args.rxr}")
    rxr = torch.load(args.rxr, map_location="cpu", weights_only=True)
    print(f"  episodes: {len(rxr)}")

    # ---- sanity checks ----
    sample_r2r = next(iter(r2r.values()))
    sample_rxr = next(iter(rxr.values()))
    if sample_r2r.shape[-1] != sample_rxr.shape[-1]:
        raise RuntimeError(
            f"state_dim mismatch: R2R={sample_r2r.shape[-1]} vs "
            f"RxR={sample_rxr.shape[-1]}. They must come from the same TSSM."
        )
    print(f"  state_dim: {sample_r2r.shape[-1]} (matches)")

    # R2R and RxR both use bare integer episode IDs starting from 1, so we
    # always prefix to make them disjoint (matches the lookup-side prefix in
    # llava/data/dataset.py).
    r2r_prefixed = {f"r2r:{k}": v for k, v in r2r.items()}
    rxr_prefixed = {f"rxr:{k}": v for k, v in rxr.items()}
    assert not (set(r2r_prefixed) & set(rxr_prefixed)), \
        "post-prefix overlap is impossible; this assert is a sanity guard"

    # ---- merge ----
    merged = dict(r2r_prefixed)
    merged.update(rxr_prefixed)
    print(f"[merge] {len(r2r)} r2r:* + {len(rxr)} rxr:* = {len(merged)} total")

    # ---- save (atomic via .tmp rename) ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    torch.save(merged, tmp)
    os.replace(tmp, args.out)
    sz_mb = os.path.getsize(args.out) / (1024 ** 2)
    print(f"[save] {args.out}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
